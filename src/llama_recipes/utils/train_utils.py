# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import random
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import contextlib


import torch
import torch.distributed as dist
from torch.cuda import nccl
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json


from llama_recipes.model_checkpointing import save_fsdp_model_checkpoint_full, save_model_and_optimizer_sharded, save_optimizer_checkpoint, save_peft_checkpoint, save_model_checkpoint, remove_old_checkpoints
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from llama_recipes.utils.flop_utils import FlopMeasure
def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

@contextlib.contextmanager
def profile(cfg, local_rank=None):
    use_profiler: bool = cfg.use_profiler
    use_flop_counter: bool = cfg.flop_counter
    if use_flop_counter and use_profiler:
        raise ValueError("Cannot use both profiler and flop counter")
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            raise ValueError(f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage, {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        print(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                cfg.profiler_dir
            ),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    elif use_flop_counter:
        if cfg.max_train_step > 0 and cfg.max_train_step <= cfg.flop_counter_start:
            raise ValueError(f"flop counter requires at least {cfg.flop_counter_start + 1} train steps, please increase the max_train_step, current max_train_step {cfg.max_train_step}")
        with FlopMeasure(rank=local_rank,warmup_step=cfg.flop_counter_start) as flop_counter:
            yield flop_counter
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


def train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    optimizer,
    lr_scheduler,
    gradient_accumulation_steps,
    train_config,
    fsdp_config=None,
    local_rank=None,
    rank=None,
    wandb_run=None,
    checkpoint_interval=500,
    max_checkpoints_to_keep=2,
):
    """
    Trains the model on the given dataloader
    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        eval_dataloader: The dataloader containing the eval data
        tokenizer: The tokenizer used to decode predictions
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing an update
        train_config: The training configuration
        fsdp_config: The FSDP configuration (optional)
        local_rank: The rank of the current node in a distributed setting
        rank: The global rank in a distributed setting
        wandb_run: The Weights & Biases run object
        checkpoint_interval: The interval (in steps) at which to save checkpoints
        max_checkpoints_to_keep: The maximum number of checkpoints to keep
    Returns:
        results: A dictionary containing average training and validation perplexity and loss
    """
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])


    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []
        val_step_pred = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        step = 0  # Initialize the step counter
        print(f"Starting epoch {epoch}/{train_config.num_epochs}")
        print(f"train_config.max_train_step: {train_config.max_train_step}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    for key in batch.keys():
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:
                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            elif torch.cuda.is_available():
                                batch[key] = batch[key].to('cuda:0')
                    with autocast():
                        loss = model(**batch).loss
                    total_loss += loss.detach().float()
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    

                    checkpoint_start_time = time.perf_counter()
                    early_stopping = False
                    if total_train_steps % checkpoint_interval == 0:
                        should_save_model = train_config.save_model
                        
                        if train_config.run_validation:
                            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity, eval_preds = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
                            if train_config.save_metrics:
                                val_step_loss.extend(temp_val_loss)
                                val_step_perplexity.extend(temp_step_perplexity)
                                val_step_pred.extend(eval_preds)
                            should_save_model = train_config.save_model and eval_epoch_loss < best_val_loss
                            
                        if should_save_model:
                            save_checkpoint_with_memory_management(
                                model, optimizer, train_config, fsdp_config, rank, epoch, step, max_checkpoints_to_keep
                            )
                                
                        if train_config.run_validation:
                            if eval_epoch_loss < best_val_loss:
                                best_val_loss = eval_epoch_loss
                                if train_config.enable_fsdp:
                                    if rank==0:
                                        print(f"best eval loss on epoch {epoch+1} step {step+1} is {best_val_loss}")
                                else:
                                        print(f"best eval loss on epoch {epoch+1} step {step+1} is {best_val_loss}")
                            val_loss.append(float(best_val_loss))
                            val_prep.append(float(eval_ppl))
                            model.train()

                        # Saving the results every epoch to plot later
                        if train_config.save_metrics:
                            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep, val_step_pred)
                            
                                                
                        if train_config.run_validation and eval_epoch_loss > best_val_loss:
                            early_stopping = True

                    if train_config.enable_fsdp:
                        dist.barrier()

                        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                        checkpoint_times.append(checkpoint_end_time)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if isinstance(profile_context, FlopMeasure):
                        if train_config.flop_counter and profile_context.is_done():
                            TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank == 0:
                            gradients = [p.grad.data.norm(2) for p in model.parameters() if p.grad is not None]
                            if gradients:  # Check if there are any gradients
                                gradient_norm = torch.norm(torch.stack(gradients)).item()
                            else:
                                gradient_norm = 0.0  # Handle case where there are no gradients
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                                'train/perplexity': float(torch.exp(loss.detach().float())),  # Logging perplexity
                                'train/lr': lr_scheduler.get_last_lr()[0] if lr_scheduler else None,  # Current learning rate
                                'train/grad_norm': torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold),  # Gradient norm
                                'train/time': time.perf_counter() - epoch_start_time,  # Time taken for the current training step
                                'train/gradient_norm': gradient_norm  # Log gradients
                            })


                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep, val_step_pred)

                    if early_stopping:
                        # Early stopping
                        if train_config.enable_fsdp:
                            if rank==0:
                                print(f"Early stopping at epoch {epoch+1} step {step+1}")
                        else:
                            print(f"Early stopping at epoch {epoch+1} step {step+1}")
                        break
                    
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()
        should_save_model = train_config.save_model
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity, eval_preds = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run)
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)
            should_save_model = train_config.save_model and eval_epoch_loss < best_val_loss
        
        checkpoint_start_time = time.perf_counter()
        if should_save_model:
            save_checkpoint_with_memory_management(
                model, optimizer, train_config, fsdp_config, rank, epoch, step, max_checkpoints_to_keep
            )
        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)

        if train_config.run_validation:
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank==0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep, val_step_pred)

    # Initialize t_flops
    TFlops = None 

    # Calculate average times and losses
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)

    # Validation metrics if applicable
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    # Store average training metrics
    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss

    # Store validation metrics if applicable
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss

    # Store average times
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    # Save metrics filename if required
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename

    # Profile and calculate t_flops if applicable
    if train_config.use_profiler or train_config.flop_counter:
        profile_context.step()
        if isinstance(profile_context, FlopMeasure):
            if train_config.flop_counter and profile_context.is_done():
                TFlops = profile_context.get_flops_per_sec() / 1e12

    # Only add t_flops to results if it has been calculated
    if train_config.flop_counter and TFlops is not None:
        results["model_tflops"] = TFlops

    # Save training parameters for reference
    if train_config.enable_fsdp and not train_config.use_peft and rank == 0:
        clear_gpu_cache(rank)
        save_train_params(train_config, fsdp_config, rank, epoch=epoch, step=step)

    return results

def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer, wandb_run):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    
    # Create a clean cache before starting evaluation
    clear_gpu_cache(local_rank)
    
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank==0:
                    print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                break
            for key in batch.keys():
                if train_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                else:
                    if is_xpu_available():
                        batch[key] = batch[key].to('xpu:0')
                    else:
                        batch[key] = batch[key].to('cuda:0')
                        
            # Get input lengths before attention mask is modified
            input_lengths = batch['attention_mask'].sum(dim=1).tolist()
            
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()
            # Decode predictions and add to evaluation predictions list
            preds = torch.argmax(outputs.logits, -1)
            # eval_preds.extend(
            #     tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            # )
            for pred, input_length in zip(preds, input_lengths):
                # Slice prediction to only include generated tokens
                generated_tokens = pred[input_length-1:]  # -1 because we want to start from the last input token
                decoded_pred = tokenizer.decode(generated_tokens.detach().cpu().numpy(), skip_special_tokens=True)
                eval_preds.append(decoded_pred)
            # Periodic cache clearing during validation
            if step % 10 == 0:  # Adjust frequency as needed
                clear_gpu_cache(local_rank)
                
    print(random.sample(eval_preds, 1))

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")
        
    # Clear cache after validation
    clear_gpu_cache(local_rank)

    if wandb_run:
        wandb_run.log({
                        'eval/perplexity': eval_ppl,
                        'eval/loss': eval_epoch_loss,
                    }, commit=False)

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity, eval_preds


def save_checkpoint_with_memory_management(
    model,
    optimizer,
    train_config,
    fsdp_config,
    rank,
    epoch,
    step,
    max_checkpoints_to_keep
):
    """
    Handles checkpoint saving with proper memory management
    """
    try:
        # Synchronize before saving
        if train_config.enable_fsdp:
            dist.barrier()
            
        # Clear GPU cache before any checkpoint operation
        clear_gpu_cache(rank)
        
        if train_config.use_peft:
            if train_config.enable_fsdp:
                if rank == 0:
                    print("We are about to save the PEFT modules")
            else:
                print("We are about to save the PEFT modules")
            save_peft_checkpoint(model, train_config.output_dir, epoch=epoch, step=step)
            clear_gpu_cache(rank)  # Clear cache after PEFT save
            
        else:
            if not train_config.enable_fsdp:
                save_model_checkpoint(model, train_config.output_dir, epoch=epoch, step=step)
                clear_gpu_cache(rank)
                
            elif fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                if rank == 0:
                    print("Saving FSDP model checkpoint using FULL_STATE_DICT")
                
                # Save model state
                save_fsdp_model_checkpoint_full(
                    model, optimizer, rank, train_config, epoch=epoch, step=step
                )
                clear_gpu_cache(rank)  # Clear cache after model save
                
                # Save optimizer state if needed
                if train_config.save_optimizer:
                    if rank == 0:
                        print("Saving FSDP optimizer state")
                    save_optimizer_checkpoint(
                        model, optimizer, rank, train_config, epoch=epoch, step=step
                    )
                    clear_gpu_cache(rank)  # Clear cache after optimizer save
                    
            elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                if rank == 0:
                    print("Saving FSDP checkpoints using SHARDED_STATE_DICT")
                    
                if train_config.save_optimizer:
                    save_model_and_optimizer_sharded(
                        model, rank, train_config, optim=optimizer, epoch=epoch, step=step
                    )
                else:
                    save_model_and_optimizer_sharded(
                        model, rank, train_config, epoch=epoch, step=step
                    )
                clear_gpu_cache(rank)
        
        # Save training parameters if needed
        if train_config.enable_fsdp and not train_config.use_peft and rank == 0:
            save_train_params(train_config, fsdp_config, rank, epoch=epoch, step=step)
            clear_gpu_cache(rank)
        
        # Remove old checkpoints if needed
        if max_checkpoints_to_keep is not None:
            remove_old_checkpoints(train_config.output_dir, max_checkpoints_to_keep)
        
        # Final synchronization
        if train_config.enable_fsdp:
            dist.barrier()
            
    except Exception as e:
        print(f"Error during checkpoint saving: {str(e)}")
        raise e
    finally:
        # Always clear cache at the end
        clear_gpu_cache(rank)


def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                print(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only available in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    if is_xpu_available():
        torch.xpu_empty_cache()
    else:
        torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")




def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""


    verify_bfloat_support = ((
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and torch.version.cuda >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    ) or
    (is_xpu_available()))


    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                print(f"FP16 enabled")
        else:
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank, epoch=1, step=0):
    """
    Saves the train_config and FSDP config into a train_params.yaml.
    This will be used by the converter script in the inference folder to fetch the HF model name or path.
    It also serves as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file.
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}

    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}

    # Construct the folder name using properties of the train_config object
    folder_name = (
        train_config.dist_checkpoint_root_folder
        + "/"
        + train_config.dist_checkpoint_folder
        + "-"
        + train_config.model_name
    )
    save_dir = Path.cwd() / folder_name / f"epoch_{epoch}-step_{step}"  # Create subdirectory
    save_dir.mkdir(parents=True, exist_ok=True)

    file_name = save_dir / "train_params.yaml"

    # Write the YAML dictionary to the file
    with open(file_name, 'w') as f:
        yaml.dump(train_params_dict, f, default_flow_style=False, indent=4)

    if rank == 0:
        print(f"Training params are saved in {file_name}")

def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl, val_step_pred):
    output_dir = os.path.dirname(output_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl,
        "val_step_prediction": val_step_pred
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
