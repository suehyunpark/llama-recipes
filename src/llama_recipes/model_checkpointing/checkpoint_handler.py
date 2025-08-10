# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from pathlib import Path
from datetime import datetime
import torch
import time
import os

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

from torch.distributed._shard.checkpoint import (
    FileSystemReader,
    FileSystemWriter,
    save_state_dict,
    load_state_dict,
)
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)


from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed._shard.checkpoint as dist_cp
import torch.distributed as dist

def get_date_of_run():
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run

# create singleton saving policies to avoid making over and over
fullstate_save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

def load_model_sharded(model, rank, cfg):
    """
    Load the latest sharded model checkpoint from the checkpoint directory,
    determined by the newest subdirectory based on modification time.
    """

    checkpoint_dir = Path(cfg.checkpoint_folder)

    if not checkpoint_dir.exists():
        if rank == 0:
            print(f"No sharded_state_dict checkpoint directory found...skipping")
        return

    # Get all subdirectories and sort them by modification time (newest first)
    subdirectories = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    
    # Check if any subdirectories exist to avoid errors when sorting
    if not subdirectories:  
        if rank == 0:
            print(f"No checkpoint subdirectories found in {checkpoint_dir}. Returning...")
        return

    subdirectories.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    latest_checkpoint_dir = subdirectories[0]  # Get the newest subdirectory

    if rank == 0:
         print(f"loading model from model path: {latest_checkpoint_dir} ")
    
    reader = FileSystemReader(latest_checkpoint_dir)

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        checkpoint = {"model": model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
      
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            print(f"checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        model.load_state_dict(checkpoint["model"]) 

    if rank == 0:
        print(f"Sharded state checkpoint loaded from {latest_checkpoint_dir}")

def save_model_and_optimizer_sharded(model, rank, cfg, optim=None, epoch=1, step=0):
    """save model and optimizer via sharded_state_dict to save_dir"""
    
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )

    save_dir = Path.cwd() / folder_name
    save_dir = save_dir / f"epoch_{epoch}-step_{step}"
    
    # Create the subdirectory here to ensure it exists
    save_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"Saving model to {save_dir}")

    distributed_writer = dist_cp.FileSystemWriter(
        save_dir,
    )
    t0 = time.perf_counter()

    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        
        state_dict = {"model": model.state_dict()}
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),
            
        )
    dist.barrier()
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {save_dir}")
        print(
            f"Checkpoint Time = {t1-t0:.4f}\n"
        )

def save_fsdp_model_checkpoint_full(
    model,
    optimizer,
    rank,
    cfg,
    epoch=1,
    step=0,
):
    """Saving model via rank0 CPU streaming and full_state_dict"""

    # Create the directory for saving
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )
    save_dir = Path.cwd() / folder_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"--> Saving model...")

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
        cpu_state = model.state_dict()
        print(f"saving process: rank {rank} done with model state_dict\n")

        # Save the model state
        save_name = f"{cfg.model_name.replace('/', '--')}-epoch_{epoch}-step_{step}.pt"
        save_full_path = str(save_dir) + "/" + save_name
        torch.save(cpu_state, save_full_path)

        # Save the optimizer state if provided
        if optimizer is not None:
            optimizer_state = optimizer.state_dict()
            optimizer_save_name = f"{cfg.model_name.replace('/', '--')}-optimizer-epoch_{epoch}-step_{step}.pt"
            optimizer_save_path = str(save_dir) + "/" + optimizer_save_name
            
            torch.save(optimizer_state, optimizer_save_path)
            print(f"Optimizer checkpoint saved for epoch {epoch} at {optimizer_save_path}\n")

    dist.barrier()

def load_model_checkpoint(model, rank, cfg):
    """
    Load the latest model checkpoint from the checkpoint directory,
    determined by the newest subdirectory based on modification time.
    """
    
    if rank != 0:
        return

    checkpoint_dir = Path(cfg.checkpoint_folder)

    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {checkpoint_dir} not found. Returning...")
        return

    # Get all subdirectories and sort them by modification time (newest first)
    subdirectories = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    subdirectories.sort(key=lambda d: d.stat().st_mtime, reverse=True)  # Newest first

    if not subdirectories:
        print(f"No checkpoint subdirectories found in {checkpoint_dir}. Returning...")
        return

    latest_checkpoint_dir = subdirectories[0]  # Get the newest subdirectory

    # Construct the full path to the model checkpoint file
    checkpoint_model_filename = f"{cfg.model_name.replace('/', '--')}.pt" 
    full_state_dict_model_path = latest_checkpoint_dir / checkpoint_model_filename

    # Is the checkpoint present?
    if not full_state_dict_model_path.is_file():
        print(f"model checkpoint {full_state_dict_model_path} not present. Returning...")
        return

    # Load the checkpoint
    model_checkpoint = torch.load(full_state_dict_model_path)

    # Integrate into loaded model
    model.load_state_dict(model_checkpoint)

    print(f"model checkpoint loaded to rank0 cpu from {full_state_dict_model_path}")

def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1, step=0):
    """save optimizer state via full state dict"""

    print(f"--> optim state call on rank {rank}\n")

    # Pull all sharded optimizer states to rank0 cpu...
    optim_state = FSDP.optim_state_dict(model, optimizer)
    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    if rank == 0:
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name
        )
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

        opt_save_name = f"optimizer-{cfg.model_name}-epoch_{epoch}-step_{step}.pt" 
        opt_save_full_path = save_dir / opt_save_name  # This path is already using save_dir
        opt_save_full_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"--> saving optimizer state...")

        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path} to disk")


def load_optimizer_checkpoint(model, optimizer, rank, cfg):
    """
    Load the latest optimizer checkpoint from the checkpoint directory,
    determined by the newest subdirectory based on modification time.
    """
    if rank != 0:
        return

    checkpoint_dir = Path(cfg.checkpoint_folder)

    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {checkpoint_dir} not found. Returning...")
        return

    # Get all subdirectories and sort them by modification time (newest first)
    subdirectories = [d for d in checkpoint_dir.iterdir() if d.is_dir()]
    subdirectories.sort(key=lambda d: d.stat().st_mtime, reverse=True)  # Newest first

    if not subdirectories:
        print(f"No checkpoint subdirectories found in {checkpoint_dir}. Returning...")
        return

    latest_checkpoint_dir = subdirectories[0]  # Get the newest subdirectory

    # Construct the full path to the optimizer checkpoint file
    optimizer_checkpoint_filename = f"optimizer-{cfg.model_name}.pt"
    optimizer_checkpoint_path = latest_checkpoint_dir / optimizer_checkpoint_filename

    if not optimizer_checkpoint_path.is_file():
        print(f"Warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning.")
        return

    # Load the optimizer state
    full_osd = torch.load(optimizer_checkpoint_path)

    # Scatter the full optimizer state dictionary
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    # Update the optimizer's state with the loaded sharded optimizer state
    optimizer.load_state_dict(sharded_osd)

    print(f"Optimizer shard loaded on rank {rank} from {optimizer_checkpoint_path}")

def load_sharded_model_single_gpu(model, model_path):
    reader = FileSystemReader(model_path)

    state_dict = {
        "model": model.state_dict()
    }
    
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=reader,  # Use the reader here
        no_dist=True,
    )
    
    model.load_state_dict(state_dict["model"])
    
    print(f"Sharded state checkpoint loaded from {model_path}")
    return model

def save_peft_checkpoint(model, model_path, epoch=1, step=0):
    """save_pretrained peft model"""

    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    model_path = os.path.join(model_path, f"peft-checkpoint-epoch_{epoch}-step_{step}") 

    
    if isinstance(model, FSDP):
        state_dict = get_model_state_dict(model, options=options)
        model.save_pretrained(model_path, state_dict=state_dict)
    else:
        model.save_pretrained(model_path)
    
    
def save_model_checkpoint(model, output_dir, epoch=1, step=0):
    """save model when not peft and on single device"""
    
    output_dir = Path(output_dir) / f"epoch_{epoch}-step_{step}"  # Create subdirectory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "model.pt"     
    state_dict = model.state_dict()
    
    torch.save(state_dict, output_file)

from pathlib import Path

def remove_old_checkpoints(base_dir, max_checkpoints_to_keep=None):
    """
    Traverse through directories in the base directory and remove old checkpoint directories.
    
    Args:
        base_dir (str): The base directory to search for checkpoint folders.
        max_checkpoints_to_keep (int or None): The maximum number of checkpoint directories to keep.
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Base directory {base_path} does not exist.")
        return

    # List to hold all checkpoint directories found
    checkpoint_dirs = []

    # Gather all checkpoint directories
    for checkpoint in base_path.rglob('*'):
        if checkpoint.is_dir() and any((checkpoint / file).exists() for file in ['train_params.yaml', '.pt', '.distcp']):
            checkpoint_dirs.append(checkpoint)  # Register the directory

    print(f"Found {len(checkpoint_dirs)} checkpoint directories.")

    # If there are no checkpoints, exit the function
    if not checkpoint_dirs:
        print("No checkpoint directories found.")
        return

    # Sort directories by modification time (oldest first)
    checkpoint_dirs.sort(key=lambda d: d.stat().st_mtime)

    # Check if max_checkpoints_to_keep is set
    if max_checkpoints_to_keep is not None and len(checkpoint_dirs) > max_checkpoints_to_keep:
        # Remove old checkpoint directories if needed
        while len(checkpoint_dirs) > max_checkpoints_to_keep:
            old_checkpoint_dir = checkpoint_dirs.pop(0)

            # Attempt to remove all files within the directory first
            try:
                for file in old_checkpoint_dir.iterdir():
                    if file.is_file():  # Check if it is a file before unlinking
                        print(f"Deleting file: {file}")
                        file.unlink()

                # Attempt to remove the old checkpoint directory itself
                old_checkpoint_dir.rmdir()  # This will succeed only if the directory is empty
                print(f"Removed old checkpoint directory: {old_checkpoint_dir.name}")
            except OSError as e:
                print(f"Could not remove {old_checkpoint_dir.name}: {e}. Skipping this directory.")

    print(f"Total remaining checkpoints: {len(checkpoint_dirs)}")