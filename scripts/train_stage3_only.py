#!/usr/bin/env python3
"""
Stage 3 only training script for CLAP2Diffusion
Loads checkpoints from Stage 2 and trains only gate parameters
"""

import os
# Set HuggingFace cache directory if not already set
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.expanduser('~/.cache/huggingface')
if 'TRANSFORMERS_CACHE' not in os.environ:
    os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.environ['HF_HOME'], 'transformers')
if 'HF_DATASETS_CACHE' not in os.environ:
    os.environ['HF_DATASETS_CACHE'] = os.path.join(os.environ['HF_HOME'], 'datasets')

import sys
import json
import argparse
import torch
import time
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train import CLAP2DiffusionTrainer

class Stage3OnlyTrainer(CLAP2DiffusionTrainer):
    def __init__(self, config_path: str, checkpoint_path: str = None):
        """Initialize trainer with Stage 2 checkpoint."""
        super().__init__(config_path)
        
        # Load Stage 2 checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
            print(f"Loaded checkpoint from: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load Stage 2 checkpoint."""
        checkpoint_dir = Path(checkpoint_path)
        
        # Load audio adapter
        audio_adapter_path = checkpoint_dir / "audio_adapter.pt"
        if audio_adapter_path.exists():
            state_dict = torch.load(audio_adapter_path, map_location=self.device)
            self.audio_adapter.load_state_dict(state_dict)
            print("Loaded audio adapter checkpoint")
        
        # Load attention adapter
        attention_adapter_path = checkpoint_dir / "attention_adapter.pt"
        if attention_adapter_path.exists():
            state_dict = torch.load(attention_adapter_path, map_location=self.device)
            self.attention_adapter.load_state_dict(state_dict)
            print("Loaded attention adapter checkpoint")
        
        # Load accelerator state (includes UNet with LoRA)
        if (checkpoint_dir / "pytorch_model.bin").exists():
            self.accelerator.load_state(checkpoint_dir)
            print("Loaded accelerator state (UNet with LoRA)")
    
    def train(self):
        """Train Stage 3 only."""
        print("\n=== Starting Stage 3 Training (Gate Optimization) ===")
        
        # Setup Stage 3
        stage = 3
        self.setup_stage(stage)
        self.current_stage = stage
        
        # Ensure gate parameter requires grad
        self.attention_adapter.gate.requires_grad_(True)
        
        # Create log directory
        log_dir = Path(self.config['training'].get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create log files
        stage3_log_path = log_dir / f"stage3_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        stage3_log = open(stage3_log_path, "w")
        stage3_log.write(f"Stage 3 Training Log (Standalone)\n")
        stage3_log.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        stage3_log.write(f"Total steps: {self.num_steps}\n")
        stage3_log.write(f"Learning rate: {self.config['stage3'].get('learning_rate', 1e-3)}\n")
        
        # Print trainable parameters info
        trainable_params = []
        for name, param in self.attention_adapter.named_parameters():
            if param.requires_grad:
                trainable_params.append(name)
                stage3_log.write(f"Trainable param: {name}, shape: {param.shape}\n")
        
        print(f"Trainable parameters: {trainable_params}")
        stage3_log.write(f"\nTrainable parameters: {trainable_params}\n\n")
        stage3_log.flush()
        
        # Training loop
        stage_start_time = time.time()
        progress_bar = tqdm(range(self.num_steps), desc="Stage 3 (Gate Optimization)")
        
        # Create data loader iterator
        train_iter = iter(self.train_loader)
        
        for step in range(self.num_steps):
            # Training step
            self.unet.train()
            self.audio_adapter.train()
            self.attention_adapter.train()
            
            # Get batch, restart iterator if needed
            try:
                batch = next(train_iter)
            except StopIteration:
                print(f"\nRestarting data loader at step {step}")
                stage3_log.write(f"\nRestarted data loader at step {step}\n")
                stage3_log.flush()
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            
            # Forward pass with accumulation
            with self.accelerator.accumulate(self.attention_adapter):
                loss = self.train_step(batch)
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.accelerator.sync_gradients:
                    params_to_clip = [p for p in self.attention_adapter.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(
                        params_to_clip,
                        self.config['training'].get('max_grad_norm', 1.0)
                    )
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress
            progress_bar.update(1)
            logs = {
                "loss": loss.detach().item(),
                "lr": self.lr_scheduler.get_last_lr()[0],
                "gate_value": self.attention_adapter.gate.item()
            }
            progress_bar.set_postfix(**logs)
            
            # Log every 10 steps
            if step % 10 == 0:
                log_entry = f"Step {step}/{self.num_steps}: loss={logs['loss']:.6f}, lr={logs['lr']:.2e}, gate={logs['gate_value']:.4f}\n"
                stage3_log.write(log_entry)
                stage3_log.flush()
            
            # Validation
            if (step + 1) % self.config['training'].get('val_steps', 500) == 0:
                val_loss = self.validate()
                print(f"\nValidation loss: {val_loss:.4f}")
                
                val_log_entry = f"\nValidation at step {step}: loss={val_loss:.4f}\n"
                stage3_log.write(val_log_entry)
                stage3_log.flush()
            
            # Save checkpoint
            if (step + 1) % self.config['training'].get('save_steps', 1000) == 0:
                self.save_checkpoint(3, step)
                checkpoint_log = f"\nCheckpoint saved at step {step}\n"
                stage3_log.write(checkpoint_log)
                stage3_log.flush()
        
        # Save final checkpoint
        self.save_checkpoint(3, self.num_steps, final=True)
        
        # Completion log
        stage_time = time.time() - stage_start_time
        completion_log = f"\nStage 3 completed in {stage_time/60:.2f} minutes\n"
        completion_log += f"Final loss: {logs['loss']:.6f}\n"
        completion_log += f"Final gate value: {logs['gate_value']:.4f}\n"
        completion_log += f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        stage3_log.write(completion_log)
        stage3_log.close()
        
        print(f"\n=== Stage 3 Training Complete! ===")
        print(f"Training log saved to: {stage3_log_path}")
        print(f"Final gate value: {logs['gate_value']:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Train CLAP2Diffusion Stage 3 Only")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.json",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/stage2_final",
        help="Path to Stage 2 checkpoint directory"
    )
    
    args = parser.parse_args()
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        # Try to find the latest Stage 2 checkpoint
        checkpoint_dir = Path("checkpoints")
        stage2_checkpoints = sorted(checkpoint_dir.glob("stage2_*"))
        if stage2_checkpoints:
            checkpoint_path = stage2_checkpoints[-1]
            print(f"Using latest Stage 2 checkpoint: {checkpoint_path}")
        else:
            print("Warning: No Stage 2 checkpoint found. Training from scratch.")
            checkpoint_path = None
    
    # Create trainer and start Stage 3 training
    trainer = Stage3OnlyTrainer(args.config, checkpoint_path)
    trainer.train()

if __name__ == "__main__":
    main()