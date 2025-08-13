#!/usr/bin/env python3
"""
Stage 1: Audio Adapter Training (Optimized)
Train only the audio projection MLP to align audio embeddings with text space
Includes caching, safe augmentation, and performance optimizations
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.train_optimized import CLAP2DiffusionTrainer  # Use optimized version
import torch
from tqdm import tqdm
import time
from datetime import datetime

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1: Audio Adapter Training")
    parser.add_argument("--config", default="configs/training_config_safe.json")  # Use optimized config
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--use_balanced", action="store_true", help="Use balanced dataset")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Stage 1: Audio Adapter Training")
    print("="*60)
    
    # Create trainer
    trainer = CLAP2DiffusionTrainer(args.config)
    
    # Override to use balanced dataset if requested
    if args.use_balanced:
        print("\nUsing balanced dataset with class weights")
        trainer.config['data']['train_metadata'] = 'train_metadata_balanced.json'
        trainer.config['data']['val_metadata'] = 'val_metadata_balanced.json'
        trainer.config['data']['test_metadata'] = 'test_metadata_balanced.json'
        # Reinitialize datasets with balanced metadata
        trainer.setup_datasets()
    
    # Resume if requested
    start_step = 0
    if args.resume:
        checkpoint_dir = Path("checkpoints")
        stage1_checkpoints = sorted(checkpoint_dir.glob("stage1_step*"))
        if stage1_checkpoints:
            latest = stage1_checkpoints[-1]
            print(f"Resuming from {latest}")
            
            state_dict = torch.load(latest / "audio_adapter.pt", map_location=trainer.device)
            trainer.audio_adapter.load_state_dict(state_dict)
            
            # Extract step number from checkpoint name
            start_step = int(latest.name.split("step")[-1]) + 1
            print(f"Starting from step {start_step}")
    
    # Setup Stage 1
    trainer.setup_stage(1)
    trainer.current_stage = 1
    
    # Training configuration
    total_steps = trainer.config['stage1'].get('num_steps', 3000)
    remaining_steps = total_steps - start_step
    
    if remaining_steps <= 0:
        print("Stage 1 already completed!")
        return
    
    print(f"\nTraining Configuration:")
    print(f"  Total steps: {total_steps}")
    print(f"  Starting from: {start_step}")
    print(f"  Remaining: {remaining_steps}")
    print(f"  Learning rate: {trainer.config['stage1'].get('learning_rate', 1e-4)}")
    print(f"  Batch size: {trainer.config['training'].get('batch_size', 4)}")
    print(f"  Gradient accumulation: {trainer.config['training'].get('gradient_accumulation_steps', 1)}")
    
    # Create log file
    log_dir = Path(trainer.config['training'].get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True, parents=True)
    log_path = log_dir / f"stage1_standalone_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "w", buffering=1)
    
    log_file.write(f"Stage 1 Training Log\n")
    log_file.write(f"="*50 + "\n")
    log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Total steps: {total_steps}\n")
    log_file.write(f"Starting from step: {start_step}\n\n")
    
    print(f"\nLog file: {log_path}")
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    # Training loop
    stage_start_time = time.time()
    progress_bar = tqdm(range(start_step, total_steps), desc="Stage 1", initial=start_step, total=total_steps)
    train_iter = iter(trainer.train_loader)
    
    for step in range(start_step, total_steps):
        trainer.unet.train()
        trainer.audio_adapter.train()
        trainer.attention_adapter.train()
        
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(trainer.train_loader)
            batch = next(train_iter)
        
        # Training step with accumulation
        with trainer.accelerator.accumulate(trainer.audio_adapter):
            loss = trainer.train_step(batch)
            trainer.accelerator.backward(loss)
            
            if trainer.accelerator.sync_gradients:
                torch.nn.utils.clip_grad_norm_(
                    list(trainer.audio_adapter.parameters()),
                    trainer.config['training'].get('max_grad_norm', 1.0)
                )
            
            trainer.optimizer.step()
            trainer.lr_scheduler.step()
            trainer.optimizer.zero_grad()
        
        # Update progress
        progress_bar.update(1)
        logs = {
            "loss": loss.detach().item(),
            "lr": trainer.lr_scheduler.get_last_lr()[0]
        }
        progress_bar.set_postfix(**logs)
        
        # Log every 100 steps
        if step % 100 == 0:
            log_msg = f"Step {step}: loss={logs['loss']:.6f}, lr={logs['lr']:.2e}"
            log_file.write(log_msg + "\n")
            
        # Save checkpoint every 500 steps
        if (step + 1) % 500 == 0:
            trainer.save_checkpoint(1, step)
            log_file.write(f"Checkpoint saved at step {step}\n")
            
        # Validation every 500 steps
        if (step + 1) % 500 == 0:
            val_loss = trainer.validate()
            val_msg = f"Validation at step {step}: loss={val_loss:.6f}"
            print(f"\n{val_msg}")
            log_file.write(val_msg + "\n")
    
    # Save final checkpoint
    trainer.save_checkpoint(1, total_steps-1, final=True)
    
    # Training complete
    stage_time = time.time() - stage_start_time
    final_msg = f"\n{'='*60}\n"
    final_msg += f"Stage 1 Training Complete!\n"
    final_msg += f"Time: {stage_time/60:.2f} minutes\n"
    final_msg += f"Final loss: {logs['loss']:.6f}\n"
    final_msg += f"{'='*60}\n"
    
    print(final_msg)
    log_file.write(final_msg)
    log_file.close()
    
    print(f"✅ Stage 1 checkpoint saved to: checkpoints/stage1_final")
    print(f"📝 Log saved to: {log_path}")

if __name__ == "__main__":
    main()