#!/usr/bin/env python3
"""
Stage 2: LoRA + Audio Adapter Fine-tuning (Optimized)
Train LoRA parameters in UNet along with audio adapter for text-audio balance
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
    parser = argparse.ArgumentParser(description="Stage 2: LoRA + Audio Adapter Training")
    parser.add_argument("--config", default="configs/training_config_safe.json")  # Use optimized config
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--stage1-checkpoint", default="checkpoints/stage1_final", 
                       help="Path to Stage 1 checkpoint")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Stage 2: LoRA + Audio Adapter Fine-tuning")
    print("="*60)
    
    # Create trainer
    trainer = CLAP2DiffusionTrainer(args.config)
    
    # Load Stage 1 checkpoint
    stage1_dir = Path(args.stage1_checkpoint)
    if not stage1_dir.exists():
        print(f"⚠️  Stage 1 checkpoint not found at {stage1_dir}")
        print("Please complete Stage 1 training first or specify correct path")
        return
    
    print(f"Loading Stage 1 checkpoint from {stage1_dir}")
    audio_adapter_state = torch.load(
        stage1_dir / "audio_adapter.pt", 
        map_location=trainer.device
    )
    trainer.audio_adapter.load_state_dict(audio_adapter_state)
    print("✓ Loaded audio adapter from Stage 1")
    
    # Also load attention adapter if it exists
    attention_adapter_path = stage1_dir / "attention_adapter.pt"
    if attention_adapter_path.exists():
        attention_state = torch.load(attention_adapter_path, map_location=trainer.device)
        trainer.attention_adapter.load_state_dict(attention_state)
        print(f"✓ Loaded attention adapter, gate: {trainer.attention_adapter.gate.item():.6f}")
    
    # Resume if requested
    start_step = 0
    if args.resume:
        checkpoint_dir = Path("checkpoints")
        stage2_checkpoints = sorted(checkpoint_dir.glob("stage2_step*"))
        if stage2_checkpoints:
            latest = stage2_checkpoints[-1]
            print(f"Resuming from {latest}")
            
            # Load states
            audio_state = torch.load(latest / "audio_adapter.pt", map_location=trainer.device)
            trainer.audio_adapter.load_state_dict(audio_state)
            
            attention_state = torch.load(latest / "attention_adapter.pt", map_location=trainer.device)
            trainer.attention_adapter.load_state_dict(attention_state)
            
            # Extract step number
            start_step = int(latest.name.split("step")[-1]) + 1
            print(f"Starting from step {start_step}")
    
    # Setup Stage 2
    trainer.setup_stage(2)
    trainer.current_stage = 2
    
    # Training configuration
    total_steps = trainer.config['stage2'].get('num_steps', 7000)
    remaining_steps = total_steps - start_step
    
    if remaining_steps <= 0:
        print("Stage 2 already completed!")
        return
    
    print(f"\nTraining Configuration:")
    print(f"  Total steps: {total_steps}")
    print(f"  Starting from: {start_step}")
    print(f"  Remaining: {remaining_steps}")
    print(f"  Learning rate: {trainer.config['stage2'].get('learning_rate', 5e-5)}")
    print(f"  LoRA rank: {trainer.config['stage2'].get('lora_rank', 8)}")
    print(f"  LoRA alpha: {trainer.config['stage2'].get('lora_alpha', 32)}")
    print(f"  Batch size: {trainer.config['training'].get('batch_size', 4)}")
    
    # Create log file
    log_dir = Path(trainer.config['training'].get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True, parents=True)
    log_path = log_dir / f"stage2_standalone_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "w", buffering=1)
    
    log_file.write(f"Stage 2 Training Log\n")
    log_file.write(f"="*50 + "\n")
    log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Total steps: {total_steps}\n")
    log_file.write(f"Starting from step: {start_step}\n")
    log_file.write(f"LoRA config: rank={trainer.config['stage2'].get('lora_rank', 8)}, "
                  f"alpha={trainer.config['stage2'].get('lora_alpha', 32)}\n\n")
    
    print(f"\nLog file: {log_path}")
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    
    # Training loop
    stage_start_time = time.time()
    progress_bar = tqdm(range(start_step, total_steps), desc="Stage 2", initial=start_step, total=total_steps)
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
        with trainer.accelerator.accumulate(trainer.unet):
            loss = trainer.train_step(batch)
            trainer.accelerator.backward(loss)
            
            if trainer.accelerator.sync_gradients:
                # Clip gradients for LoRA and audio adapter
                params_to_clip = [p for p in trainer.unet.parameters() if p.requires_grad]
                params_to_clip.extend(list(trainer.audio_adapter.parameters()))
                
                torch.nn.utils.clip_grad_norm_(
                    params_to_clip,
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
            
        # Save checkpoint every 1000 steps
        if (step + 1) % 1000 == 0:
            trainer.save_checkpoint(2, step)
            log_file.write(f"Checkpoint saved at step {step}\n")
            print(f"\n💾 Checkpoint saved at step {step}")
            
        # Validation every 1000 steps
        if (step + 1) % 1000 == 0:
            val_loss = trainer.validate()
            val_msg = f"Validation at step {step}: loss={val_loss:.6f}"
            print(f"📊 {val_msg}")
            log_file.write(val_msg + "\n")
    
    # Save final checkpoint
    trainer.save_checkpoint(2, total_steps-1, final=True)
    
    # Training complete
    stage_time = time.time() - stage_start_time
    final_msg = f"\n{'='*60}\n"
    final_msg += f"Stage 2 Training Complete!\n"
    final_msg += f"Time: {stage_time/60:.2f} minutes\n"
    final_msg += f"Final loss: {logs['loss']:.6f}\n"
    final_msg += f"{'='*60}\n"
    
    print(final_msg)
    log_file.write(final_msg)
    log_file.close()
    
    print(f"✅ Stage 2 checkpoint saved to: checkpoints/stage2_final")
    print(f"📝 Log saved to: {log_path}")
    print(f"\n🎯 Ready for Stage 3 (Gate Optimization)")

if __name__ == "__main__":
    main()