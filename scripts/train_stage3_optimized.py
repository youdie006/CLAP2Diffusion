#!/usr/bin/env python3
"""
Stage 3: Gate Parameter Optimization (Optimized)
Fine-tune only the gate parameter to balance audio-text contributions
Includes early stopping and improved monitoring
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
    parser = argparse.ArgumentParser(description="Stage 3: Gate Parameter Optimization")
    parser.add_argument("--config", default="configs/training_config_safe.json")  # Use optimized config
    parser.add_argument("--lr", type=float, default=0.1,  # Changed from 0.01 to 0.1 based on troubleshooting 
                       help="Learning rate for gate (default: 0.01)")
    parser.add_argument("--steps", type=int, default=2000,
                       help="Number of training steps (default: 2000)")
    parser.add_argument("--init-gate", type=float, default=None,
                       help="Initialize gate to specific value")
    parser.add_argument("--target-min", type=float, default=0.35,
                       help="Minimum target for tanh(gate)")
    parser.add_argument("--target-max", type=float, default=0.45,
                       help="Maximum target for tanh(gate)")
    parser.add_argument("--stage2-checkpoint", default="checkpoints/stage2_final",
                       help="Path to Stage 2 checkpoint")
    parser.add_argument("--early-stop", action="store_true",
                       help="Stop when gate reaches target range")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("Stage 3: Gate Parameter Optimization")
    print("="*60)
    
    # Create trainer
    trainer = CLAP2DiffusionTrainer(args.config)
    
    # Load Stage 2 checkpoint
    stage2_dir = Path(args.stage2_checkpoint)
    if not stage2_dir.exists():
        print(f"⚠️  Stage 2 checkpoint not found at {stage2_dir}")
        print("Please complete Stage 2 training first or specify correct path")
        return
    
    print(f"Loading Stage 2 checkpoint from {stage2_dir}")
    
    # Load audio adapter
    audio_adapter_state = torch.load(
        stage2_dir / "audio_adapter.pt",
        map_location=trainer.device
    )
    trainer.audio_adapter.load_state_dict(audio_adapter_state)
    print("✓ Loaded audio adapter from Stage 2")
    
    # Load attention adapter
    attention_adapter_state = torch.load(
        stage2_dir / "attention_adapter.pt",
        map_location=trainer.device
    )
    trainer.attention_adapter.load_state_dict(attention_adapter_state)
    initial_gate = trainer.attention_adapter.gate.item()
    print(f"✓ Loaded attention adapter, gate: {initial_gate:.6f}")
    
    # Initialize gate if requested
    if args.init_gate is not None:
        print(f"Initializing gate to {args.init_gate}")
        with torch.no_grad():
            trainer.attention_adapter.gate.data = torch.tensor(
                args.init_gate, device=trainer.device, dtype=trainer.dtype
            )
    
    # Setup Stage 3
    trainer.current_stage = 3
    trainer.unet.requires_grad_(False)
    trainer.audio_adapter.requires_grad_(False)
    trainer.attention_adapter.requires_grad_(False)
    trainer.attention_adapter.gate.requires_grad_(True)
    
    # Custom optimizer for gate with specified learning rate
    trainer.optimizer = torch.optim.Adam(
        [trainer.attention_adapter.gate],
        lr=args.lr,
        betas=(0.9, 0.999)
    )
    
    trainer.num_steps = args.steps
    
    # Prepare models (but NOT optimizer)
    trainer.unet, trainer.audio_adapter, trainer.attention_adapter, trainer.train_loader = \
        trainer.accelerator.prepare(
            trainer.unet, trainer.audio_adapter, trainer.attention_adapter,
            trainer.train_loader
        )
    
    # Training configuration
    print(f"\nTraining Configuration:")
    print(f"  Total steps: {args.steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Initial gate: {trainer.attention_adapter.gate.item():.6f}")
    print(f"  Target range: tanh(gate) ∈ [{args.target_min:.2f}, {args.target_max:.2f}]")
    print(f"  Early stopping: {args.early_stop}")
    print(f"  Batch size: {trainer.config['training'].get('batch_size', 4)}")
    
    # Create log file with buffering for real-time writes
    log_dir = Path(trainer.config['training'].get('log_dir', 'logs'))
    log_dir.mkdir(exist_ok=True, parents=True)
    log_path = log_dir / f"stage3_standalone_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "w", buffering=1)  # Line buffering
    
    log_file.write(f"Stage 3 Gate Optimization Log\n")
    log_file.write(f"="*50 + "\n")
    log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    log_file.write(f"Learning rate: {args.lr}\n")
    log_file.write(f"Initial gate: {trainer.attention_adapter.gate.item():.6f}\n")
    log_file.write(f"Target range: [{args.target_min:.2f}, {args.target_max:.2f}]\n\n")
    
    print(f"\nLog file: {log_path}")
    print("\n" + "="*60)
    print("Starting gate optimization...")
    print("="*60 + "\n")
    
    # Training loop
    stage_start_time = time.time()
    progress_bar = tqdm(range(args.steps), desc="Stage 3")
    train_iter = iter(trainer.train_loader)
    
    best_loss = float('inf')
    best_gate = trainer.attention_adapter.gate.item()
    
    for step in range(args.steps):
        trainer.unet.train()
        trainer.audio_adapter.train()
        trainer.attention_adapter.train()
        
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(trainer.train_loader)
            batch = next(train_iter)
        
        # Training step
        loss = trainer.train_step(batch)
        
        # Backward
        trainer.optimizer.zero_grad()
        loss.backward()
        
        # Check gradient
        if trainer.attention_adapter.gate.grad is not None:
            grad_norm = trainer.attention_adapter.gate.grad.norm().item()
        else:
            grad_norm = 0.0
        
        # Clip gradient
        torch.nn.utils.clip_grad_norm_([trainer.attention_adapter.gate], max_norm=0.5)
        
        # Update
        trainer.optimizer.step()
        
        # Get values
        gate_value = trainer.attention_adapter.gate.item()
        gate_tanh = torch.tanh(trainer.attention_adapter.gate).item()
        loss_val = loss.detach().item()
        
        # Track best
        if loss_val < best_loss:
            best_loss = loss_val
            best_gate = gate_value
        
        logs = {
            "loss": loss_val,
            "gate": gate_value,
            "tanh": gate_tanh,
            "grad": grad_norm
        }
        progress_bar.set_postfix(**logs)
        
        # Log every 50 steps
        if step % 50 == 0:
            log_msg = f"Step {step:4d} | Loss: {loss_val:.6f} | Gate: {gate_value:+.6f} | "
            log_msg += f"Tanh: {gate_tanh:+.6f} | Grad: {grad_norm:.6f}"
            
            # Check status
            if gate_tanh < args.target_min:
                status = "Below target"
            elif gate_tanh > args.target_max:
                status = "Above target"
            elif args.target_min <= gate_tanh <= args.target_max:
                status = "✓ IN TARGET RANGE"
            else:
                status = "Optimizing..."
            
            log_msg += f" | Status: {status}"
            print(f"\n{log_msg}")
            log_file.write(log_msg + "\n")
        
        # Save checkpoint every 500 steps
        if (step + 1) % 500 == 0:
            trainer.save_checkpoint(3, step)
            checkpoint_msg = f"\n--- Checkpoint at step {step+1} ---"
            checkpoint_msg += f"\n  Best loss: {best_loss:.6f}"
            checkpoint_msg += f"\n  Best gate: {best_gate:.6f}"
            checkpoint_msg += f"\n  Current gate: {gate_value:.6f}"
            checkpoint_msg += f"\n  Current tanh: {gate_tanh:.6f}\n"
            
            print(checkpoint_msg)
            log_file.write(checkpoint_msg + "\n")
        
        # Early stopping if gate is in target range
        if args.early_stop and args.target_min <= gate_tanh <= args.target_max and step > 100:
            success_msg = f"\n{'='*60}\n"
            success_msg += f"🎯 SUCCESS! Gate reached target range at step {step}!\n"
            success_msg += f"  Final gate: {gate_value:.6f}\n"
            success_msg += f"  Final tanh: {gate_tanh:.6f}\n"
            success_msg += f"  Final loss: {loss_val:.6f}\n"
            success_msg += f"{'='*60}\n"
            
            print(success_msg)
            log_file.write(success_msg + "\n")
            break
    
    # Save final checkpoint
    trainer.save_checkpoint(3, step, final=True)
    
    # Training complete
    stage_time = time.time() - stage_start_time
    final_gate = trainer.attention_adapter.gate.item()
    final_tanh = torch.tanh(trainer.attention_adapter.gate).item()
    
    final_msg = f"\n{'='*60}\n"
    final_msg += f"Stage 3 Gate Optimization Complete!\n"
    final_msg += f"Time: {stage_time/60:.2f} minutes\n"
    final_msg += f"Total steps: {step + 1}\n"
    final_msg += f"Final gate: {final_gate:.6f}\n"
    final_msg += f"Final tanh: {final_tanh:.6f}\n"
    final_msg += f"Best loss: {best_loss:.6f}\n"
    
    if args.target_min <= final_tanh <= args.target_max:
        final_msg += f"✅ Gate in target range [{args.target_min:.2f}, {args.target_max:.2f}]\n"
    else:
        final_msg += f"⚠️  Gate outside target range [{args.target_min:.2f}, {args.target_max:.2f}]\n"
    
    final_msg += f"{'='*60}\n"
    
    print(final_msg)
    log_file.write(final_msg)
    log_file.close()
    
    print(f"✅ Stage 3 checkpoint saved to: checkpoints/stage3_final")
    print(f"📝 Log saved to: {log_path}")
    print(f"\n🚀 Model training complete! Ready for inference.")

if __name__ == "__main__":
    main()