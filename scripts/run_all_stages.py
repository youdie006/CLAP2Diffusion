#!/usr/bin/env python3
"""
Run all training stages sequentially
Automatically proceeds to next stage after completion
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

def run_stage(stage_num, script_name, use_balanced=True):
    """Run a single training stage"""
    print("\n" + "="*60)
    print(f"Starting Stage {stage_num} Training")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Build command
    cmd = ["python", f"scripts/{script_name}"]
    if use_balanced:
        cmd.append("--use_balanced")
    
    # Run the training script
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=False  # Show output in real-time
        )
        
        print(f"\n✅ Stage {stage_num} completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Stage {stage_num} failed with error:")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ Stage {stage_num} interrupted by user")
        return False

def check_checkpoint(stage_num):
    """Check if stage checkpoint exists"""
    checkpoint_dir = Path("checkpoints")
    stage_checkpoint = checkpoint_dir / f"stage{stage_num}_final"
    
    if stage_checkpoint.exists():
        print(f"✅ Found checkpoint: {stage_checkpoint}")
        return True
    return False

def main():
    print("\n" + "="*80)
    print("CLAP2Diffusion Full Training Pipeline")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Configuration:")
    print("  - Using balanced dataset with class weights")
    print("  - Stage 1: Audio Adapter (3k steps)")
    print("  - Stage 2: LoRA + Adapter (7k steps)") 
    print("  - Stage 3: Gate Optimization (2k steps)")
    print("="*80)
    
    # Create log directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Log file for overall progress
    log_file = log_dir / f"full_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    with open(log_file, "w") as f:
        f.write("CLAP2Diffusion Full Training Log\n")
        f.write("="*50 + "\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    start_time = time.time()
    
    # Stage 1: Audio Adapter Training
    if not check_checkpoint(1):
        print("\n📚 Stage 1: Audio Adapter Training")
        success = run_stage(1, "train_stage1_optimized.py", use_balanced=True)
        if not success:
            print("\n❌ Stage 1 failed. Stopping pipeline.")
            sys.exit(1)
    else:
        print("\n⏭️ Stage 1 already completed, skipping...")
    
    # Brief pause between stages
    print("\n⏸️ Pausing 10 seconds before Stage 2...")
    time.sleep(10)
    
    # Stage 2: LoRA + Adapter Training
    if not check_checkpoint(2):
        print("\n📚 Stage 2: LoRA + Adapter Training")
        success = run_stage(2, "train_stage2_optimized.py", use_balanced=True)
        if not success:
            print("\n❌ Stage 2 failed. Stopping pipeline.")
            sys.exit(1)
    else:
        print("\n⏭️ Stage 2 already completed, skipping...")
    
    # Brief pause between stages
    print("\n⏸️ Pausing 10 seconds before Stage 3...")
    time.sleep(10)
    
    # Stage 3: Gate Optimization
    if not check_checkpoint(3):
        print("\n📚 Stage 3: Gate Parameter Optimization")
        success = run_stage(3, "train_stage3_optimized.py", use_balanced=True)
        if not success:
            print("\n❌ Stage 3 failed. Stopping pipeline.")
            sys.exit(1)
    else:
        print("\n⏭️ Stage 3 already completed, skipping...")
    
    # Training complete
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    
    print("\n" + "="*80)
    print("🎉 ALL STAGES COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"Total training time: {hours}h {minutes}m")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nCheckpoints saved in: ./checkpoints/")
    print("Logs saved in: ./logs/")
    print("="*80)
    
    # Write final log
    with open(log_file, "a") as f:
        f.write(f"\n\nTraining completed successfully!\n")
        f.write(f"Total time: {hours}h {minutes}m\n")
        f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

if __name__ == "__main__":
    main()