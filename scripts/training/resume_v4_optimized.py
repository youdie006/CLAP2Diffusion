#!/usr/bin/env python
"""
Resume V4 Hybrid training with optimized settings
Starts from step 276 with improved performance
"""

import os
import sys
import json
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_v4_hybrid import CLAP2DiffusionV4Trainer

def main():
    # Update config with resume information
    config_path = "/mnt/d/MyProject/CLAP2Diffusion/configs/training_config_v4_hybrid.json"
    
    # Create trainer with optimized settings
    trainer = CLAP2DiffusionV4Trainer(
        config_path=config_path,
        output_dir="./outputs/v4_hybrid_optimized"
    )
    
    # Manually set the step count to resume from
    trainer.global_step = 276
    print(f"Resuming from step {trainer.global_step}")
    
    # Start training with optimized settings
    print("\n" + "="*50)
    print("Starting optimized training")
    print(f"Batch size: 2")
    print(f"Gradient accumulation: 1")
    print(f"Mixed precision: Disabled")
    print(f"XFormers: Disabled")
    print(f"Expected speed: ~10-20 seconds/step")
    print("="*50 + "\n")
    
    trainer.train()

if __name__ == "__main__":
    main()