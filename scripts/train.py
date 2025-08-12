#!/usr/bin/env python3
"""
Training script for CLAP2Diffusion
Implements 3-stage training: Audio Adapter -> LoRA+Adapter -> Gate Optimization
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, TaskType

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.dataset import AudioImageDataset
from src.models.audio_encoder import CLAPAudioEncoder
from src.models.audio_adapter import AudioProjectionMLP
from src.models.attention_adapter import AudioAdapterAttention
from src.models.unet_with_audio import AudioConditionedUNet


class CLAP2DiffusionTrainer:
    def __init__(self, config_path: str):
        """Initialize trainer with configuration."""
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision='fp16' if self.config['training'].get('mixed_precision', True) else 'no',
            gradient_accumulation_steps=self.config['training'].get('gradient_accumulation_steps', 1)
        )
        
        # Set device
        self.device = self.accelerator.device
        
        # Initialize models
        self.setup_models()
        
        # Initialize datasets
        self.setup_datasets()
        
        # Training state
        self.current_stage = 1
        self.global_step = 0
        
    def setup_models(self):
        """Initialize all model components."""
        print("Loading models...")
        
        # Load base Stable Diffusion
        model_id = self.config['model'].get('base_model', 'runwayml/stable-diffusion-v1-5')
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.config['training'].get('mixed_precision', True) else torch.float32
        )
        
        # Extract components
        self.vae = self.pipe.vae
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer
        self.unet = self.pipe.unet
        self.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        
        # Initialize CLAP audio encoder
        self.audio_encoder = CLAPAudioEncoder(
            model_name=self.config['model'].get('clap_model', 'laion/larger_clap_music_and_speech')
        )
        
        # Initialize audio adapter
        self.audio_adapter = AudioProjectionMLP(
            audio_dim=self.config['model'].get('audio_dim', 512),
            text_dim=self.config['model'].get('text_dim', 768),
            num_tokens=self.config['model'].get('num_audio_tokens', 8)
        )
        
        # Initialize attention adapter
        self.attention_adapter = AudioAdapterAttention(
            dim=self.config['model'].get('attention_dim', 768),
            audio_dim=self.config['model'].get('audio_dim', 512)
        )
        
        # Move to device
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)
        self.audio_encoder = self.audio_encoder.to(self.device)
        self.audio_adapter = self.audio_adapter.to(self.device)
        self.attention_adapter = self.attention_adapter.to(self.device)
        
        # Freeze base models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.audio_encoder.model.requires_grad_(False)
        
        print("Models loaded successfully!")
        
    def setup_datasets(self):
        """Initialize training and validation datasets."""
        print("Loading datasets...")
        
        # Training dataset
        self.train_dataset = AudioImageDataset(
            data_dir=Path(self.config['data']['data_dir']),
            metadata_file=Path(self.config['data']['train_metadata']),
            tokenizer=self.tokenizer,
            image_size=self.config['data'].get('image_size', 512),
            audio_sample_rate=self.config['data'].get('sample_rate', 48000),
            audio_duration=self.config['data'].get('audio_duration', 10.0),
            augment=True
        )
        
        # Validation dataset
        self.val_dataset = AudioImageDataset(
            data_dir=Path(self.config['data']['data_dir']),
            metadata_file=Path(self.config['data']['val_metadata']),
            tokenizer=self.tokenizer,
            image_size=self.config['data'].get('image_size', 512),
            audio_sample_rate=self.config['data'].get('sample_rate', 48000),
            audio_duration=self.config['data'].get('audio_duration', 10.0),
            augment=False
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training'].get('batch_size', 4),
            shuffle=True,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training'].get('batch_size', 4),
            shuffle=False,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True
        )
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
        
    def setup_stage(self, stage: int):
        """Setup training for specific stage."""
        print(f"\n=== Setting up Stage {stage} ===")
        
        if stage == 1:
            # Stage 1: Train only audio adapter
            print("Stage 1: Training Audio Adapter")
            self.unet.requires_grad_(False)
            self.audio_adapter.requires_grad_(True)
            self.attention_adapter.requires_grad_(False)
            
            # Optimizer for audio adapter
            self.optimizer = torch.optim.AdamW(
                self.audio_adapter.parameters(),
                lr=self.config['stage1'].get('learning_rate', 1e-4),
                weight_decay=self.config['stage1'].get('weight_decay', 0.01)
            )
            
            self.num_steps = self.config['stage1'].get('num_steps', 3000)
            
        elif stage == 2:
            # Stage 2: Train LoRA + Audio Adapter
            print("Stage 2: Training LoRA + Audio Adapter")
            
            # Apply LoRA to UNet
            lora_config = LoraConfig(
                r=self.config['stage2'].get('lora_rank', 8),
                lora_alpha=self.config['stage2'].get('lora_alpha', 32),
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
                lora_dropout=self.config['stage2'].get('lora_dropout', 0.1)
            )
            
            self.unet = get_peft_model(self.unet, lora_config)
            self.unet.print_trainable_parameters()
            
            self.audio_adapter.requires_grad_(True)
            self.attention_adapter.requires_grad_(False)
            
            # Optimizer for LoRA + audio adapter
            params = list(self.unet.parameters()) + list(self.audio_adapter.parameters())
            self.optimizer = torch.optim.AdamW(
                params,
                lr=self.config['stage2'].get('learning_rate', 5e-5),
                weight_decay=self.config['stage2'].get('weight_decay', 0.01)
            )
            
            self.num_steps = self.config['stage2'].get('num_steps', 7000)
            
        elif stage == 3:
            # Stage 3: Train gate parameters
            print("Stage 3: Training Gate Parameters")
            self.unet.requires_grad_(False)
            self.audio_adapter.requires_grad_(False)
            self.attention_adapter.requires_grad_(True)
            
            # Only train gate parameter
            self.optimizer = torch.optim.AdamW(
                [self.attention_adapter.gate],
                lr=self.config['stage3'].get('learning_rate', 1e-3),
                weight_decay=0
            )
            
            self.num_steps = self.config['stage3'].get('num_steps', 2000)
        
        # Learning rate scheduler
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=min(500, self.num_steps // 10),
            num_training_steps=self.num_steps
        )
        
        # Prepare with accelerator
        self.unet, self.audio_adapter, self.attention_adapter, self.optimizer, self.train_loader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.unet, self.audio_adapter, self.attention_adapter, 
                self.optimizer, self.train_loader, self.lr_scheduler
            )
        
    def train_step(self, batch):
        """Single training step."""
        # Get batch data
        images = batch['image'].to(self.device)
        audio = batch['audio'].to(self.device)
        text_input_ids = batch['text_input_ids'].to(self.device)
        text_attention_mask = batch['text_attention_mask'].to(self.device)
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, 
            (latents.shape[0],), device=self.device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)
        
        # Get text embeddings
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_input_ids,
                attention_mask=text_attention_mask
            )[0]
        
        # Get audio embeddings
        with torch.no_grad():
            audio_embeddings = self.audio_encoder(audio)
        
        # Project audio to text space
        audio_tokens = self.audio_adapter(audio_embeddings)
        
        # Combine text and audio embeddings
        combined_embeddings = torch.cat([text_embeddings, audio_tokens], dim=1)
        
        # Predict noise
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=combined_embeddings
        ).sample
        
        # Calculate loss
        loss = nn.functional.mse_loss(model_pred, noise, reduction="mean")
        
        return loss
    
    def validate(self):
        """Run validation."""
        self.unet.eval()
        self.audio_adapter.eval()
        self.attention_adapter.eval()
        
        val_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                loss = self.train_step(batch)
                val_loss += loss.item()
                num_batches += 1
                
                if num_batches >= 50:  # Validate on subset
                    break
        
        avg_val_loss = val_loss / num_batches
        
        self.unet.train()
        self.audio_adapter.train()
        self.attention_adapter.train()
        
        return avg_val_loss
    
    def train(self):
        """Main training loop."""
        print("\n=== Starting Training ===")
        
        # Initialize wandb
        if self.config['training'].get('use_wandb', False):
            wandb.init(
                project="clap2diffusion",
                config=self.config,
                name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Train each stage
        for stage in range(1, 4):
            self.setup_stage(stage)
            self.current_stage = stage
            
            # Training loop for current stage
            progress_bar = tqdm(range(self.num_steps), desc=f"Stage {stage}")
            
            for step in range(self.num_steps):
                # Training step
                self.unet.train()
                self.audio_adapter.train()
                self.attention_adapter.train()
                
                batch = next(iter(self.train_loader))
                
                with self.accelerator.accumulate(self.unet):
                    loss = self.train_step(batch)
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.optimizer.param_groups[0]['params'],
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
                    "stage": stage
                }
                progress_bar.set_postfix(**logs)
                
                # Log to wandb
                if self.config['training'].get('use_wandb', False):
                    wandb.log(logs, step=self.global_step)
                
                # Validation
                if (step + 1) % self.config['training'].get('val_steps', 500) == 0:
                    val_loss = self.validate()
                    print(f"\nValidation loss: {val_loss:.4f}")
                    
                    if self.config['training'].get('use_wandb', False):
                        wandb.log({"val_loss": val_loss}, step=self.global_step)
                
                # Save checkpoint
                if (step + 1) % self.config['training'].get('save_steps', 1000) == 0:
                    self.save_checkpoint(stage, step)
                
                self.global_step += 1
            
            # Save stage checkpoint
            self.save_checkpoint(stage, self.num_steps, final=True)
        
        print("\n=== Training Complete! ===")
    
    def save_checkpoint(self, stage: int, step: int, final: bool = False):
        """Save training checkpoint."""
        checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if final:
            checkpoint_path = checkpoint_dir / f"stage{stage}_final"
        else:
            checkpoint_path = checkpoint_dir / f"stage{stage}_step{step}"
        
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        self.accelerator.save_state(checkpoint_path)
        
        # Save audio adapter and attention adapter separately
        torch.save(
            self.audio_adapter.state_dict(),
            checkpoint_path / "audio_adapter.pt"
        )
        torch.save(
            self.attention_adapter.state_dict(),
            checkpoint_path / "attention_adapter.pt"
        )
        
        print(f"Checkpoint saved to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CLAP2Diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.json",
        help="Path to training configuration file"
    )
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = CLAP2DiffusionTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()