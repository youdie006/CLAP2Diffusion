#!/usr/bin/env python
"""
Optimized V4 Training Script
Simplified architecture for faster training
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import wandb
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hierarchical_audio_v4 import HierarchicalAudioEncoder
from src.data.audiocaps_hierarchical_v4 import AudioCapsHierarchicalDataLoader


class OptimizedV4Trainer:
    """Optimized trainer with simplified audio conditioning"""
    
    def __init__(self, config_path: str, output_dir: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file
        self.log_file = self.output_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.loss_history = []
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision='fp16',  # Enable mixed precision for speed
            gradient_accumulation_steps=1
        )
        
        self.device = self.accelerator.device
        
        # Initialize models
        self._init_models()
        
        # Initialize data loaders
        self._init_data_loaders()
        
        # Initialize training components
        self._init_training()
        
        # Initialize logging
        if self.accelerator.is_main_process:
            self._init_logging()
    
    def _init_models(self):
        """Initialize all model components"""
        config = self.config['model']
        
        # Load base models
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config['base_model'], subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            config['base_model'], subfolder="text_encoder"
        )
        self.text_encoder.requires_grad_(False)
        
        # Load base UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            config['base_model'], subfolder="unet"
        )
        
        # Apply LoRA
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        )
        
        self.unet = get_peft_model(self.unet, lora_config)
        
        # Audio encoder
        self.audio_encoder = HierarchicalAudioEncoder(
            freeze_clap=True,
            audio_dim=512,
            text_dim=768
        )
        
        # VAE
        self.vae = AutoencoderKL.from_pretrained(
            config['base_model'], subfolder="vae"
        )
        self.vae.requires_grad_(False)
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config['base_model'], subfolder="scheduler"
        )
        
        # Enable xformers for speed
        try:
            self.unet.enable_xformers_memory_efficient_attention()
            if self.accelerator.is_main_process:
                print("✓ XFormers enabled for memory efficient attention")
        except:
            if self.accelerator.is_main_process:
                print("⚠ XFormers not available, using default attention")
        
        # Move models to device
        self.text_encoder = self.text_encoder.to(self.device)
        self.vae = self.vae.to(self.device)
        self.unet = self.unet.to(self.device)
        self.audio_encoder = self.audio_encoder.to(self.device)
        
        if self.accelerator.is_main_process:
            self.unet.print_trainable_parameters()
    
    def _init_data_loaders(self):
        """Initialize data loaders"""
        data_config = self.config['data']
        
        # Custom collate function
        def custom_collate_fn(batch):
            audio = torch.stack([item['audio'] for item in batch])
            images = torch.stack([item['image'] for item in batch])
            captions = [item['caption'] for item in batch]
            return {
                'audio': audio,
                'images': images,
                'captions': captions
            }
        
        # Create data loader wrapper
        self.data_loader_wrapper = AudioCapsHierarchicalDataLoader(
            data_root=data_config['data_root'],
            metadata_path=data_config['metadata_path'],
            batch_size=4,  # Increase batch size since we optimized
            num_workers=2,  # Enable workers for faster loading
            composition_strategy='matching',
            max_samples=5000
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.data_loader_wrapper.train_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )
        
        if self.accelerator.is_main_process:
            self.data_loader_wrapper.print_statistics()
    
    def _init_training(self):
        """Initialize training components"""
        # Collect trainable parameters
        trainable_params = []
        trainable_params.extend(self.unet.parameters())
        trainable_params.extend(self.audio_encoder.decomposer.parameters())
        
        # Optimizer
        self.optimizer = AdamW(
            trainable_params,
            lr=1e-4,
            weight_decay=0.01
        )
        
        # Prepare for distributed training
        self.unet, self.audio_encoder, self.train_loader, self.optimizer = \
            self.accelerator.prepare(
                self.unet, self.audio_encoder, self.train_loader, self.optimizer
            )
        
        # Training state
        self.global_step = 0
    
    def _init_logging(self):
        """Initialize logging"""
        # Try to initialize WandB with environment variables
        try:
            # Get credentials from environment variables
            entity = os.getenv('WANDB_ENTITY')
            project = os.getenv('WANDB_PROJECT', 'CLAP2Diffusion-V4')
            api_key = os.getenv('WANDB_API_KEY')
            
            if not api_key:
                print("⚠ WANDB_API_KEY not found in environment variables")
                print("⚠ Please create a .env file with your WandB credentials")
                self.wandb_run = None
                return
            
            # Set API key
            os.environ['WANDB_API_KEY'] = api_key
            
            self.wandb_run = wandb.init(
                entity=entity,
                project=project,
                name=f"v4_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "learning_rate": 1e-4,
                    "batch_size": 4,
                    "num_steps": 3000,
                    "architecture": "V4 Optimized",
                    "audio_weight": 0.3,
                    **self.config
                },
                reinit=True
            )
            print(f"✓ WandB logging initialized successfully")
            print(f"  Entity: {entity}")
            print(f"  Project: {project}")
        except Exception as e:
            print(f"⚠ WandB initialization failed: {e}")
            print("⚠ Continuing without WandB logging")
            self.wandb_run = None
    
    def train(self):
        """Main training loop"""
        num_steps = 3000
        
        print(f"\n{'='*50}")
        print(f"Starting Optimized V4 Training")
        print(f"Steps: {num_steps}")
        print(f"Batch size: 4")
        print(f"Expected speed: 10-20 seconds/step")
        print(f"{'='*50}\n")
        
        progress_bar = tqdm(
            total=num_steps,
            desc="Training",
            disable=not self.accelerator.is_main_process
        )
        
        self.unet.train()
        self.audio_encoder.train()
        
        while self.global_step < num_steps:
            for batch in self.train_loader:
                if self.global_step >= num_steps:
                    break
                
                with self.accelerator.accumulate(self.unet):
                    # Training step
                    loss = self.training_step(batch)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    self.accelerator.clip_grad_norm_(
                        self.unet.parameters(),
                        1.0
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Update progress
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    # Logging
                    if self.accelerator.is_main_process:
                        if self.global_step % 10 == 0:
                            loss_value = loss.item()
                            lr_value = self.optimizer.param_groups[0]['lr']
                            
                            # Log to WandB if available
                            if self.wandb_run is not None:
                                self.wandb_run.log({
                                    'loss': loss_value,
                                    'lr': lr_value,
                                    'step': self.global_step
                                })
                            
                            # Log to file
                            log_entry = f"Step {self.global_step}: loss={loss_value:.4f}, lr={lr_value:.2e}\n"
                            with open(self.log_file, 'a') as f:
                                f.write(log_entry)
                            
                            # Store in history
                            self.loss_history.append({
                                'step': self.global_step,
                                'loss': loss_value,
                                'lr': lr_value,
                                'timestamp': datetime.now().isoformat()
                            })
                            
                            print(f"Step {self.global_step}: loss={loss_value:.4f}")
                    
                    # Save checkpoint
                    if self.global_step % 500 == 0:
                        self.save_checkpoint()
        
        progress_bar.close()
    
    def training_step(self, batch):
        """Single training step"""
        # Move batch to device
        images = batch['images'].to(self.device)
        audio = batch['audio'].to(self.device)
        captions = batch['captions']
        
        # Encode text
        text_inputs = self.tokenizer(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
        
        # Process audio and get tokens
        audio_tokens, _ = self.audio_encoder(audio, return_hierarchy=True)
        
        # Ensure correct shape [B, 77, 768]
        if audio_tokens.shape[1] != 77:
            # Interpolate to 77 tokens if needed
            audio_tokens = F.interpolate(
                audio_tokens.transpose(1, 2),
                size=77,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        # Combine embeddings (key optimization)
        alpha = 0.3
        enhanced_embeddings = text_embeddings + alpha * audio_tokens
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=self.device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=enhanced_embeddings,
            return_dict=False
        )[0]
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        return loss
    
    def save_checkpoint(self):
        """Save training checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint = {
            'global_step': self.global_step,
            'unet_state_dict': self.accelerator.unwrap_model(self.unet).state_dict(),
            'audio_encoder_state_dict': self.accelerator.unwrap_model(self.audio_encoder).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        path = self.output_dir / f'checkpoint_step_{self.global_step}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")


def main():
    # Create trainer
    trainer = OptimizedV4Trainer(
        config_path="/mnt/d/MyProject/CLAP2Diffusion/configs/training_config_v4_hybrid.json",
        output_dir="./outputs/v4_optimized"
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()