#!/usr/bin/env python3
"""
Training script for CLAP2Diffusion
Implements 3-stage training: Audio Adapter -> LoRA+Adapter -> Gate Optimization
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
import time
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
        
        # Initialize accelerator with BF16 for better stability
        self.accelerator = Accelerator(
            mixed_precision='bf16' if self.config['training'].get('mixed_precision', True) else 'no',
            gradient_accumulation_steps=self.config['training'].get('gradient_accumulation_steps', 1)
        )
        
        # Set device and dtype (use BF16 for better stability)
        self.device = self.accelerator.device
        self.dtype = torch.bfloat16 if self.config['training'].get('mixed_precision', True) else torch.float32
        
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
        
        # Load base Stable Diffusion with BF16
        model_id = self.config['model'].get('base_model', 'runwayml/stable-diffusion-v1-5')
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.config['training'].get('mixed_precision', True) else torch.float32
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
            input_dim=self.config['model'].get('audio_dim', 512),
            output_dim=self.config['model'].get('text_dim', 768),
            num_tokens=self.config['model'].get('num_audio_tokens', 8)
        )
        
        # Initialize attention adapter
        self.attention_adapter = AudioAdapterAttention(
            dim=self.config['model'].get('attention_dim', 768)
        )
        
        # Move to device with correct dtype
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)
        self.unet = self.unet.to(self.device)
        self.audio_encoder = self.audio_encoder.to(self.device)
        self.audio_adapter = self.audio_adapter.to(self.device, dtype=self.dtype)
        self.attention_adapter = self.attention_adapter.to(self.device, dtype=self.dtype)
        
        # Freeze base models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.audio_encoder.clap_model.requires_grad_(False)
        
        print("Models loaded successfully!")
        
    def collate_fn(self, batch):
        """Custom collate function to handle text tokenization."""
        # Extract data from batch
        audio = torch.stack([item['audio'] for item in batch])
        images = torch.stack([item['image'] for item in batch])
        texts = [item['text'] for item in batch]
        
        # Tokenize text
        text_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        )
        
        return {
            'audio': audio,
            'image': images,
            'text': texts,
            'text_input_ids': text_inputs.input_ids,
            'text_attention_mask': text_inputs.attention_mask
        }
    
    def setup_datasets(self):
        """Initialize training and validation datasets."""
        print("Loading datasets...")
        
        # Training dataset
        self.train_dataset = AudioImageDataset(
            data_root=self.config['data']['data_dir'],
            split="train",
            audio_sample_rate=self.config['data'].get('sample_rate', 48000),
            audio_duration=self.config['data'].get('audio_duration', 10.0),
            image_size=self.config['data'].get('image_size', 512),
            augment=True
        )
        
        # Validation dataset
        self.val_dataset = AudioImageDataset(
            data_root=self.config['data']['data_dir'],
            split="val",
            audio_sample_rate=self.config['data'].get('sample_rate', 48000),
            audio_duration=self.config['data'].get('audio_duration', 10.0),
            image_size=self.config['data'].get('image_size', 512),
            augment=False
        )
        
        # Create dataloaders with optimization
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training'].get('batch_size', 4),
            shuffle=True,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True,
            collate_fn=self.collate_fn,
            prefetch_factor=2,
            persistent_workers=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training'].get('batch_size', 4),
            shuffle=False,
            num_workers=self.config['training'].get('num_workers', 4),
            pin_memory=True,
            collate_fn=self.collate_fn
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
        
        # Learning rate scheduler with more warmup
        self.lr_scheduler = get_scheduler(
            "cosine",
            optimizer=self.optimizer,
            num_warmup_steps=min(1000, self.num_steps // 5),  # Increased warmup
            num_training_steps=self.num_steps
        )
        
        # Prepare with accelerator (exclude optimizer to avoid FP16 gradient issues)
        self.unet, self.audio_adapter, self.attention_adapter, self.train_loader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.unet, self.audio_adapter, self.attention_adapter, 
                self.train_loader, self.lr_scheduler
            )
        
    def train_step(self, batch):
        """Single training step."""
        # Get batch data
        images = batch['image'].to(self.device, dtype=self.dtype)
        audio = batch['audio'].to(self.device, dtype=self.dtype) 
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
        
        # Debug: Check shapes
        if audio_embeddings.dim() == 1:
            audio_embeddings = audio_embeddings.unsqueeze(0)
        
        # Ensure batch dimension matches
        if audio_embeddings.shape[0] != text_embeddings.shape[0]:
            # If audio batch is smaller, expand it
            if audio_embeddings.shape[0] == 1:
                audio_embeddings = audio_embeddings.expand(text_embeddings.shape[0], -1)
        
        # Convert to correct dtype before projection
        audio_embeddings = audio_embeddings.to(dtype=text_embeddings.dtype)
        
        # Project audio to text space
        audio_tokens = self.audio_adapter(audio_embeddings)
        
        # Ensure audio_tokens has correct dtype
        audio_tokens = audio_tokens.to(dtype=text_embeddings.dtype)
        
        # Combine text and audio embeddings
        combined_embeddings = torch.cat([text_embeddings, audio_tokens], dim=1)
        
        # Predict noise
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=combined_embeddings
        ).sample
        
        # Calculate loss (in float32 for stability)
        loss = nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
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
        
        # Create main log file
        log_dir = Path(self.config['training'].get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True, parents=True)
        main_log_path = log_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        main_log = open(main_log_path, "w")
        main_log.write(f"CLAP2Diffusion Training Log\n")
        main_log.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        main_log.write(f"Configuration:\n")
        main_log.write(json.dumps(self.config, indent=2) + "\n\n")
        main_log.flush()
        
        # Train each stage
        for stage in range(1, 4):
            self.setup_stage(stage)
            self.current_stage = stage
            
            # Stage-specific log
            stage_log_path = log_dir / f"stage{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            stage_log = open(stage_log_path, "w")
            stage_log.write(f"Stage {stage} Training Log\n")
            stage_log.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            stage_log.write(f"Total steps: {self.num_steps}\n")
            stage_log.write(f"Learning rate: {self.lr_scheduler.get_last_lr()[0]}\n\n")
            stage_log.flush()
            
            stage_start_time = time.time()
            
            # Training loop for current stage
            progress_bar = tqdm(range(self.num_steps), desc=f"Stage {stage}")
            
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
                    stage_log.write(f"\nRestarted data loader at step {step}\n")
                    stage_log.flush()
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)
                
                # Use appropriate model for accumulation based on stage
                accumulate_model = self.audio_adapter if self.current_stage == 1 else self.unet
                
                with self.accelerator.accumulate(accumulate_model):
                    loss = self.train_step(batch)
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping (native PyTorch since optimizer is not prepared)
                    if self.accelerator.sync_gradients:
                        # Clip gradients for trainable parameters only
                        if self.current_stage == 1:
                            params_to_clip = self.audio_adapter.parameters()
                        elif self.current_stage == 2:
                            # Only clip trainable LoRA parameters
                            params_to_clip = [p for p in self.unet.parameters() if p.requires_grad]
                            params_to_clip.extend(self.audio_adapter.parameters())
                        else:  # stage 3
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
                    "stage": stage
                }
                progress_bar.set_postfix(**logs)
                
                # Write to log files
                if step % 10 == 0:  # Log every 10 steps
                    log_entry = f"Step {step}/{self.num_steps}: loss={logs['loss']:.6f}, lr={logs['lr']:.2e}\n"
                    stage_log.write(log_entry)
                    stage_log.flush()
                    main_log.write(f"Stage {stage} - {log_entry}")
                    main_log.flush()
                
                # Log to wandb
                if self.config['training'].get('use_wandb', False):
                    wandb.log(logs, step=self.global_step)
                
                # Validation
                if (step + 1) % self.config['training'].get('val_steps', 500) == 0:
                    val_loss = self.validate()
                    print(f"\nValidation loss: {val_loss:.4f}")
                    
                    val_log_entry = f"\nValidation at step {step}: loss={val_loss:.4f}\n"
                    stage_log.write(val_log_entry)
                    stage_log.flush()
                    main_log.write(f"Stage {stage} - {val_log_entry}")
                    main_log.flush()
                    
                    if self.config['training'].get('use_wandb', False):
                        wandb.log({"val_loss": val_loss}, step=self.global_step)
                
                # Save checkpoint
                if (step + 1) % self.config['training'].get('save_steps', 1000) == 0:
                    self.save_checkpoint(stage, step)
                    checkpoint_log = f"\nCheckpoint saved at step {step}\n"
                    stage_log.write(checkpoint_log)
                    stage_log.flush()
                    main_log.write(f"Stage {stage} - {checkpoint_log}")
                    main_log.flush()
                
                self.global_step += 1
            
            # Save stage checkpoint
            self.save_checkpoint(stage, self.num_steps, final=True)
            
            # Stage completion log
            stage_time = time.time() - stage_start_time
            completion_log = f"\nStage {stage} completed in {stage_time/60:.2f} minutes\n"
            completion_log += f"Final loss: {logs['loss']:.6f}\n"
            completion_log += f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            stage_log.write(completion_log)
            stage_log.close()
            
            main_log.write(completion_log)
            main_log.flush()
            
            print(f"\nStage {stage} completed in {stage_time/60:.2f} minutes")
        
        print("\n=== Training Complete! ===")
        
        # Final log entry
        main_log.write(f"\nTraining completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        main_log.close()
        
        print(f"Training logs saved to: {log_dir}")
    
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