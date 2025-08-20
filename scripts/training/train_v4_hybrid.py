"""
Training script for CLAP2Diffusion V4 Hybrid Architecture
Combines hierarchical audio decomposition with gated attention
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator
from diffusers import DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import wandb
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict, List, Optional, Tuple, Union

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet_hybrid_audio_v4 import HybridAudioConditionedUNet
from src.data.audiocaps_hierarchical_v4 import AudioCapsHierarchicalDataLoader


class CLAP2DiffusionV4Trainer:
    """
    Trainer for V4 Hybrid Architecture
    Implements multi-stage training strategy
    """
    
    def __init__(
        self,
        config_path: str,
        output_dir: str,
        resume_from: Optional[str] = None
    ):
        """
        Initialize trainer with configuration
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            mixed_precision='fp16' if self.config.get('mixed_precision', True) else 'no',
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 4)
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
        
        # Resume if checkpoint provided
        if resume_from:
            self.load_checkpoint(resume_from)
    
    def _init_models(self):
        """Initialize all model components"""
        config = self.config['model']
        
        # Load base models
        self.tokenizer = CLIPTokenizer.from_pretrained(
            config['base_model'], 
            subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            config['base_model'],
            subfolder="text_encoder"
        )
        self.text_encoder.requires_grad_(False)
        
        # Load base UNet
        self.base_unet = UNet2DConditionModel.from_pretrained(
            config['base_model'],
            subfolder="unet"
        )
        
        # Apply LoRA if enabled
        if self.config['training'].get('use_lora', False):
            from peft import LoraConfig, get_peft_model
            
            # Important: Don't freeze base model before applying LoRA
            # PEFT will handle freezing the right parameters
            
            lora_config = LoraConfig(
                r=self.config['training'].get('lora_rank', 16),
                lora_alpha=self.config['training'].get('lora_alpha', 32),
                lora_dropout=self.config['training'].get('lora_dropout', 0.1),
                target_modules=["to_k", "to_q", "to_v", "to_out.0"]
                # Note: TaskType.DIFFUSION not available in all PEFT versions
            )
            
            # Apply LoRA to base UNet
            self.base_unet = get_peft_model(self.base_unet, lora_config)
            if self.accelerator.is_main_process:
                self.base_unet.print_trainable_parameters()
        
        # Create hybrid UNet with audio conditioning
        self.unet = HybridAudioConditionedUNet(
            base_unet=self.base_unet,
            audio_dim=config.get('audio_dim', 512),
            use_adapter_list=config.get('use_adapter_list', [False, True, True]),
            hierarchy_levels=config.get('hierarchy_levels', ["foreground", "full", "ambience"]),
            freeze_base_unet=True,  # Always freeze base, LoRA handles trainable params
            use_4layer_projection=config.get('use_4layer_projection', True),
            initial_gate_values=config.get('initial_gate_values', [0.0, 0.0, 0.0])
        )
        
        # Noise scheduler
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            config['base_model'],
            subfolder="scheduler"
        )
        
        # VAE for encoding/decoding images
        from diffusers import AutoencoderKL
        self.vae = AutoencoderKL.from_pretrained(
            config['base_model'],
            subfolder="vae"
        )
        self.vae.requires_grad_(False)
        
        # Disable xformers for now - causing performance issues with custom audio features
        # XFormers cannot handle our custom cross_attention_kwargs properly
        if self.accelerator.is_main_process:
            print("⚠ xformers disabled - using default attention for better compatibility")
        
        # Enable gradient checkpointing if specified
        if self.config['training'].get('gradient_checkpointing', False):
            self.base_unet.enable_gradient_checkpointing()
            if self.accelerator.is_main_process:
                print("✓ Gradient checkpointing enabled")
        
        # Move models to device
        self.text_encoder = self.text_encoder.to(self.device)
        self.vae = self.vae.to(self.device)
        self.unet = self.unet.to(self.device)
        
        # Print trainable parameters
        if self.accelerator.is_main_process:
            self.unet.print_trainable_params()
    
    def _init_data_loaders(self):
        """Initialize data loaders"""
        data_config = self.config['data']
        
        # Create custom collate function with tokenization
        def custom_collate_fn(batch):
            """Collate function that includes text tokenization"""
            # Stack tensors
            audio = torch.stack([item['audio'] for item in batch])
            images = torch.stack([item['image'] for item in batch])
            
            # Tokenize captions
            captions = [item['caption'] for item in batch]
            text_inputs = self.tokenizer(
                captions,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            
            # Collect other data
            hierarchies = [item['hierarchy'] for item in batch]
            composition_types = [item['composition_type'] for item in batch]
            metadata = [item['metadata'] for item in batch]
            
            return {
                'audio': audio,
                'images': images,
                'text_inputs': text_inputs,
                'captions': captions,
                'hierarchies': hierarchies,
                'composition_types': composition_types,
                'metadata': metadata
            }
        
        # Create data loader wrapper
        self.data_loader_wrapper = AudioCapsHierarchicalDataLoader(
            data_root=data_config['data_root'],
            metadata_path=data_config['metadata_path'],
            batch_size=data_config['batch_size'],
            num_workers=data_config.get('num_workers', 4),
            composition_strategy=data_config.get('composition_strategy', 'balanced'),
            max_samples=data_config.get('max_samples', None)  # For debugging
        )
        
        # Create custom data loaders with our collate function
        from torch.utils.data import DataLoader
        
        self.train_loader = DataLoader(
            self.data_loader_wrapper.train_dataset,
            batch_size=data_config['batch_size'],
            shuffle=True,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            drop_last=True,
            collate_fn=custom_collate_fn
        )
        
        self.val_loader = DataLoader(
            self.data_loader_wrapper.val_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        self.test_loader = DataLoader(
            self.data_loader_wrapper.test_dataset,
            batch_size=data_config['batch_size'],
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=custom_collate_fn
        )
        
        # Print statistics
        if self.accelerator.is_main_process:
            self.data_loader_wrapper.print_statistics()
    
    def _init_training(self):
        """Initialize training components"""
        train_config = self.config['training']
        
        # Get current stage configuration
        self.current_stage = train_config.get('current_stage', 1)
        stage_key = f"stage{self.current_stage}"
        stage_config = self.config[stage_key]
        
        # Optimizer
        self.optimizer = AdamW(
            self.unet.get_trainable_parameters(),
            lr=stage_config['learning_rate'],
            weight_decay=stage_config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=stage_config['num_steps'],
            eta_min=stage_config['learning_rate'] * 0.1
        )
        
        # Prepare for distributed training
        # IMPORTANT: Don't prepare optimizer to avoid gradient scaler conflicts
        # See TROUBLESHOOTING.md section 1.1
        self.unet, self.train_loader = \
            self.accelerator.prepare(
                self.unet, self.train_loader
            )
        
        # Ensure model is in training mode
        self.unet.train()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
    
    def _init_logging(self):
        """Initialize logging (WandB)"""
        if self.config.get('use_wandb', True):
            wandb.init(
                project="CLAP2Diffusion-V4",
                name=f"v4_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config
            )
    
    def train(self):
        """Main training loop"""
        stage_config = self.config[f"stage{self.current_stage}"]
        num_steps = stage_config['num_steps']
        
        print(f"\n{'='*50}")
        print(f"Starting Stage {self.current_stage} Training")
        print(f"Steps: {num_steps}")
        print(f"Learning rate: {stage_config['learning_rate']}")
        print(f"{'='*50}\n")
        
        # Training loop
        progress_bar = tqdm(
            total=num_steps,
            desc=f"Stage {self.current_stage}",
            disable=not self.accelerator.is_main_process
        )
        
        while self.global_step < num_steps:
            for batch in self.train_loader:
                if self.global_step >= num_steps:
                    break
                
                # Use accelerator's accumulate context for proper gradient accumulation
                with self.accelerator.accumulate(self.unet):
                    # Training step
                    loss, loss_dict = self.training_step(batch)
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Only perform optimizer operations when gradients are synced
                    if self.accelerator.sync_gradients:
                        # Gradient clipping - use native PyTorch since optimizer isn't prepared
                        if self.config['training'].get('max_grad_norm', 1.0) > 0:
                            # Get only parameters that require gradients
                            trainable_params = [p for p in self.unet.parameters() if p.requires_grad]
                            if trainable_params:
                                torch.nn.utils.clip_grad_norm_(
                                    trainable_params,
                                    self.config['training']['max_grad_norm']
                                )
                        
                        # Optimizer step and zero gradients
                        self.optimizer.step()
                        self.lr_scheduler.step()
                        self.optimizer.zero_grad()
                        
                        # Update progress after actual step
                        self.global_step += 1
                        progress_bar.update(1)
                        
                        # Logging - log every step for better monitoring
                        if self.accelerator.is_main_process:
                            self.log_metrics(loss_dict)
                            
                            # Also print to console every 10 steps
                            if self.global_step % 10 == 0:
                                print(f"Step {self.global_step}: loss={loss_dict.get('total_loss', 0):.4f}, "
                                      f"lr={self.lr_scheduler.get_last_lr()[0]:.2e}")
                        
                        # Validation
                        if self.global_step % 500 == 0:
                            self.validate()
                        
                        # Save checkpoint
                        if self.global_step % 1000 == 0:
                            self.save_checkpoint()
        
        progress_bar.close()
        
        # Stage transition
        if self.current_stage < 3:
            self.transition_to_next_stage()
    
    def training_step(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """Single training step"""
        # Move batch to device
        images = batch['images'].to(self.device)
        audio = batch['audio'].to(self.device)
        captions = batch['captions']
        hierarchies = batch['hierarchies']
        composition_types = batch['composition_types']
        
        # Encode text
        text_inputs = self.tokenizer(
            captions,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_inputs.input_ids)[0]
        
        # Process audio and combine with text embeddings
        # This is the key optimization - directly combine embeddings
        if audio is not None:
            # Get audio tokens from the encoder
            audio_tokens, audio_hierarchy = self.unet.audio_encoder(
                audio, 
                return_hierarchy=True
            )
            
            # Ensure audio_tokens has the right shape [B, 77, 768]
            if audio_tokens.dim() == 2:
                audio_tokens = audio_tokens.unsqueeze(1).expand(-1, 77, -1)
            elif audio_tokens.shape[1] != 77:
                # If not 77 tokens, interpolate or truncate
                if audio_tokens.shape[1] > 77:
                    audio_tokens = audio_tokens[:, :77, :]
                else:
                    # Pad with zeros if less than 77
                    padding = torch.zeros(
                        audio_tokens.shape[0], 
                        77 - audio_tokens.shape[1], 
                        audio_tokens.shape[2],
                        device=audio_tokens.device
                    )
                    audio_tokens = torch.cat([audio_tokens, padding], dim=1)
            
            # Combine text and audio embeddings using weighted addition
            # This is much faster than complex adapter systems
            alpha = 0.3  # Audio influence weight
            enhanced_embeddings = text_embeddings + alpha * audio_tokens
        else:
            enhanced_embeddings = text_embeddings
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Sample noise
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        
        # Add noise to latents
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Get f_multiplier based on composition type
        f_multipliers = self._get_f_multipliers(composition_types)
        
        # Predict noise - now using enhanced embeddings
        # No need to pass audio separately since it's already combined
        unet_output = self.unet(
            sample=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=enhanced_embeddings,
            audio_input=None,  # Don't pass audio separately
            f_multiplier=f_multipliers.mean().item()  # Use mean for batch
        )
        
        # Extract noise prediction - handle both dict and tensor outputs
        if isinstance(unet_output, dict):
            noise_pred = unet_output['sample']
        else:
            noise_pred = unet_output
        
        # Ensure noise_pred requires grad if any model parameters do
        if not noise_pred.requires_grad and any(p.requires_grad for p in self.unet.parameters()):
            # This shouldn't happen, but let's add a check
            print(f"Warning: noise_pred doesn't require grad but model has trainable params")
            print(f"noise_pred.requires_grad: {noise_pred.requires_grad}")
            print(f"noise_pred.is_leaf: {noise_pred.is_leaf}")
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise, reduction="mean")
        
        # Additional losses based on stage
        loss_dict = {'mse_loss': loss.item()}
        
        if self.current_stage == 2:
            # Add hierarchy consistency loss
            hierarchy_loss = self._calculate_hierarchy_loss(noise_pred, hierarchies)
            loss = loss + 0.1 * hierarchy_loss
            loss_dict['hierarchy_loss'] = hierarchy_loss.item()
        
        loss_dict['total_loss'] = loss.item()
        
        return loss, loss_dict
    
    def _get_f_multipliers(self, composition_types: List[str]) -> torch.Tensor:
        """Get f_multiplier values based on composition types"""
        multipliers = []
        for comp_type in composition_types:
            if comp_type == "matching":
                multipliers.append(0.8)
            elif comp_type == "complementary":
                multipliers.append(0.6)
            elif comp_type == "creative":
                multipliers.append(0.4)
            else:  # contradictory
                multipliers.append(0.2)
        
        return torch.tensor(multipliers, device=self.device)
    
    def _calculate_hierarchy_loss(self, pred: torch.Tensor, hierarchies: List[Dict]) -> torch.Tensor:
        """Calculate hierarchy consistency loss"""
        # Placeholder for hierarchy consistency loss
        # In practice, this would check if the model is properly using
        # different hierarchy levels at different UNet depths
        return torch.tensor(0.0, device=self.device)
    
    def validate(self):
        """Validation step"""
        self.unet.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", 
                            disable=not self.accelerator.is_main_process):
                loss, _ = self.training_step(batch)
                val_losses.append(loss.item())
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        if self.accelerator.is_main_process:
            print(f"\nValidation Loss: {avg_val_loss:.4f}")
            
            if self.config.get('use_wandb', True):
                wandb.log({
                    'val_loss': avg_val_loss,
                    'global_step': self.global_step
                })
            
            # Save best model
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_checkpoint(is_best=True)
        
        self.unet.train()
    
    def log_metrics(self, loss_dict: Dict):
        """Log metrics to WandB"""
        if self.config.get('use_wandb', True):
            log_dict = {
                **loss_dict,
                'learning_rate': self.lr_scheduler.get_last_lr()[0],
                'global_step': self.global_step,
                'epoch': self.epoch
            }
            
            # Add gate values
            gate_values = self.unet.get_gate_values()
            for name, value in gate_values.items():
                log_dict[f'gate/{name}'] = value
            
            wandb.log(log_dict)
    
    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint"""
        if not self.accelerator.is_main_process:
            return
        
        checkpoint = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'current_stage': self.current_stage,
            'model_state_dict': self.accelerator.unwrap_model(self.unet).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'config': self.config
        }
        
        # Save checkpoint
        if is_best:
            path = self.output_dir / 'best_model.pt'
        else:
            path = self.output_dir / f'checkpoint_step_{self.global_step}.pt'
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
        
        # Save gate parameters separately (like SonicDiffusion)
        gate_path = self.output_dir / f'gates_stage{self.current_stage}.pt'
        self.accelerator.unwrap_model(self.unet).save_gates(gate_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.current_stage = checkpoint['current_stage']
        
        self.unet.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming from step {self.global_step}, stage {self.current_stage}")
    
    def transition_to_next_stage(self):
        """Transition to next training stage"""
        print(f"\n{'='*50}")
        print(f"Transitioning from Stage {self.current_stage} to Stage {self.current_stage + 1}")
        print(f"{'='*50}\n")
        
        self.current_stage += 1
        self.global_step = 0
        
        # Reinitialize training components for new stage
        self._init_training()
        
        # Continue training
        self.train()


def main():
    parser = argparse.ArgumentParser(description="Train CLAP2Diffusion V4 Hybrid")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='./outputs/v4_hybrid',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = CLAP2DiffusionV4Trainer(
        config_path=args.config,
        output_dir=args.output_dir,
        resume_from=args.resume
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()