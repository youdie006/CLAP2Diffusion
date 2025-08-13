#!/usr/bin/env python3
"""
Test image generation with different checkpoints
Compare quality across training stages
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import json
import argparse
from datetime import datetime
import torchaudio
from diffusers import StableDiffusionPipeline, DDIMScheduler

from src.models.audio_encoder import CLAPAudioEncoder
from src.models.audio_adapter import AudioProjectionMLP
from src.models.attention_adapter import AudioAdapterAttention
from src.models.unet_with_audio import AudioConditionedUNet


class CheckpointTester:
    def __init__(self, device="cuda"):
        self.device = device
        self.base_model = "runwayml/stable-diffusion-v1-5"
        
        # Load base pipeline
        print("Loading base Stable Diffusion pipeline...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        # Set scheduler
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        # Load CLAP
        print("Loading CLAP encoder...")
        self.clap_encoder = CLAPAudioEncoder(
            model_name="laion/larger_clap_music_and_speech",
            device=device
        )
        
        # Initialize components (will load weights for each checkpoint)
        self.audio_adapter = None
        self.attention_adapter = None
        
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint weights"""
        checkpoint_path = Path(checkpoint_path)
        print(f"\nLoading checkpoint: {checkpoint_path.name}")
        
        # Load audio adapter
        audio_adapter_path = checkpoint_path / "audio_adapter.pt"
        if audio_adapter_path.exists():
            self.audio_adapter = AudioProjectionMLP(
                audio_dim=512,
                text_dim=768,
                num_tokens=8
            ).to(self.device)
            self.audio_adapter.load_state_dict(
                torch.load(audio_adapter_path, map_location=self.device)
            )
            self.audio_adapter.eval()
            print(f"  Loaded audio adapter")
        
        # Load attention adapter
        attention_adapter_path = checkpoint_path / "attention_adapter.pt"
        if attention_adapter_path.exists():
            # Note: This is simplified - actual implementation would need to integrate with UNet
            attention_state = torch.load(attention_adapter_path, map_location=self.device)
            gate_value = attention_state.get('gate', torch.zeros(1)).item()
            print(f"  Gate value: {gate_value:.4f}")
            
        # Load LoRA weights if exist (Stage 2+)
        lora_path = checkpoint_path / "pytorch_lora_weights.safetensors"
        if lora_path.exists():
            from peft import PeftModel
            print(f"  Loading LoRA weights...")
            # This would need proper LoRA loading implementation
            
        return checkpoint_path.name
        
    def generate_image(self, audio_path, text_prompt="", seed=42, num_steps=50):
        """Generate image from audio and text"""
        # Load and encode audio
        audio_embeds = self.encode_audio(audio_path)
        
        # Project audio to text space
        if self.audio_adapter is not None:
            audio_tokens = self.audio_adapter(audio_embeds)
            audio_tokens = audio_tokens.view(1, 8, 768)  # [batch, num_tokens, dim]
        else:
            audio_tokens = None
            
        # Generate image
        torch.manual_seed(seed)
        
        with torch.no_grad():
            # If we have audio tokens, concatenate with text embeddings
            if audio_tokens is not None and text_prompt:
                # Encode text
                text_inputs = self.pipe.tokenizer(
                    text_prompt,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                text_embeds = self.pipe.text_encoder(text_inputs.input_ids)[0]
                
                # Combine audio and text (simplified)
                # In real implementation, this would go through AudioConditionedUNet
                combined_embeds = torch.cat([audio_tokens[:, :8, :], text_embeds[:, 8:, :]], dim=1)
            elif audio_tokens is not None:
                combined_embeds = audio_tokens
            else:
                # Text only
                combined_embeds = self.pipe.encode_prompt(text_prompt)[0]
                
        # Generate
        image = self.pipe(
            prompt_embeds=combined_embeds,
            num_inference_steps=num_steps,
            guidance_scale=7.5
        ).images[0]
        
        return image
        
    def encode_audio(self, audio_path):
        """Encode audio file to embeddings"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(sr, 48000)
            waveform = resampler(waveform)
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        # Encode with CLAP
        with torch.no_grad():
            audio_embeds = self.clap_encoder.encode_audio(waveform)
            
        return audio_embeds
        
    def test_all_checkpoints(self, audio_path, output_dir="results/checkpoint_comparison"):
        """Test all available checkpoints"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define checkpoints to test
        checkpoints = [
            # Stage 1
            "checkpoints/stage1_step499",
            "checkpoints/stage1_step999", 
            "checkpoints/stage1_step1499",
            "checkpoints/stage1_step1999",
            "checkpoints/stage1_step2499",
            "checkpoints/stage1_step2999",
            "checkpoints/stage1_final",
            # Stage 2
            "checkpoints/stage2_step999",
            "checkpoints/stage2_step1999",
            "checkpoints/stage2_step2999",
            "checkpoints/stage2_step3999",
            "checkpoints/stage2_step4999",
            "checkpoints/stage2_step5999",
        ]
        
        # Add Stage 3 if exists
        stage3_path = Path("checkpoints/stage3_final")
        if stage3_path.exists():
            checkpoints.append(str(stage3_path))
            
        # Test each checkpoint
        results = []
        audio_name = Path(audio_path).stem
        
        for checkpoint_path in checkpoints:
            if not Path(checkpoint_path).exists():
                print(f"Skipping {checkpoint_path} - not found")
                continue
                
            # Load checkpoint
            checkpoint_name = self.load_checkpoint(checkpoint_path)
            
            # Generate images with different settings
            settings = [
                {"text": "", "suffix": "audio_only"},
                {"text": "high quality, detailed", "suffix": "with_text"},
            ]
            
            for setting in settings:
                print(f"  Generating with {setting['suffix']}...")
                
                # Generate image
                image = self.generate_image(
                    audio_path,
                    text_prompt=setting["text"],
                    seed=42,
                    num_steps=50
                )
                
                # Save image
                filename = f"{audio_name}_{checkpoint_name}_{setting['suffix']}.png"
                save_path = output_dir / filename
                image.save(save_path)
                print(f"  Saved: {save_path}")
                
                results.append({
                    "checkpoint": checkpoint_name,
                    "audio": audio_name,
                    "text": setting["text"],
                    "image": str(save_path)
                })
                
        # Save results metadata
        metadata_path = output_dir / f"results_{audio_name}.json"
        with open(metadata_path, "w") as f:
            json.dump(results, f, indent=2)
            
        print(f"\nResults saved to {output_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Test checkpoints with audio generation")
    parser.add_argument("--audio", type=str, help="Audio file path", 
                       default="data/vggsound/audio/E9ulxBBGUVc.wav")
    parser.add_argument("--checkpoint", type=str, help="Specific checkpoint to test")
    parser.add_argument("--all", action="store_true", help="Test all checkpoints")
    parser.add_argument("--output", type=str, default="results/checkpoint_comparison",
                       help="Output directory")
    args = parser.parse_args()
    
    # Create tester
    tester = CheckpointTester()
    
    if args.all:
        # Test all checkpoints
        results = tester.test_all_checkpoints(args.audio, args.output)
    elif args.checkpoint:
        # Test specific checkpoint
        tester.load_checkpoint(args.checkpoint)
        image = tester.generate_image(args.audio)
        
        output_path = Path(args.output) / f"test_{Path(args.checkpoint).name}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"Saved to {output_path}")
    else:
        # Test with latest checkpoint
        latest = "checkpoints/stage2_step5999"  # or stage3_final when ready
        tester.load_checkpoint(latest)
        image = tester.generate_image(args.audio)
        
        output_path = Path(args.output) / "test_latest.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()