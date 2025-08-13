#!/usr/bin/env python3
"""
Simple test script for CLAP2Diffusion image generation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import torchaudio
import argparse
from datetime import datetime
from diffusers import StableDiffusionPipeline, DDIMScheduler
import json

from src.models.audio_encoder import CLAPAudioEncoder
from src.models.audio_adapter import AudioProjectionMLP
from src.models.attention_adapter import AudioAdapterAttention


def load_models(checkpoint_path="checkpoints/stage3_final", device="cuda"):
    """Load trained models"""
    print(f"Loading models from {checkpoint_path}...")
    
    # Load base SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    # Load CLAP encoder
    print("Loading CLAP encoder...")
    clap_encoder = CLAPAudioEncoder(
        model_name="laion/larger_clap_music_and_speech",
        device=device
    )
    
    # Load audio adapter
    print("Loading audio adapter...")
    audio_adapter = AudioProjectionMLP(
        input_dim=512,
        output_dim=768,
        num_tokens=8
    ).to(device)
    
    checkpoint_path = Path(checkpoint_path)
    audio_adapter_state = torch.load(
        checkpoint_path / "audio_adapter.pt",
        map_location=device
    )
    audio_adapter.load_state_dict(audio_adapter_state)
    audio_adapter.eval()
    
    # Load attention adapter (for gate value)
    attention_state = torch.load(
        checkpoint_path / "attention_adapter.pt",
        map_location=device
    )
    gate_value = attention_state.get('gate', torch.zeros(1)).item()
    print(f"Gate value: {gate_value:.4f}, tanh(gate): {np.tanh(gate_value):.4f}")
    
    return pipe, clap_encoder, audio_adapter, gate_value


def encode_audio(audio_path, clap_encoder, device="cuda"):
    """Encode audio file to CLAP embeddings"""
    print(f"Encoding audio: {audio_path}")
    
    # Load audio
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample to 48kHz if needed (CLAP requirement)
    if sr != 48000:
        resampler = torchaudio.transforms.Resample(sr, 48000)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Move to device
    waveform = waveform.to(device)
    
    # Encode with CLAP
    with torch.no_grad():
        audio_embeds = clap_encoder.encode_audio(waveform)
    
    return audio_embeds


def generate_image(
    pipe,
    audio_adapter,
    audio_embeds,
    text_prompt="",
    negative_prompt="",
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42,
    gate_scale=1.0
):
    """Generate image from audio embeddings and optional text"""
    
    # Set seed for reproducibility
    generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    # Project audio to text space
    with torch.no_grad():
        audio_tokens = audio_adapter(audio_embeds)
        audio_tokens = audio_tokens.view(1, 8, 768)  # [batch, num_tokens, dim]
        
        # Apply gate scaling (simulate gate effect)
        audio_tokens = audio_tokens * gate_scale
    
    # Handle text encoding
    if text_prompt:
        # Encode text
        text_inputs = pipe.tokenizer(
            text_prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(pipe.device)
        
        with torch.no_grad():
            text_embeds = pipe.text_encoder(text_inputs.input_ids)[0]
        
        # Combine audio and text embeddings
        # Simple concatenation strategy (can be improved)
        combined_embeds = torch.cat([audio_tokens[:, :4, :], text_embeds[:, 4:, :]], dim=1)
    else:
        # Audio only - pad to match text encoder output size
        padding = torch.zeros(1, 77-8, 768, device=pipe.device, dtype=audio_tokens.dtype)
        combined_embeds = torch.cat([audio_tokens, padding], dim=1)
    
    # Handle negative prompt
    if negative_prompt:
        negative_inputs = pipe.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        ).to(pipe.device)
        
        with torch.no_grad():
            negative_embeds = pipe.text_encoder(negative_inputs.input_ids)[0]
    else:
        negative_embeds = torch.zeros_like(combined_embeds)
    
    # Generate image
    with torch.no_grad():
        image = pipe(
            prompt_embeds=combined_embeds,
            negative_prompt_embeds=negative_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
    
    return image


def main():
    parser = argparse.ArgumentParser(description="Test CLAP2Diffusion generation")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file")
    parser.add_argument("--text", type=str, default="", help="Text prompt (optional)")
    parser.add_argument("--negative", type=str, default="", help="Negative prompt")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stage3_final",
                       help="Checkpoint directory")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gate-scale", type=float, default=1.0, 
                       help="Gate scaling factor (0.0-1.0)")
    args = parser.parse_args()
    
    # Load models
    pipe, clap_encoder, audio_adapter, gate_value = load_models(
        checkpoint_path=args.checkpoint
    )
    
    # Use actual gate value if not specified
    if args.gate_scale == 1.0:
        args.gate_scale = np.tanh(gate_value)
        print(f"Using trained gate scale: {args.gate_scale:.4f}")
    
    # Encode audio
    audio_embeds = encode_audio(args.audio, clap_encoder)
    
    # Generate image
    print("\nGenerating image...")
    print(f"  Audio: {Path(args.audio).name}")
    print(f"  Text: '{args.text}'" if args.text else "  Text: (none)")
    print(f"  Steps: {args.steps}")
    print(f"  Guidance: {args.guidance}")
    print(f"  Seed: {args.seed}")
    
    image = generate_image(
        pipe,
        audio_adapter,
        audio_embeds,
        text_prompt=args.text,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        gate_scale=args.gate_scale
    )
    
    # Save image
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_name = Path(args.audio).stem
        output_path = Path("results") / f"{audio_name}_{timestamp}.png"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"\n✓ Image saved to: {output_path}")
    
    # Save metadata
    metadata = {
        "audio": str(args.audio),
        "text": args.text,
        "negative": args.negative,
        "checkpoint": args.checkpoint,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "gate_value": gate_value,
        "gate_scale": args.gate_scale,
        "timestamp": datetime.now().isoformat()
    }
    
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()