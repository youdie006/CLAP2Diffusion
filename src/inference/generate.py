"""
Simple inference script for CLAP2Diffusion
Command-line interface for audio-conditioned image generation
"""

import argparse
import torch
import sys
from pathlib import Path
from PIL import Image
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.pipeline_audio import AudioDiffusionPipeline


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Generate images with audio and text conditioning")
    
    # Input arguments
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for generation")
    parser.add_argument("--audio", type=str, default=None,
                       help="Path to audio file for conditioning")
    parser.add_argument("--negative_prompt", type=str, 
                       default="low quality, blurry, distorted",
                       help="Negative prompt")
    
    # Generation parameters
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Text guidance scale")
    parser.add_argument("--audio_scale", type=float, default=3.0,
                       help="Audio guidance scale")
    parser.add_argument("--audio_strength", type=float, default=1.0,
                       help="Audio conditioning strength")
    parser.add_argument("--seed", type=int, default=-1,
                       help="Random seed (-1 for random)")
    
    # Output settings
    parser.add_argument("--output", type=str, default="output.png",
                       help="Output image path")
    parser.add_argument("--width", type=int, default=512,
                       help="Image width")
    parser.add_argument("--height", type=int, default=512,
                       help="Image height")
    parser.add_argument("--num_images", type=int, default=1,
                       help="Number of images to generate")
    
    # Model settings
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda/cpu)")
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16 precision")
    
    args = parser.parse_args()
    
    # Set device and dtype
    device = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.fp16 and device == "cuda" else torch.float32
    
    print(f"Initializing pipeline on {device}...")
    
    # Initialize pipeline
    pipeline = AudioDiffusionPipeline(
        device=device,
        dtype=dtype,
        enable_xformers=True
    )
    
    # Load checkpoint if provided
    if args.model_path:
        print(f"Loading checkpoint from {args.model_path}...")
        checkpoint = torch.load(args.model_path, map_location=device)
        if "audio_adapter" in checkpoint:
            pipeline.unet.load_adapter_state_dict(checkpoint["audio_adapter"])
        if "audio_projection" in checkpoint:
            pipeline.audio_projection.load_state_dict(checkpoint["audio_projection"])
    
    # Set seed
    if args.seed == -1:
        args.seed = torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    
    print(f"Generating image with seed {args.seed}...")
    print(f"Prompt: {args.prompt}")
    if args.audio:
        print(f"Audio: {args.audio}")
    
    # Generate images
    result = pipeline(
        prompt=args.prompt,
        audio=args.audio,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        audio_guidance_scale=args.audio_scale,
        audio_strength=args.audio_strength,
        width=args.width,
        height=args.height,
        num_images_per_prompt=args.num_images,
        generator=generator
    )
    
    # Save images
    images = result["images"]
    
    if args.num_images == 1:
        # Save single image
        images[0].save(args.output)
        print(f"Image saved to {args.output}")
    else:
        # Save multiple images
        output_path = Path(args.output)
        base_name = output_path.stem
        extension = output_path.suffix
        
        for i, img in enumerate(images):
            output_file = output_path.parent / f"{base_name}_{i}{extension}"
            img.save(output_file)
            print(f"Image {i+1} saved to {output_file}")
    
    # Save metadata
    metadata = {
        "prompt": args.prompt,
        "audio": args.audio,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "audio_scale": args.audio_scale,
        "audio_strength": args.audio_strength,
        "width": args.width,
        "height": args.height,
        "num_images": args.num_images,
        "timestamp": datetime.now().isoformat()
    }
    
    metadata_path = Path(args.output).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")
    
    print("Generation complete!")


if __name__ == "__main__":
    main()