"""
Gradio Demo Application for CLAP2Diffusion
Interactive web interface for audio-conditioned image generation
"""

import gradio as gr
import torch
import numpy as np
import sys
import os
from pathlib import Path
from PIL import Image
import librosa
import soundfile as sf
from typing import Optional, Tuple, List
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.pipeline_audio import AudioDiffusionPipeline, AudioInpaintingPipeline
from configs import model_config, training_config


class CLAP2DiffusionDemo:
    """
    Gradio demo application for CLAP2Diffusion.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_fp16: bool = True
    ):
        """
        Initialize demo application.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run on
            use_fp16: Whether to use FP16 precision
        """
        self.device = device
        self.dtype = torch.float16 if use_fp16 and device == "cuda" else torch.float32
        
        print(f"Initializing CLAP2Diffusion on {device}...")
        
        # Initialize pipelines
        self.generation_pipeline = AudioDiffusionPipeline(
            device=device,
            dtype=self.dtype,
            enable_xformers=True
        )
        
        self.inpainting_pipeline = AudioInpaintingPipeline(
            device=device,
            dtype=self.dtype,
            enable_xformers=True
        )
        
        # Load checkpoint if provided
        if model_path and os.path.exists(model_path):
            self.load_checkpoint(model_path)
        
        # Example audio descriptions for presets
        self.audio_examples = {
            "Thunder": "Sound of thunder and lightning during a storm",
            "Ocean Waves": "Peaceful ocean waves on a beach",
            "Fire Crackling": "Campfire crackling and burning",
            "Rain": "Heavy rain falling on surfaces",
            "Birds Chirping": "Birds singing in nature",
            "City Traffic": "Urban traffic and city sounds",
            "Wind": "Strong wind blowing through trees",
            "Applause": "Crowd applauding and cheering"
        }
        
        print("Demo initialized successfully!")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "audio_adapter" in checkpoint:
            self.generation_pipeline.unet.load_adapter_state_dict(checkpoint["audio_adapter"])
        if "audio_projection" in checkpoint:
            self.generation_pipeline.audio_projection.load_state_dict(checkpoint["audio_projection"])
        
        print("Checkpoint loaded successfully!")
    
    def process_audio(
        self,
        audio_file: str,
        duration: float = 10.0
    ) -> Tuple[np.ndarray, int]:
        """
        Process uploaded audio file.
        
        Args:
            audio_file: Path to audio file
            duration: Target duration in seconds
            
        Returns:
            Audio array and sample rate
        """
        if audio_file is None:
            return None, None
        
        # Load audio
        audio, sr = librosa.load(audio_file, sr=None, duration=duration)
        
        # Resample to 48kHz for CLAP
        if sr != 48000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
            sr = 48000
        
        return audio, sr
    
    def generate_image(
        self,
        prompt: str,
        audio_file: Optional[str],
        negative_prompt: str,
        num_steps: int,
        guidance_scale: float,
        audio_scale: float,
        audio_strength: float,
        seed: int,
        width: int,
        height: int
    ) -> Tuple[Image.Image, dict]:
        """
        Generate image with audio and text conditioning.
        
        Returns:
            Generated image and metadata
        """
        # Set seed for reproducibility
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Process audio if provided
        audio_array = None
        if audio_file:
            audio_array, sr = self.process_audio(audio_file)
            if audio_array is None:
                return None, {"error": "Failed to process audio file"}
        
        # Generate image
        try:
            result = self.generation_pipeline(
                prompt=prompt,
                audio=audio_array,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                audio_guidance_scale=audio_scale,
                audio_strength=audio_strength,
                generator=generator,
                width=width,
                height=height
            )
            
            image = result["images"][0]
            
            # Create metadata
            metadata = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "num_steps": num_steps,
                "guidance_scale": guidance_scale,
                "audio_scale": audio_scale,
                "audio_strength": audio_strength,
                "width": width,
                "height": height,
                "audio_file": audio_file if audio_file else "None",
                "timestamp": datetime.now().isoformat()
            }
            
            return image, metadata
            
        except Exception as e:
            print(f"Generation error: {e}")
            return None, {"error": str(e)}
    
    def inpaint_image(
        self,
        prompt: str,
        image: Image.Image,
        mask: Image.Image,
        audio_file: Optional[str],
        negative_prompt: str,
        strength: float,
        num_steps: int,
        guidance_scale: float,
        audio_scale: float,
        audio_strength: float,
        seed: int
    ) -> Tuple[Image.Image, dict]:
        """
        Perform audio-conditioned inpainting.
        
        Returns:
            Inpainted image and metadata
        """
        if image is None or mask is None:
            return None, {"error": "Please provide both image and mask"}
        
        # Set seed
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Process audio
        audio_array = None
        if audio_file:
            audio_array, sr = self.process_audio(audio_file)
        
        # Inpaint
        try:
            result = self.inpainting_pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                audio=audio_array,
                negative_prompt=negative_prompt,
                strength=strength,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                audio_guidance_scale=audio_scale,
                audio_strength=audio_strength,
                generator=generator
            )
            
            inpainted = result["images"][0]
            
            metadata = {
                "mode": "inpainting",
                "prompt": prompt,
                "strength": strength,
                "seed": seed,
                "audio_file": audio_file if audio_file else "None",
                "timestamp": datetime.now().isoformat()
            }
            
            return inpainted, metadata
            
        except Exception as e:
            print(f"Inpainting error: {e}")
            return None, {"error": str(e)}
    
    def create_interface(self) -> gr.Blocks:
        """
        Create Gradio interface.
        
        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(title="CLAP2Diffusion Demo", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # CLAP2Diffusion: Audio-Conditioned Image Generation
                
                Generate and edit images using both audio and text descriptions.
                The audio provides additional context and atmosphere to guide the generation.
                """
            )
            
            with gr.Tabs():
                # Generation Tab
                with gr.TabItem("Generate"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            # Input controls
                            prompt_input = gr.Textbox(
                                label="Text Prompt",
                                placeholder="Describe the image you want to generate...",
                                lines=3
                            )
                            
                            audio_input = gr.Audio(
                                label="Audio Input (Optional)",
                                type="filepath",
                                elem_id="audio_upload"
                            )
                            
                            with gr.Accordion("Audio Examples", open=False):
                                audio_preset = gr.Dropdown(
                                    choices=list(self.audio_examples.keys()),
                                    label="Preset Audio Descriptions",
                                    info="Select a preset to see example prompts"
                                )
                                preset_description = gr.Textbox(
                                    label="Description",
                                    interactive=False
                                )
                            
                            negative_prompt_input = gr.Textbox(
                                label="Negative Prompt",
                                placeholder="What to avoid in the image...",
                                value="low quality, blurry, distorted, ugly"
                            )
                            
                            with gr.Accordion("Advanced Settings", open=False):
                                with gr.Row():
                                    steps_slider = gr.Slider(
                                        minimum=1,
                                        maximum=100,
                                        value=50,
                                        step=1,
                                        label="Inference Steps"
                                    )
                                    guidance_slider = gr.Slider(
                                        minimum=1.0,
                                        maximum=20.0,
                                        value=7.5,
                                        step=0.5,
                                        label="Guidance Scale"
                                    )
                                
                                with gr.Row():
                                    audio_scale_slider = gr.Slider(
                                        minimum=0.0,
                                        maximum=10.0,
                                        value=3.0,
                                        step=0.5,
                                        label="Audio Guidance Scale"
                                    )
                                    audio_strength_slider = gr.Slider(
                                        minimum=0.0,
                                        maximum=2.0,
                                        value=1.0,
                                        step=0.1,
                                        label="Audio Strength"
                                    )
                                
                                with gr.Row():
                                    width_slider = gr.Slider(
                                        minimum=256,
                                        maximum=1024,
                                        value=512,
                                        step=64,
                                        label="Width"
                                    )
                                    height_slider = gr.Slider(
                                        minimum=256,
                                        maximum=1024,
                                        value=512,
                                        step=64,
                                        label="Height"
                                    )
                                
                                seed_input = gr.Number(
                                    label="Seed (-1 for random)",
                                    value=-1,
                                    precision=0
                                )
                            
                            generate_btn = gr.Button(
                                "Generate Image",
                                variant="primary",
                                elem_id="generate_button"
                            )
                        
                        with gr.Column(scale=1):
                            # Output
                            output_image = gr.Image(
                                label="Generated Image",
                                type="pil",
                                elem_id="output_image"
                            )
                            
                            metadata_output = gr.JSON(
                                label="Generation Metadata"
                            )
                    
                    # Examples
                    gr.Examples(
                        examples=[
                            ["A stormy night with lightning strikes", None, "low quality", 50, 7.5, 3.0, 1.0],
                            ["Ocean waves crashing on a tropical beach at sunset", None, "low quality", 50, 7.5, 3.0, 1.0],
                            ["Cozy campfire in a forest clearing under stars", None, "low quality", 50, 7.5, 3.0, 1.0],
                        ],
                        inputs=[prompt_input, audio_input, negative_prompt_input, 
                               steps_slider, guidance_slider, audio_scale_slider, audio_strength_slider],
                        outputs=[output_image, metadata_output],
                        fn=self.generate_image,
                        cache_examples=False
                    )
                
                # Inpainting Tab
                with gr.TabItem("Inpaint"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            inpaint_prompt = gr.Textbox(
                                label="Text Prompt",
                                placeholder="Describe what to inpaint...",
                                lines=2
                            )
                            
                            inpaint_image = gr.Image(
                                label="Input Image",
                                type="pil",
                                tool="sketch"
                            )
                            
                            inpaint_audio = gr.Audio(
                                label="Audio Input (Optional)",
                                type="filepath"
                            )
                            
                            inpaint_negative = gr.Textbox(
                                label="Negative Prompt",
                                value="low quality, blurry"
                            )
                            
                            with gr.Accordion("Inpainting Settings", open=True):
                                inpaint_strength = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.8,
                                    step=0.05,
                                    label="Inpainting Strength"
                                )
                                
                                inpaint_steps = gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    value=50,
                                    step=1,
                                    label="Steps"
                                )
                                
                                with gr.Row():
                                    inpaint_guidance = gr.Slider(
                                        minimum=1.0,
                                        maximum=20.0,
                                        value=7.5,
                                        step=0.5,
                                        label="Guidance Scale"
                                    )
                                    inpaint_audio_scale = gr.Slider(
                                        minimum=0.0,
                                        maximum=10.0,
                                        value=3.0,
                                        step=0.5,
                                        label="Audio Scale"
                                    )
                                
                                inpaint_audio_strength = gr.Slider(
                                    minimum=0.0,
                                    maximum=2.0,
                                    value=1.0,
                                    step=0.1,
                                    label="Audio Strength"
                                )
                                
                                inpaint_seed = gr.Number(
                                    label="Seed",
                                    value=-1,
                                    precision=0
                                )
                            
                            inpaint_btn = gr.Button(
                                "Inpaint",
                                variant="primary"
                            )
                        
                        with gr.Column(scale=1):
                            inpaint_output = gr.Image(
                                label="Inpainted Result",
                                type="pil"
                            )
                            
                            inpaint_metadata = gr.JSON(
                                label="Inpainting Metadata"
                            )
            
            # Event handlers
            audio_preset.change(
                lambda x: self.audio_examples.get(x, ""),
                inputs=[audio_preset],
                outputs=[preset_description]
            )
            
            generate_btn.click(
                self.generate_image,
                inputs=[
                    prompt_input, audio_input, negative_prompt_input,
                    steps_slider, guidance_slider, audio_scale_slider,
                    audio_strength_slider, seed_input, width_slider, height_slider
                ],
                outputs=[output_image, metadata_output]
            )
            
            inpaint_btn.click(
                self.inpaint_image,
                inputs=[
                    inpaint_prompt, inpaint_image, inpaint_image,  # Use sketch as mask
                    inpaint_audio, inpaint_negative, inpaint_strength,
                    inpaint_steps, inpaint_guidance, inpaint_audio_scale,
                    inpaint_audio_strength, inpaint_seed
                ],
                outputs=[inpaint_output, inpaint_metadata]
            )
        
        return demo


def main():
    """Main function to run the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CLAP2Diffusion Demo")
    parser.add_argument("--model_path", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda/cpu)")
    parser.add_argument("--share", action="store_true",
                       help="Share the demo publicly")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the demo on")
    parser.add_argument("--no_fp16", action="store_true",
                       help="Disable FP16 precision")
    
    args = parser.parse_args()
    
    # Create demo
    demo_app = CLAP2DiffusionDemo(
        model_path=args.model_path,
        device=args.device,
        use_fp16=not args.no_fp16
    )
    
    # Create and launch interface
    interface = demo_app.create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )


if __name__ == "__main__":
    main()