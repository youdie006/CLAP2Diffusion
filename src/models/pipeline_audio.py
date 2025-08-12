"""
Audio-Conditioned Diffusion Pipeline
Main pipeline for generating images with audio and text conditioning
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any, Callable
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionInpaintPipeline,
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import numpy as np

from .audio_encoder import CLAPAudioEncoder
from .audio_adapter import AudioProjectionMLP, AudioTextCombiner
from .unet_with_audio import AudioConditionedUNet


class AudioDiffusionPipeline:
    """
    Complete pipeline for audio-conditioned image generation.
    Combines CLAP audio encoding, text encoding, and diffusion generation.
    """
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        clap_model: str = "laion/clap-htsat-unfused",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        use_audio_adapter: bool = True,
        num_audio_tokens: int = 8,
        enable_xformers: bool = True
    ):
        """
        Initialize audio diffusion pipeline.
        
        Args:
            model_id: Base Stable Diffusion model ID
            clap_model: CLAP model for audio encoding
            device: Device to run on
            dtype: Data type for models
            use_audio_adapter: Whether to use audio adapters
            num_audio_tokens: Number of audio pseudo tokens
            enable_xformers: Whether to enable xformers memory efficient attention
        """
        self.device = device
        self.dtype = dtype
        
        # Load base pipeline components
        self.vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=dtype
        ).to(device)
        
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype
        ).to(device)
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )
        
        # Load audio-conditioned UNet
        self.unet = AudioConditionedUNet(
            base_model_name=model_id,
            use_audio_adapter=use_audio_adapter,
            device=device
        )
        
        # Load audio encoder and projection
        self.audio_encoder = CLAPAudioEncoder(
            model_name=clap_model,
            device=device,
            freeze=False
        )
        
        self.audio_projection = AudioProjectionMLP(
            input_dim=self.audio_encoder.embedding_dim,
            output_dim=768,  # CLIP dimension
            num_tokens=num_audio_tokens
        ).to(device)
        
        self.audio_text_combiner = AudioTextCombiner(
            combination_method="concat"
        ).to(device)
        
        # Load scheduler
        self.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        
        # Enable xformers if requested
        if enable_xformers:
            try:
                self.unet.unet.enable_xformers_memory_efficient_attention()
                self.vae.enable_xformers_memory_efficient_attention()
            except:
                print("xformers not available, using standard attention")
        
        # Freeze models as needed
        self.vae.eval()
        self.text_encoder.eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
    
    @torch.no_grad()
    def encode_audio(
        self,
        audio: Union[np.ndarray, torch.Tensor, str],
        sample_rate: Optional[int] = None
    ) -> torch.Tensor:
        """
        Encode audio to conditioning tokens.
        
        Args:
            audio: Audio input (waveform, tensor, or file path)
            sample_rate: Sample rate of audio
            
        Returns:
            Audio conditioning tokens
        """
        # Load audio from file if path provided
        if isinstance(audio, str):
            audio_embeds = self.audio_encoder.get_audio_embeds_from_file(audio)
        else:
            audio_embeds = self.audio_encoder(audio, sample_rate)
        
        # Project to pseudo tokens
        audio_tokens = self.audio_projection(audio_embeds)
        
        return audio_tokens
    
    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        batch_size: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompts to embeddings.
        
        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s)
            batch_size: Batch size
            
        Returns:
            Prompt embeddings and negative prompt embeddings
        """
        # Handle batch
        if not isinstance(prompt, list):
            prompt = [prompt] * batch_size
        
        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode
        text_input_ids = text_inputs.input_ids.to(self.device)
        prompt_embeds = self.text_encoder(text_input_ids)[0]
        
        # Handle negative prompt
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif not isinstance(negative_prompt, list):
            negative_prompt = [negative_prompt] * batch_size
        
        # Tokenize negative
        negative_text_inputs = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Encode negative
        negative_input_ids = negative_text_inputs.input_ids.to(self.device)
        negative_prompt_embeds = self.text_encoder(negative_input_ids)[0]
        
        return prompt_embeds, negative_prompt_embeds
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        audio: Optional[Union[np.ndarray, torch.Tensor, str]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        audio_guidance_scale: float = 3.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        audio_strength: float = 1.0,
        callback: Optional[Callable] = None,
        callback_steps: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate images with audio and text conditioning.
        
        Args:
            prompt: Text prompt
            audio: Audio input for conditioning
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Text guidance scale
            audio_guidance_scale: Audio guidance scale
            negative_prompt: Negative prompt
            num_images_per_prompt: Number of images to generate
            generator: Random generator
            audio_strength: Strength of audio conditioning
            callback: Progress callback
            callback_steps: Steps between callbacks
            
        Returns:
            Dictionary with generated images and metadata
        """
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        total_batch_size = batch_size * num_images_per_prompt
        
        # Encode prompts
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt, negative_prompt, batch_size
        )
        
        # Encode and combine with audio if provided
        audio_context = None
        if audio is not None:
            # Encode audio
            audio_tokens = self.encode_audio(audio)
            
            # Expand for batch
            if audio_tokens.shape[0] == 1 and total_batch_size > 1:
                audio_tokens = audio_tokens.repeat(total_batch_size, 1, 1)
            
            # Combine with text
            combined_embeds, _ = self.audio_text_combiner(
                audio_tokens, prompt_embeds
            )
            
            # Also combine with negative (using zero audio)
            zero_audio = torch.zeros_like(audio_tokens)
            combined_negative, _ = self.audio_text_combiner(
                zero_audio, negative_prompt_embeds
            )
            
            # Set audio context for UNet
            audio_context = audio_tokens
            
            # Set audio strength
            self.unet.set_audio_scale(audio_strength)
            
            # Use combined embeddings
            prompt_embeds = combined_embeds
            negative_prompt_embeds = combined_negative
        
        # Repeat for multiple images per prompt
        if num_images_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat(num_images_per_prompt, 1, 1)
            negative_prompt_embeds = negative_prompt_embeds.repeat(num_images_per_prompt, 1, 1)
            if audio_context is not None:
                audio_context = audio_context.repeat(num_images_per_prompt, 1, 1)
        
        # Prepare latents
        latents_shape = (
            total_batch_size,
            self.unet.unet.config.in_channels,
            height // 8,
            width // 8
        )
        latents = torch.randn(
            latents_shape,
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        
        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = latents * self.scheduler.init_noise_sigma
        
        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            with torch.no_grad():
                # Combine positive and negative embeddings
                combined_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
                
                # Combine audio context if available
                combined_audio = None
                if audio_context is not None:
                    combined_audio = torch.cat([
                        torch.zeros_like(audio_context),  # No audio for negative
                        audio_context * audio_guidance_scale  # Scaled audio for positive
                    ])
                
                # UNet forward
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=combined_embeds,
                    audio_context=combined_audio,
                    return_dict=False
                )[0]
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            
            # Callback
            if callback is not None and i % callback_steps == 0:
                callback(i, t, latents)
        
        # Decode latents
        latents = 1 / self.vae.config.scaling_factor * latents
        with torch.no_grad():
            images = self.vae.decode(latents).sample
        
        # Post-process images
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).astype(np.uint8)
        
        # Convert to PIL
        pil_images = [Image.fromarray(img) for img in images]
        
        return {
            "images": pil_images,
            "latents": latents.cpu(),
            "audio_tokens": audio_tokens.cpu() if audio is not None else None,
            "prompt_embeds": prompt_embeds.cpu(),
            "nsfw_content_detected": [False] * len(pil_images)
        }
    
    def enable_attention_slicing(self, slice_size: Optional[int] = "auto"):
        """Enable attention slicing for memory efficiency."""
        self.unet.unet.enable_attention_slicing(slice_size)
    
    def disable_attention_slicing(self):
        """Disable attention slicing."""
        self.unet.unet.disable_attention_slicing()
    
    def enable_vae_slicing(self):
        """Enable VAE slicing for memory efficiency."""
        self.vae.enable_slicing()
    
    def disable_vae_slicing(self):
        """Disable VAE slicing."""
        self.vae.disable_slicing()


class AudioInpaintingPipeline(AudioDiffusionPipeline):
    """
    Pipeline for audio-conditioned image inpainting.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize inpainting pipeline."""
        # Override model_id for inpainting model
        if "model_id" not in kwargs:
            kwargs["model_id"] = "runwayml/stable-diffusion-inpainting"
        super().__init__(*args, **kwargs)
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        image: Union[Image.Image, torch.Tensor],
        mask_image: Union[Image.Image, torch.Tensor],
        audio: Optional[Union[np.ndarray, torch.Tensor, str]] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        audio_guidance_scale: float = 3.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        generator: Optional[torch.Generator] = None,
        audio_strength: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform audio-conditioned inpainting.
        
        Args:
            prompt: Text prompt
            image: Input image
            mask_image: Mask indicating areas to inpaint
            audio: Audio input for conditioning
            strength: Inpainting strength
            num_inference_steps: Number of denoising steps
            guidance_scale: Text guidance scale
            audio_guidance_scale: Audio guidance scale
            negative_prompt: Negative prompt
            generator: Random generator
            audio_strength: Strength of audio conditioning
            
        Returns:
            Dictionary with inpainted images and metadata
        """
        # TODO: Implement inpainting logic
        # This would involve encoding the image and mask,
        # then using them in the denoising process
        
        # For now, fallback to regular generation
        return super().__call__(
            prompt=prompt,
            audio=audio,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            negative_prompt=negative_prompt,
            generator=generator,
            audio_strength=audio_strength,
            **kwargs
        )