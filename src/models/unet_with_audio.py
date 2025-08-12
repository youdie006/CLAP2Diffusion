"""
Audio-Conditioned UNet for CLAP2Diffusion
Modified UNet that accepts audio context for conditioning
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple
from diffusers.models.unet_2d_condition import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from .attention_adapter import AudioAdapterAttention, AudioAdapterBlock


class AudioConditionedCrossAttention(nn.Module):
    """
    Modified cross-attention block that handles both text and audio conditioning.
    """
    
    def __init__(
        self,
        original_attention: Attention,
        dim: int,
        use_audio_adapter: bool = True,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize audio-conditioned cross-attention.
        
        Args:
            original_attention: Original attention module
            dim: Dimension of the attention
            use_audio_adapter: Whether to use audio adapter
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.original_attention = original_attention
        self.use_audio_adapter = use_audio_adapter
        
        if use_audio_adapter:
            self.audio_adapter = AudioAdapterAttention(
                dim=dim,
                num_heads=num_heads,
                head_dim=dim // num_heads,
                dropout=dropout
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        audio_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass with both text and audio conditioning.
        
        Args:
            hidden_states: Input hidden states
            encoder_hidden_states: Text encoder hidden states
            audio_context: Audio context embeddings
            attention_mask: Attention mask
            
        Returns:
            Output hidden states
        """
        # Apply original text cross-attention
        hidden_states = self.original_attention(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Apply audio adapter if enabled and audio context is provided
        if self.use_audio_adapter and audio_context is not None:
            hidden_states = self.audio_adapter(
                hidden_states,
                audio_context=audio_context,
                attention_mask=attention_mask
            )
        
        return hidden_states


class AudioConditionedUNet(nn.Module):
    """
    UNet with audio conditioning capabilities.
    Wraps a standard UNet2DConditionModel and adds audio adapter layers.
    """
    
    def __init__(
        self,
        base_model_name: str = "runwayml/stable-diffusion-v1-5",
        use_audio_adapter: bool = True,
        adapter_config: Optional[Dict[str, Any]] = None,
        device: str = "cuda"
    ):
        """
        Initialize audio-conditioned UNet.
        
        Args:
            base_model_name: Base diffusion model name
            use_audio_adapter: Whether to use audio adapters
            adapter_config: Configuration for audio adapters
            device: Device to run on
        """
        super().__init__()
        
        # Load base UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            base_model_name,
            subfolder="unet"
        ).to(device)
        
        self.use_audio_adapter = use_audio_adapter
        self.device = device
        
        # Default adapter config
        if adapter_config is None:
            adapter_config = {
                "num_heads": 8,
                "dropout": 0.1,
                "gate_init": 0.0
            }
        
        # Add audio adapters to cross-attention layers if enabled
        if use_audio_adapter:
            self._add_audio_adapters(adapter_config)
    
    def _add_audio_adapters(self, config: Dict[str, Any]):
        """
        Add audio adapter layers to the UNet.
        
        Args:
            config: Adapter configuration
        """
        self.audio_adapters = nn.ModuleDict()
        
        # Add adapters to down blocks
        for i, down_block in enumerate(self.unet.down_blocks):
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                for j, layer in enumerate(down_block.attentions):
                    adapter_name = f"down_{i}_{j}"
                    self.audio_adapters[adapter_name] = AudioAdapterBlock(
                        dim=layer.transformer_blocks[0].norm1.normalized_shape[0],
                        num_heads=config["num_heads"],
                        dropout=config["dropout"],
                        use_self_attn=False,
                        use_feedforward=False
                    )
        
        # Add adapter to mid block
        if hasattr(self.unet.mid_block, "has_cross_attention") and self.unet.mid_block.has_cross_attention:
            for j, layer in enumerate([self.unet.mid_block.attentions[0]]):
                adapter_name = f"mid_{j}"
                self.audio_adapters[adapter_name] = AudioAdapterBlock(
                    dim=layer.transformer_blocks[0].norm1.normalized_shape[0],
                    num_heads=config["num_heads"],
                    dropout=config["dropout"],
                    use_self_attn=False,
                    use_feedforward=False
                )
        
        # Add adapters to up blocks
        for i, up_block in enumerate(self.unet.up_blocks):
            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                for j, layer in enumerate(up_block.attentions):
                    adapter_name = f"up_{i}_{j}"
                    self.audio_adapters[adapter_name] = AudioAdapterBlock(
                        dim=layer.transformer_blocks[0].norm1.normalized_shape[0],
                        num_heads=config["num_heads"],
                        dropout=config["dropout"],
                        use_self_attn=False,
                        use_feedforward=False
                    )
    
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        audio_context: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs
    ) -> Union[Tuple, Dict]:
        """
        Forward pass through audio-conditioned UNet.
        
        Args:
            sample: Noisy latents [batch_size, channels, height, width]
            timestep: Denoising timestep
            encoder_hidden_states: Text encoder hidden states
            audio_context: Audio context embeddings
            class_labels: Optional class labels
            attention_mask: Optional attention mask
            return_dict: Whether to return dictionary
            
        Returns:
            Model output
        """
        # Store audio context for use in adapters
        self.current_audio_context = audio_context
        
        # Forward through base UNet
        # We'll need to modify the forward hooks to inject audio context
        if self.use_audio_adapter and audio_context is not None:
            # Register forward hooks for audio injection
            hooks = self._register_audio_hooks(audio_context)
            
            # Forward pass
            output = self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=class_labels,
                attention_mask=attention_mask,
                return_dict=return_dict,
                **kwargs
            )
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
        else:
            # Standard forward without audio
            output = self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=class_labels,
                attention_mask=attention_mask,
                return_dict=return_dict,
                **kwargs
            )
        
        return output
    
    def _register_audio_hooks(self, audio_context: torch.Tensor):
        """
        Register forward hooks to inject audio context.
        
        Args:
            audio_context: Audio context to inject
            
        Returns:
            List of hook handles
        """
        hooks = []
        
        def make_audio_hook(adapter_name):
            def hook(module, input, output):
                if adapter_name in self.audio_adapters:
                    # Apply audio adapter
                    adapter = self.audio_adapters[adapter_name]
                    output = adapter(output, audio_context=audio_context)
                return output
            return hook
        
        # Register hooks for down blocks
        for i, down_block in enumerate(self.unet.down_blocks):
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                for j, layer in enumerate(down_block.attentions):
                    adapter_name = f"down_{i}_{j}"
                    if adapter_name in self.audio_adapters:
                        hook = layer.register_forward_hook(make_audio_hook(adapter_name))
                        hooks.append(hook)
        
        # Register hook for mid block
        if hasattr(self.unet.mid_block, "has_cross_attention") and self.unet.mid_block.has_cross_attention:
            adapter_name = "mid_0"
            if adapter_name in self.audio_adapters:
                hook = self.unet.mid_block.attentions[0].register_forward_hook(make_audio_hook(adapter_name))
                hooks.append(hook)
        
        # Register hooks for up blocks
        for i, up_block in enumerate(self.unet.up_blocks):
            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                for j, layer in enumerate(up_block.attentions):
                    adapter_name = f"up_{i}_{j}"
                    if adapter_name in self.audio_adapters:
                        hook = layer.register_forward_hook(make_audio_hook(adapter_name))
                        hooks.append(hook)
        
        return hooks
    
    def freeze_base_model(self):
        """Freeze the base UNet parameters."""
        for param in self.unet.parameters():
            param.requires_grad = False
    
    def unfreeze_base_model(self):
        """Unfreeze the base UNet parameters."""
        for param in self.unet.parameters():
            param.requires_grad = True
    
    def freeze_adapters(self):
        """Freeze the audio adapter parameters."""
        if hasattr(self, 'audio_adapters'):
            for param in self.audio_adapters.parameters():
                param.requires_grad = False
    
    def unfreeze_adapters(self):
        """Unfreeze the audio adapter parameters."""
        if hasattr(self, 'audio_adapters'):
            for param in self.audio_adapters.parameters():
                param.requires_grad = True
    
    def set_audio_scale(self, scale: float):
        """
        Set the audio conditioning scale for all adapters.
        
        Args:
            scale: Audio conditioning scale (f_multiplier)
        """
        if hasattr(self, 'audio_adapters'):
            for adapter in self.audio_adapters.values():
                if hasattr(adapter.audio_attn, 'set_audio_multiplier'):
                    adapter.audio_attn.set_audio_multiplier(scale)
    
    def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get state dict for audio adapters only.
        
        Returns:
            State dict containing adapter parameters
        """
        if hasattr(self, 'audio_adapters'):
            return self.audio_adapters.state_dict()
        return {}
    
    def load_adapter_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """
        Load state dict for audio adapters.
        
        Args:
            state_dict: State dict to load
        """
        if hasattr(self, 'audio_adapters'):
            self.audio_adapters.load_state_dict(state_dict)