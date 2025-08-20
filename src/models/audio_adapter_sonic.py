"""
Audio Adapter based on SonicDiffusion architecture
Conv1D-based adapter for audio-to-spatial feature conversion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SonicAdapter(nn.Module):
    """
    SonicDiffusion-style adapter converting audio to spatial features
    """
    def __init__(
        self,
        channels=[320, 640, 1280, 1280],
        grid_dims=[64, 32, 16, 8],
        audio_dim=1024,
        device="cuda"
    ):
        super().__init__()
        self.channels = channels
        self.grid_dims = grid_dims
        
        self.input_adapters = nn.ModuleList()
        
        # Input Block 1: 320 channels, 64x64 grid
        ch = channels[0]
        grid_dim = grid_dims[0]
        layer = nn.Sequential(
            nn.ConvTranspose1d(1, ch // 4, 4, stride=4, bias=True, padding=0),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        self.input_adapters.append(layer.to(device))
        
        # Input Block 2: 640 channels, 32x32 grid
        ch = channels[1]
        grid_dim = grid_dims[1]
        layer = nn.Sequential(
            nn.Conv1d(1, ch // 4, 17, stride=1, bias=True, padding=8),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        self.input_adapters.append(layer.to(device))
        
        # Input Block 3: 1280 channels, 16x16 grid
        ch = channels[2]
        grid_dim = grid_dims[2]
        layer = nn.Sequential(
            nn.Conv1d(1, ch // 4, 17, stride=4, bias=True, padding=8),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        self.input_adapters.append(layer.to(device))
        
        # Input Block 4: 1280 channels, 8x8 grid
        ch = channels[3]
        grid_dim = grid_dims[3]
        layer = nn.Sequential(
            nn.Conv1d(1, ch // 4, 17, stride=16, bias=True, padding=2),
            nn.GELU(),
            nn.Unflatten(2, (grid_dim, grid_dim)),
            nn.Conv2d(ch // 4, ch, 3, 1, 1)
        )
        self.input_adapters.append(layer.to(device))
        
    def forward(self, audio_features, level=0):
        """
        Convert audio features to spatial features for specific UNet level
        
        Args:
            audio_features: [batch_size, audio_dim] CLAP embeddings
            level: which adapter level to use (0-3)
        
        Returns:
            spatial_features: [batch_size, channels[level], grid_dim, grid_dim]
        """
        # Add channel dimension
        x = audio_features.unsqueeze(1)  # [B, 1, audio_dim]
        
        # Apply appropriate adapter
        return self.input_adapters[level](x)


class SonicAudioProjector(nn.Module):
    """
    SonicDiffusion-style audio projector with cross-attention
    """
    def __init__(
        self,
        audio_dim=1024,
        context_dim=768,
        audio_token_count=77,  # Match text token count
        initial_channel_dim=1,
        transformer_layer_count=4,
        dropout=0.1,
        h=8,
        dim_head=40,
        device="cuda"
    ):
        super().__init__()
        
        self.h = h
        inner_dim = dim_head * h
        
        # Audio embedding projection (similar to SonicDiffusion)
        self.audio_emb_projection = nn.Sequential(
            nn.Conv1d(initial_channel_dim, audio_token_count, kernel_size=17, stride=1, padding=8),
            nn.GELU(),
            nn.Conv1d(audio_token_count, audio_token_count, kernel_size=17, stride=1, padding=8),
            nn.GELU(),
            nn.LayerNorm([audio_token_count, audio_dim]),
            nn.Conv1d(audio_token_count, audio_token_count, kernel_size=17, stride=1, padding=8),
            nn.GELU(),
            nn.LayerNorm([audio_token_count, audio_dim]),
            nn.ConvTranspose1d(audio_token_count, audio_token_count, kernel_size=17, stride=3, padding=7),
            nn.GELU(),
            nn.LayerNorm([audio_token_count, 3 * audio_dim]),
            nn.Conv1d(audio_token_count, audio_token_count, kernel_size=17, stride=4, padding=7),
            nn.Dropout(dropout)
        )
        
        # Cross-attention layers
        self.cross_attention = nn.ModuleList()
        self.between_attention = nn.ModuleList()
        
        for _ in range(transformer_layer_count):
            # Cross-attention (Q, K, V all from audio)
            self.cross_attention.append(
                SonicCrossAttention(audio_dim, context_dim, dropout, h, dim_head)
            )
            
            # FFN between attention layers
            self.between_attention.append(nn.Sequential(
                nn.Linear(inner_dim, inner_dim),
                nn.GELU(),
                nn.Linear(inner_dim, context_dim),
                nn.Dropout(dropout)
            ))
        
        self.to_out_adapter = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, audio_context):
        """
        Process audio through projection and attention layers
        
        Args:
            audio_context: [batch_size, audio_dim] CLAP embeddings
            
        Returns:
            audio_tokens: [batch_size, 77, 768] matching text format
        """
        # Add channel dimension
        audio_context = audio_context.unsqueeze(1)  # [B, 1, audio_dim]
        
        # Project audio
        audio_proj = self.audio_emb_projection(audio_context)  # [B, 77, 768]
        
        # Apply transformer layers
        out = audio_proj
        for cross_attn, ffn in zip(self.cross_attention, self.between_attention):
            out = cross_attn(out)
            out = ffn(out)
        
        out = self.to_out_adapter(out)  # [B, 77, 768]
        
        return out


class SonicCrossAttention(nn.Module):
    """
    Cross-attention module from SonicDiffusion
    Q, K, V all from audio (self-attention on audio features)
    """
    def __init__(self, audio_dim=1024, context_dim=768, dropout=0.0, h=8, dim_head=40):
        super().__init__()
        self.h = h
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * h
        
        self.to_q_adapter = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_k_adapter = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_adapter = nn.Linear(context_dim, inner_dim, bias=False)
        
    def forward(self, audio):
        """
        Self-attention on audio features
        
        Args:
            audio: [batch_size, num_tokens, dim]
        """
        from einops import rearrange, einsum
        
        q_adapter = self.to_q_adapter(audio)
        k_adapter = self.to_k_adapter(audio)
        v_adapter = self.to_v_adapter(audio)
        
        q_adapter, k_adapter, v_adapter = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.h),
            (q_adapter, k_adapter, v_adapter)
        )
        
        sim_adapter = einsum('b i d, b j d -> b i j', q_adapter, k_adapter) * self.scale
        attn_adapter = sim_adapter.softmax(dim=-1)
        
        out = einsum('b i j, b j d -> b i d', attn_adapter, v_adapter)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=self.h)
        
        return out