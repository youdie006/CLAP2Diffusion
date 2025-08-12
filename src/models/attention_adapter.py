"""
Audio Adapter Attention Module (SonicDiffusion Style)
Implements learnable audio conditioning for diffusion models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from einops import rearrange


class AudioAdapterAttention(nn.Module):
    """
    Audio adapter attention layer that can be added to existing attention blocks.
    Uses a learnable gate parameter to control audio feature contribution.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
        use_bias: bool = False,
        upcast_attention: bool = False,
        use_layer_norm: bool = True
    ):
        """
        Initialize audio adapter attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dropout: Dropout rate
            use_bias: Whether to use bias in linear layers
            upcast_attention: Whether to upcast attention computation
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.upcast_attention = upcast_attention
        
        # Learnable gate parameter for controlling audio contribution
        self.gate = nn.Parameter(torch.zeros(1))
        self.f_multiplier = 1.0  # Can be adjusted during inference
        
        # Query, Key, Value projections for audio cross-attention
        self.to_q = nn.Linear(dim, self.inner_dim, bias=use_bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=use_bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=use_bias)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Optional layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(dim)
        
        # Dropout for attention
        self.attn_dropout = nn.Dropout(dropout)
    
    def set_audio_multiplier(self, multiplier: float):
        """Set the audio feature multiplier for inference control."""
        self.f_multiplier = multiplier
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of audio adapter attention.
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, dim]
            audio_context: Audio context embeddings [batch, audio_seq_len, dim]
            attention_mask: Optional attention mask
            
        Returns:
            Output hidden states with audio conditioning
        """
        if audio_context is None:
            # If no audio context, return input unchanged
            return hidden_states
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute queries from hidden states
        queries = self.to_q(hidden_states)
        queries = rearrange(queries, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute keys and values from audio context
        keys = self.to_k(audio_context)
        values = self.to_v(audio_context)
        keys = rearrange(keys, 'b n (h d) -> b h n d', h=self.num_heads)
        values = rearrange(values, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute attention scores
        if self.upcast_attention:
            queries = queries.float()
            keys = keys.float()
        
        scale = self.head_dim ** -0.5
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Compute attention weights
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)
        
        # Apply attention to values
        if self.upcast_attention:
            attention_probs = attention_probs.to(values.dtype)
        
        hidden_states_audio = torch.matmul(attention_probs, values)
        hidden_states_audio = rearrange(hidden_states_audio, 'b h n d -> b n (h d)')
        
        # Output projection
        hidden_states_audio = self.to_out(hidden_states_audio)
        
        # Apply layer norm if enabled
        if self.use_layer_norm:
            hidden_states_audio = self.norm(hidden_states_audio)
        
        # Apply gate with tanh activation
        gate = torch.tanh(self.gate) * self.f_multiplier
        
        # Combine with original hidden states
        output = hidden_states + gate * hidden_states_audio
        
        return output


class AudioAdapterBlock(nn.Module):
    """
    Complete adapter block with self-attention and audio cross-attention.
    Can be inserted into existing transformer blocks.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.1,
        use_self_attn: bool = True,
        use_feedforward: bool = True,
        ff_mult: int = 4
    ):
        """
        Initialize audio adapter block.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
            dropout: Dropout rate
            use_self_attn: Whether to include self-attention
            use_feedforward: Whether to include feedforward
            ff_mult: Feedforward multiplier
        """
        super().__init__()
        
        self.use_self_attn = use_self_attn
        self.use_feedforward = use_feedforward
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Self-attention (optional)
        if use_self_attn:
            self.self_attn = nn.MultiheadAttention(
                dim, 
                num_heads, 
                dropout=dropout,
                batch_first=True
            )
            self.norm_self = nn.LayerNorm(dim)
        
        # Audio cross-attention
        self.audio_attn = AudioAdapterAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout
        )
        
        # Feedforward (optional)
        if use_feedforward:
            self.norm3 = nn.LayerNorm(dim)
            self.ff = nn.Sequential(
                nn.Linear(dim, dim * ff_mult),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * ff_mult, dim),
                nn.Dropout(dropout)
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of audio adapter block.
        
        Args:
            hidden_states: Input hidden states
            audio_context: Audio context embeddings
            attention_mask: Optional attention mask
            
        Returns:
            Output hidden states
        """
        residual = hidden_states
        
        # Self-attention
        if self.use_self_attn:
            hidden_states = self.norm_self(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states, 
                hidden_states, 
                hidden_states,
                attn_mask=attention_mask
            )
            hidden_states = residual + hidden_states
            residual = hidden_states
        
        # Audio cross-attention
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.audio_attn(
            hidden_states,
            audio_context=audio_context,
            attention_mask=attention_mask
        )
        hidden_states = residual + hidden_states
        
        # Feedforward
        if self.use_feedforward:
            residual = hidden_states
            hidden_states = self.norm3(hidden_states)
            hidden_states = self.ff(hidden_states)
            hidden_states = residual + hidden_states
        
        return hidden_states


class MultiLevelAudioAdapter(nn.Module):
    """
    Multi-level audio adapter for UNet architecture.
    Applies audio conditioning at different resolution levels.
    """
    
    def __init__(
        self,
        dims: Dict[str, int],
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize multi-level audio adapter.
        
        Args:
            dims: Dictionary mapping level names to dimensions
                  e.g., {'down_64': 320, 'down_32': 640, 'mid': 1280}
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.adapters = nn.ModuleDict()
        
        for level_name, dim in dims.items():
            self.adapters[level_name] = AudioAdapterAttention(
                dim=dim,
                num_heads=num_heads,
                head_dim=dim // num_heads,
                dropout=dropout
            )
    
    def forward(
        self,
        hidden_states_dict: Dict[str, torch.Tensor],
        audio_context: torch.Tensor,
        level_name: str
    ) -> torch.Tensor:
        """
        Apply audio conditioning at specific level.
        
        Args:
            hidden_states_dict: Dictionary of hidden states at different levels
            audio_context: Audio context embeddings
            level_name: Name of the current level
            
        Returns:
            Conditioned hidden states
        """
        if level_name not in self.adapters:
            return hidden_states_dict.get(level_name, None)
        
        hidden_states = hidden_states_dict[level_name]
        return self.adapters[level_name](hidden_states, audio_context)