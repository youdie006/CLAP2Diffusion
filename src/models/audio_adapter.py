"""
Audio Adapter Module - Projection MLP for audio-to-text token conversion
Converts CLAP audio embeddings to pseudo text tokens for diffusion conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class AudioProjectionMLP(nn.Module):
    """
    MLP that projects CLAP audio embeddings to pseudo text tokens.
    Converts audio features to a format compatible with text conditioning.
    """
    
    def __init__(
        self,
        input_dim: int = 512,      # CLAP audio embedding dimension
        hidden_dim: int = 1024,     # Hidden layer dimension
        output_dim: int = 768,      # CLIP text embedding dimension
        num_tokens: int = 8,        # Number of pseudo tokens to generate
        dropout: float = 0.1,
        activation: str = "gelu",
        use_residual: bool = False
    ):
        """
        Initialize audio projection MLP.
        
        Args:
            input_dim: Input dimension (CLAP audio embedding)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (CLIP text embedding)
            num_tokens: Number of pseudo tokens to generate
            dropout: Dropout rate
            activation: Activation function type
            use_residual: Whether to use residual connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.use_residual = use_residual
        
        # Choose activation function
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Main projection layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim * num_tokens)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ln_out = nn.LayerNorm(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Optional: learnable positional embeddings for tokens
        self.token_pos_emb = nn.Parameter(
            torch.randn(1, num_tokens, output_dim) * 0.02
        )
        
        # Optional: token type embeddings to distinguish audio tokens
        self.token_type_emb = nn.Parameter(
            torch.randn(1, 1, output_dim) * 0.02
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        for module in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        audio_embeds: torch.Tensor,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Forward pass to convert audio embeddings to pseudo tokens.
        
        Args:
            audio_embeds: CLAP audio embeddings [batch_size, input_dim]
            return_dict: Whether to return additional information
            
        Returns:
            Pseudo text tokens [batch_size, num_tokens, output_dim]
        """
        batch_size = audio_embeds.shape[0]
        
        # First projection layer
        hidden = self.fc1(audio_embeds)
        hidden = self.ln1(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        
        # Second projection layer (with optional residual)
        hidden2 = self.fc2(hidden)
        hidden2 = self.ln2(hidden2)
        hidden2 = self.activation(hidden2)
        hidden2 = self.dropout(hidden2)
        
        if self.use_residual and hidden.shape == hidden2.shape:
            hidden = hidden + hidden2
        else:
            hidden = hidden2
        
        # Final projection to tokens
        tokens = self.fc3(hidden)  # [batch_size, output_dim * num_tokens]
        tokens = tokens.view(batch_size, self.num_tokens, self.output_dim)
        
        # Apply layer norm per token
        tokens = self.ln_out(tokens)
        
        # Add positional embeddings
        tokens = tokens + self.token_pos_emb
        
        # Add token type embeddings
        tokens = tokens + self.token_type_emb
        
        if return_dict:
            return {
                "tokens": tokens,
                "positional_embeddings": self.token_pos_emb,
                "type_embeddings": self.token_type_emb
            }
        
        return tokens


class AudioTextCombiner(nn.Module):
    """
    Combines audio pseudo tokens with text embeddings for joint conditioning.
    """
    
    def __init__(
        self,
        audio_dim: int = 768,
        text_dim: int = 768,
        max_text_len: int = 77,
        combination_method: str = "concat",  # "concat", "add", "weighted"
        use_gate: bool = True
    ):
        """
        Initialize audio-text combiner.
        
        Args:
            audio_dim: Audio token dimension
            text_dim: Text embedding dimension
            max_text_len: Maximum text sequence length
            combination_method: How to combine audio and text
            use_gate: Whether to use gating mechanism
        """
        super().__init__()
        
        self.audio_dim = audio_dim
        self.text_dim = text_dim
        self.max_text_len = max_text_len
        self.combination_method = combination_method
        self.use_gate = use_gate
        
        # Ensure dimensions match
        assert audio_dim == text_dim, "Audio and text dimensions must match"
        
        # Gating mechanism for balancing audio/text
        if use_gate:
            self.audio_gate = nn.Parameter(torch.ones(1))
            self.text_gate = nn.Parameter(torch.ones(1))
        
        # Weighted combination learnable weights
        if combination_method == "weighted":
            self.audio_weight = nn.Parameter(torch.ones(1) * 0.5)
            self.text_weight = nn.Parameter(torch.ones(1) * 0.5)
        
        # Optional projection for dimension matching
        self.projection = nn.Identity()
        if audio_dim != text_dim:
            self.projection = nn.Linear(audio_dim, text_dim)
    
    def forward(
        self,
        audio_tokens: torch.Tensor,
        text_embeds: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combine audio tokens with text embeddings.
        
        Args:
            audio_tokens: Audio pseudo tokens [batch_size, num_tokens, dim]
            text_embeds: Text embeddings [batch_size, seq_len, dim]
            text_mask: Text attention mask [batch_size, seq_len]
            
        Returns:
            Combined embeddings and attention mask
        """
        batch_size = audio_tokens.shape[0]
        num_audio_tokens = audio_tokens.shape[1]
        
        # Apply gating if enabled
        if self.use_gate:
            audio_tokens = audio_tokens * torch.sigmoid(self.audio_gate)
            text_embeds = text_embeds * torch.sigmoid(self.text_gate)
        
        if self.combination_method == "concat":
            # Concatenate audio tokens at the beginning
            combined = torch.cat([audio_tokens, text_embeds], dim=1)
            
            # Update attention mask
            if text_mask is not None:
                audio_mask = torch.ones(
                    batch_size, num_audio_tokens,
                    dtype=text_mask.dtype,
                    device=text_mask.device
                )
                combined_mask = torch.cat([audio_mask, text_mask], dim=1)
            else:
                combined_mask = None
                
        elif self.combination_method == "add":
            # Add audio tokens to first N text tokens
            min_len = min(num_audio_tokens, text_embeds.shape[1])
            combined = text_embeds.clone()
            combined[:, :min_len] = combined[:, :min_len] + audio_tokens[:, :min_len]
            combined_mask = text_mask
            
        elif self.combination_method == "weighted":
            # Weighted combination
            audio_weight = torch.sigmoid(self.audio_weight)
            text_weight = torch.sigmoid(self.text_weight)
            
            # Normalize weights
            total_weight = audio_weight + text_weight
            audio_weight = audio_weight / total_weight
            text_weight = text_weight / total_weight
            
            # Pad or truncate to match dimensions
            if num_audio_tokens < text_embeds.shape[1]:
                audio_tokens = F.pad(
                    audio_tokens,
                    (0, 0, 0, text_embeds.shape[1] - num_audio_tokens)
                )
            else:
                audio_tokens = audio_tokens[:, :text_embeds.shape[1]]
            
            combined = audio_weight * audio_tokens + text_weight * text_embeds
            combined_mask = text_mask
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        return combined, combined_mask


class AudioConditioner(nn.Module):
    """
    Complete audio conditioning module combining encoder, projection, and combination.
    """
    
    def __init__(
        self,
        clap_model_name: str = "laion/clap-htsat-unfused",
        num_pseudo_tokens: int = 8,
        combination_method: str = "concat",
        device: str = "cuda"
    ):
        """
        Initialize complete audio conditioner.
        
        Args:
            clap_model_name: CLAP model to use
            num_pseudo_tokens: Number of pseudo tokens to generate
            combination_method: How to combine audio and text
            device: Device to run on
        """
        super().__init__()
        
        # Import here to avoid circular dependency
        from .audio_encoder import CLAPAudioEncoder
        
        # Audio encoder
        self.audio_encoder = CLAPAudioEncoder(
            model_name=clap_model_name,
            device=device
        )
        
        # Projection MLP
        self.projection = AudioProjectionMLP(
            input_dim=self.audio_encoder.embedding_dim,
            output_dim=768,  # CLIP dimension
            num_tokens=num_pseudo_tokens
        )
        
        # Combiner
        self.combiner = AudioTextCombiner(
            combination_method=combination_method
        )
        
        self.device = device
    
    def forward(
        self,
        audio: torch.Tensor,
        text_embeds: torch.Tensor,
        sample_rate: int = 48000,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete audio conditioning pipeline.
        
        Args:
            audio: Raw audio waveform
            text_embeds: Text embeddings
            sample_rate: Audio sample rate
            text_mask: Text attention mask
            
        Returns:
            Combined embeddings and mask
        """
        # Encode audio
        audio_embeds = self.audio_encoder(audio, sample_rate)
        
        # Project to pseudo tokens
        audio_tokens = self.projection(audio_embeds)
        
        # Combine with text
        combined, combined_mask = self.combiner(
            audio_tokens, text_embeds, text_mask
        )
        
        return combined, combined_mask