"""MERaLiON-SER-v1 downstream network, adapted for offline inference.
Copyright 2025 AGENCY FOR SCIENCE TECHNOLOGY AND RESEARCH.
Source: MERaLiON/MERaLiON-SER-v1 @ 7e3ee6fa4534dea8316e8ca43e377e2fbb58496b.
Modifications: removed training imports and unused decoder; local config and strict weights.
See LICENSE.pdf and Notice in this directory for the full public license.
"""
import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

class HierarchicalAttentionPooling(nn.Module):
    """
    Multi-level attention pooling that captures both local and global patterns.
    Pools at different temporal scales and combines them.
    """
    def __init__(self, embed_dim: int, num_levels: int = 3, reduction_factor: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.reduction_factor = reduction_factor

        # Attention layers for each level
        self.level_attentions = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_levels)
        ])

        # Combination layer to merge multi-level features
        self.combiner = nn.Linear(embed_dim * num_levels, embed_dim)


    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len) - 1 for valid positions
        Returns:
            (batch, embed_dim) - hierarchical pooled representation
        """
        batch_size, seq_len, embed_dim = x.shape
        level_features = []

        current_x = x
        current_mask = mask

        for level in range(self.num_levels):
            # Apply attention pooling at current resolution
            attn_scores = self.level_attentions[level](current_x).squeeze(-1)

            if current_mask is not None:
                attn_scores = attn_scores.masked_fill(current_mask == 0, -1e9)

            attn_weights = torch.softmax(attn_scores, dim=1)
            level_feature = torch.sum(current_x * attn_weights.unsqueeze(-1), dim=1)
            level_features.append(level_feature)

            # Downsample for next level (if not last level)
            if level < self.num_levels - 1:
                # Average pooling to reduce temporal resolution
                kernel_size = min(self.reduction_factor, current_x.size(1))
                if kernel_size > 1:
                    current_x = F.avg_pool1d(
                        current_x.transpose(1, 2),
                        kernel_size=kernel_size,
                        stride=kernel_size
                    ).transpose(1, 2)

                    if current_mask is not None:
                        # Downsample mask too
                        current_mask = F.max_pool1d(
                            current_mask.float().unsqueeze(1),
                            kernel_size=kernel_size,
                            stride=kernel_size
                        ).squeeze(1).bool()

        # Combine all level features
        combined_features = torch.cat(level_features, dim=-1)
        output = self.combiner(combined_features)

        return output
class LearnableMultiResolutionPooling(nn.Module):
    """
    Learns to attend to different temporal resolutions based on the input.
    The model decides which time scales are most important for each utterance.
    """
    def __init__(self, embed_dim: int, num_resolutions: int = 4, min_kernel: int = 1, max_kernel: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_resolutions = num_resolutions

        # Generate kernel sizes in log space
        kernel_sizes = [int(k) for k in torch.logspace(
            math.log10(min_kernel), math.log10(max_kernel), num_resolutions, device="cpu"
        )]
        self.kernel_sizes = kernel_sizes

        # Resolution-specific processing
        self.resolution_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, embed_dim, kernel_size=k, padding=k//2, groups=embed_dim//8 if embed_dim >= 8 else 1)
            for k in kernel_sizes
        ])

        # Learnable resolution weights (depends on input)
        self.resolution_weight_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_resolutions),
            nn.Softmax(dim=-1)
        )

        # Attention for each resolution
        self.resolution_attentions = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_resolutions)
        ])

        # Final combination
        self.final_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, embed_dim)
            mask: (batch, seq_len)
        Returns:
            (batch, embed_dim) - adaptive multi-resolution pooled representation
        """
        batch_size, seq_len, embed_dim = x.shape

        # Compute global context for resolution weighting
        global_context = torch.mean(x, dim=1)  # (batch, embed_dim)
        resolution_weights = self.resolution_weight_net(global_context)  # (batch, num_resolutions)

        resolution_features = []

        for i, kernel_size in enumerate(self.kernel_sizes):
            # Apply resolution-specific convolution
            x_conv = self.resolution_convs[i](x.transpose(1, 2)).transpose(1, 2)

            # Apply attention pooling
            attn_scores = self.resolution_attentions[i](x_conv).squeeze(-1)

            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

            attn_weights = torch.softmax(attn_scores, dim=1)
            resolution_feature = torch.sum(x_conv * attn_weights.unsqueeze(-1), dim=1)
            resolution_features.append(resolution_feature)

        # Stack resolution features: (batch, num_resolutions, embed_dim)
        resolution_stack = torch.stack(resolution_features, dim=1)

        # Apply learnable resolution weights
        weighted_features = resolution_stack * resolution_weights.unsqueeze(-1)  # (batch, num_resolutions, embed_dim)

        # Combine weighted resolution features
        combined_output = torch.sum(weighted_features, dim=1)  # (batch, embed_dim)
        final_output = self.final_proj(combined_output)

        return final_output

class HybridEmotionPooling(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.multiscale = LearnableMultiResolutionPooling(embed_dim)
        self.segment = HierarchicalAttentionPooling(embed_dim)
        self.combiner = nn.Linear(embed_dim * 2, embed_dim)

    def forward(self, x, mask=None):
        ms_out = self.multiscale(x, mask)
        seg_out = self.segment(x, mask)
        return self.combiner(torch.cat([ms_out, seg_out], dim=-1))

class SE1d(nn.Module):
    """Squeeze-and-Excitation block for 1D convolutions"""
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        w = self.fc(self.avg(x))
        return x * w

class Res2Block1d(nn.Module):
    """Res2Net block adapted for 1D convolutions - BatchNorm free"""
    def __init__(self, channels: int, scale: int = 4, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        assert channels % scale == 0, f"channels ({channels}) must be divisible by scale ({scale})"
        self.scale = scale
        self.width = channels // scale
        pad = (kernel_size // 2) * dilation

        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size, padding=pad, dilation=dilation, bias=True)
            for _ in range(scale - 1)
        ])
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        xs = torch.split(x, self.width, dim=1)
        out = [xs[0]]
        for i, conv in enumerate(self.convs, start=1):
            if i == 1:
                s = xs[i]
            else:
                s = xs[i] + out[-1]  # Fixed: proper residual connection
            out.append(conv(s))
        y = torch.cat(out, dim=1)
        return self.act(self.norm(y))

class ECAPABlock(nn.Module):
    """Enhanced ECAPA block with proper residual connections - BatchNorm free"""
    def __init__(self, channels: int, scale: int = 4, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 1, bias=True)
        self.norm1 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.act1 = nn.ReLU(inplace=True)

        self.res2 = Res2Block1d(channels, scale=scale, kernel_size=kernel_size, dilation=dilation)
        self.se = SE1d(channels)

        self.conv2 = nn.Conv1d(channels, channels, 1, bias=True)
        self.norm2 = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels)
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        residual = x

        y = self.act1(self.norm1(self.conv1(x)))
        y = self.res2(y)
        y = self.se(y)
        y = self.norm2(self.conv2(y))

        return self.act2(y + residual)

class EmotionECAPATDNN(nn.Module):
    """ECAPA-TDNN optimized for emotion recognition with hierarchical attention"""
    def __init__(
        self,
        input_dim: int,
        channels: int = 512,
        output_dim: int = 256,
        num_blocks: int = 3,
        dilations: tuple = (1, 2, 3),
        embed_dim: int = 512,
        num_emotions: int = 8,  # Common emotion categories
        dropout: float = 0.2,
        pooling_type="attention"
    ):
        super().__init__()
        self.output_dim = output_dim
        # Input projection with layer norm for stability
        self.proj_in = nn.Sequential(
            nn.Linear(input_dim, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # ECAPA blocks with different dilations
        self.blocks = nn.ModuleList([
            ECAPABlock(channels, scale=4, kernel_size=3, dilation=d)
            for d in dilations
        ])

        # Multi-scale feature aggregation
        self.mfa = nn.Sequential(
            nn.Conv1d(channels * (len(dilations) + 1), channels, 1, bias=True),
            nn.GroupNorm(num_groups=min(32, channels), num_channels=channels),
            nn.GELU()
        )

        self.pooling = HybridEmotionPooling(channels)
        # Final embedding layers
        self.embed = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(channels, channels // 2),
            nn.LayerNorm(channels // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_embeddings: bool = False):
        """
        Forward pass

        Args:
            x: Input tensor (batch, time, features) - from Whisper encoder
            attention_mask: Attention mask (batch, time)
            return_embeddings: Whether to return embeddings instead of logits

        Returns:
            If return_embeddings=False: emotion logits (batch, num_emotions)
            If return_embeddings=True: feature embeddings (batch, embed_dim // 2)
        """
        # Project input and transpose for conv1d
        x = self.proj_in(x)  # (batch, time, channels)
        x_conv = x.transpose(1, 2)  # (batch, channels, time)

        # Apply ECAPA blocks and collect multi-scale features
        features = [x_conv]
        for block in self.blocks:
            x_conv = block(x_conv)
            features.append(x_conv)

        # Multi-scale feature aggregation
        y = torch.cat(features, dim=1)  # (batch, channels * (n_blocks + 1), time)
        y = self.mfa(y)  # (batch, channels, time)
        y = y.transpose(1, 2)  # (batch, time, channels)

        # Hierarchical attention pooling
        #pooled = self.norm_layer(self.pooling(y))  # (batch, channels)
        pooled = self.pooling(y)  # (batch, channels)

        # Generate embeddings
        embeddings = self.embed(pooled)  # (batch, embed_dim // 2)

        #if return_embeddings:
        return embeddings


WHISPER_CONFIG = {'d_model': 1024, 'encoder_attention_heads': 16, 'encoder_ffn_dim': 4096, 'encoder_layers': 24, 'num_mel_bins': 80, 'max_source_positions': 1500, 'vocab_size': 51865}

class MeralionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        config = WhisperConfig(**WHISPER_CONFIG)
        config._attn_implementation = "sdpa"
        self.whisper = nn.Module()
        self.whisper.encoder = WhisperEncoder(config)
        self.downstream_model = EmotionECAPATDNN(input_dim=config.d_model, pooling_type="hybrid")
        self.dim_layer = nn.Sequential(nn.Linear(256, 256), nn.RMSNorm(256), nn.GELU(),
                                       nn.Linear(256, 3), nn.Sigmoid())
        self.emotion_layer = nn.Linear(256, 256)
        self.emotion_classification_layer = nn.Sequential(nn.RMSNorm(256), nn.GELU(), nn.Linear(256, 7))

    def forward(self, input_features):
        encoded = self.whisper.encoder(input_features).last_hidden_state
        shared = self.downstream_model(encoded)
        return self.dim_layer(shared), self.emotion_classification_layer(self.emotion_layer(shared))
