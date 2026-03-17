"""WavLM-to-audio vocoder — baseline (no GAN).

Architecture:
    WavLM-Base+ (frozen)
    └─ last_hidden_state  [B, T', 768]
       └─ WavLMAdapter    [B, 256, T']
          └─ HiFiGenerator (320× upsample)  [B, T]

The generator uses progressive ConvTranspose1d upsampling with multi-dilation
residual blocks, matching the HiFi-GAN generator design.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weight initialisation helpers
# ---------------------------------------------------------------------------

def _init_conv(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Adapter: WavLM dim (768) → hidden_dim (256)
# ---------------------------------------------------------------------------

class WavLMAdapter(nn.Module):
    """Residual Conv1D stack that projects WavLM features to hidden_dim.

    Input:  (B, T', wavlm_dim)
    Output: (B, hidden_dim, T')
    """

    def __init__(
        self,
        wavlm_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 6,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(wavlm_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(
                {
                    "conv": nn.Conv1d(
                        hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2
                    ),
                    "norm": nn.BatchNorm1d(hidden_dim),
                    "drop": nn.Dropout(dropout),
                }
            )
            for _ in range(num_layers)
        )
        self.apply(_init_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T', D)
        x = self.input_proj(x).transpose(1, 2)  # (B, hidden_dim, T')
        for blk in self.blocks:
            res = x
            x = F.gelu(blk["norm"](blk["conv"](x)))
            x = blk["drop"](x) + res
        return x


# ---------------------------------------------------------------------------
# Residual block with dilated convolutions
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation, padding=pad)
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.norm2 = nn.BatchNorm1d(channels)
        self.apply(_init_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm1(F.leaky_relu(self.conv1(x), 0.2))
        x = self.norm2(F.leaky_relu(self.conv2(x), 0.2))
        return x + res


# ---------------------------------------------------------------------------
# HiFi-like generator (320× upsampling)
# ---------------------------------------------------------------------------

class HiFiGenerator(nn.Module):
    """Progressive upsampling generator inspired by HiFi-GAN.

    Upsampling schedule: [8, 5, 4, 2]  →  8×5×4×2 = 320×

    Input:  (B, hidden_dim, T')
    Output: (B, 1, T'×320)
    """

    UPSAMPLE_RATES   = (8, 5, 4, 2)
    UPSAMPLE_KERNELS = (16, 10, 8, 4)
    RESBLOCK_KERNELS = (3, 7, 11)
    RESBLOCK_DILS    = ((1, 3, 5), (1, 3, 5), (1, 3, 5))

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        self.pre = nn.Conv1d(hidden_dim, 512, 7, padding=3)

        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        ch = 512
        for rate, kernel in zip(self.UPSAMPLE_RATES, self.UPSAMPLE_KERNELS):
            out_ch = ch // 2
            self.ups.append(
                nn.ConvTranspose1d(ch, out_ch, kernel, stride=rate,
                                   padding=(kernel - rate) // 2)
            )
            self.resblocks.append(
                nn.ModuleList(
                    ResBlock(out_ch, k, d)
                    for k, dils in zip(self.RESBLOCK_KERNELS, self.RESBLOCK_DILS)
                    for d in dils
                )
            )
            ch = out_ch

        self.post = nn.Conv1d(ch, 1, 7, padding=3)
        self.apply(_init_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.pre(x), 0.2)
        for up, rbs in zip(self.ups, self.resblocks):
            x = F.leaky_relu(up(x), 0.2)
            xs = sum(rb(x) for rb in rbs) / len(rbs)
            x = xs
        return torch.tanh(self.post(x))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class WavLM2Audio(nn.Module):
    """WavLM (frozen) → Adapter → HiFiGenerator → waveform.

    Args:
        wavlm_model_name:  HuggingFace model ID or local path.
        hidden_dim:        Adapter and generator channel width.
        num_adapter_layers: Number of residual conv blocks in the adapter.
        kernel_size:       Conv kernel size in the adapter.
        freeze_wavlm:      If True, WavLM weights are frozen and kept in eval mode.
        dropout:           Dropout rate in the adapter.
    """

    def __init__(
        self,
        wavlm_model_name: str = "microsoft/wavlm-base-plus",
        hidden_dim: int = 256,
        num_adapter_layers: int = 6,
        kernel_size: int = 7,
        freeze_wavlm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.freeze_wavlm = bool(freeze_wavlm)

        # Load WavLM (try local cache first)
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                wavlm_model_name, local_files_only=True
            )
            self.wavlm = WavLMModel.from_pretrained(
                wavlm_model_name, local_files_only=True
            )
        except Exception:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                wavlm_model_name
            )
            self.wavlm = WavLMModel.from_pretrained(wavlm_model_name)

        wavlm_dim = self.wavlm.config.hidden_size

        if self.freeze_wavlm:
            for p in self.wavlm.parameters():
                p.requires_grad_(False)
            self.wavlm.eval()

        self.adapter   = WavLMAdapter(wavlm_dim, hidden_dim, num_adapter_layers,
                                       kernel_size, dropout)
        self.generator = HiFiGenerator(hidden_dim)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("WavLM2Audio: %d trainable params (WavLM frozen=%s)",
                    n_trainable, freeze_wavlm)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Reconstruct audio from raw waveform.

        Args:
            audio: (B, T) float32 in [-1, 1]
        Returns:
            (B, T) reconstructed waveform
        """
        B, T = audio.shape

        ctx = torch.no_grad() if self.freeze_wavlm else torch.enable_grad()
        with ctx:
            feats = self.wavlm(audio).last_hidden_state  # (B, T', 768)

        adapted = self.adapter(feats)            # (B, 256, T')
        out = self.generator(adapted).squeeze(1) # (B, T'*320)

        # Deterministic length correction (crop or pad — no interpolation)
        if out.shape[1] > T:
            out = out[:, :T]
        elif out.shape[1] < T:
            out = F.pad(out, (0, T - out.shape[1]))

        return out
