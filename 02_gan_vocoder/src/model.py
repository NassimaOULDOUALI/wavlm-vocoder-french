import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, remove_weight_norm
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

logger = logging.getLogger(__name__)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


class Snake(nn.Module):
    def __init__(self, channels, alpha=1.0, alpha_logscale=False):
        super().__init__()
        if alpha_logscale:
            self.alpha = nn.Parameter(torch.zeros(1, channels, 1) + math.log(alpha))
            self.alpha_logscale = True
        else:
            self.alpha = nn.Parameter(torch.ones(1, channels, 1) * alpha)
            self.alpha_logscale = False

    def forward(self, x):
        alpha = self.alpha.exp() if self.alpha_logscale else self.alpha
        return x + (1.0 / (alpha + 1e-8)) * torch.sin(alpha * x).pow(2)


class WavLMAdapterImproved(nn.Module):
    def __init__(
        self,
        wavlm_dim=768,
        hidden_dim=256,
        num_layers=3,
        kernel_size=7,
        dropout=0.1,
        use_snake=False,
    ):
        super().__init__()
        self.wavlm_dim = wavlm_dim
        self.hidden_dim = hidden_dim
        self.use_snake = use_snake

        self.input_proj = nn.Linear(wavlm_dim, hidden_dim)

        self.conv_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_blocks.append(
                nn.ModuleDict({
                    "conv": weight_norm(nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )),
                    "dropout": nn.Dropout(dropout),
                })
            )

        if use_snake:
            self.activations = nn.ModuleList([Snake(hidden_dim) for _ in range(num_layers)])

        self.apply(init_weights)

    def forward(self, x):
        # x: (B, T', D)
        x = self.input_proj(x)      # (B, T', hidden_dim)
        x = x.transpose(1, 2)       # (B, hidden_dim, T')

        for i, block in enumerate(self.conv_blocks):
            residual = x
            x = block["conv"](x)
            if self.use_snake:
                x = self.activations[i](x)
            else:
                x = F.gelu(x)
            x = block["dropout"](x)
            x = x + residual

        return x

    def remove_weight_norm(self):
        for block in self.conv_blocks:
            remove_weight_norm(block["conv"])


class ResBlockImproved(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), use_snake=False):
        super().__init__()
        self.use_snake = use_snake

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        if use_snake:
            self.activations1 = nn.ModuleList()
            self.activations2 = nn.ModuleList()

        for d in dilation:
            self.convs1.append(
                weight_norm(nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=d,
                    padding=(kernel_size * d - d) // 2,
                ))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=1,
                    padding=(kernel_size - 1) // 2,
                ))
            )
            if use_snake:
                self.activations1.append(Snake(channels))
                self.activations2.append(Snake(channels))

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x):
        for i, (c1, c2) in enumerate(zip(self.convs1, self.convs2)):
            xt = x

            if self.use_snake:
                xt = self.activations1[i](xt)
            else:
                xt = F.leaky_relu(xt, 0.1)
            xt = c1(xt)

            if self.use_snake:
                xt = self.activations2[i](xt)
            else:
                xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)

            x = xt + x

        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class HiFiGeneratorImproved(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        upsample_rates=(8, 5, 4, 2),
        upsample_kernel_sizes=(16, 10, 8, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilations=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        use_snake=False,
    ):
        super().__init__()
        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.use_snake = use_snake

        self.conv_pre = weight_norm(nn.Conv1d(hidden_dim, 512, 7, 1, 3))

        self.ups = nn.ModuleList()
        channels = 512
        for rate, kernel in zip(upsample_rates, upsample_kernel_sizes):
            out_channels = channels // 2
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(
                    channels,
                    out_channels,
                    kernel,
                    stride=rate,
                    padding=(kernel - rate) // 2,
                ))
            )
            channels = out_channels

        # ResBlocks
        self.resblocks = nn.ModuleList()
        ch = 512
        for i in range(len(self.ups)):
            ch = ch // 2
            for k, d in zip(resblock_kernel_sizes, resblock_dilations):
                self.resblocks.append(ResBlockImproved(ch, k, d, use_snake=use_snake))

        # Activations for upsampling
        if use_snake:
            self.up_activations = nn.ModuleList([Snake(512 // (2 ** (i + 1))) for i in range(len(self.ups))])
            self.post_act = Snake(ch)

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, 3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)

        for i, up in enumerate(self.ups):
            if self.use_snake:
                x = self.up_activations[i](x)
            else:
                x = F.leaky_relu(x, 0.1)

            x = up(x)

            xs = None
            for j in range(self.num_kernels):
                idx = i * self.num_kernels + j
                if xs is None:
                    xs = self.resblocks[idx](x)
                else:
                    xs = xs + self.resblocks[idx](x)
            x = xs / self.num_kernels

        if self.use_snake:
            x = self.post_act(x)
        else:
            x = F.leaky_relu(x, 0.1)

        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_post)


class WavLM2AudioImproved(nn.Module):
    """
    feature_mode:
      - "last"            : outputs.last_hidden_state
      - "last_n_mean"     : moyenne uniforme des N dernières couches (1..12), exclut embedding idx=0
      - "weighted_all"    : somme pondérée (softmax) apprenable sur TOUTES les hidden_states (idx=0..12)
      - "weighted_last_n" : somme pondérée (softmax) apprenable sur les N dernières couches (1..12), exclut idx=0
    """
    def __init__(
        self,
        wavlm_model_name="microsoft/wavlm-base-plus",
        hidden_dim=256,
        num_adapter_layers=3,
        kernel_size=7,
        freeze_wavlm=True,
        dropout=0.1,
        feature_mode="last_n_mean",
        wavlm_last_n=1,
        use_snake=False,
    ):
        super().__init__()

        self.freeze_wavlm = bool(freeze_wavlm)
        self.feature_mode = str(feature_mode)
        self.wavlm_last_n = int(wavlm_last_n)

        # Load WavLM (local first)
        try:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                wavlm_model_name, local_files_only=True
            )
            self.wavlm = WavLMModel.from_pretrained(
                wavlm_model_name, local_files_only=True
            )
        except Exception:
            self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wavlm_model_name)
            self.wavlm = WavLMModel.from_pretrained(wavlm_model_name)

        self.wavlm_dim = self.wavlm.config.hidden_size
        self.num_wavlm_layers = self.wavlm.config.num_hidden_layers + 1  # 13 for base-plus

        # Freeze WavLM
        if self.freeze_wavlm:
            for p in self.wavlm.parameters():
                p.requires_grad = False
            self.wavlm.eval()

        # Create learnable layer weights ONLY if used (avoid unused params in DDP)
        if self.feature_mode == "weighted_all":
            # logits (softmax) over all hidden_states (idx 0..12)
            self.layer_weights = nn.Parameter(torch.zeros(self.num_wavlm_layers))
            logger.info(f"Created {self.num_wavlm_layers} learnable logits for weighted_all")

        elif self.feature_mode == "weighted_last_n":
            # logits (softmax) over last-N transformer layers (exclude embedding idx=0)
            max_n = self.num_wavlm_layers - 1  # 12
            n = max(1, min(self.wavlm_last_n, max_n))
            self.layer_weights = nn.Parameter(torch.zeros(n))
            logger.info(f"Created {n} learnable logits for weighted_last_n (wavlm_last_n={self.wavlm_last_n})")

        else:
            self.layer_weights = None

        # Adapter + Generator
        self.adapter = WavLMAdapterImproved(
            wavlm_dim=self.wavlm_dim,
            hidden_dim=hidden_dim,
            num_layers=num_adapter_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            use_snake=use_snake,
        )

        self.generator = HiFiGeneratorImproved(
            hidden_dim=hidden_dim,
            use_snake=use_snake,
        )

        logger.info(
            f"[WavLM2AudioImproved] feature_mode={self.feature_mode}, wavlm_last_n={self.wavlm_last_n}, "
            f"freeze_wavlm={self.freeze_wavlm}"
        )

    def _select_features(self, outputs):
        """
        returns: features (B, T', D)
        """
        if self.feature_mode == "last":
            return outputs.last_hidden_state

        hidden_states = outputs.hidden_states
        assert hidden_states is not None, "hidden_states required but None"

        if self.feature_mode == "weighted_all":
            # Weighted sum over ALL hidden_states (including embedding idx=0)
            assert self.layer_weights is not None, "layer_weights missing for weighted_all"
            w = F.softmax(self.layer_weights, dim=0)  # (L,)
            return sum(wi * hi for wi, hi in zip(w, hidden_states))

        if self.feature_mode == "weighted_last_n":
            # Weighted sum (softmax) over last-N layers, exclude embedding idx=0
            assert self.layer_weights is not None, "layer_weights missing for weighted_last_n"

            L = len(hidden_states)              # typically 13
            max_n = L - 1                       # 12 (exclude embedding)
            n_cfg = max(1, min(self.wavlm_last_n, max_n))

            # tie n to actual parameter size (must match across ranks)
            n = int(self.layer_weights.numel())
            if n != n_cfg:
                raise ValueError(f"Mismatch: layer_weights has n={n}, but wavlm_last_n implies n={n_cfg}")

            start = L - n                        # N=2 -> start=11 -> idx 11,12
            idxs = list(range(start, L))
            idxs = [i for i in idxs if i >= 1]   # never include embedding 0

            hs = [hidden_states[i] for i in idxs]
            w = F.softmax(self.layer_weights, dim=0)  # (N,)

            if len(hs) != w.numel():
                raise ValueError(f"Mismatch: selected {len(hs)} layers, but have {w.numel()} weights.")

            return sum(wi * hi for wi, hi in zip(w, hs))

        if self.feature_mode == "last_n_mean":
            # hidden_states indices: 0=embedding, 1..12=transformer layers
            L = len(hidden_states)   # 13
            max_n = L - 1            # 12 (exclude embedding idx=0)
            n = max(1, min(self.wavlm_last_n, max_n))

            start = L - n            # n=1 -> 12 ; n=2 -> 11 ; ... ; n=12 -> 1
            idxs = list(range(start, L))
            idxs = [i for i in idxs if i >= 1]  # safety: never include 0

            hs = [hidden_states[i] for i in idxs]
            return torch.stack(hs, dim=0).mean(dim=0)

        raise ValueError(f"Unknown feature_mode={self.feature_mode}")

    def forward(self, audio):
        """
        audio: (B, T) float in [-1, 1]
        returns: (B, T) reconstructed waveform
        """
        B, T = audio.shape

        need_hs = (self.feature_mode != "last")

        if self.freeze_wavlm:
            with torch.no_grad():
                outputs = self.wavlm(audio, output_hidden_states=need_hs)
        else:
            outputs = self.wavlm(audio, output_hidden_states=need_hs)

        features = self._select_features(outputs)        # (B, T', D)
        adapted = self.adapter(features)                 # (B, C, T')
        reconstructed = self.generator(adapted).squeeze(1)  # (B, Tout)

        # Deterministic crop/pad (NO interpolation)
        if reconstructed.shape[1] > T:
            reconstructed = reconstructed[:, :T]
        elif reconstructed.shape[1] < T:
            reconstructed = F.pad(reconstructed, (0, T - reconstructed.shape[1]))

        return reconstructed

    def remove_weight_norm(self):
        self.adapter.remove_weight_norm()
        self.generator.remove_weight_norm()
