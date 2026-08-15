"""Phase 6: LCNN-9 (Light CNN with Max-Feature-Map activations).

Layer sizes follow the original LCNN recipe, which is also the backbone of the
official ASVspoof LFCC-LCNN baseline. Keeping the backend identical to that
baseline is deliberate: it means the eventual comparison isolates exactly ONE
variable -- the front-end (our CQT vs their LFCC) -- which is the thesis's central
claim. Swapping in a different backbone would change two things at once and make
the difference unattributable.

Nine conv layers, four 2x2 pools. Spatial trace at T=250:
    (1, 90, 250) -> 45x125 -> 22x62 -> 11x31 -> 5x15
Receptive field of a final unit is 64x64 input pixels: 64 of 90 frequency bins and
64 frames (~1.0s of audio) -- the scale a loudspeaker's frequency-response
signature actually lives at.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import config


class MFM(nn.Module):
    """Max-Feature-Map: split the channels in half and take the elementwise max.

    Replaces ReLU. Where ReLU compares each activation against a FIXED threshold of
    zero (destroying everything negative), MFM compares two LEARNED feature maps
    against each other and keeps the winner -- a competitive, data-driven selection
    that preserves informative negative-going responses. That matters here because
    replay artifacts are subtle, small-magnitude spectral deviations that a hard
    zero threshold could discard. It also halves the channel count, which is what
    makes LCNN "light": the next conv sees half the input channels.
    """

    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)


class MFMLinear(nn.Module):
    """MFM over a flat feature vector (chunks along the feature axis)."""

    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)


def conv_mfm(in_ch: int, out_ch: int, k: int) -> nn.Sequential:
    """out_ch is the conv's width BEFORE MFM halves it, matching how the LCNN paper
    quotes filter counts (e.g. 'Conv 3x3, 96' yields 48 channels downstream)."""
    return nn.Sequential(nn.Conv2d(in_ch, out_ch, k, padding=k // 2), MFM())


class LCNN(nn.Module):
    def __init__(self, n_bins: int = config.CQT_N_BINS,
                 n_frames: int = config.CQT_FIXED_FRAMES,
                 head: str = config.LCNN_HEAD,
                 dropout: float = config.LCNN_DROPOUT):
        super().__init__()
        self.head_kind = head

        self.features = nn.Sequential(
            conv_mfm(1, 64, 5),                              # -> 32 ch
            nn.MaxPool2d(2),                                 # (1)
            conv_mfm(32, 64, 1), nn.BatchNorm2d(32),         # 1x1 = channel remix
            conv_mfm(32, 96, 3),                             # -> 48 ch
            nn.MaxPool2d(2), nn.BatchNorm2d(48),             # (2)
            conv_mfm(48, 96, 1), nn.BatchNorm2d(48),
            conv_mfm(48, 128, 3),                            # -> 64 ch
            nn.MaxPool2d(2),                                 # (3)
            conv_mfm(64, 128, 1), nn.BatchNorm2d(64),
            conv_mfm(64, 64, 3), nn.BatchNorm2d(32),         # -> 32 ch
            conv_mfm(32, 64, 1), nn.BatchNorm2d(32),
            conv_mfm(32, 64, 3),                             # -> 32 ch
            nn.MaxPool2d(2),                                 # (4)
        )

        # Infer the post-conv shape rather than hardcoding it, so changing T or the
        # bin count cannot silently produce a wrong Linear size.
        with torch.no_grad():
            c, f, t = self.features(torch.zeros(1, 1, n_bins, n_frames)).shape[1:]

        if head == "timepool":
            # Pool the TIME axis only -> (C, F, 1). Two reasons over flattening:
            # the feature count becomes independent of T (so the T sweep compares
            # context, not model capacity), and the FREQUENCY axis survives -- which
            # is where the replay fingerprint lives, so pooling it away would
            # discard the evidence.
            self.pool = nn.AdaptiveAvgPool2d((f, 1))
            fc_in = c * f
        elif head == "flatten":
            # The original LCNN paper's head, kept as an ablation. Note this makes
            # the FC (and thus the parameter count) scale with T.
            self.pool = nn.Identity()
            fc_in = c * f * t
        else:
            raise ValueError(f"unknown head {head!r}")

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_in, 160), MFMLinear(),   # -> 80
            nn.BatchNorm1d(80),
            nn.Dropout(dropout),
            nn.Linear(80, 1),                     # raw logit; higher = more bonafide
        )
        self.fc_in = fc_in

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x).squeeze(-1)   # (B,) logits

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
