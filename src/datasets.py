"""Phase 6: torch Dataset over the packed CQT store.

This is where the fixed-length windowing described in PROJECT_PLAN.md 4.3b/4.3c
actually happens. Phase 4 deliberately cached CQTs at their natural length so that
the crop could be re-drawn every epoch; baking one crop in at cache time would have
made "random crop while training" not actually random.

Per sample:
    seek/read from the packed blob  ->  (90, n_frames) uint8
    -> tile-pad or crop to CQT_FIXED_FRAMES
    -> SpecAugment (train only)
    -> float32, normalised
    -> (1, 90, T) tensor + label
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config


class CQTDataset(Dataset):
    def __init__(self, split: str, train: bool, n_frames: int | None = None,
                 norm: str | None = None, augment: bool | None = None,
                 index: pd.DataFrame | None = None, seed: int = config.RANDOM_SEED,
                 wav_aug_copies: int = 0):
        self.split = split
        self.train = train
        self.T = n_frames if n_frames is not None else config.CQT_FIXED_FRAMES
        self.norm = norm if norm is not None else config.LCNN_NORM
        # Augmentation follows `train` unless overridden -- dev/eval must never be
        # augmented, and their crop must be deterministic so reported EER is stable.
        self.augment = train if augment is None else augment

        self.index = pd.read_parquet(config.PACKED_INDEX[split]) if index is None else index
        self.blob_path = config.PACKED_BLOB[split]

        # Pre-computed waveform-augmented copies (train only). All copies share this
        # index -- augment_waveform.py trims each augmented waveform back to the
        # original length precisely so offsets and frame counts stay identical.
        # The CLEAN blob stays in the pool, so ~1/(n+1) of draws are unmodified:
        # training exclusively on perturbed audio would adapt the model to an input
        # distribution that neither dev nor the 2021 eval set actually has.
        self.blob_paths = [self.blob_path]
        if self.augment and split == "train" and wav_aug_copies > 0:
            from .augment_waveform import aug_blob_path
            for c in range(1, wav_aug_copies + 1):
                p = aug_blob_path(c)
                if not p.exists():
                    raise FileNotFoundError(f"missing augmented copy {c}: {p}")
                self.blob_paths.append(p)
        # 1 = bonafide (positive class), 0 = spoof -- the convention fixed in metrics.py.
        self.labels = (self.index["label"] == "bonafide").to_numpy().astype(np.float32)
        self.offsets = self.index["offset"].to_numpy()
        self.n_frames = self.index["n_frames"].to_numpy()

        self._seed = seed
        self._fh = None   # opened lazily, per worker process (file handles do not fork)
        self._rng = None

    def __len__(self):
        return len(self.index)

    def _ensure_open(self):
        if self._fh is None:
            self._fh = [open(p, "rb", buffering=0) for p in self.blob_paths]
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng(self._seed + 9973 * wid)

    def _read(self, i: int) -> np.ndarray:
        n = int(self.n_frames[i])
        # Draw a fresh copy each access, so across epochs the same file is seen
        # clean and under several different augmentation chains -- composed on top
        # of the random crop and SpecAugment applied downstream.
        fh = self._fh[0] if len(self._fh) == 1 else self._fh[int(self._rng.integers(0, len(self._fh)))]
        fh.seek(int(self.offsets[i]))
        raw = fh.read(config.CQT_N_BINS * n)
        return np.frombuffer(raw, dtype=np.uint8).reshape(config.CQT_N_BINS, n)

    def _fit_length(self, x: np.ndarray) -> np.ndarray:
        """Pad short clips by TILING, not zero-padding: a block of digital silence
        never occurs in real speech and gives the CNN an artificial boundary to key
        on. Crop long ones randomly when training (fresh window each epoch = free
        augmentation) and centrally otherwise (deterministic, reproducible EER)."""
        T, n = self.T, x.shape[1]
        if n < T:
            reps = int(np.ceil(T / n))
            x = np.tile(x, (1, reps))
            n = x.shape[1]
        if n > T:
            start = int(self._rng.integers(0, n - T + 1)) if self.train else (n - T) // 2
            x = x[:, start:start + T]
        return x

    def _specaugment(self, x: np.ndarray) -> np.ndarray:
        """Masks are written as 0 == the quietest dB value in our uint8 encoding,
        i.e. 'silence in this band/instant', which is the standard formulation."""
        x = x.copy()
        for _ in range(config.SPECAUG_N_FREQ_MASKS):
            w = int(self._rng.integers(0, config.SPECAUG_FREQ_MASK_MAX + 1))
            if w:
                f0 = int(self._rng.integers(0, max(1, config.CQT_N_BINS - w)))
                x[f0:f0 + w, :] = 0
        for _ in range(config.SPECAUG_N_TIME_MASKS):
            w = int(self._rng.integers(0, config.SPECAUG_TIME_MASK_MAX + 1))
            if w:
                t0 = int(self._rng.integers(0, max(1, self.T - w)))
                x[:, t0:t0 + w] = 0
        return x

    def _normalise(self, x: np.ndarray) -> np.ndarray:
        if self.norm == "unit":
            # uint8 0-255 encodes dB in [-80, 0]; a fixed global rescale that
            # preserves everything, including absolute level.
            return x.astype(np.float32) / 255.0
        if self.norm == "cmvn":
            # Per-utterance, per-frequency-bin mean/variance normalisation. Standard
            # in speech for removing channel effects -- but the replay fingerprint IS
            # a channel effect, so this may erase the evidence. Run as an ablation.
            x = x.astype(np.float32)
            mu = x.mean(axis=1, keepdims=True)
            sd = x.std(axis=1, keepdims=True)
            return (x - mu) / (sd + 1e-5)
        raise ValueError(f"unknown normalisation {self.norm!r}")

    def __getitem__(self, i: int):
        self._ensure_open()
        x = self._fit_length(self._read(i))
        if self.augment:
            x = self._specaugment(x)
        x = self._normalise(x)
        return torch.from_numpy(x).unsqueeze(0), torch.tensor(self.labels[i])

    def pos_weight(self) -> float:
        """n_spoof / n_bonafide -- passed to BCEWithLogitsLoss so the minority
        (bonafide) class is not drowned out. Same strategy as Phase 5's
        class_weight='balanced'."""
        n_pos = float(self.labels.sum())
        return (len(self.labels) - n_pos) / n_pos

    def __getstate__(self):
        # File handles and Generators do not survive being sent to a worker process.
        s = self.__dict__.copy()
        s["_fh"] = None
        s["_rng"] = None
        return s
