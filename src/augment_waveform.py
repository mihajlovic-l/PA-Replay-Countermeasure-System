"""Phase 6: pre-computed waveform-domain augmentation (PROJECT_PLAN.md phase 6, option b).

WHY waveform-domain at all, when SpecAugment is free? Because the two target
different problems. SpecAugment perturbs the spectrogram and guards against
memorising specific files. Waveform augmentation perturbs the *physics*, and is
aimed at the simulation shortcut: 2019 PA is entirely simulated (shoebox RIRs,
synthetic device responses), so a high-capacity CNN can score well in-domain by
keying on simulator regularities that do not exist in 2021's real re-recordings.
Perturbing with the effects the simulator omits makes those regularities
unreliable, forcing the model onto physics that actually transfers.

WHY pre-compute instead of augmenting on the fly: applying these needs the
waveform, but Phase 4 cached CQTs. Re-decoding and re-CQT-ing every epoch costs
~1h/epoch (~20h per run at Phase 4's measured 155ms/file), and Phase 6 involves
many runs. Pre-computing is a one-time ~3-4h and zero per-run cost. See
PROGRESS_REPORT.md for the full argument.

Each copy is an INDEPENDENT random draw: every file gets its own randomly-sampled
chain (and randomly-sampled parameters) from the menu below, so chains like
"noise -> room -> codec" occur rather than each effect only ever in isolation.

Augmented waveforms are trimmed/padded back to the ORIGINAL length before the CQT,
so every copy shares the existing packed index (same offsets, same n_frames). That
keeps the Dataset trivial and rules out a whole class of offset-mismatch bugs.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from tqdm import tqdm

from . import config
from .features import extract_cqt_uint8, load_audio

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def aug_blob_path(copy: int) -> Path:
    return config.PACKED_DIR / f"cqt_train_aug{copy}.dat"


# ---------------------------------------------------------------------------
# Individual effects. Each takes and returns a float32 waveform at SAMPLE_RATE.
# ---------------------------------------------------------------------------

def add_noise(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Additive noise at a random SNR -- stands in for real microphone self-noise,
    which the 2019 simulation does not model at all."""
    snr_db = float(rng.uniform(10.0, 30.0))
    sig_power = float(np.mean(y ** 2)) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    # Mix white and low-passed ("pink-ish") noise so it is not always spectrally flat.
    noise = rng.standard_normal(len(y)).astype(np.float32)
    if rng.random() < 0.5:
        k = int(rng.integers(3, 16))
        noise = np.convolve(noise, np.ones(k, dtype=np.float32) / k, mode="same")
        noise /= (noise.std() + 1e-9)
    return y + noise * np.sqrt(noise_power)


def add_reverb(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Convolve with a synthetic exponentially-decaying-noise RIR.

    Deliberately NOT a shoebox model: the point is to add room behaviour whose
    structure differs from the simulator that generated the training data, so the
    model cannot treat 2019's particular RIR family as the definition of "a room".
    """
    rt60 = float(rng.uniform(0.08, 0.5))
    n = int(rt60 * config.SAMPLE_RATE)
    if n < 8:
        return y
    t = np.arange(n, dtype=np.float32) / config.SAMPLE_RATE
    rir = rng.standard_normal(n).astype(np.float32) * np.exp(-6.9078 * t / rt60)
    rir[0] += 1.0                      # direct path
    rir /= (np.abs(rir).sum() + 1e-9)  # preserve loudness
    return np.convolve(y, rir, mode="full")[:len(y)].astype(np.float32)


def soft_clip(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Nonlinear distortion -- real loudspeakers are not linear, the simulated
    device responses are. tanh gives smooth saturation; the hard branch models a
    driven amplifier clipping."""
    peak = float(np.abs(y).max()) + 1e-9
    drive = float(rng.uniform(1.5, 6.0))
    if rng.random() < 0.5:
        out = np.tanh(y / peak * drive) / np.tanh(drive) * peak
    else:
        thr = peak * float(rng.uniform(0.3, 0.8))
        out = np.clip(y, -thr, thr)
    return out.astype(np.float32)


def codec_roundtrip(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Encode and decode through a lossy codec -- stands in for real
    recording/transmission chain processing. This is the slow effect (two ffmpeg
    subprocess calls), so it is sampled less often than the others."""
    codec, ext, rate = [
        ("libmp3lame", "mp3", f"{int(rng.choice([64, 96, 128, 192]))}k"),
        ("aac", "m4a", f"{int(rng.choice([64, 96, 128]))}k"),
    ][int(rng.integers(0, 2))]

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.wav"
        enc = Path(td) / f"mid.{ext}"
        sf.write(src, y, config.SAMPLE_RATE)
        for cmd in (
            [FFMPEG, "-v", "quiet", "-y", "-i", str(src), "-c:a", codec, "-b:a", rate, str(enc)],
            [FFMPEG, "-v", "quiet", "-i", str(enc), "-f", "f32le", "-ac", "1",
             "-ar", str(config.SAMPLE_RATE), "-"],
        ):
            if cmd[-1] == "-":
                p = subprocess.run(cmd, capture_output=True)
                if p.returncode != 0 or not p.stdout:
                    return y
                return np.frombuffer(p.stdout, dtype=np.float32).copy()
            if subprocess.run(cmd, capture_output=True).returncode != 0:
                return y
    return y


EFFECTS = [
    ("noise", add_noise, 0.60),
    ("reverb", add_reverb, 0.50),
    ("clip", soft_clip, 0.35),
    ("codec", codec_roundtrip, 0.30),
]


def augment(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random chain. Order is shuffled so e.g. clipping-then-room and
    room-then-clipping both occur. At least one effect is always applied -- the
    clean version is already available to the Dataset as the original blob."""
    order = rng.permutation(len(EFFECTS))
    applied = False
    for i in order:
        name, fn, p = EFFECTS[i]
        if rng.random() < p:
            y = fn(y, rng)
            applied = True
    if not applied:
        y = add_noise(y, rng)
    peak = float(np.abs(y).max())
    if peak > 1.0:            # keep in range; CQT is magnitude-based but be safe
        y = y / peak
    return y.astype(np.float32)


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    """Back to the original sample count so the CQT frame count -- and therefore
    the packed index -- matches the clean copy exactly."""
    if len(y) > n:
        return y[:n]
    if len(y) < n:
        return np.pad(y, (0, n - len(y)))
    return y


def process_one(args) -> tuple[int, bytes] | tuple[int, None]:
    i, filepath, n_samples, seed = args
    try:
        rng = np.random.default_rng(seed)
        y = load_audio(filepath)
        y = fit_length(augment(y, rng), len(y))
        return i, extract_cqt_uint8(y).tobytes()
    except Exception:
        return i, None


def resume_point(tmp_path: Path, rows: pd.DataFrame) -> int:
    """How many files of a partially-written copy are already complete.

    Because every copy shares the clean blob's layout, a file's byte offset is
    known in advance -- so the partial file's SIZE alone tells us exactly where we
    stopped. Any trailing incomplete record is truncated back to the last clean
    boundary, so resumption can never leave a half-written CQT in the blob (which
    would silently misalign every subsequent sample).
    """
    if not tmp_path.exists():
        return 0
    size = tmp_path.stat().st_size
    offs = rows["offset"].to_numpy()
    done = max(0, int(np.searchsorted(offs, size, side="right")) - 1)
    if offs[done] != size:
        with open(tmp_path, "r+b") as f:
            f.truncate(int(offs[done]))
    return done


def build_copy(copy: int, rows: pd.DataFrame, n_jobs: int, force: bool):
    out_path = aug_blob_path(copy)
    tmp_path = out_path.with_suffix(".partial")
    if out_path.exists() and not force:
        print(f"  copy {copy}: exists ({out_path.stat().st_size/1e9:.2f} GB) -- skipping")
        return
    if force and tmp_path.exists():
        tmp_path.unlink()

    expected = (rows["n_frames"].to_numpy() * config.CQT_N_BINS)
    tasks = [(i, rows.filepath.iloc[i], None,
              config.RANDOM_SEED + 100_000 * copy + i) for i in range(len(rows))]

    # Resume WITHIN a copy, not just between copies -- each copy is ~26 min of work
    # and this session has lost background jobs to crashes more than once.
    done = resume_point(tmp_path, rows)
    if done:
        print(f"  copy {copy}: resuming at file {done:,}/{len(tasks):,} "
              f"({done/len(tasks)*100:.1f}% already written)")
    tasks = tasks[done:]

    fails = 0
    with open(tmp_path, "ab" if done else "wb") as blob:
        for start in tqdm(range(0, len(tasks), 2000), desc=f"copy {copy}", unit="chunk"):
            chunk = tasks[start:start + 2000]
            results = Parallel(n_jobs=n_jobs)(delayed(process_one)(t) for t in chunk)
            for (i, raw) in sorted(results, key=lambda r: r[0]):
                if raw is None or len(raw) != expected[i]:
                    # Fall back to the clean CQT so offsets stay exactly aligned.
                    # A length mismatch here would corrupt every later sample.
                    fails += 1
                    raw = read_clean(i, rows)
                blob.write(raw)
            blob.flush()   # so the resume point is accurate after an abrupt kill
    tmp_path.replace(out_path)
    print(f"  copy {copy}: {out_path.stat().st_size/1e9:.2f} GB written"
          + (f"  ({fails} files fell back to clean)" if fails else ""))


_clean_fh = None


def read_clean(i: int, rows: pd.DataFrame) -> bytes:
    global _clean_fh
    if _clean_fh is None:
        _clean_fh = open(config.PACKED_BLOB["train"], "rb", buffering=0)
    _clean_fh.seek(int(rows.offset.iloc[i]))
    return _clean_fh.read(config.CQT_N_BINS * int(rows.n_frames.iloc[i]))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--n-copies", type=int, default=3)
    p.add_argument("--n-jobs", type=int, default=config.FEATURE_EXTRACTION_N_JOBS)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    index = pd.read_parquet(config.PACKED_INDEX["train"])
    split = pd.read_csv(config.SPLITS_DIR / "train_2019.csv")[["filename", "filepath"]]
    rows = index.merge(split, on="filename", how="left")
    if rows.filepath.isna().any():
        raise RuntimeError("some packed files have no filepath in train_2019.csv")

    print(f"Augmenting {len(rows):,} training files into {args.n_copies} copies "
          f"({args.n_jobs} workers)")
    print("  dev/eval are NOT augmented -- they are always evaluated clean.")
    for copy in range(1, args.n_copies + 1):
        build_copy(copy, rows, args.n_jobs, args.force)

    total = sum(aug_blob_path(c).stat().st_size for c in range(1, args.n_copies + 1))
    print(f"\nDone. {total/1e9:.2f} GB of augmented copies in {config.PACKED_DIR}")


if __name__ == "__main__":
    main()
