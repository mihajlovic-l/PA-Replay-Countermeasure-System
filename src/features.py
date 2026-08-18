"""Phase 4: DSP feature extraction (MFCC + CQT), cached to E:\\ASVspoof\\features.

Only touches the 2019 pool (train_2019.csv + dev_2019.csv, 241,056 files combined).
2021 features are extracted on-the-fly during Phase 7 evaluation and never cached --
see PROJECT_PLAN.md section 4.4 and PROGRESS_REPORT.md for the disk-cost reasoning
(caching CQT for all 721,332 2021-eval files would cost ~27GB for a set that's only
scored once).

Every file is decoded via ffmpeg (see config.py's FEATURE_EXTRACTION_N_JOBS comment
for why), not soundfile. CQT is cached at each file's natural length -- uncapped,
unpadded, uncropped (confirmed across all 241,056 files that only 0.75% exceed 10s
and capping would save a negligible ~17MB out of ~6.1GB total). The fixed-length
pad/random-crop/center-crop handling described in PROJECT_PLAN.md happens later, in
Phase 6's Dataset class, operating on these variable-length cached arrays -- baking a
single crop in now would make "random crop while training" not actually random
across epochs.

Resumable by design: re-running this script skips any file whose MFCC row is already
in the checkpointed parquet AND whose CQT .npy already exists, and checkpoints after
every chunk -- this script previously survived an unrelated environment crash mid-run
without needing to restart, and there's no reason to assume the next multi-hour run
won't hit the same kind of interruption.
"""
from __future__ import annotations

import subprocess
import sys
import warnings

import imageio_ffmpeg
import librosa
import numpy as np
import pandas as pd
import soundfile
from joblib import Parallel, delayed

from . import config

CQT_DIR = config.FEATURES_DIR / "cqt"
MFCC_DIR = config.FEATURES_DIR / "mfcc"
CQT_DIR.mkdir(parents=True, exist_ok=True)
MFCC_DIR.mkdir(parents=True, exist_ok=True)

MFCC_PARQUET = MFCC_DIR / "pooled_mfcc.parquet"
FAILURES_CSV = config.FEATURES_DIR / "extraction_failures.csv"

# librosa.cqt downsamples the signal for each successive octave, so on very short
# clips the lowest octave is left with fewer samples than its FFT window and librosa
# zero-pads, warning each time. Benign here: it perturbs only the lowest CQT bins
# (~32-65Hz), far below the high-frequency band the replay fingerprint occupies, and
# the output is still a valid CQT. But 2021 contains clips down to 0.63s (39 frames,
# tiled ~10x at T=400) and roughly 0.6% of the corpus is under 1s, so this fires
# thousands of times -- each with a DIFFERENT signal length in the message, which
# defeats Python's once-per-location dedup and buries the progress bar of a 6-hour
# run. Filtered narrowly by message rather than blanket-suppressing UserWarning.
#
# Nothing is hidden by this: every file's duration_s and n_frames are recorded in the
# Phase 7 score shards, so exactly which files were affected stays measurable after
# the fact.
warnings.filterwarnings("ignore", message=r"n_fft=\d+ is too large for input signal",
                        category=UserWarning)

FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()
CHUNK_SIZE = 5000
MFCC_COLS = [f"mfcc_{i}" for i in range(120)]


def load_audio_ffmpeg(path: str) -> np.ndarray:
    """Decode by spawning ffmpeg. Universal -- handles every file in both corpora --
    but costs ~20.9 ms/file at 8 processes, of which ~20.2 ms is the SPAWN itself
    (measured with `ffmpeg -version`, i.e. no input file at all). Process creation is
    a system-wide serialisation point here (kernel + Defender scanning the binary on
    each launch), so it barely parallelises: 8 workers buy only 2.0x over 1. That
    caps decode near ~48 files/s no matter how many cores are thrown at it."""
    cmd = [
        FFMPEG_EXE, "-v", "quiet", "-i", path,
        "-f", "f32le", "-ac", "1", "-ar", str(config.SAMPLE_RATE), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exit {proc.returncode}: {proc.stderr[-300:].decode(errors='replace')}")
    y = np.frombuffer(proc.stdout, dtype=np.float32)
    if len(y) == 0:
        raise RuntimeError("ffmpeg produced zero samples")
    return y


def load_audio(path: str, prefer_soundfile: bool = True) -> np.ndarray:
    """Decode to mono float32 at config.SAMPLE_RATE.

    Tries soundfile first (in-process, no spawn), falling back to ffmpeg for anything
    it cannot read. This is NOT a change of numerical behaviour, and that was verified
    rather than assumed: on 120 files where both decoders succeed the outputs are
    **bit-identical, max absolute difference exactly 0.0**. FLAC is lossless, so any
    correct decoder must agree -- which is why the fallback is free of the numerical
    inconsistency that Phase 4's "one uniform decode path" rule existed to prevent.
    Re-running Phase 4 through this function would reproduce its cache exactly.

    Why the change: Phase 4 measured soundfile failing on ~46% of the **2019** corpus
    ("flac decoder lost sync", a libsndfile decoder limitation) and switched
    everything to ffmpeg. That figure was never re-tested on 2021 -- where soundfile
    reads **500/500** sampled files. Since ffmpeg's cost is almost entirely process
    spawn, avoiding it where possible is the single biggest lever on Phase 7 runtime.

    Any file soundfile cannot read, or delivers at an unexpected sample rate, falls
    through to ffmpeg, which also handles resampling. So correctness never depends on
    soundfile's coverage -- only speed does.
    """
    if prefer_soundfile:
        try:
            y, sr = soundfile.read(path, dtype="float32", always_2d=False)
            if y.ndim > 1:                      # mixdown, matching ffmpeg's -ac 1
                y = y.mean(axis=1, dtype=np.float32)
            if sr == config.SAMPLE_RATE and len(y):
                return np.ascontiguousarray(y, dtype=np.float32)
        except Exception:                       # noqa: BLE001 -- any failure -> ffmpeg
            pass
    return load_audio_ffmpeg(path)


def extract_mfcc_vector(y: np.ndarray) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCC,
        n_fft=config.MFCC_N_FFT, hop_length=config.MFCC_HOP_LENGTH,
    )
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    feat = np.concatenate([mfcc, delta1, delta2], axis=0)  # (60, T)
    return np.concatenate([feat.mean(axis=1), feat.std(axis=1)]).astype(np.float32)  # (120,)


def extract_cqt_uint8(y: np.ndarray) -> np.ndarray:
    cqt = librosa.cqt(
        y, sr=config.SAMPLE_RATE, hop_length=config.CQT_HOP_LENGTH,
        n_bins=config.CQT_N_BINS, bins_per_octave=config.CQT_BINS_PER_OCTAVE,
    )
    db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max, top_db=config.CQT_TOP_DB)
    scaled = (db + config.CQT_TOP_DB) / config.CQT_TOP_DB * 255.0
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def process_one(row: dict) -> dict:
    filename = row["filename"]
    cqt_path = CQT_DIR / f"{filename}.npy"
    try:
        y = load_audio(row["filepath"])
        mfcc_vec = extract_mfcc_vector(y)
        if not cqt_path.exists():
            np.save(cqt_path, extract_cqt_uint8(y))
        return {"filename": filename, "mfcc": mfcc_vec, "error": None}
    except Exception as e:
        return {"filename": filename, "mfcc": None, "error": str(e)}


def extract_for_eval(row: dict) -> dict:
    """Decode + MFCC + CQT, returning the CQT instead of writing a .npy.

    Phase 7's streaming worker. It lives HERE, not in evaluate_2021.py, and that is
    load-bearing rather than tidiness: joblib's loky workers import the module a
    function is DEFINED in so they can unpickle it, and evaluate_2021 imports torch.
    Defining it there made all 8 workers load torch + the CUDA runtime (~300MB
    each), which on this 5.9GB machine failed with Windows error 1455
    (ERROR_COMMITMENT_LIMIT) on ~11% of files -- the same commit-limit ceiling that
    rules out DataLoader workers in Phase 6. This module imports no torch, so a
    worker here costs what Phase 4's did, which is known to fit.
    """
    try:
        y = load_audio(row["filepath"])
        return {"filename": row["filename"], "mfcc": extract_mfcc_vector(y),
                "cqt": extract_cqt_uint8(y), "n_samples": len(y), "error": None}
    except Exception as e:  # noqa: BLE001
        return {"filename": row["filename"], "mfcc": None, "cqt": None,
                "n_samples": 0, "error": f"{type(e).__name__}: {e}"}


def load_pool() -> pd.DataFrame:
    train_df = pd.read_csv(config.SPLITS_DIR / "train_2019.csv")
    train_df["subset"] = "train"
    dev_df = pd.read_csv(config.SPLITS_DIR / "dev_2019.csv")
    dev_df["subset"] = "dev"
    return pd.concat([train_df, dev_df], ignore_index=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    pool = load_pool()
    print(f"Pool: {len(pool)} files")

    cached = pd.read_parquet(MFCC_PARQUET) if MFCC_PARQUET.exists() else pd.DataFrame(columns=["filename"])
    done = set(cached["filename"]) & {p.stem for p in CQT_DIR.glob("*.npy")}
    todo = pool[~pool["filename"].isin(done)].reset_index(drop=True)
    print(f"Already cached: {len(done)}. Remaining: {len(todo)}")

    failures = []
    accumulated = cached[cached["filename"].isin(done)].reset_index(drop=True)

    for start in range(0, len(todo), CHUNK_SIZE):
        chunk = todo.iloc[start:start + CHUNK_SIZE]
        records = chunk.to_dict("records")

        results = Parallel(n_jobs=config.FEATURE_EXTRACTION_N_JOBS, verbose=5)(
            delayed(process_one)(row) for row in records
        )

        new_rows = []
        for meta, res in zip(records, results):
            if res["error"] is not None:
                failures.append({"filename": res["filename"], "error": res["error"]})
                continue
            row_out = {
                "filename": meta["filename"], "speaker_id": meta["speaker_id"],
                "label": meta["label"], "subset": meta["subset"],
                "env_id": meta["env_id"], "attack_id": meta["attack_id"], "split": meta["split"],
            }
            row_out.update(zip(MFCC_COLS, res["mfcc"]))
            new_rows.append(row_out)

        if new_rows:
            accumulated = pd.concat([accumulated, pd.DataFrame(new_rows)], ignore_index=True)
            accumulated.to_parquet(MFCC_PARQUET, index=False)

        done_so_far = start + len(chunk)
        print(f"chunk {start}-{done_so_far}/{len(todo)}: {len(accumulated)} total cached rows, {len(failures)} failures so far")

    if failures:
        pd.DataFrame(failures).to_csv(FAILURES_CSV, index=False)
        print(f"{len(failures)} failures written to {FAILURES_CSV}")

    print(f"\nDone. {len(accumulated)} rows in {MFCC_PARQUET}, CQT arrays in {CQT_DIR}")


if __name__ == "__main__":
    main()
