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
import zlib

import imageio_ffmpeg
import librosa
import numpy as np
import pandas as pd
import soundfile
from joblib import Parallel, delayed
from scipy.fft import dct as scipy_dct

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


_LFCC_FB = None


def _linear_filterbank() -> np.ndarray:
    """Triangular filters on a LINEAR frequency axis -- the whole of what separates
    LFCC from MFCC. Mel spacing spends its resolution on the low octaves and
    compresses the top ones; the replay fingerprint lives in the top ones, which is
    why LFCC beats MFCC on PA and why the official B02 baseline uses it.

    Cached at module level: it depends only on constants, and rebuilding a
    (70, 257) matrix inside every one of ~1.1 M worker calls is pure waste.
    """
    global _LFCC_FB
    if _LFCC_FB is None:
        n_f = config.LFCC_N_FILTERS
        edges = np.linspace(0.0, config.SAMPLE_RATE / 2.0, n_f + 2)
        freqs = np.fft.rfftfreq(config.LFCC_N_FFT, 1.0 / config.SAMPLE_RATE)
        fb = np.zeros((n_f, len(freqs)), dtype=np.float32)
        for i in range(n_f):
            lo, ctr, hi = edges[i], edges[i + 1], edges[i + 2]
            up = (freqs - lo) / max(ctr - lo, 1e-9)
            dn = (hi - freqs) / max(hi - ctr, 1e-9)
            fb[i] = np.clip(np.minimum(up, dn), 0.0, None)
        _LFCC_FB = fb
    return _LFCC_FB


def _with_deltas(c: np.ndarray) -> np.ndarray:
    """Stack [static, delta, delta-delta] -> (3*n_coeff, T).

    librosa.feature.delta needs at least `width` frames and raises otherwise. 2021
    contains clips down to 0.63 s, and while that is still ~63 frames at a 10 ms hop,
    the guard costs nothing and turns a crash on some future short file into a
    slightly cruder derivative.
    """
    t = c.shape[1]
    w = min(9, t if t % 2 else t - 1)
    if w < 3:
        z = np.zeros_like(c)
        return np.concatenate([c, z, z], axis=0)
    d1 = librosa.feature.delta(c, width=w)
    d2 = librosa.feature.delta(c, width=w, order=2)
    return np.concatenate([c, d1, d2], axis=0)


def extract_lfcc(y: np.ndarray) -> np.ndarray:
    """LFCC, standard ASVspoof recipe -> (60, T) float32. Frame-level, NOT pooled.

    Note the contrast with extract_mfcc_vector above, which pools to one vector per
    file: a GMM models a distribution over FRAMES, so nothing here may be collapsed.
    """
    S = np.abs(librosa.stft(y, n_fft=config.LFCC_N_FFT,
                            hop_length=config.LFCC_HOP_LENGTH,
                            win_length=config.LFCC_WIN_LENGTH)) ** 2
    energies = _linear_filterbank() @ S
    log_e = np.log(energies + 1e-10)
    c = scipy_dct(log_e, type=2, axis=0, norm="ortho")[:config.LFCC_N_COEFF]
    return _with_deltas(c).astype(np.float32)


def extract_cqt_dct(cqt_uint8: np.ndarray) -> np.ndarray:
    """Constant-Q cepstral features from the CACHED uint8 CQT -> (60, T) float32.

    Reads Phase 4's cache, so no audio is decoded -- which is the entire reason this
    partner is cheap, and (9.8c) also the reason it is expected to be the more
    REDUNDANT one: it consumes the identical array our own LCNN consumes.

    NOT CQCC. Real CQCC resamples the geometrically-spaced bins onto a uniform scale
    before the DCT; this cache cannot support that. The dequantisation below inverts
    features.extract_cqt_uint8 exactly: [0,255] -> [-CQT_TOP_DB, 0] dB. dB is already
    logarithmic, so the DCT applies directly with no further log.
    """
    db = cqt_uint8.astype(np.float32) / 255.0 * config.CQT_TOP_DB - config.CQT_TOP_DB
    c = scipy_dct(db, type=2, axis=0, norm="ortho")[:config.CQTDCT_N_COEFF]
    return _with_deltas(c).astype(np.float32)


def extract_cqt_uint8(y: np.ndarray) -> np.ndarray:
    cqt = librosa.cqt(
        y, sr=config.SAMPLE_RATE, hop_length=config.CQT_HOP_LENGTH,
        n_bins=config.CQT_N_BINS, bins_per_octave=config.CQT_BINS_PER_OCTAVE,
    )
    db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max, top_db=config.CQT_TOP_DB)
    scaled = (db + config.CQT_TOP_DB) / config.CQT_TOP_DB * 255.0
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8)


def recycle_workers() -> None:
    """Tear down the loky pool so the next Parallel() spawns fresh workers.

    `Parallel` has no public teardown -- the executor is process-global and is reached
    through loky, not through the Parallel object. Full rationale for why this is
    needed at all (Windows error 1450 after a few hundred chunks) is in
    evaluate_2021.recycle_workers; duplicated here as five lines rather than imported,
    because evaluate_2021 imports torch and pulling ~300 MB of CUDA runtime into a
    process that is about to hold a 432 MB frame matrix is not worth the deduplication.
    """
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
    except Exception:                                        # noqa: BLE001
        pass


def sample_frames_for_gmm(row: dict) -> dict:
    """joblib worker for 9.8c: extract frame-level features, keep only `quota` frames.

    Lives HERE rather than in train_gmm.py for the reason recorded in CLAUDE.md and
    demonstrated by extract_for_eval below: loky imports the module a callable is
    DEFINED in so it can unpickle it. A worker defined beside a torch import makes
    every one of the 8 workers load torch + the CUDA runtime (~300 MB each), which
    on 5.9 GB fails with Windows error 1455.

    Sampling is seeded from a STABLE hash of the filename -- crc32, not Python's
    builtin hash(), which is randomised per process unless PYTHONHASHSEED is set and
    would therefore make a resumed run sample differently from the run it resumed.
    """
    try:
        if row["feat"] == "lfcc":
            f = extract_lfcc(load_audio(row["filepath"]))
        else:                                   # cqtdct: reads the Phase 4 cache
            f = extract_cqt_dct(np.load(CQT_DIR / f"{row['filename']}.npy"))
        t, q = f.shape[1], row["quota"]
        if t <= q:
            sel = np.arange(t)
        else:
            rng = np.random.default_rng(zlib.crc32(row["filename"].encode()))
            sel = np.sort(rng.choice(t, size=q, replace=False))
        return {"filename": row["filename"], "frames": f[:, sel].T.copy(),
                "error": None}
    except Exception as e:
        return {"filename": row["filename"], "frames": None, "error": str(e)}


def extract_frames_for_scoring(row: dict) -> dict:
    """joblib worker for 9.8c scoring: ALL frames for one file, never sampled.

    Separate from the sampler above because the two do genuinely different things:
    fitting wants an unbiased, duration-neutral sample of the training distribution,
    while scoring needs every frame of the file being scored. Same module, same
    reason.
    """
    try:
        if row["feat"] == "lfcc":
            f = extract_lfcc(load_audio(row["filepath"]))
        else:
            f = extract_cqt_dct(np.load(row["cqt_path"]) if "cqt_path" in row
                                else CQT_DIR / f"{row['filename']}.npy")
        return {"filename": row["filename"], "frames": f.T, "error": None}
    except Exception as e:
        return {"filename": row["filename"], "frames": None, "error": str(e)}


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
