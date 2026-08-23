"""9.8c pass 2: score the in-house GMMs on 2019 dev and 2021, streaming.

    python -m src.score_gmm --validate               # the declared batching control
    python -m src.score_gmm --feat lfcc --source dev
    python -m src.score_gmm                          # all four combinations

Emits one per-file log-likelihood ratio per (feature, corpus) to
`GMM_DIR/<feat>_<source>_scores.parquet`. 2019 **dev** is scored because that is what
makes a ZERO-SHOT fusion possible: weights fitted there mean the system never sees a
2021 label (9.8c). 2021 is scored on all three partitions, as Phase 7 does.

NOTHING FRAME-LEVEL IS EVER STORED FOR 2021. At ~149 frames/file x 943,110 files x 60
dims that would be ~79 GB; features are extracted, scored and discarded per chunk.

FOUR PATHS, because the two features get their input from different places:

    lfcc   + dev/2021  -> decode audio (~36 ms/file: 32 decode, 4 LFCC), 8 workers
    cqtdct + dev       -> read the Phase 4 per-file .npy cache
    cqtdct + 2021      -> read the Phase 7 blob shards sequentially

The cqtdct paths decode no audio at all, which is why that partner is cheap -- and
(9.8c) also why it is predicted to be the more redundant one: it consumes the very
array our own LCNN consumes.

BATCHED SCORING. Frames from a whole chunk are concatenated and scored in ONE pass,
then reduced back per file with `np.add.reduceat`. Scoring file-by-file would repeat a
(n_frames x 512) matmul setup ~1 M times; batching turns it into a few thousand large
matmuls. `--validate` checks the two give the same answer, which is 9.8c.1(b).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from . import config, features
from .gmm import DiagGMM, llr

CHUNK = 2000
POOL_REBUILD_EVERY = 20
N_SHARDS = 236


def scores_path(feat: str, source: str):
    return config.GMM_DIR / f"{feat}_{source}_scores.parquet"


def load_pair(feat: str) -> tuple[DiagGMM, DiagGMM]:
    from .train_gmm import model_path
    for lab in ("bonafide", "spoof"):
        if not model_path(feat, lab).exists():
            sys.exit(f"missing {model_path(feat, lab).name} -- run train_gmm first")
    return (DiagGMM.load(model_path(feat, "bonafide")),
            DiagGMM.load(model_path(feat, "spoof")))


def score_batch(bona: DiagGMM, spoof: DiagGMM, frames: list) -> np.ndarray:
    """Mean per-frame LLR for each file, computed as one batched pass."""
    lens = np.array([len(f) for f in frames])
    out = np.full(len(frames), np.nan)
    keep = np.flatnonzero(lens > 0)
    if not len(keep):
        return out
    X = np.concatenate([frames[i] for i in keep])
    d = bona.score_frames(X) - spoof.score_frames(X)
    edges = np.concatenate([[0], np.cumsum(lens[keep])[:-1]])
    out[keep] = np.add.reduceat(d, edges) / lens[keep]
    return out


# --- the four input paths ---------------------------------------------------------

def _audio_chunks(df: pd.DataFrame, feat: str, n_jobs: int, start: int = 0):
    """Decode + extract in parallel. Yields (chunk_index, filenames, frames)."""
    rows = [{"filename": r.filename, "filepath": r.filepath, "feat": feat}
            for r in df.itertuples()]
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    for ci in range(start, len(chunks)):
        # Recycle the pool periodically: loky reuses one across Parallel() calls and
        # the workers accumulate handles until a large pipe write fails with Windows
        # error 1450. This killed a Phase 7 run at chunk 117. There is no teardown on
        # the Parallel object -- the executor is process-global (features.recycle_workers).
        if ci % POOL_REBUILD_EVERY == 0:
            features.recycle_workers()
        ch = chunks[ci]
        try:
            res = Parallel(n_jobs=n_jobs)(
                delayed(features.extract_frames_for_scoring)(r) for r in ch)
        except Exception as e:
            tqdm.write(f"  pool failed on chunk {ci} ({type(e).__name__}: "
                       f"{str(e)[:80]}); recycling and retrying")
            features.recycle_workers()
            time.sleep(10)
            res = Parallel(n_jobs=n_jobs)(
                delayed(features.extract_frames_for_scoring)(r) for r in ch)
        yield (ci, [r["filename"] for r in res],
               [r["frames"] if r["error"] is None else np.empty((0, 60), np.float32)
                for r in res])
    features.recycle_workers()


def _cqt_npy_chunks(df: pd.DataFrame, start: int = 0):
    """2019: one cached .npy per file."""
    rows = list(df.itertuples())
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    for ci in range(start, len(chunks)):
        names, frames = [], []
        for r in chunks[ci]:
            names.append(r.filename)
            try:
                a = np.load(features.CQT_DIR / f"{r.filename}.npy")
                frames.append(features.extract_cqt_dct(a).T)
            except Exception:                                # noqa: BLE001
                frames.append(np.empty((0, 60), np.float32))
        yield ci, names, frames


def _cqt_shard_chunks(start: int = 0):
    """2021: sequential blob shards, the layout Phase 7 already wrote."""
    for c in range(start, N_SHARDS):
        blob = config.PA2021_CQT_SHARD_DIR / f"cqt_{c:05d}.dat"
        index = config.PA2021_CQT_SHARD_DIR / f"index_{c:05d}.parquet"
        if not index.exists():
            continue
        idx = pd.read_parquet(index)
        names, frames = list(idx["filename"]), []
        with open(blob, "rb", buffering=0) as fh:
            for off, nf in zip(idx["offset"].to_numpy(), idx["n_frames"].to_numpy()):
                fh.seek(int(off))
                raw = fh.read(config.CQT_N_BINS * int(nf))
                a = np.frombuffer(raw, dtype=np.uint8).reshape(config.CQT_N_BINS,
                                                               int(nf))
                frames.append(features.extract_cqt_dct(a).T)
        yield c, names, frames


def run(feat: str, source: str, n_jobs: int, force: bool) -> None:
    """Score one (feature, corpus) pair.

    Resumable per chunk: each chunk's scores are written to their own parquet under
    `<feat>_<source>_parts/` and the final table is a concat of those. This mirrors
    the Phase 7 shard pattern, and it matters most on the 2021 LFCC pass, which
    decodes 943,110 files over 1-2 h -- losing that to an interrupt at 95% would be
    the same mistake twice.
    """
    out_p = scores_path(feat, source)
    if out_p.exists() and not force:
        print(f"{out_p.name} exists -- skipping (--force to rescore)")
        return
    parts = config.GMM_DIR / f"{feat}_{source}_parts"
    parts.mkdir(exist_ok=True)
    if force:
        for f in parts.glob("*.parquet"):
            f.unlink()

    tag = config.PHASE7_INHOUSE_GMM_BY_FEAT[feat]
    bona, spoof = load_pair(feat)

    if source == "dev":
        df = pd.read_csv(config.SPLITS_DIR / "dev_2019.csv")
        total, n_chunks = len(df), (len(df) + CHUNK - 1) // CHUNK
    else:
        total = len(pd.read_parquet(config.PA2021_CQT_INDEX, columns=["filename"]))
        df = (pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet",
                              columns=["filename", "filepath"])
              if feat == "lfcc" else None)
        n_chunks = ((total + CHUNK - 1) // CHUNK) if feat == "lfcc" else N_SHARDS

    # Resume from the first chunk with no part file. Trailing parts are kept: chunk
    # indices are stable, so a gap can only be a chunk that genuinely never ran.
    done = {int(f.stem.split("_")[-1]) for f in parts.glob("part_*.parquet")}
    start = 0
    while start in done:
        start += 1
    if start:
        print(f"  resuming at chunk {start}/{n_chunks} ({len(done)} parts on disk)")

    if source == "dev":
        gen = (_audio_chunks(df, feat, n_jobs, start) if feat == "lfcc"
               else _cqt_npy_chunks(df, start))
    else:
        gen = (_audio_chunks(df, feat, n_jobs, start) if feat == "lfcc"
               else _cqt_shard_chunks(start))

    print(f"\n=== {feat} on {source}: {total:,} files, {n_chunks} chunks ===")
    t0 = time.time()
    bar = tqdm(total=n_chunks, desc=f"{feat}/{source}", unit="chunk", initial=start)
    for ci, names, frames in gen:
        s = score_batch(bona, spoof, frames)
        pd.DataFrame({"filename": names, tag: s.astype(np.float32)}).to_parquet(
            parts / f"part_{ci:05d}.parquet", index=False)
        bar.update(1)
        bar.set_postfix_str(f"{len(names)} files, {(time.time() - t0) / 60:.1f} min")
    bar.close()

    d = pd.concat([pd.read_parquet(f) for f in sorted(parts.glob("part_*.parquet"))],
                  ignore_index=True)
    d.to_parquet(out_p, index=False)
    bad = int(d[tag].isna().sum())
    print(f"  {len(d):,} scored ({bad} failed) in "
          f"{(time.time() - t0) / 60:.1f} min -> {out_p.name}")
    if len(d) != total:
        print(f"  WARNING: expected {total:,} rows, got {len(d):,}")


# --- the declared control ----------------------------------------------------------

def validate(feat: str = "cqtdct", n: int = 200) -> bool:
    """Batched scoring must equal per-file scoring (9.8c.1b).

    The batching is an optimisation; if it does not reproduce the one-file-at-a-time
    answer then every score in the parquet is wrong in a way nothing downstream would
    reveal.

    WHAT "EQUAL" MEANS HERE, AND WHY IT IS NOT BIT-EQUALITY. Concatenating files
    changes the SHAPE of the (n_frames x 60) @ (60 x 512) matmul, and BLAS blocks and
    sums a tall matrix differently from a short one. Measured, that moves a per-frame
    log-likelihood of magnitude ~2170 by ~1.7e-10 (relative 8e-14, i.e. ordinary
    float64 reassociation); worst case over a few hundred files it reaches ~1.7e-6 on
    the file score, driven by frames where two mixture components are nearly tied and
    the logsumexp amplifies. Disabling chunking entirely reproduces the same figure,
    which is what identifies the matmul shape rather than the chunk boundaries as the
    source.

    So the primary assertion is RANK PRESERVATION, not bit-equality: these scores feed
    EER, which reads only the ordering of scores and never their magnitudes (the same
    property that makes Spearman the right correlation in 9.8b.2). A numeric bound is
    kept as a secondary check, set at 1e-4 -- ~60x above the observed worst case, ~5
    orders below the score spread of ~21, and comfortably under the ~1e-6 EER artifact
    P3 already documents for every number in this project.
    """
    bona, spoof = load_pair(feat)
    df = pd.read_csv(config.SPLITS_DIR / "dev_2019.csv").head(n)
    frames = []
    for r in df.itertuples():
        a = np.load(features.CQT_DIR / f"{r.filename}.npy")
        frames.append(features.extract_cqt_dct(a).T)
    batched = score_batch(bona, spoof, frames)
    one_at_a_time = np.array([llr(bona, spoof, f) for f in frames])

    worst = float(np.abs(batched - one_at_a_time).max())
    inversions = int((np.argsort(np.argsort(batched))
                      != np.argsort(np.argsort(one_at_a_time))).sum())
    spread = float(np.ptp(one_at_a_time))
    print(f"batched vs per-file over {n} files:")
    print(f"  rank inversions   {inversions}        <- the one that matters")
    print(f"  worst |diff|      {worst:.3e}  (score spread {spread:.1f})")
    ok = inversions == 0 and worst < 1e-4
    print("  PASS" if ok else "  FAIL -- batching changes the ordering")
    return ok


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("OMP_NUM_THREADS", str(config.FEATURE_EXTRACTION_N_JOBS))

    p = argparse.ArgumentParser()
    p.add_argument("--feat", choices=["lfcc", "cqtdct"], default=None)
    p.add_argument("--source", choices=["dev", "2021"], default=None)
    p.add_argument("--n-jobs", type=int, default=config.FEATURE_EXTRACTION_N_JOBS)
    p.add_argument("--force", action="store_true")
    p.add_argument("--validate", action="store_true",
                   help="check batched == per-file scoring, then exit")
    a = p.parse_args()

    if a.validate:
        sys.exit(0 if validate() else 1)
    for feat in ([a.feat] if a.feat else ["lfcc", "cqtdct"]):
        for src in ([a.source] if a.source else ["dev", "2021"]):
            run(feat, src, a.n_jobs, a.force)
    print("\ndone. next: python -m src.fuse_inhouse")


if __name__ == "__main__":
    main()
