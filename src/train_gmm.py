"""9.8c pass 1: fit in-house cepstral GMMs as fusion partners.

    python -m src.train_gmm --feat lfcc      # ~1.5-2 h (decodes audio)
    python -m src.train_gmm --feat cqtdct    # ~0.5 h  (reads the Phase 4 CQT cache)
    python -m src.train_gmm                  # both, LFCC first

Two GMMs per feature -- one bonafide, one spoof -- scored later as the mean per-frame
log-likelihood ratio. Protocol, candidates and predictions were declared in
PROJECT_PLAN.md 9.8c.1 before any of this ran.

WHY IN-HOUSE PARTNERS AT ALL. Fusing with the official baselines (9.8b) cost four
things, and the deepest was that the result stopped being zero-shot: its weights were
fitted on 87,048 labelled 2021 trials. That cannot be repaired with the official
baselines, whose score files exist ONLY for 2021 -- there is nowhere else to fit. Our
own GMMs can be scored on 2019 dev, so the fusion weights can be fitted there and the
whole system never sees a 2021 label.

FRAME SAMPLING, AND WHY IT IS PER FILE. A GMM scores a file by the MEAN frame
log-likelihood, so every file counts once at test time. Pooling all frames and drawing
uniformly would instead weight the fit by duration -- and duration is not class-neutral
here: 6.10 measured bonafide at 323 frames against spoof at 274, and duration alone
scores 41.5% EER. So each file contributes an equal quota. The quota is set PER CLASS
(~48 frames/file bonafide, ~13 spoof at the ~1:9 split) because the two classes are fitted
independently, so there the criterion is fit quality and runtime, not equality: both land
near 1.8 M frames, ~3.5 k per component at k=512.

RESUMABLE at the granularity of (feature, class): sampled frames and fitted models are
each skipped if already on disk. The extraction pool is rebuilt every 20 chunks, which
is the Phase 7 fix for Windows error 1450 -- loky reuses one pool across Parallel()
calls and the workers accumulate handles until a large pipe write fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from . import config, features
from .gmm import DiagGMM

CHUNK = 2000            # files per Parallel() call
POOL_REBUILD_EVERY = 20  # chunks -- see error 1450 above


def frames_path(feat: str, label: str):
    return config.GMM_DIR / f"{feat}_{label}_frames.npy"


def state_path(feat: str, label: str):
    """Sidecar recording the valid prefix length and how far extraction got."""
    return config.GMM_DIR / f"{feat}_{label}_frames.json"


def model_path(feat: str, label: str):
    return config.GMM_DIR / f"{feat}_{label}_gmm.npz"


def collect_frames(df: pd.DataFrame, feat: str, label: str, quota: int,
                   n_jobs: int) -> tuple[np.ndarray, int]:
    """Extract and quota-sample every file of one class into a float32 matrix.

    Written straight into a memmapped .npy with a JSON sidecar recording how many
    chunks are done, so an interrupt costs at most the chunk in flight rather than
    the whole class. The first version of this saved only at the end and lost five
    minutes of completed work to a crash in the teardown -- exactly the failure mode
    CLAUDE.md's "resumable at fine granularity, write progress continuously" exists
    to prevent.

    The array stays at its full nominal size; `n` (the sidecar's frame count) is the
    valid prefix. Trimming would mean rewriting ~430 MB for the sake of the handful
    of frames short files fail to supply.
    """
    rows = [{"filename": r.filename, "filepath": r.filepath,
             "feat": feat, "quota": quota} for r in df.itertuples()]
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    n_dims = 3 * (config.LFCC_N_COEFF if feat == "lfcc" else config.CQTDCT_N_COEFF)
    fp, sp = frames_path(feat, label), state_path(feat, label)

    start, n, failures = 0, 0, []
    if fp.exists() and sp.exists():
        st = json.loads(sp.read_text())
        if st.get("quota") == quota and st.get("n_chunks") == len(chunks):
            start, n = st["chunks_done"], st["n_frames"]
            failures = [tuple(x) for x in st.get("failures", [])]
            if start:
                print(f"  {label}: resuming at chunk {start}/{len(chunks)} "
                      f"({n:,} frames already extracted)")
    if start:
        out = np.lib.format.open_memmap(fp, mode="r+")
    else:
        out = np.lib.format.open_memmap(fp, mode="w+", dtype=np.float32,
                                        shape=(len(rows) * quota, n_dims))

    bar = tqdm(total=len(rows), desc=f"{feat}/{label} extract", unit="file",
               initial=start * CHUNK)
    for ci in range(start, len(chunks)):
        if ci % POOL_REBUILD_EVERY == 0:
            features.recycle_workers()
        ch = chunks[ci]
        try:
            res = Parallel(n_jobs=n_jobs)(
                delayed(features.sample_frames_for_gmm)(r) for r in ch)
        except Exception as e:                      # pool failure -> recycle, retry
            tqdm.write(f"  pool failed on chunk {ci} ({type(e).__name__}: "
                       f"{str(e)[:80]}); recycling and retrying")
            features.recycle_workers()
            time.sleep(10)
            res = Parallel(n_jobs=n_jobs)(
                delayed(features.sample_frames_for_gmm)(r) for r in ch)
        for r in res:
            if r["error"] is not None:
                failures.append((r["filename"], r["error"]))
                continue
            f = r["frames"]
            out[n:n + len(f)] = f
            n += len(f)
        out.flush()
        sp.write_text(json.dumps({"quota": quota, "n_chunks": len(chunks),
                                  "chunks_done": ci + 1, "n_frames": n,
                                  "complete": ci + 1 == len(chunks),
                                  "failures": failures[:500]}), encoding="utf-8")
        bar.update(len(ch))
        bar.set_postfix_str(f"{n:,} frames, {len(failures)} failed")
    bar.close()
    features.recycle_workers()

    if failures:
        pd.DataFrame(failures, columns=["filename", "error"]).to_csv(
            config.GMM_DIR / f"{feat}_{label}_failures.csv", index=False)
        print(f"  {len(failures)} files failed -> {feat}_{label}_failures.csv")
    return out, n


def build(feat: str, n_jobs: int, force: bool, limit: int = 0,
          n_components: int = config.GMM_N_COMPONENTS) -> None:
    df = pd.read_csv(config.SPLITS_DIR / "train_2019.csv")
    print(f"\n=== {feat}: {len(df):,} training files "
          f"({n_components} components, diagonal) ===")

    for label in ("bonafide", "spoof"):
        sub = df[df["label"] == label]
        # Quota is derived from the FULL class size even under --limit, so a smoke
        # test exercises the real per-file quota instead of collapsing to "all frames"
        # and preallocating a 432 MB memmap for a hundred files.
        quota = max(config.GMM_MIN_FRAMES_PER_FILE,
                    round(config.GMM_TARGET_FRAMES_PER_CLASS / len(sub)))
        if limit:                       # smoke-test path; never used for a real fit
            sub = sub.head(limit)
        fp, mp = frames_path(feat, label), model_path(feat, label)

        if mp.exists() and not force:
            g = DiagGMM.load(mp)
            print(f"  {label}: model exists ({g.n_iter_} iters, "
                  f"logL/frame {g.lower_bound_:.5f}) -- skipping")
            continue

        sp = state_path(feat, label)
        done = (fp.exists() and sp.exists()
                and json.loads(sp.read_text()).get("complete") and not force)
        if done:
            n = json.loads(sp.read_text())["n_frames"]
            X = np.load(fp, mmap_mode="r")
            print(f"  {label}: frames cached, {n:,} x {X.shape[1]}")
        else:
            print(f"  {label}: {len(sub):,} files x quota {quota} "
                  f"-> ~{len(sub) * quota:,} frames")
            t0 = time.time()
            X, n = collect_frames(sub, feat, label, quota, n_jobs)
            print(f"  {label}: {n:,} frames in "
                  f"{(time.time() - t0) / 60:.1f} min -> {fp.name}")

        # CONTROL (9.8c.1c): every file contributes exactly its quota, unless it is
        # shorter than the quota or failed. A silent shortfall would mean the
        # duration-neutrality argument above does not actually hold.
        print(f"  {label}: quota realised {n / max(len(sub) * quota, 1):.4%} of nominal")

        # Materialise the valid prefix: the fit streams it in chunks anyway, but a
        # memmap would re-read from disk on every one of ~30 EM iterations.
        X = np.ascontiguousarray(X[:n])
        g = DiagGMM(n_components=n_components).fit(X, desc=f"{feat}/{label} EM")
        g.save(mp)
        print(f"  {label}: fitted in {g.n_iter_} iters, "
              f"logL/frame {g.lower_bound_:.5f} -> {mp.name}")
        del X


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    # The EM is one big BLAS matmul per chunk; the extraction workers are separate
    # processes. Threads are left alone here (unlike fuse/bootstrap, which pin to 1)
    # because the fit is single-process and genuinely benefits.
    os.environ.setdefault("OMP_NUM_THREADS", str(config.FEATURE_EXTRACTION_N_JOBS))

    p = argparse.ArgumentParser()
    p.add_argument("--feat", choices=["lfcc", "cqtdct"], default=None,
                   help="default: both, LFCC first (the better partner, 9.8c)")
    p.add_argument("--n-jobs", type=int, default=config.FEATURE_EXTRACTION_N_JOBS)
    p.add_argument("--force", action="store_true", help="refit even if cached")
    p.add_argument("--limit", type=int, default=0,
                   help="smoke test: use only N files per class")
    p.add_argument("--components", type=int, default=config.GMM_N_COMPONENTS,
                   help="smoke test: fewer mixture components")
    a = p.parse_args()

    for feat in ([a.feat] if a.feat else ["lfcc", "cqtdct"]):
        build(feat, a.n_jobs, a.force, a.limit, a.components)
    print("\ndone. next: python -m src.score_gmm")


if __name__ == "__main__":
    main()
