"""Score-level fusion of our CQT-LCNN with the official ASVspoof baselines.

    python -m src.fuse                  # development only: speaker-disjoint CV on progress
    python -m src.fuse --confirm-eval   # refit on all progress, apply ONCE to eval

Protocol declared in PROJECT_PLAN.md 9.8b.1 before any eval number existed: method,
candidate set, selection rule (one-standard-error) and three predictions.

WHY FUSION, AND WHY THESE PARTNERS. Measured on progress, per-file, LFCC-GMM's errors
are nearly INDEPENDENT of ours (rescue 58.9% against an independence reference of
60.2%) while its rank correlation is the lowest of any candidate (rho 0.101) -- despite
it being the WORST system available at 39.8% EER. Our own second-best system is the
opposite: rho 0.797 and a rescue ratio of 0.44, i.e. it agrees with us AND fails where
we fail. Fusion gain is roughly decorrelation x partner strength, which is why 9.8's
attempt to fuse our own systems returned ~0.13 pp and why this one should not.

WHAT THIS IS NOT. The fused system is **not zero-shot** -- its weights are fitted on
87,048 labelled 2021 trials, whereas every single system in this project saw only 2019.
It also contains systems we did not build, so it cannot be compared against the official
baselines. Reported as a separate extension; the thesis's primary result stays the
single CQT-LCNN.

DESIGN NOTES.

*   **Logistic regression, not averaging.** The partners are 7-17 pp worse, so they need
    down-weighting. Equal-weight fusion is computed too, as a control on whether the
    trained weights are doing real work.
*   **Folds split by SPEAKER, not by trial.** P3 established that trials from one voice
    are not independent; a trial-level split would leak between fit and test and flatter
    every candidate.
*   **z-normalisation statistics come from the fitting data only** -- from the training
    folds during CV, and from `progress` when applying to eval. No eval information
    enters the model at any point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from . import config, metrics, tdcf

PRIMARY = "timepool_T150_aug"

# Fixed in PROJECT_PLAN 9.8b.1 before eval was touched -- NOT chosen from results.
CANDIDATES = {
    "ours":        [PRIMARY],
    "ours+2GMM":   [PRIMARY, "CQCC-GMM", "LFCC-GMM"],
    "ours+4base":  [PRIMARY, "CQCC-GMM", "LFCC-GMM", "LFCC-LCNN", "RawNet2"],
    "ours+all":    [PRIMARY, "flatten_T400_aug", "CQCC-GMM", "LFCC-GMM",
                    "LFCC-LCNN", "RawNet2"],
}
N_FOLDS = 5


def load(partitions=("progress", "eval")) -> pd.DataFrame:
    ph = pd.read_parquet(config.PA2021_POSTHOC_SCORES).drop(columns=["partition"])
    lc = pd.read_parquet(config.PA2021_LCNN_SCORES)
    man = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet",
                          columns=["filename", "label", "partition", "speaker_id"])
    d = ph.merge(lc, on="filename").merge(man, on="filename")
    for name, path in config.PA2021_BASELINE_SCORE_FILES.items():
        d = d.merge(pd.read_csv(path, sep=r"\s+", names=["filename", name]), on="filename")
    d = d[d["partition"].isin(partitions)].reset_index(drop=True)
    d["y"] = (d["label"] == "bonafide").astype(int)
    return d


def fuse_scores(Xfit, yfit, Xapply, trained=True):
    """z-norm on the FITTING data only, then combine."""
    mu, sd = Xfit.mean(0), Xfit.std(0)
    sd[sd == 0] = 1.0
    Zf, Za = (Xfit - mu) / sd, (Xapply - mu) / sd
    if not trained:
        return Za.mean(axis=1)                      # equal-weight control
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Zf, yfit)
    return lr.decision_function(Za)


def evaluate(y, s, asv) -> tuple[float, float]:
    e, _ = metrics.eer_from_labels(y, s)
    t, _ = tdcf.min_tdcf(y, s, asv)
    return e * 100, t


def cross_validate(d: pd.DataFrame, asv: dict, seed: int) -> pd.DataFrame:
    """Speaker-disjoint K-fold CV on progress."""
    g = d[d["partition"] == "progress"].reset_index(drop=True)
    y, spk = g["y"].to_numpy(), g["speaker_id"].to_numpy()
    uniq = np.unique(spk)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, N_FOLDS)

    # (candidate, trained?) combinations; averaging a single system is itself, so skip
    configs = [(name, cols, tr) for name, cols in CANDIDATES.items()
               for tr in (True, False) if tr or len(cols) > 1]

    rows = []
    bar = tqdm(configs, desc="CV", unit="config")
    for name, cols, trained in bar:
        bar.set_postfix_str(f"{name} ({'trained' if trained else 'equal'})")
        eers, tdcfs = [], []
        X = g[cols].to_numpy()
        for f in folds:
            te = np.isin(spk, f)
            s = fuse_scores(X[~te], y[~te], X[te], trained)
            e, t = evaluate(y[te], s, asv)
            eers.append(e); tdcfs.append(t)
        rows.append({"candidate": name, "n_systems": len(cols),
                     "weights": "trained" if trained else "equal",
                     "cv_eer": float(np.mean(eers)),
                     "cv_eer_se": float(np.std(eers, ddof=1) / np.sqrt(N_FOLDS)),
                     "cv_tdcf": float(np.mean(tdcfs))})
    return pd.DataFrame(rows)


def select(cv: pd.DataFrame) -> str:
    """One-standard-error rule: among candidates within 1 SE of the best CV EER,
    take the one with the fewest systems. Stops a marginally-better but more complex
    combination winning on noise -- the failure mode a fusion sweep invites."""
    t = cv[cv["weights"] == "trained"].sort_values("cv_eer").reset_index(drop=True)
    best = t.iloc[0]
    within = t[t["cv_eer"] <= best["cv_eer"] + best["cv_eer_se"]]
    return within.sort_values(["n_systems", "cv_eer"]).iloc[0]["candidate"]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")

    p = argparse.ArgumentParser()
    p.add_argument("--confirm-eval", action="store_true",
                   help="refit the selected candidate on all progress and apply ONCE to eval")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = p.parse_args()

    print("loading scores ...", flush=True)
    d = load()
    print("reading the ASV protocol (2.5M rows, ~30s) ...", flush=True)
    asv = tdcf.asv_error_rates(tdcf.ASV_OPERATING_PARTITION)
    prog = d[d["partition"] == "progress"]
    print(f"progress {len(prog):,} trials, {prog['speaker_id'].nunique()} speakers, "
          f"{N_FOLDS}-fold speaker-disjoint CV")

    cv = cross_validate(d, asv, args.seed)
    cv.to_csv(config.PHASE7_POSTHOC_DIR / "fusion_cv_progress.csv", index=False)
    print("\n=== development: cross-validated on progress (eval untouched) ===")
    print(f"  {'candidate':12s} {'weights':8s} {'n':>2s} {'CV EER':>8s} {'±SE':>6s} {'CV t-DCF':>9s}")
    for _, r in cv.sort_values(["cv_eer"]).iterrows():
        print(f"  {r.candidate:12s} {r.weights:8s} {r.n_systems:2d} "
              f"{r.cv_eer:8.3f} {r.cv_eer_se:6.3f} {r.cv_tdcf:9.4f}")

    chosen = select(cv)
    best = cv[(cv.candidate == chosen) & (cv.weights == "trained")].iloc[0]
    print(f"\n  1-SE rule selects: **{chosen}** "
          f"({best.n_systems} systems, CV EER {best.cv_eer:.3f} ± {best.cv_eer_se:.3f})")

    if not args.confirm_eval:
        print("\n  development only. Re-run with --confirm-eval to apply it once to eval.")
        return

    # --- the single eval application ---------------------------------------
    cols = CANDIDATES[chosen]
    fit = d[d["partition"] == "progress"]
    ev = d[d["partition"] == "eval"]
    s = fuse_scores(fit[cols].to_numpy(), fit["y"].to_numpy(), ev[cols].to_numpy())
    e, t = evaluate(ev["y"].to_numpy(), s, asv)
    e0, t0 = evaluate(ev["y"].to_numpy(), ev[PRIMARY].to_numpy(), asv)

    out = pd.DataFrame({"filename": ev["filename"].to_numpy(), chosen: s.astype(np.float32)})
    out.to_parquet(config.PA2021_WORK_DIR / "fusion_eval_scores.parquet", index=False)
    out.to_csv(config.PHASE7_POSTHOC_SCORES_DIR / f"fusion_{chosen}.score.txt",
               sep=" ", header=False, index=False, float_format="%.6f")

    print(f"\n=== EVAL (single application, {len(ev):,} trials) ===")
    print(f"  {PRIMARY:22s} EER {e0:6.3f}%   t-DCF {t0:.4f}")
    print(f"  fusion {chosen:15s} EER {e:6.3f}%   t-DCF {t:.4f}")
    print(f"  {'delta':22s}     {e-e0:+6.3f} pp        {t-t0:+.4f}")
    summary = {"selected": chosen, "systems": cols, "n_fit_trials": int(len(fit)),
               "cv_eer": float(best.cv_eer), "cv_eer_se": float(best.cv_eer_se),
               "eval_eer": e, "eval_tdcf": t,
               "single_eval_eer": e0, "single_eval_tdcf": t0,
               "zero_shot": False,
               "note": "weights fitted on labelled 2021 progress; contains systems we "
                       "did not build; not comparable against the official baselines"}
    (config.PHASE7_POSTHOC_DIR / "fusion_eval.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
