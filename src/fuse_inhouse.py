"""9.8c: fusion of our CQT-LCNN with our OWN cepstral GMMs.

    python -m src.fuse_inhouse                 # development only
    python -m src.fuse_inhouse --confirm-eval  # ONE eval application per arm

Protocol in PROJECT_PLAN.md 9.8c.1, normalisation amendment in 9.8c.2, both declared
before any number here existed.

WHAT MAKES THIS DIFFERENT FROM 9.8b. That fusion contained systems we did not build and
was fitted on 87,048 labelled 2021 trials, so it was neither ours nor zero-shot. Every
component here is ours, and the `dev` arm fits its weights on **2019 dev** -- so that
system never sees a 2021 label and the zero-shot claim survives fusion. The `progress`
arm is the 9.8b-comparable one; the gap between the arms prices what labelled
target-domain data is worth.

THREE DEVELOPMENT SURFACES, AND WHY THE THIRD EXISTS.

  1. dev-arm CV        -- speaker-disjoint 5-fold within 2019 dev (27 speakers)
  2. progress-arm CV   -- speaker-disjoint 5-fold within 2021 progress (67 speakers)
  3. dev -> progress   -- fit on ALL of dev, apply to progress

The third is not redundant. 9.8c.2's normalisation question CANNOT be answered by CV:
dev-arm CV fits and tests within dev, so both z-norm schemes draw on dev statistics and
the 21.7x dev-vs-2021 scale mismatch never appears. It only bites when a dev-fitted model
meets 2021 -- so it is measured by applying to `progress`, which is what 9.0 reserves
progress for. **Eval is untouched by all three**, and the eval spend stays at the two
applications 9.8c.1 declared.

TWO NORMALISATION SCHEMES (9.8c.2):
  fit-norm    z-norm statistics from the fitting partition -- as 9.8b.1 declared
  apply-norm  z-norm statistics from the partition being scored; reads NO labels, so
              the zero-shot property is untouched, at the cost of a transductive
              assumption (a batch of target data must exist at inference)
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
LFCC = "our-LFCC-GMM"
CQTDCT = "our-CQT-DCT-GMM"

# Fixed in 9.8c.1 before anything was scored -- NOT chosen from results.
CANDIDATES = {
    "ours":              [PRIMARY],
    "ours+LFCC":         [PRIMARY, LFCC],
    "ours+CQTDCT":       [PRIMARY, CQTDCT],
    "ours+2GMM-inhouse": [PRIMARY, LFCC, CQTDCT],
}
NORMS = ("fit-norm", "apply-norm")
N_FOLDS = 5


# --- data -------------------------------------------------------------------------

def load_dev() -> pd.DataFrame:
    """2019 dev: our LCNN's saved dev scores plus both in-house GMMs."""
    d = pd.read_csv(config.SPLITS_DIR / "dev_2019.csv",
                    usecols=["filename", "label", "speaker_id"])
    lc = pd.read_csv(config.PHASE6_DIR / PRIMARY / "dev_scores.csv",
                     usecols=["filename", "score"]).rename(columns={"score": PRIMARY})
    d = d.merge(lc, on="filename")
    for feat in ("lfcc", "cqtdct"):
        d = d.merge(pd.read_parquet(config.GMM_DIR / f"{feat}_dev_scores.parquet"),
                    on="filename")
    d["y"] = (d["label"] == "bonafide").astype(int)
    d["partition"] = "dev"
    return d


def load_2021() -> pd.DataFrame:
    man = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet",
                          columns=["filename", "label", "partition", "speaker_id"])
    ph = pd.read_parquet(config.PA2021_POSTHOC_SCORES).drop(columns=["partition"])
    d = man.merge(ph[["filename", PRIMARY]], on="filename")
    for feat in ("lfcc", "cqtdct"):
        d = d.merge(pd.read_parquet(config.GMM_DIR / f"{feat}_2021_scores.parquet"),
                    on="filename")
    d["y"] = (d["label"] == "bonafide").astype(int)
    return d


# --- fusion ------------------------------------------------------------------------

def fuse(Xfit, yfit, Xapply, norm: str, trained: bool = True):
    """Returns (scores, info). See 9.8c.2 for what the two norms mean and why.

    Under `apply-norm` each set is standardised by its OWN statistics, so a system whose
    score scale differs between fitting and application domains still arrives at the
    weights with unit variance. Under `fit-norm` both use the fitting set's, which is
    what 9.8b.1 declared and what silently switches off a partner whose scale shifted.
    """
    mu, sd = Xfit.mean(0), Xfit.std(0)
    sd[sd == 0] = 1.0
    Zf = (Xfit - mu) / sd
    if norm == "apply-norm":
        amu, asd = Xapply.mean(0), Xapply.std(0)
        asd[asd == 0] = 1.0
        Za = (Xapply - amu) / asd
    else:
        Za = (Xapply - mu) / sd
    if not trained:
        return Za.mean(axis=1), None
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Zf, yfit)
    return lr.decision_function(Za), {"coef": lr.coef_[0].copy(),
                                      "intercept": float(lr.intercept_[0])}


def evaluate(y, s, asv):
    # Cast at the source. The score columns are float32 on disk, and sklearn
    # PRESERVES float32 through LogisticRegression, so metrics computed off them come
    # back as numpy float32 -- which json.dumps refuses (unlike float64, which is a
    # float subclass and serialises silently). Caught only because the summary write
    # is the last statement in the run.
    e, _ = metrics.eer_from_labels(y, s)
    t, _ = tdcf.min_tdcf(y, s, asv)
    return float(e) * 100, float(t)


def cross_validate(g: pd.DataFrame, asv: dict, arm: str, seed: int) -> pd.DataFrame:
    """Speaker-disjoint K-fold within one fitting partition, both norms, both weightings."""
    y, spk = g["y"].to_numpy(), g["speaker_id"].to_numpy()
    uniq = np.unique(spk)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = [f for f in np.array_split(uniq, N_FOLDS) if len(f)]

    combos = [(n, c, nm, tr) for n, c in CANDIDATES.items() for nm in NORMS
              for tr in (True, False) if (tr or len(c) > 1)]
    rows = []
    for name, cols, norm, trained in tqdm(combos, desc=f"CV[{arm}]", unit="cfg"):
        X = g[cols].to_numpy()
        eers, tdcfs = [], []
        for f in folds:
            te = np.isin(spk, f)
            if y[te].min() == y[te].max():
                continue
            s, _ = fuse(X[~te], y[~te], X[te], norm, trained)
            e, t = evaluate(y[te], s, asv)
            eers.append(e); tdcfs.append(t)
        rows.append({"arm": arm, "candidate": name, "n_systems": len(cols),
                     "norm": norm, "weights": "trained" if trained else "equal",
                     "cv_eer": float(np.mean(eers)),
                     "cv_eer_se": float(np.std(eers, ddof=1) / np.sqrt(len(eers))),
                     "cv_tdcf": float(np.mean(tdcfs)), "n_folds": len(eers)})
    return pd.DataFrame(rows)


def select(cv: pd.DataFrame, norm: str) -> str:
    """One-standard-error rule, unchanged from 9.8b.1: lowest CV EER, then fewest
    systems among everything within 1 SE of it."""
    t = (cv[(cv["weights"] == "trained") & (cv["norm"] == norm)]
         .sort_values("cv_eer").reset_index(drop=True))
    best = t.iloc[0]
    within = t[t["cv_eer"] <= best["cv_eer"] + best["cv_eer_se"]]
    return within.sort_values(["n_systems", "cv_eer"]).iloc[0]["candidate"]


def transfer_check(dev: pd.DataFrame, prog: pd.DataFrame, asv: dict) -> pd.DataFrame:
    """Fit on ALL of dev, apply to progress -- where the 9.8c.2 question is decided.

    This is the only surface on which the normalisation schemes can differ for the dev
    arm, because it is the only one where a dev-fitted model meets 2021 data. Progress,
    not eval, so it costs nothing against 9.0.
    """
    rows = []
    for name, cols in CANDIDATES.items():
        for norm in NORMS:
            s, info = fuse(dev[cols].to_numpy(), dev["y"].to_numpy(),
                           prog[cols].to_numpy(), norm)
            e, t = evaluate(prog["y"].to_numpy(), s, asv)
            rows.append({"candidate": name, "n_systems": len(cols), "norm": norm,
                         "progress_eer": e, "progress_tdcf": t,
                         "coef": None if info is None else
                                 [round(float(x), 4) for x in info["coef"]]})
    return pd.DataFrame(rows)


# --- main --------------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")

    p = argparse.ArgumentParser()
    p.add_argument("--confirm-eval", action="store_true",
                   help="apply the selected system from EACH arm once to eval")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    a = p.parse_args()

    print("loading scores ...", flush=True)
    dev, d21 = load_dev(), load_2021()
    prog = d21[d21["partition"] == "progress"].reset_index(drop=True)
    print("reading the ASV protocol (2.5M rows, ~30s) ...", flush=True)
    asv = tdcf.asv_error_rates(tdcf.ASV_OPERATING_PARTITION)
    print(f"dev {len(dev):,} trials / {dev.speaker_id.nunique()} speakers | "
          f"progress {len(prog):,} / {prog.speaker_id.nunique()} speakers")

    # --- development ---------------------------------------------------------
    cv = pd.concat([cross_validate(dev, asv, "dev", a.seed),
                    cross_validate(prog, asv, "progress", a.seed)], ignore_index=True)
    cv.to_csv(config.PHASE7_POSTHOC_DIR / "inhouse_fusion_cv.csv", index=False)

    for arm in ("dev", "progress"):
        print(f"\n=== development CV, {arm} arm (eval untouched) ===")
        print(f"  {'candidate':19s} {'norm':11s} {'wt':8s} {'n':>2s} "
              f"{'CV EER':>8s} {'±SE':>6s} {'CV t-DCF':>9s}")
        for _, r in cv[cv.arm == arm].sort_values("cv_eer").iterrows():
            print(f"  {r.candidate:19s} {r.norm:11s} {r.weights:8s} {r.n_systems:2d} "
                  f"{r.cv_eer:8.3f} {r.cv_eer_se:6.3f} {r.cv_tdcf:9.4f}")

    # --- the 9.8c.2 question, answered on progress ----------------------------
    tc = transfer_check(dev, prog, asv)
    tc.to_csv(config.PHASE7_POSTHOC_DIR / "inhouse_fusion_transfer.csv", index=False)
    print("\n=== dev-fitted, applied to PROGRESS: does the norm scheme matter? ===")
    print(f"  {'candidate':19s} {'fit-norm':>9s} {'apply-norm':>11s} {'delta':>8s}")
    for name in CANDIDATES:
        f_ = tc[(tc.candidate == name) & (tc.norm == "fit-norm")].iloc[0]
        a_ = tc[(tc.candidate == name) & (tc.norm == "apply-norm")].iloc[0]
        print(f"  {name:19s} {f_.progress_eer:9.3f} {a_.progress_eer:11.3f} "
              f"{a_.progress_eer - f_.progress_eer:+8.3f}")

    chosen = {arm: {nm: select(cv[cv.arm == arm], nm) for nm in NORMS}
              for arm in ("dev", "progress")}
    print("\n=== 1-SE selection ===")
    for arm in chosen:
        for nm in NORMS:
            print(f"  {arm:9s} / {nm:11s} -> {chosen[arm][nm]}")

    summary = {"selected": chosen, "n_dev": int(len(dev)), "n_progress": int(len(prog))}
    (config.PHASE7_POSTHOC_DIR / "inhouse_fusion_dev.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    if not a.confirm_eval:
        print("\n  development only. --confirm-eval applies each arm once to eval.")
        return

    # --- the two declared eval applications -----------------------------------
    ev = d21[d21["partition"] == "eval"].reset_index(drop=True)
    yv = ev["y"].to_numpy()
    e0, t0 = evaluate(yv, ev[PRIMARY].to_numpy(), asv)
    print(f"\n=== EVAL ({len(ev):,} trials) ===")
    print(f"  {'single system':34s} EER {e0:6.3f}%  t-DCF {t0:.4f}")

    out = {"single_eval_eer": e0, "single_eval_tdcf": t0, "arms": {}}
    for arm, fitset in (("dev", dev), ("progress", prog)):
        # 9.8c.2's amended scheme for both arms. The fit-norm alternative is not
        # spent on eval: its cost is measured on progress by transfer_check, which
        # is what keeps this at the two eval applications 9.8c.1 declared.
        norm = "apply-norm"
        cand = chosen[arm][norm]
        cols = CANDIDATES[cand]
        s, info = fuse(fitset[cols].to_numpy(), fitset["y"].to_numpy(),
                       ev[cols].to_numpy(), norm)
        e, t = evaluate(yv, s, asv)
        pd.DataFrame({"filename": ev["filename"].to_numpy(),
                      f"inhouse_{arm}": s.astype(np.float32)}).to_parquet(
            config.PA2021_WORK_DIR / f"inhouse_fusion_{arm}_eval.parquet", index=False)
        out["arms"][arm] = {
            "candidate": cand, "systems": cols, "norm": norm,
            "zero_shot": arm == "dev", "n_fit_trials": int(len(fitset)),
            "eval_eer": e, "eval_tdcf": t,
            "coef": [float(x) for x in info["coef"]],
            "intercept": float(info["intercept"]),
        }
        print(f"  {arm + ' arm: ' + cand:34s} EER {e:6.3f}%  t-DCF {t:.4f}   "
              f"({e - e0:+.3f} pp)  zero-shot={arm == 'dev'}")

    z, pr = out["arms"]["dev"]["eval_eer"], out["arms"]["progress"]["eval_eer"]
    out["price_of_target_labels_pp"] = float(z - pr)
    print(f"\n  price of 87,048 labelled 2021 trials: {z - pr:+.3f} pp EER")
    (config.PHASE7_POSTHOC_DIR / "inhouse_fusion_eval.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
