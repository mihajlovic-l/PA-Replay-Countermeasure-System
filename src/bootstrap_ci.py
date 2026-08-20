"""Bootstrap confidence intervals for EER and min t-DCF on 2021 PA eval.

    python -m src.bootstrap_ci              # the real run (B=2000, ~5 min)
    python -m src.bootstrap_ci --quick      # B=200, for checking wiring
    python -m src.bootstrap_ci --validate   # fast path vs naive, then exit

WHY. Every claim in this project is a DIFFERENCE -- augmentation vs clean, T250 vs
T400, ours vs the best official baseline -- and none of them carried an error bar. This
module supplies them, and it changes at least one verdict (see 9.6 in PROJECT_PLAN).

TWO INTERVALS ARE REPORTED, AND THE GAP BETWEEN THEM IS ITSELF A RESULT.

*   **Speaker-clustered (HEADLINE).** Trials are not independent: 721,332 eval trials
    come from **67 speakers**, and every one of a speaker's ~14,472 trials shares that
    speaker's voice. The unit that actually repeats is the *speaker*, so resampling
    draws 67 speakers with replacement and takes each drawn speaker's trials as an
    indivisible block. The experiment being imagined is "recruit 67 more speakers",
    not "collect 721,332 more trials".

*   **Trial-level (CONTRAST).** The conventional thing to report, and far too narrow --
    it treats 14,472 trials from one voice as 14,472 independent observations. Shown
    beside the honest interval to make the difference visible; the design effect is
    roughly 1 + (m-1)*ICC with m ~ 10,766, so even an intra-speaker correlation of 0.001
    inflates the variance ~12x.

Both are STRATIFIED by class: bonafide are resampled from bonafide, spoof from spoof.
The design is perfectly regular -- all 67 speakers have exactly 1,404 bonafide trials,
and all 48 spoof-bearing speakers exactly 13,068 spoof trials -- so class-stratified
speaker resampling reproduces 94,068 / 627,264 EXACTLY in every replicate.

PAIRED DIFFERENCES. Comparisons use the SAME resample for both systems, so the shared
"was this a hard draw?" component cancels. This is not a refinement: for prediction 1
the marginal intervals OVERLAP by 1.46 pp (suggesting no effect) while the paired
interval excludes zero decisively. Comparing marginal CIs is systematically
conservative and will discard real effects -- the CI of the difference is the test.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from . import config, metrics, tdcf

# Comparisons that carry the thesis's claims. Fixed here rather than chosen from
# results, so the set is not selected on whichever differences came out largest.
COMPARISONS = [
    ("flatten_T400_aug", "flatten_T400", "pred 2b: aggressive waveform augmentation"),
    ("flatten_T400_aug1", "flatten_T400", "pred 2a: mild waveform augmentation"),
    ("baseline_T250", "T400", "pred 1: shorter T transfers better"),
    ("cmvn_T400", "T400", "pred 3: CMVN transfers better"),
    ("flatten_T400_aug", "CQCC-GMM", "headline: best vs best official baseline"),
    ("flatten_T400_aug", "MFCC-SVM", "central claim: CQT-LCNN vs MFCC-SVM"),
    ("flatten_T400", "LFCC-LCNN", "front-end: CQT vs LFCC (both unaugmented)"),
    ("flatten_T400", "T400", "head: flatten vs timepool at T=400"),
    ("T150", "T400", "T axis: 150 vs 400"),
]


# --- data ----------------------------------------------------------------------

def load_eval() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    d = pd.read_parquet(config.PA2021_LCNN_SCORES)
    if config.PA2021_CLASSICAL_SCORES.exists():
        d = d.merge(pd.read_parquet(config.PA2021_CLASSICAL_SCORES), on="filename", how="left")
    man = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet",
                          columns=["filename", "speaker_id", "label", "partition"])
    d = d.merge(man, on="filename")
    d = d[d["partition"] == config.PA2021_REPORTED_PARTITION].reset_index(drop=True)

    systems = [c for c in list(config.PHASE7_LCNN_SYSTEMS) +
               list(config.PHASE7_CLASSICAL_SYSTEMS) if c in d.columns]
    for name, path in config.PA2021_BASELINE_SCORE_FILES.items():
        if path.exists():
            s = pd.read_csv(path, sep=r"\s+", names=["filename", name])
            d = d.merge(s, on="filename", how="left")
            systems.append(name)

    y = (d["label"] == "bonafide").to_numpy().astype(np.int8)
    spk = d["speaker_id"].to_numpy()
    return d, y, spk, systems


# --- weighted curves (the fast path) -------------------------------------------

def prepare(scores: np.ndarray, y: np.ndarray | None = None) -> dict:
    """Sort once per system. Every replicate is then a reweighting of this order.

    Thresholds are evaluated only at UNIQUE score boundaries, which is what
    sklearn's roc_curve does. That matters here: MFCC-RF has just 236 distinct
    values across 721,332 trials (a 300-tree vote), so treating tied scores as
    separate thresholds would silently produce a different curve.
    """
    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    out = {"order": order, "cut": last}
    if y is not None:                     # y[order] never changes across
        out["y_ord"] = y[order]           # replicates -- gather it once
    return out



def curves(prep: dict, y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted (Pmiss_cm, Pfa_cm) at every unique-score threshold."""
    wo = w[prep["order"]]
    yo = prep["y_ord"]
    wy = wo * yo
    wn = wo - wy
    cb, cs = np.cumsum(wy)[prep["cut"]], np.cumsum(wn)[prep["cut"]]
    tot_b, tot_s = cb[-1], cs[-1]
    return 1.0 - cb / tot_b, cs / tot_s          # Pmiss_cm, Pfa_cm


def eer_from_curves(pmiss: np.ndarray, pfa: np.ndarray) -> float:
    i = int(np.nanargmin(np.abs(pmiss - pfa)))
    return float((pmiss[i] + pfa[i]) / 2)


def tdcf_from_curves(pmiss: np.ndarray, pfa: np.ndarray, coef) -> float:
    C0, C1, C2 = coef
    return float(np.nanmin((C0 + C1 * pmiss + C2 * pfa) / (C0 + min(C1, C2))))


# --- resampling ----------------------------------------------------------------

def make_weighter(y: np.ndarray, spk: np.ndarray, clustered: bool):
    """Return a function rng -> integer weight per trial.

    Clustered: draw speakers with replacement, separately within each class, and give
    every trial of a drawn speaker that speaker's draw count. Trial-level: draw trials
    directly, again within class.
    """
    n = len(y)
    bmask, smask = y == 1, y == 0
    if clustered:
        bcode, buniq = pd.factorize(spk[bmask])
        scode, suniq = pd.factorize(spk[smask])
        nb, ns = len(buniq), len(suniq)

        def weights(rng):
            w = np.empty(n, dtype=np.int32)
            w[bmask] = np.bincount(rng.integers(0, nb, nb), minlength=nb)[bcode]
            w[smask] = np.bincount(rng.integers(0, ns, ns), minlength=ns)[scode]
            return w
    else:
        nb, ns = int(bmask.sum()), int(smask.sum())

        def weights(rng):
            w = np.empty(n, dtype=np.int32)
            w[bmask] = np.bincount(rng.integers(0, nb, nb), minlength=nb)
            w[smask] = np.bincount(rng.integers(0, ns, ns), minlength=ns)
            return w
    return weights


# --- validation ----------------------------------------------------------------

def validate(d, y, spk, systems, n_rep: int = 25) -> bool:
    """Fast weighted path vs naive resample-and-recompute, on real replicates.

    The fast path is only an optimisation; if it does not reproduce the naive answer
    exactly it is worthless, so this is checked rather than assumed.

    The reference uses `drop_intermediate=False`. sklearn's DEFAULT is True, which
    prunes threshold points it considers non-informative for plotting -- harmless for a
    ROC picture, but it can move the EER crossing by ~1e-6 because the exact crossing
    point may sit on a pruned threshold. Validating against the thinned curve would
    therefore compare our exact computation against a slightly lossy one and report a
    spurious failure. Against the complete threshold list the two agree to 0.0.

    (Consequence worth knowing: every EER in this project comes from
    `metrics.eer_from_labels`, i.e. the thinned path, so all reported EERs carry a
    ~1e-6 == 0.0001 pp thinning artifact. Utterly negligible against CI widths of
    2-5 pp, but it is the reason this function does not demand bit-equality with
    metrics.py.)
    """
    from sklearn.metrics import roc_curve

    def eer_exact(yy, ss):
        fpr, tpr, _ = roc_curve(yy, ss, drop_intermediate=False)
        fnr = 1.0 - tpr
        i = int(np.nanargmin(np.abs(fnr - fpr)))
        return (fnr[i] + fpr[i]) / 2

    print("validating fast weighted path against naive resampling...")
    rng = np.random.default_rng(0)
    weights = make_weighter(y, spk, clustered=True)
    worst = 0.0
    for tag in systems[:4]:
        sc = d[tag].to_numpy()
        prep = prepare(sc, y)
        for _ in range(n_rep):
            w = weights(rng)
            fast = eer_from_curves(*curves(prep, y, w))
            idx = np.repeat(np.arange(len(w)), w)      # materialise the same resample
            worst = max(worst, abs(fast - eer_exact(y[idx], sc[idx])))
    print(f"  worst |fast - naive| over {n_rep * 4} replicates: {worst:.3e}")
    ok = worst < 1e-12
    print("  PASS — identical" if ok else "  FAIL — fast path is wrong, do not use")
    return ok


# --- main ----------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")

    p = argparse.ArgumentParser()
    p.add_argument("-B", "--replicates", type=int, default=2000)
    p.add_argument("--quick", action="store_true", help="B=200")
    p.add_argument("--validate", action="store_true", help="check the fast path and exit")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = p.parse_args()
    B = 200 if args.quick else args.replicates

    d, y, spk, systems = load_eval()
    print(f"{len(d):,} eval trials, {len(np.unique(spk))} speakers, {len(systems)} systems")

    if args.validate:
        sys.exit(0 if validate(d, y, spk, systems) else 1)
    if not validate(d, y, spk, systems, n_rep=10):
        sys.exit(1)

    asv = tdcf.asv_error_rates(tdcf.ASV_OPERATING_PARTITION)
    coef = tdcf.tdcf_coefficients(asv)
    preps = {t: prepare(d[t].to_numpy(), y) for t in systems}

    rows, diffs = [], []
    for scheme, clustered in (("speaker-clustered", True), ("trial-level", False)):
        weights = make_weighter(y, spk, clustered)
        rng = np.random.default_rng(args.seed)
        eer = {t: np.empty(B) for t in systems}
        tdc = {t: np.empty(B) for t in systems}

        t0 = time.time()
        for b in range(B):
            w = weights(rng)                       # ONE resample, shared by all systems
            for t in systems:
                pm, pf = curves(preps[t], y, w)
                eer[t][b] = eer_from_curves(pm, pf) * 100
                tdc[t][b] = tdcf_from_curves(pm, pf, coef)
        print(f"{scheme}: {B} replicates x {len(systems)} systems in "
              f"{(time.time()-t0)/60:.1f} min")

        q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for t in systems:
            pm, pf = curves(preps[t], y, np.ones(len(y), dtype=np.int32))
            lo, hi = q(eer[t]); tlo, thi = q(tdc[t])
            rows.append({"scheme": scheme, "system": t,
                         "eer": eer_from_curves(pm, pf) * 100, "eer_lo": lo, "eer_hi": hi,
                         "eer_ci_width": hi - lo,
                         "min_tdcf": tdcf_from_curves(pm, pf, coef),
                         "tdcf_lo": tlo, "tdcf_hi": thi, "tdcf_ci_width": thi - tlo})
        for a_, b_, label in COMPARISONS:
            if a_ not in systems or b_ not in systems:
                continue
            de, dt = eer[a_] - eer[b_], tdc[a_] - tdc[b_]
            lo, hi = q(de); tlo, thi = q(dt)
            diffs.append({
                "scheme": scheme, "comparison": label, "system": a_, "control": b_,
                "eer_diff": float(de.mean()), "eer_lo": lo, "eer_hi": hi,
                "eer_excludes_zero": bool(hi < 0 or lo > 0),
                "tdcf_diff": float(dt.mean()), "tdcf_lo": tlo, "tdcf_hi": thi,
                "tdcf_excludes_zero": bool(thi < 0 or tlo > 0),
                "corr_eer": float(np.corrcoef(eer[a_], eer[b_])[0, 1]),
                "sign_consistency": float(max((de < 0).mean(), (de > 0).mean())),
            })

    t = pd.DataFrame(rows); c = pd.DataFrame(diffs)
    t.to_csv(config.PHASE7_DIR / "bootstrap_ci_systems.csv", index=False)
    c.to_csv(config.PHASE7_DIR / "bootstrap_ci_comparisons.csv", index=False)

    for scheme in ("speaker-clustered", "trial-level"):
        s = t[t.scheme == scheme].sort_values("eer")
        print(f"\n=== marginal 95% CI — {scheme}"
              f"{'  [HEADLINE]' if scheme.startswith('speaker') else '  [contrast]'} ===")
        for _, r in s.iterrows():
            print(f"  {r.system:18s} EER {r.eer:6.2f} [{r.eer_lo:6.2f},{r.eer_hi:6.2f}] "
                  f"±{r.eer_ci_width/2:4.2f}   t-DCF {r.min_tdcf:.4f} "
                  f"[{r.tdcf_lo:.4f},{r.tdcf_hi:.4f}]")

    print("\n=== paired difference CIs (speaker-clustered) ===")
    for _, r in c[c.scheme == "speaker-clustered"].iterrows():
        v = "SIGNIFICANT" if r["eer_excludes_zero"] else "not distinguishable"
        print(f"  {r.comparison}\n    {r.system} - {r.control}: "
              f"{r.eer_diff:+6.2f} pp [{r.eer_lo:+6.2f},{r.eer_hi:+6.2f}]  "
              f"corr {r['corr_eer']:.3f}  -> {v}")

    widths = t.groupby("scheme")["eer_ci_width"].mean()
    ratio = widths["speaker-clustered"] / widths["trial-level"]
    print(f"\nmean EER CI width: speaker-clustered {widths['speaker-clustered']:.3f} pp "
          f"vs trial-level {widths['trial-level']:.3f} pp — **{ratio:.1f}x wider**")
    (config.PHASE7_DIR / "bootstrap_ci_summary.json").write_text(json.dumps({
        "replicates": B, "seed": args.seed, "n_speakers": int(len(np.unique(spk))),
        "mean_ci_width_speaker": float(widths["speaker-clustered"]),
        "mean_ci_width_trial": float(widths["trial-level"]),
        "width_ratio": float(ratio),
    }, indent=2), encoding="utf-8")
    print(f"written to {config.PHASE7_DIR}")


if __name__ == "__main__":
    main()
