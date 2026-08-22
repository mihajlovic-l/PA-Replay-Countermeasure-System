"""Score-level fusion of our CQT-LCNN with the official ASVspoof baselines.

    python -m src.fuse                  # development only: speaker-disjoint CV on progress
    python -m src.fuse --confirm-eval   # refit on all progress, apply ONCE to eval
    python -m src.fuse --weights-only   # recover the fitted weights, WITHOUT touching eval
    python -m src.fuse --seed-sweep 20  # how fragile was the 1-SE selection? (progress only)

Protocol declared in PROJECT_PLAN.md 9.8b.1 before any eval number existed: method,
candidate set, selection rule (one-standard-error) and three predictions. The two extra
modes were declared in 9.8b.4, likewise before their numbers existed.

THE EVAL BOUNDARY. `--confirm-eval` was spent ONCE and is not repeated. Neither of the
other two modes SCORES eval: `--seed-sweep` never reads eval at all, and `--weights-only`
reads eval *features* solely to verify that the recovered coefficients reproduce the
fused scores already on disk -- no eval labels, no eval metric, no decision. It does not
even load the ASV protocol, so it cannot compute a t-DCF on anything. Both compute freely
on `progress`, which is what that partition is for (9.0).

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
    """z-norm on the FITTING data only, then combine. Returns (scores, info).

    `info` carries the fitted weights, or None for the equal-weight control, which has
    none. Persisting it closes 9.8b.4 A: 9.8b.1a names direct evidence of complementarity
    between the CQT-LCNN and cepstral-GMM front-ends as one of the three things fusion
    contributes WITHOUT damaging the thesis, and the coefficients are that evidence.

    They can serve as evidence only because every column was z-normed to unit variance on
    the fitting data, which puts the coefficients on a common scale and makes them
    COMPARABLE ACROSS SYSTEMS. On raw scores they would just reflect each system's
    arbitrary dynamic range.
    """
    mu, sd = Xfit.mean(0), Xfit.std(0)
    sd[sd == 0] = 1.0
    Zf, Za = (Xfit - mu) / sd, (Xapply - mu) / sd
    if not trained:
        return Za.mean(axis=1), None                # equal-weight control
    lr = LogisticRegression(max_iter=2000)
    lr.fit(Zf, yfit)
    info = {"coef": lr.coef_[0].copy(), "intercept": float(lr.intercept_[0]),
            "mu": mu, "sd": sd}
    return lr.decision_function(Za), info


def evaluate(y, s, asv) -> tuple[float, float]:
    e, _ = metrics.eer_from_labels(y, s)
    t, _ = tdcf.min_tdcf(y, s, asv)
    return e * 100, t


def cross_validate(d: pd.DataFrame, asv: dict, seed: int, n_folds: int = N_FOLDS,
                   progress_bar: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Speaker-disjoint K-fold CV on progress. Returns (summary, per-fold weights).

    K is a parameter rather than a constant because 9.8b.1 declared "speaker-disjoint
    K-fold" WITHOUT fixing K, so 5 was an implementation choice -- and K is what the
    1-SE band is most sensitive to, since SE ~ 1/sqrt(K). See `seed_sweep`.
    """
    g = d[d["partition"] == "progress"].reset_index(drop=True)
    y, spk = g["y"].to_numpy(), g["speaker_id"].to_numpy()
    uniq = np.unique(spk)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = np.array_split(uniq, n_folds)

    # A fold needs both classes for an EER to exist. 19 of the 67 progress speakers
    # carry no spoof trials, so at large K an all-bonafide fold is possible (~1e-5 per
    # fold at K=10). Vanishingly rare, but skipped explicitly and counted rather than
    # left to produce a silent nan.
    usable = [f for f in folds if y[np.isin(spk, f)].min() == 0
              and y[np.isin(spk, f)].max() == 1]
    n_skipped = len(folds) - len(usable)

    # (candidate, trained?) combinations; averaging a single system is itself, so skip
    configs = [(name, cols, tr) for name, cols in CANDIDATES.items()
               for tr in (True, False) if tr or len(cols) > 1]

    rows, wrows = [], []
    bar = tqdm(configs, desc="CV", unit="config", disable=not progress_bar)
    for name, cols, trained in bar:
        bar.set_postfix_str(f"{name} ({'trained' if trained else 'equal'})")
        eers, tdcfs = [], []
        X = g[cols].to_numpy()
        for fi, f in enumerate(usable):
            te = np.isin(spk, f)
            s, info = fuse_scores(X[~te], y[~te], X[te], trained)
            e, t = evaluate(y[te], s, asv)
            eers.append(e); tdcfs.append(t)
            if info is not None:
                denom = float(np.abs(info["coef"]).sum())
                for sys_name, c in zip(cols, info["coef"]):
                    wrows.append({"candidate": name, "n_folds": n_folds, "seed": seed,
                                  "fold": fi, "system": sys_name, "coef": float(c),
                                  "coef_norm": float(c) / denom,
                                  "intercept": info["intercept"]})
        rows.append({"candidate": name, "n_systems": len(cols),
                     "weights": "trained" if trained else "equal",
                     "cv_eer": float(np.mean(eers)),
                     "cv_eer_se": float(np.std(eers, ddof=1) / np.sqrt(len(usable))),
                     "cv_tdcf": float(np.mean(tdcfs)),
                     "n_folds_used": len(usable), "n_folds_skipped": n_skipped})
    return pd.DataFrame(rows), pd.DataFrame(wrows)


def select(cv: pd.DataFrame) -> str:
    """One-standard-error rule: among candidates within 1 SE of the best CV EER,
    take the one with the fewest systems. Stops a marginally-better but more complex
    combination winning on noise -- the failure mode a fusion sweep invites."""
    t = cv[cv["weights"] == "trained"].sort_values("cv_eer").reset_index(drop=True)
    best = t.iloc[0]
    within = t[t["cv_eer"] <= best["cv_eer"] + best["cv_eer_se"]]
    return within.sort_values(["n_systems", "cv_eer"]).iloc[0]["candidate"]


def seed_sweep(d: pd.DataFrame, asv: dict, n_seeds: int,
               k_values=(5, 10)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """How fragile was the 1-SE selection? (9.8b.4 B)

    `ours+2GMM` cleared the 1-SE threshold by 0.162 pp at one fold split -- thin enough
    that the parsimony arm of the rule may have been decided by the shuffle. This
    resamples the split and records what the DECLARED rule would have chosen each time.
    K is swept alongside the seed because 9.8b.1 fixed "K-fold" but not K.

    THIS IS NOT A RE-SELECTION. The decision was made and eval was spent at seed 42,
    K=5. If most splits would have chosen otherwise, that is reported as a limitation on
    the existing result; eval is not re-applied.

    Leave-one-speaker-out is deliberately not offered as a third K, for a structural
    reason rather than a budgetary one: a single speaker contributes ~1.3k progress
    trials, and 19 of the 67 speakers carry no spoof trials at all, so most single-speaker
    folds have no EER to compute.

    Ordered K=5 first with seed 42 leading, so the control (this split must reproduce
    `fusion_cv_progress.csv`) is the first thing computed and an early interrupt still
    checks it.
    """
    rows, wframes = [], []
    combos = [(k, config.RANDOM_SEED + i) for k in k_values for i in range(n_seeds)]
    for k, seed in tqdm(combos, desc="seed sweep", unit="split"):
        cv, w = cross_validate(d, asv, seed, n_folds=k, progress_bar=False)
        t = cv[cv["weights"] == "trained"].sort_values("cv_eer").reset_index(drop=True)
        best = t.iloc[0]
        thr = float(best["cv_eer"] + best["cv_eer_se"])
        chosen = select(cv)
        for _, r in t.iterrows():
            rows.append({"seed": seed, "n_folds": k, "candidate": r["candidate"],
                         "n_systems": int(r["n_systems"]), "cv_eer": r["cv_eer"],
                         "cv_eer_se": r["cv_eer_se"], "threshold": thr,
                         "headroom": thr - r["cv_eer"],
                         "in_1se_band": bool(r["cv_eer"] <= thr),
                         "is_best": bool(r["candidate"] == best["candidate"]),
                         "selected": bool(r["candidate"] == chosen)})
        wframes.append(w)
    return pd.DataFrame(rows), pd.concat(wframes, ignore_index=True)


def report_sweep(sw: pd.DataFrame, sww: pd.DataFrame) -> None:
    """Print the sweep, and check the declared control against the recorded run."""
    n_splits = sw[["seed", "n_folds"]].drop_duplicates().shape[0]
    print(f"\n=== selection robustness over {n_splits} fold splits "
          f"(progress only; eval untouched) ===")

    sel = sw[sw["selected"]]
    print(f"  {'candidate':12s} {'n':>2s} {'selected':>9s}   {'in 1-SE band':>13s}")
    for cand, cols in CANDIDATES.items():
        s = (sel["candidate"] == cand).sum()
        b = sw[(sw["candidate"] == cand) & sw["in_1se_band"]].shape[0]
        print(f"  {cand:12s} {len(cols):2d} {s:6d}/{n_splits:<3d} "
              f"{b:10d}/{n_splits:<3d}")

    print(f"\n  by K:  " + "   ".join(
        f"K={k}: " + ", ".join(
            f"{c} {(g['candidate'] == c).sum()}/{g['seed'].nunique()}"
            for c in sel[sel["n_folds"] == k]["candidate"].unique())
        for k, g in sel.groupby("n_folds")))

    h = sw[sw["candidate"] == "ours+2GMM"]["headroom"]
    print(f"\n  ours+2GMM headroom to the 1-SE threshold: median {h.median():+.3f} pp, "
          f"range [{h.min():+.3f}, {h.max():+.3f}], inside in {(h >= 0).mean():.0%}")

    # Weight stability (9.8b.4 A, prediction 3), measured over every fold of every split
    w = sww[sww["candidate"] == "ours+2GMM"]
    if len(w):
        print(f"\n  ours+2GMM weights over {w['fold'].count() // 3} folds "
              f"(coefficient on z-normed scores):")
        for s, g in w.groupby("system", sort=False):
            print(f"    {s:22s} {g['coef'].mean():+8.4f} ± {g['coef'].std():.4f}   "
                  f"share {100 * g['coef_norm'].mean():5.1f}%   "
                  f"spread ±{100 * g['coef'].std() / abs(g['coef'].mean()):.0f}% of mean")

    # CONTROL: the split that was actually used must reproduce the recorded run.
    ref = config.PHASE7_POSTHOC_DIR / "fusion_cv_progress.csv"
    got = sw[(sw["seed"] == config.RANDOM_SEED) & (sw["n_folds"] == N_FOLDS)]
    if ref.exists() and len(got):
        old = pd.read_csv(ref)
        old = old[old["weights"] == "trained"][["candidate", "cv_eer", "cv_eer_se"]]
        m = got.merge(old, on="candidate", suffixes=("", "_ref"))
        worst = float((m["cv_eer"] - m["cv_eer_ref"]).abs().max())
        ok = worst < 1e-9 and len(m) == len(old)
        print(f"\n  CONTROL: seed {config.RANDOM_SEED}, K={N_FOLDS} vs the recorded run, "
              f"worst |diff| {worst:.2e}  ->  {'PASS' if ok else 'FAIL'}")

    print("\n  This measures how fragile an ALREADY-MADE decision was. It is not a "
          "re-selection:\n  eval was spent once, at seed "
          f"{config.RANDOM_SEED}, K={N_FOLDS}, and is not re-applied.")


def weights_only(d: pd.DataFrame) -> None:
    """Recover the coefficients behind the reported eval number, without touching eval.

    The eval refit's mu, sd and coefficients all come from `progress` -- eval enters only
    at `decision_function` -- so the weights are recoverable by fitting on progress alone
    and `--confirm-eval` does not need re-running.

    CONTROL, declared in 9.8b.4: applying the recovered weights to the eval FEATURES
    already on disk must reproduce the stored fused scores. No labels are read, no metric
    is computed and no decision is made; it only proves these are the weights that
    actually produced the reported number rather than a plausible-looking refit.
    """
    summary_path = config.PHASE7_POSTHOC_DIR / "fusion_eval.json"
    if not summary_path.exists():
        sys.exit(f"no {summary_path.name} -- run --confirm-eval first")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    chosen = summary["selected"]        # the RECORDED decision, never a re-derived one
    cols = CANDIDATES[chosen]

    fit = d[d["partition"] == "progress"]
    X = fit[cols].to_numpy()
    _, info = fuse_scores(X, fit["y"].to_numpy(), X)
    denom = float(np.abs(info["coef"]).sum())

    print(f"\n=== fitted weights: {chosen}, fitted on {len(fit):,} progress trials ===")
    print("  (logistic-regression coefficients on z-normed scores -- unit variance per")
    print("   column, so these are directly comparable across systems)")
    print(f"  {'system':22s} {'coef':>8s} {'share':>7s}   {'z-mean':>9s} {'z-std':>8s}")
    for s, c, mu, sd in zip(cols, info["coef"], info["mu"], info["sd"]):
        print(f"  {s:22s} {c:8.4f} {100*c/denom:6.1f}%   {mu:9.4f} {sd:8.4f}")
    print(f"  {'intercept':22s} {info['intercept']:8.4f}")

    worst = float("nan")
    if config.PA2021_FUSION_SCORES.exists():
        ev = d[d["partition"] == "eval"]
        Za = (ev[cols].to_numpy() - info["mu"]) / info["sd"]
        recomputed = Za @ info["coef"] + info["intercept"]
        stored = pd.read_parquet(config.PA2021_FUSION_SCORES)
        m = pd.DataFrame({"filename": ev["filename"].to_numpy(),
                          "recomputed": recomputed}).merge(stored, on="filename")
        worst = float(np.abs(m["recomputed"] - m[chosen]).max())
        # stored scores were written as float32, so ~1e-6 of rounding is expected
        ok = worst < 1e-4 and len(m) == len(ev)
        print(f"\n  CONTROL: reproduces the {len(m):,} stored eval scores, "
              f"worst |diff| {worst:.2e}  ->  {'PASS' if ok else 'FAIL'}")
    else:
        print("\n  CONTROL SKIPPED: no stored eval scores to check against")

    summary["weights"] = {
        "note": "logistic-regression coefficients on z-normed scores, fitted on the "
                "87,048 labelled progress trials only; comparable across systems "
                "because every column was scaled to unit variance",
        "intercept": info["intercept"],
        "per_system": [{"system": s, "coef": float(c), "coef_share": float(c) / denom,
                        "z_mean": float(mu), "z_std": float(sd)}
                       for s, c, mu, sd in zip(cols, info["coef"],
                                               info["mu"], info["sd"])],
        "reproduces_stored_eval_scores": bool(worst < 1e-4),
        "max_abs_diff_vs_stored": worst,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  weights written to {summary_path}")

    # The evidence needed to READ those weights. A logistic-regression coefficient is a
    # PARTIAL one -- each partner's contribution conditional on the others -- so the two
    # GMMs' correlation WITH EACH OTHER governs how weight is split between them, and
    # 9.8b.2's table (each partner vs ours) cannot on its own explain the split.
    # Strength is measured on `progress`, the partition the weights are fitted on, which
    # is not the partition 9.8b.2 quoted.
    rho = fit[cols].corr(method="spearman")
    diag = []
    for a in cols:
        e, _ = metrics.eer_from_labels(fit["y"].to_numpy(), fit[a].to_numpy())
        diag.append({"system": a, "progress_eer": e * 100,
                     "coef": float(info["coef"][cols.index(a)]),
                     **{f"rho_{b}": float(rho.loc[a, b]) for b in cols}})
    dg = pd.DataFrame(diag)
    dg.to_csv(config.PHASE7_POSTHOC_DIR / "fusion_partner_diagnostics.csv", index=False)
    print(f"\n  partner diagnostics (progress; rho = Spearman):")
    print("   " + dg.round(3).to_string(index=False).replace("\n", "\n   "))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(v, "1")

    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--confirm-eval", action="store_true",
                      help="refit the selected candidate on all progress and apply ONCE to eval")
    mode.add_argument("--weights-only", action="store_true",
                      help="recover the fitted weights from progress; never scores eval")
    mode.add_argument("--seed-sweep", type=int, metavar="N", default=0,
                      help="robustness of the 1-SE selection over N fold splits x K in {5,10}")
    p.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = p.parse_args()

    print("loading scores ...", flush=True)
    d = load()

    # Deliberately BEFORE the ASV read: --weights-only never loads the protocol, so it
    # structurally cannot compute an EER or a t-DCF on anything.
    if args.weights_only:
        weights_only(d)
        return

    print("reading the ASV protocol (2.5M rows, ~30s) ...", flush=True)
    asv = tdcf.asv_error_rates(tdcf.ASV_OPERATING_PARTITION)
    prog = d[d["partition"] == "progress"]
    print(f"progress {len(prog):,} trials, {prog['speaker_id'].nunique()} speakers, "
          f"{N_FOLDS}-fold speaker-disjoint CV")

    if args.seed_sweep:
        sw, sww = seed_sweep(d, asv, args.seed_sweep)
        sw.to_csv(config.PHASE7_POSTHOC_DIR / "fusion_seed_sensitivity.csv", index=False)
        sww.to_csv(config.PHASE7_POSTHOC_DIR / "fusion_weights_sweep.csv", index=False)
        report_sweep(sw, sww)
        return

    cv, w = cross_validate(d, asv, args.seed)
    cv.to_csv(config.PHASE7_POSTHOC_DIR / "fusion_cv_progress.csv", index=False)
    w.to_csv(config.PHASE7_POSTHOC_DIR / "fusion_weights_cv.csv", index=False)
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
    s, _ = fuse_scores(fit[cols].to_numpy(), fit["y"].to_numpy(), ev[cols].to_numpy())
    e, t = evaluate(ev["y"].to_numpy(), s, asv)
    e0, t0 = evaluate(ev["y"].to_numpy(), ev[PRIMARY].to_numpy(), asv)

    out = pd.DataFrame({"filename": ev["filename"].to_numpy(), chosen: s.astype(np.float32)})
    out.to_parquet(config.PA2021_FUSION_SCORES, index=False)
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
