"""Phase 5: classical MFCC baseline -- RBF SVM + Random Forest.

This is the deliberately-weak comparison point the thesis's central argument rests
on (see PROJECT_PLAN.md's feature-combination table), so it is tuned properly
rather than strawmanned: an under-trained baseline would undermine the eventual
"CQT finds the replay fingerprint that MFCC misses" claim.

Structure: a FULL FACTORIAL sweep of subsample size x C x gamma, plus a Random
Forest fitted at every one of those same sizes and on the full train split.
Tuning is done on the speaker-disjoint 2019 dev split, never by cross-validation
on train -- CV folds would share speakers, making them a strictly worse
generalisation estimate than the honest dev split already built in Phase 2.

The sweep replaces the earlier "learning curve at fixed hyperparameters, then grid
search at one chosen size" design. It costs considerably more compute but removes
the caveat that the chosen size depended on one arbitrary hyperparameter setting,
and the resulting table shows how the optimal C/gamma themselves shift with data
quantity.

Everything is resumable: the sweep CSV is written after every single fit and
completed (n_train, C, gamma) points are skipped on re-run, and RF checkpoints are
keyed by size so they are never silently reused under the wrong label. Pass
--force to recompute from scratch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from . import config, metrics

MFCC_PARQUET = config.FEATURES_DIR / "mfcc" / "pooled_mfcc.parquet"
MFCC_COLS = [f"mfcc_{i}" for i in range(120)]

SWEEP_CSV = config.RESULTS_DIR / "phase5_svm_sweep.csv"
LEGACY_GRID_CSV = config.RESULTS_DIR / "phase5_grid_search.csv"
RF_CURVE_CSV = config.RESULTS_DIR / "phase5_rf_curve.csv"
SWEEP_PNG = config.RESULTS_DIR / "phase5_svm_sweep.png"
RF_IMPORTANCE_PNG = config.RESULTS_DIR / "phase5_rf_feature_importance.png"
SUMMARY_MD = config.RESULTS_DIR / "phase5_summary.md"
SUMMARY_JSON = config.RESULTS_DIR / "phase5_summary.json"

SVM_MODEL = config.MODELS_DIR / "svm_mfcc.joblib"
RF_FULL_MODEL = config.MODELS_DIR / "rf_mfcc_full.joblib"


def rf_sub_model_path(n_train: int):
    """RF checkpoint keyed by n_train, so a size change can never silently reuse
    a model trained at a different size under the new label."""
    return config.MODELS_DIR / f"rf_mfcc_sub_{n_train}.joblib"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_xy():
    df = pd.read_parquet(MFCC_PARQUET)
    out = {}
    for subset in ("train", "dev"):
        part = df[df["subset"] == subset]
        X = part[MFCC_COLS].to_numpy(dtype=np.float64)
        # 1 = bonafide (positive class), 0 = spoof -- see metrics.py's convention note.
        y = (part["label"] == "bonafide").to_numpy().astype(int)
        out[subset] = (X, y, part["filename"].to_numpy())
    return out


def subsample_indices(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    """One stratified draw of size n from the full train split.

    Independent per size (not nested) -- see the note in config.SVM_SWEEP_SIZES
    for why that matters for resumability.
    """
    if n >= len(y):
        return np.arange(len(y))
    idx, _ = train_test_split(np.arange(len(y)), train_size=n, stratify=y, random_state=seed)
    return idx


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_in_chunks(model, X, kind: str, desc: str, chunk: int = 5000) -> np.ndarray:
    """Chunked purely for a moving progress bar instead of one opaque multi-minute
    call. SVM scoring is O(n_support_vectors x n_test), which is the slow part."""
    parts = []
    for i in tqdm(range(0, len(X), chunk), desc=desc, leave=False, unit="chunk"):
        block = X[i:i + chunk]
        if kind == "decision":
            parts.append(model.decision_function(block))
        else:
            # predict_proba col 1 = P(bonafide); higher = more bonafide, matching
            # the project-wide score convention in metrics.py.
            parts.append(model.predict_proba(block)[:, 1])
    return np.concatenate(parts)


def build_svm(C: float, gamma) -> Pipeline:
    return Pipeline([
        # All 120 dimensions -- see the correction note in config.py for why an
        # earlier version wrongly dropped the mean(delta) block here.
        ("scale", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=C, gamma=gamma,
                    class_weight="balanced", cache_size=1000)),
    ])


def fit_and_eval_svm(C, gamma, Xtr, ytr, Xdv, ydv, desc: str) -> dict:
    t0 = time.time()
    pipe = build_svm(C, gamma).fit(Xtr, ytr)
    fit_s = time.time() - t0

    t0 = time.time()
    scores = score_in_chunks(pipe, Xdv, "decision", f"{desc} dev")
    score_s = time.time() - t0

    eer, threshold = metrics.eer_from_labels(ydv, scores)
    return {
        "pipeline": pipe, "scores": scores, "eer": eer, "threshold": threshold,
        "fit_seconds": fit_s, "score_seconds": score_s,
        "n_support": int(pipe.named_steps["svc"].n_support_.sum()),
    }


# ---------------------------------------------------------------------------
# Step 1 -- full factorial SVM sweep
# ---------------------------------------------------------------------------

def migrate_legacy_grid():
    """The previous design's grid CSV holds 16 valid points at n_train=50,000
    (all 120 features, same seed, and -- verified -- the same 50k subsample that
    independent drawing reproduces). Fold them in rather than recomputing ~2.4h."""
    if SWEEP_CSV.exists() or not LEGACY_GRID_CSV.exists():
        return
    old = pd.read_csv(LEGACY_GRID_CSV)
    if not len(old):
        return
    old["n_train"] = 50_000
    old.to_csv(SWEEP_CSV, index=False)
    print(f"  migrated {len(old)} completed points from {LEGACY_GRID_CSV.name} (n_train=50,000)")


def resolved_sizes(n_total: int) -> list[int]:
    """Sweep sizes clamped to the actual train split, plus the full split itself
    when configured. Clamping means an over-large configured size collapses onto
    the full split rather than being recorded under a size that does not exist."""
    sizes = [min(n, n_total) for n in config.SVM_SWEEP_SIZES]
    if config.SVM_SWEEP_INCLUDE_FULL:
        sizes.append(n_total)
    return sorted(set(sizes))


def sweep_work_items(sizes: list[int], done_keys: set) -> list[tuple[int, float, object]]:
    """Ordered cheapest-first: sizes ascending, and within each size the very slow
    gamma=0.1 points last -- so partial results are informative early on, and an
    early interrupt still leaves every useful (C, gamma) pair computed."""
    items = []
    for n in sorted(sizes):
        # The `and` short-circuits so float("scale") is never evaluated.
        gammas = sorted(config.SVM_GAMMA_GRID, key=lambda g: (g != "scale" and float(g) >= 0.1, str(g)))
        for gamma in gammas:
            for C in config.SVM_C_GRID:
                if (n, C, str(gamma)) not in done_keys:
                    items.append((n, C, gamma))
    return items


def run_svm_sweep(data, force: bool) -> pd.DataFrame:
    Xtr, ytr, _ = data["train"]
    Xdv, ydv, _ = data["dev"]

    if force and SWEEP_CSV.exists():
        SWEEP_CSV.unlink()
    migrate_legacy_grid()

    done = pd.read_csv(SWEEP_CSV) if SWEEP_CSV.exists() else pd.DataFrame()
    if len(done):
        # Normalise gamma to str on read. Rows loaded from CSV come back as
        # strings while freshly-computed rows hold Python floats, and mixing the
        # two silently splits groupby(["C","gamma"]) into duplicate groups --
        # which is exactly what scattered the migrated 50k points off their
        # curves in the first sweep's plot.
        done["gamma"] = done["gamma"].astype(str)
    done_keys = {(int(r["n_train"]), float(r["C"]), str(r["gamma"])) for _, r in done.iterrows()} if len(done) else set()
    rows = done.to_dict("records") if len(done) else []

    sizes = resolved_sizes(len(ytr))
    todo = sweep_work_items(sizes, done_keys)
    total = len(sizes) * len(config.SVM_C_GRID) * len(config.SVM_GAMMA_GRID)
    print(f"\n[1/2] SVM sweep: {len(sizes)} sizes x {len(config.SVM_C_GRID)} C x "
          f"{len(config.SVM_GAMMA_GRID)} gamma = {total} points "
          f"({len(done_keys)} already done, {len(todo)} to run)")
    print("      No per-fit progress bar is possible -- SVC.fit is one opaque call")
    print("      into libsvm. Per-point timings below keep the ETA meaningful.")

    idx_cache: dict[int, np.ndarray] = {}
    for n, C, gamma in tqdm(todo, desc="svm sweep", unit="pt"):
        if n not in idx_cache:
            idx_cache[n] = subsample_indices(ytr, n, config.RANDOM_SEED)
        idx = idx_cache[n]

        res = fit_and_eval_svm(C, gamma, Xtr[idx], ytr[idx], Xdv, ydv, f"n={n},C={C},g={gamma}")
        rows.append({
            "n_train": n, "C": C, "gamma": str(gamma), "eer": res["eer"],
            "threshold": res["threshold"], "fit_seconds": res["fit_seconds"],
            "score_seconds": res["score_seconds"], "n_support": res["n_support"],
        })
        pd.DataFrame(rows).to_csv(SWEEP_CSV, index=False)
        tqdm.write(f"  n={n:>6} C={C:<6} gamma={str(gamma):<6}: dev EER={res['eer']*100:.3f}%  "
                   f"(fit {res['fit_seconds']:.0f}s, score {res['score_seconds']:.0f}s, "
                   f"{res['n_support']:,} SVs)")

    out = pd.DataFrame(rows)
    out["gamma"] = out["gamma"].astype(str)  # keep in-memory dtype consistent with the CSV
    return out


def refit_best(sweep: pd.DataFrame, data) -> dict:
    """Refit the winning configuration so the saved model provably matches the
    reported table. Needed because on a resumed run the winner may have been
    computed in an earlier process and is not held in memory."""
    Xtr, ytr, _ = data["train"]
    Xdv, ydv, dv_names = data["dev"]
    best_row = sweep.loc[sweep["eer"].idxmin()]

    n = int(best_row["n_train"])
    C = float(best_row["C"])
    gamma = best_row["gamma"]
    gamma = gamma if gamma == "scale" else float(gamma)

    print(f"\n  refitting best config (n={n:,}, C={C}, gamma={gamma}) to save the model")
    idx = subsample_indices(ytr, n, config.RANDOM_SEED)
    res = fit_and_eval_svm(C, gamma, Xtr[idx], ytr[idx], Xdv, ydv, "best refit")

    joblib.dump(res["pipeline"], SVM_MODEL)
    pd.DataFrame({"filename": dv_names, "score": res["scores"],
                  "label": np.where(ydv == 1, "bonafide", "spoof")}
                 ).to_csv(config.RESULTS_DIR / "phase5_dev_scores_svm.csv", index=False)
    return {"n_train": n, "C": C, "gamma": gamma, "scores": res["scores"],
            "eer": res["eer"], "is_full_train": n == len(ytr)}


def check_edges(sweep: pd.DataFrame, best: dict) -> list[str]:
    """A winner on the boundary of a searched range means the true optimum probably
    lies outside it -- say so rather than presenting a boundary hit as 'the best'."""
    warnings = []
    if best["C"] in (min(config.SVM_C_GRID), max(config.SVM_C_GRID)):
        where = "lower" if best["C"] == min(config.SVM_C_GRID) else "upper"
        warnings.append(f"best C={best['C']} sits at the {where} edge of the C grid")
    numeric = [g for g in config.SVM_GAMMA_GRID if g != "scale"]
    if best["gamma"] != "scale" and best["gamma"] in (min(numeric), max(numeric)):
        where = "lower" if best["gamma"] == min(numeric) else "upper"
        warnings.append(f"best gamma={best['gamma']} sits at the {where} edge of the gamma grid")
    max_swept = int(sweep["n_train"].max())
    if best["n_train"] == max_swept:
        if best.get("is_full_train"):
            warnings.append(
                f"best n_train={best['n_train']:,} is the ENTIRE train split -- dev EER was still "
                "improving with data, but no more training data exists to add")
        else:
            warnings.append(
                f"best n_train={best['n_train']:,} is the largest size swept -- dev EER had not "
                "plateaued, so more training data may still help")
    return warnings


# ---------------------------------------------------------------------------
# Step 2 -- Random Forest at every sweep size, plus full train
# ---------------------------------------------------------------------------

def run_rf_curve(data, force: bool) -> pd.DataFrame:
    Xtr, ytr, _ = data["train"]
    Xdv, ydv, dv_names = data["dev"]

    sizes = resolved_sizes(len(ytr))
    print(f"\n[2/2] Random Forest at every sweep size {sizes}")
    print("      Size-matched RF rows are what make the SVM-vs-RF comparison honest:")
    print("      without them an RF win could just be a larger training set.")

    done = pd.read_csv(RF_CURVE_CSV) if (RF_CURVE_CSV.exists() and not force) else pd.DataFrame()
    done_sizes = set(done["n_train"]) if len(done) else set()
    rows = done.to_dict("records") if len(done) else []

    jobs = [(n, RF_FULL_MODEL if n == len(ytr) else rf_sub_model_path(n)) for n in sizes]

    for n, path in tqdm(jobs, desc="rf curve", unit="size"):
        if n in done_sizes:
            continue
        idx = subsample_indices(ytr, n, config.RANDOM_SEED)
        Xs, ys = Xtr[idx], ytr[idx]

        if path.exists() and not force:
            tqdm.write(f"  n={n:,}: loading cached {path.name}")
            rf = joblib.load(path)
        else:
            t0 = time.time()
            rf = RandomForestClassifier(
                n_estimators=config.RF_N_ESTIMATORS, class_weight="balanced",
                n_jobs=config.RF_N_JOBS, random_state=config.RANDOM_SEED,
            ).fit(Xs, ys)
            tqdm.write(f"  n={n:,}: fit in {time.time()-t0:.0f}s")
            joblib.dump(rf, path)

        scores = score_in_chunks(rf, Xdv, "proba", f"RF n={n}")
        report = metrics.full_report(ydv, scores)
        tqdm.write(f"  n={n:,}: dev EER={report['eer']*100:.3f}%")

        tag = "full" if n == len(ytr) else str(n)
        pd.DataFrame({"filename": dv_names, "score": scores,
                      "label": np.where(ydv == 1, "bonafide", "spoof")}
                     ).to_csv(config.RESULTS_DIR / f"phase5_dev_scores_rf_{tag}.csv", index=False)

        rows.append({"n_train": n, "is_full": n == len(ytr), **report})
        pd.DataFrame(rows).to_csv(RF_CURVE_CSV, index=False)

        if n == len(ytr):
            plot_rf_importance(rf)

    return pd.DataFrame(rows).sort_values("n_train").reset_index(drop=True)


def plot_rf_importance(rf):
    imp = rf.feature_importances_
    blocks = [("mean(MFCC)", 0, 20), ("mean(delta)", 20, 40), ("mean(delta2)", 40, 60),
              ("std(MFCC)", 60, 80), ("std(delta)", 80, 100), ("std(delta2)", 100, 120)]
    colors = plt.cm.tab10(np.linspace(0, 1, len(blocks)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for (name, a, b), c in zip(blocks, colors):
        axes[0].bar(range(a, b), imp[a:b], color=c, label=name)
    axes[0].set_xlabel("Feature index")
    axes[0].set_ylabel("Gini importance")
    axes[0].set_title(f"RF feature importance, all 120 dims (full train, {config.RF_N_ESTIMATORS} trees)")
    axes[0].legend(fontsize=8)

    totals = [imp[a:b].sum() for _, a, b in blocks]
    axes[1].bar([n for n, _, _ in blocks], totals, color=colors)
    axes[1].set_ylabel("Summed Gini importance")
    axes[1].set_title("Importance by feature block\n(mean(delta) ranks 2nd — the block wrongly dropped in the first run)")
    axes[1].tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(RF_IMPORTANCE_PNG, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_sweep(sweep: pd.DataFrame, rf_curve: pd.DataFrame, best: dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # A: best-per-size envelope for SVM, plus the RF curve
    a = axes[0]
    env = sweep.groupby("n_train")["eer"].min().reset_index()
    a.plot(env["n_train"], env["eer"] * 100, marker="o", label="SVM (best C/gamma at each size)")
    sub = rf_curve[~rf_curve["is_full"]]
    if len(sub):
        a.plot(sub["n_train"], sub["eer"] * 100, marker="s", label="Random Forest")
    full = rf_curve[rf_curve["is_full"]]
    if len(full):
        a.axhline(float(full["eer"].iloc[0]) * 100, ls=":", c="gray",
                  label=f"RF full train ({int(full['n_train'].iloc[0]):,})")
    a.axvline(best["n_train"], ls="--", c="tab:red", alpha=0.6)
    a.set_xlabel("Training subsample size")
    a.set_ylabel("Dev EER (%)")
    a.set_title("Learning curves\n(dev = speaker-disjoint 2019 dev)")
    a.legend(fontsize=8)
    a.grid(alpha=0.3)

    # B: every hyperparameter combination across sizes
    a = axes[1]
    for (C, gamma), grp in sweep.groupby(["C", "gamma"]):
        grp = grp.sort_values("n_train")
        a.plot(grp["n_train"], grp["eer"] * 100, marker=".", alpha=0.75,
               lw=1, label=f"C={C}, g={gamma}")
    a.set_xlabel("Training subsample size")
    a.set_ylabel("Dev EER (%)")
    a.set_title("Every (C, gamma) combination\nacross sizes")
    # Legend below the axes: 16 entries overlap the data if placed inside.
    a.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    a.grid(alpha=0.3)

    # C: heatmap at the winning size
    a = axes[2]
    at_best = sweep[sweep["n_train"] == best["n_train"]].copy()
    at_best["gamma"] = at_best["gamma"].astype(str)
    piv = at_best.pivot_table(index="C", columns="gamma", values="eer") * 100
    # Order columns as configured rather than alphabetically ("scale" would sort last).
    piv = piv.reindex(columns=[str(g) for g in config.SVM_GAMMA_GRID if str(g) in piv.columns])
    # Plain viridis (NOT _r): viridis runs dark -> bright with increasing value, so
    # low EER (better) renders dark and high EER (worse) renders bright yellow.
    im = a.imshow(piv.values, cmap="viridis", aspect="auto")
    a.set_xticks(range(len(piv.columns)))
    a.set_xticklabels([str(c) for c in piv.columns])
    a.set_yticks(range(len(piv.index)))
    a.set_yticklabels([str(i) for i in piv.index])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                # White text on the dark (low-EER) cells, black on the bright ones.
                a.text(j, i, f"{v:.2f}", ha="center", va="center",
                       color="white" if v < np.nanmedian(piv.values) else "black", fontsize=9)
    a.set_xlabel("gamma")
    a.set_ylabel("C")
    a.set_title(f"Dev EER (%) at n_train={best['n_train']:,}\n(darker = lower EER = better)")
    fig.colorbar(im, ax=a, label="Dev EER (%)")

    fig.tight_layout()
    # bbox_inches="tight" so the legend sitting below panel B is not clipped.
    fig.savefig(SWEEP_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_summary(sweep, rf_curve, best, edges, data):
    _, ydv, _ = data["dev"]
    svm_report = metrics.full_report(ydv, best["scores"])
    full_row = rf_curve[rf_curve["is_full"]].iloc[0]
    matched = rf_curve[rf_curve["n_train"] == best["n_train"]]

    summary = {
        "svm_best": {"n_train": best["n_train"], "C": best["C"], "gamma": str(best["gamma"]), **svm_report},
        "rf_full": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in full_row.to_dict().items()},
        "sweep_points": int(len(sweep)),
        "edge_warnings": edges,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Phase 5 — Classical MFCC baseline (SVM + Random Forest)",
        "",
        "Primary metric is EER on the speaker-disjoint resplit 2019 dev set. These are",
        "*tuning* numbers, not the final result — 2021 PA eval is still untouched and gets",
        "scored once in Phase 7.",
        "",
        f"A full factorial sweep of {len(config.SVM_SWEEP_SIZES)} subsample sizes × "
        f"{len(config.SVM_C_GRID)} C × {len(config.SVM_GAMMA_GRID)} gamma "
        f"= **{len(sweep)} SVM fits**, plus a Random Forest at every one of those sizes and",
        "on the full train split. Per-size subsamples are independent stratified draws.",
        "",
        "## Best configuration",
        "",
        f"**n_train={best['n_train']:,}, C={best['C']}, gamma={best['gamma']}** → dev EER "
        f"**{svm_report['eer']*100:.3f}%**",
        "",
    ]
    if edges:
        lines += ["> **Boundary warning(s)** — the optimum may lie outside the searched ranges:", ""]
        lines += [f"> - {w}" for w in edges] + [""]

    lines += ["## Best SVM per training size", "",
              "| n_train | best C | best gamma | dev EER | support vectors |", "|---|---|---|---|---|"]
    for n, grp in sweep.groupby("n_train"):
        r = grp.loc[grp["eer"].idxmin()]
        lines.append(f"| {int(n):,} | {r['C']} | {r['gamma']} | {r['eer']*100:.3f}% | {int(r['n_support']):,} |")

    lines += ["", "## Random Forest per training size", "",
              "| n_train | dev EER | ROC-AUC |", "|---|---|---|"]
    for _, r in rf_curve.iterrows():
        tag = f"{int(r['n_train']):,}" + (" (full)" if r["is_full"] else "")
        lines.append(f"| {tag} | {r['eer']*100:.3f}% | {r['roc_auc']:.4f} |")

    lines += ["", "## Head-to-head at the winning size", "",
              "| System | n_train | Dev EER |", "|---|---|---|",
              f"| MFCC-SVM (tuned) | {best['n_train']:,} | {svm_report['eer']*100:.3f}% |"]
    if len(matched):
        lines.append(f"| MFCC-RF (size-matched) | {best['n_train']:,} | {float(matched['eer'].iloc[0])*100:.3f}% |")
    lines.append(f"| MFCC-RF (full train) | {int(full_row['n_train']):,} | {float(full_row['eer'])*100:.3f}% |")

    lines += [
        "",
        "## SVM supplementary metrics (at the EER threshold)",
        "",
        "| Metric | Value |", "|---|---|",
        f"| ROC-AUC | {svm_report['roc_auc']:.4f} |",
        f"| Accuracy | {svm_report['accuracy']:.4f} |",
        f"| Precision (bonafide) | {svm_report['precision_bonafide']:.4f} |",
        f"| Recall (bonafide) | {svm_report['recall_bonafide']:.4f} |",
        f"| F1 (bonafide) | {svm_report['f1_bonafide']:.4f} |",
        "",
        "### Confusion matrix at the EER threshold",
        "",
    ]
    tp = svm_report["tp_bonafide_accepted"]
    fn = svm_report["fn_bonafide_rejected"]
    tn = svm_report["tn_spoof_rejected"]
    fp = svm_report["fp_spoof_accepted"]
    lines += [
        "| | predicted **bonafide** | predicted **spoof** | total |",
        "|---|---|---|---|",
        f"| **actual bonafide** | {tp:,} <br>*(TP — correctly accepted)* | {fn:,} <br>*(FN — genuine user rejected)* | {tp+fn:,} |",
        f"| **actual spoof** | {fp:,} <br>*(FP — attack got through)* | {tn:,} <br>*(TN — correctly rejected)* | {fp+tn:,} |",
        f"| **total** | {tp+fp:,} | {fn+tn:,} | {tp+fn+fp+tn:,} |",
        "",
        f"Read row-wise: {fn/(tp+fn)*100:.2f}% of genuine speech was rejected (FNR) and "
        f"{fp/(fp+tn)*100:.2f}% of replay attacks were accepted (FPR) — equal by construction, "
        "since this is the EER operating point.",
        "",
        "## Artifacts",
        "",
        "- `models/svm_mfcc.joblib` — winning Pipeline (scaler + SVC), all 120 features",
        "- `models/rf_mfcc_full.joblib`, `models/rf_mfcc_sub_<n>.joblib`",
        "- `results/phase5_svm_sweep.csv` — every sweep point (the authoritative table)",
        "- `results/phase5_rf_curve.csv`, `results/phase5_dev_scores_*.csv`",
        "- `results/phase5_svm_sweep.png`, `results/phase5_rf_feature_importance.png`",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute everything, ignoring cached results")
    args = parser.parse_args()

    print("Loading pooled MFCC features...")
    data = load_xy()
    print(f"  train {data['train'][0].shape}, dev {data['dev'][0].shape}")
    print("  SVM and RF both use all 120 pooled-MFCC features")

    sweep = run_svm_sweep(data, args.force)
    best = refit_best(sweep, data)
    edges = check_edges(sweep, best)
    print(f"\n  -> best: n={best['n_train']:,}, C={best['C']}, gamma={best['gamma']}, "
          f"dev EER={best['eer']*100:.3f}%")
    for w in edges:
        print(f"  !! {w}")

    rf_curve = run_rf_curve(data, args.force)
    plot_sweep(sweep, rf_curve, best)
    write_summary(sweep, rf_curve, best, edges, data)
    print(f"\nDone. Summary written to {SUMMARY_MD}")


if __name__ == "__main__":
    main()
