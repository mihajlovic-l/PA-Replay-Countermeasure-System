"""Phase 7, pass 3: turn the 2021 scores into the results chapter.

    python -m src.report_2021

Cheap and re-runnable -- it reads score tables, never audio, so every table and
figure can be regenerated in seconds without re-touching the corpus.

Covers PROJECT_PLAN.md 7.1-7.5:
  7.1  EER per system on 2021 (headline = partition "eval"), beside 2019 dev
  7.2  confusion matrix at the EER threshold
  7.3  precision / recall / F1 / accuracy / ROC-AUC at that same threshold
  7.4  comparison against the four official ASVspoof baselines, computed by the
       identical code path on the identical trials
  7.5  EER by recording condition
plus the two controls Phase 5/6 flagged, and an explicit verdict on each of the
three predictions registered in advance.

Every metric comes from metrics.py, the same module Phases 5 and 6 used, so the
2019 and 2021 numbers in the thesis are produced by one implementation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_curve

from . import config, metrics, tdcf

# Conditions whose groups contain BOTH classes, so each gets a genuine within-group
# EER using its own bonafide -- answering "how does the system do in room R6".
WITHIN_GROUP_COLS = ["room", "mic"]
# Conditions that describe the REPLAY DEVICE, which bonafide recordings do not have
# (their fields are "-"). No bonafide in the group means no FRR curve and hence no
# EER, so all bonafide are pooled against each condition's spoof -- see the module
# docstring of condition_breakdown for what that buys and costs.
POOLED_COLS = ["dist", "r", "m", "s", "c"]

MANIFEST_COLS = ["filename", "label", "partition", "speaker_id",
                 "room", "mic", "dist", "r", "m", "s", "c"]


# --- loading -------------------------------------------------------------------

def load_scores() -> tuple[pd.DataFrame, list[str]]:
    lcnn = pd.read_parquet(config.PA2021_LCNN_SCORES)
    df = lcnn
    if config.PA2021_CLASSICAL_SCORES.exists():
        df = df.merge(pd.read_parquet(config.PA2021_CLASSICAL_SCORES),
                      on="filename", how="left")

    manifest = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet")
    df = df.merge(manifest[MANIFEST_COLS], on="filename", how="left")
    df["y"] = (df["label"] == "bonafide").astype(int)

    ours = [c for c in list(config.PHASE7_LCNN_SYSTEMS) +
            list(config.PHASE7_CLASSICAL_SYSTEMS) if c in df.columns]
    return df, ours


def load_official(filenames: pd.Series) -> dict[str, np.ndarray]:
    """The four official baseline score.txt files, aligned to our row order.

    Same format we export (`FILENAME SCORE`, higher = bonafide), so they feed the
    identical EER function -- which is what makes 7.4 a like-for-like table rather
    than our metric against their published one.
    """
    out = {}
    for name, path in config.PA2021_BASELINE_SCORE_FILES.items():
        if not path.exists():
            print(f"  ! missing official baseline: {path}")
            continue
        s = pd.read_csv(path, sep=r"\s+", names=["filename", "score"])
        out[name] = s.set_index("filename")["score"].reindex(filenames).to_numpy()
    return out


# --- 7.1 / 7.2 / 7.3 -----------------------------------------------------------

def system_table(df: pd.DataFrame, systems: list[str],
                 official: dict[str, np.ndarray], asv: dict | None = None) -> pd.DataFrame:
    """EER (7.1) plus the supplementary metrics at the EER threshold (7.2/7.3).

    `asv` adds **min t-DCF**, the challenge's PRIMARY metric for PA. Reported alongside
    EER rather than instead of it, because the two disagree in places: a system can post
    a fair EER while being badly shaped in the cost-weighted region of the DET curve.
    """
    dev = {**config.PHASE7_LCNN_SYSTEMS, **config.PHASE7_CLASSICAL_SYSTEMS}
    rows = []
    for tag in systems:
        m = df[tag].notna()
        y, sc = df.loc[m, "y"].to_numpy(), df.loc[m, tag].to_numpy()
        rep = metrics.full_report(y, sc)
        if asv:
            rep["min_tdcf"], rep["tdcf_threshold"] = tdcf.min_tdcf(y, sc, asv)
        rows.append({"system": tag, "kind": "this project", "n_trials": int(m.sum()),
                     "dev_eer_2019": dev.get(tag), **rep})
    for name, s in official.items():
        m = ~np.isnan(s)
        y, sc = df.loc[m, "y"].to_numpy(), s[m]
        rep = metrics.full_report(y, sc)
        if asv:
            rep["min_tdcf"], rep["tdcf_threshold"] = tdcf.min_tdcf(y, sc, asv)
        rows.append({"system": name, "kind": "official baseline",
                     "n_trials": int(m.sum()), "dev_eer_2019": None, **rep})
    return pd.DataFrame(rows).sort_values("eer").reset_index(drop=True)


def posthoc_table(asv: dict | None) -> pd.DataFrame | None:
    """Post-hoc systems, in their OWN table — never merged into the Phase 7 one.

    These were trained after 2021 results were seen (PROJECT_PLAN.md 9.3.1), so they
    carry no pre-registration guarantee and must not be tabulated alongside systems
    that do. Each is reported on exactly the partitions its whitelist permits, so the
    table itself shows that the losing candidate was never scored on eval.
    """
    if not config.PA2021_POSTHOC_SCORES.exists():
        return None
    d = pd.read_parquet(config.PA2021_POSTHOC_SCORES)
    man = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet",
                          columns=["filename", "label"])
    d = d.merge(man, on="filename")
    d["y"] = (d["label"] == "bonafide").astype(int)

    rows = []
    for tag, parts in config.PHASE7_POSTHOC_SYSTEMS.items():
        if tag not in d.columns:
            continue
        for part in parts:
            sub = d[(d["partition"] == part) & d[tag].notna()]
            if not len(sub) or sub["y"].nunique() < 2:
                continue
            y, sc = sub["y"].to_numpy(), sub[tag].to_numpy()
            rep = metrics.full_report(y, sc)
            if asv:
                rep["min_tdcf"], _ = tdcf.min_tdcf(y, sc, asv)
            rows.append({"system": tag, "kind": "post-hoc (not pre-registered)",
                         "partition": part, "n_trials": len(sub), **rep})
    return pd.DataFrame(rows) if rows else None


# --- 7.5 -----------------------------------------------------------------------

def condition_breakdown(df: pd.DataFrame, system: str) -> pd.DataFrame:
    """EER per recording condition, by whichever convention the data supports.

    WITHIN-GROUP (room, mic): the group has its own bonafide and spoof, so this is
    an ordinary EER restricted to that condition.

    POOLED-BONAFIDE (replay-device factors): those groups are spoof-only, because a
    bonafide recording was never replayed through anything. Pooling ALL bonafide
    against each condition's spoof makes the FRR curve identical in every group and
    lets only the FAR curve move, so any difference between conditions is
    attributable purely to how detectable that condition's attacks are, and cannot
    be an artefact of one group holding easier genuine speech. The cost: the shared
    bonafide half makes these EERs statistically CORRELATED, so they must not be
    fed to a test that assumes independent groups.

    Also reports FAR at the single global EER threshold -- no pooling convention to
    explain, and it reads directly as "this share of that condition's attacks got
    through at the operating point we report".
    """
    d = df[df[system].notna()]
    y, sc = d["y"].to_numpy(), d[system].to_numpy()
    _, global_thr = metrics.eer_from_labels(y, sc)
    bona = sc[y == 1]

    rows = []
    for col in WITHIN_GROUP_COLS + POOLED_COLS:
        pooled = col in POOLED_COLS
        for level, g in d.groupby(col, sort=True):
            if level == "-":
                continue
            gy, gs = g["y"].to_numpy(), g[system].to_numpy()
            spoof = gs[gy == 0]
            if pooled:
                if len(spoof) == 0:      # a bonafide-only level, e.g. dist "D4"
                    continue
                eer, thr = metrics.compute_eer(bona, spoof)
                n_bona = len(bona)
            else:
                if gy.min() == gy.max():
                    continue
                eer, thr = metrics.eer_from_labels(gy, gs)
                n_bona = int(gy.sum())
            rows.append({
                "factor": col, "level": level,
                "convention": "pooled-bonafide" if pooled else "within-group",
                "eer": eer, "threshold": thr,
                "n_bonafide": n_bona, "n_spoof": len(spoof),
                "far_at_global_thr": float((spoof >= global_thr).mean()) if len(spoof) else np.nan,
            })
    return pd.DataFrame(rows)


# --- controls ------------------------------------------------------------------

def short_clip_control() -> pd.DataFrame:
    """Separate the MECHANICAL part of any 2021 degradation from domain shift.

    Pooled MFCC statistics average over ~453 frames on 2019 but only ~149 on 2021,
    and the standard error of a pooled statistic scales as 1/sqrt(T) -- so 2021's
    features are inherently ~1.38x noisier before replay realism enters the picture
    (PROGRESS_REPORT.md, Phase 5). Restricting 2019 dev to 2021-like durations
    isolates that: whatever EER rise appears here is mechanical, and only the
    remainder is evidence about simulated-vs-real replay.
    """
    idx = pd.read_parquet(config.PACKED_INDEX["dev"])[["filename", "n_frames"]]
    sources = [("MFCC-SVM", config.PHASE5_SVM_DIR / "dev_scores.csv")]
    sources += [(t, config.PHASE6_DIR / t / "dev_scores.csv")
                for t in config.PHASE7_LCNN_SYSTEMS]

    rows = []
    for tag, path in sources:
        if not path.exists():
            continue
        s = pd.read_csv(path).merge(idx, on="filename", how="inner")
        y = (s["label"] == "bonafide").astype(int).to_numpy()
        sc = s["score"].to_numpy()
        full, _ = metrics.eer_from_labels(y, sc)
        row = {"system": tag, "dev_eer_all": full, "n_all": len(s)}
        for cap in (150, 200, 250):
            m = s["n_frames"].to_numpy() <= cap
            if m.sum() > 100 and len(np.unique(y[m])) == 2:
                e, _ = metrics.eer_from_labels(y[m], sc[m])
                row[f"dev_eer_le{cap}f"] = e
                row[f"n_le{cap}f"] = int(m.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def duration_control(df: pd.DataFrame) -> pd.DataFrame:
    """Repeat on 2021 the confound Phase 6 tested on 2019.

    Short clips are tile-padded to T, which creates exact periodicity a CNN could
    detect -- so if durations differ by class, "is this repeating?" could proxy for
    "is this spoof?". Phase 6 measured duration alone at 41.5% EER on 2019 (near
    chance) and dismissed it. Same check here, since 2021 is tiled ~2.7x at T=400.
    """
    rows = []
    for part in ("all", config.PA2021_REPORTED_PARTITION):
        d = df if part == "all" else df[df["partition"] == part]
        y = d["y"].to_numpy()
        for name, sc in (("duration_s", d["duration_s"].to_numpy()),
                         ("n_frames", d["n_frames"].to_numpy().astype(float)),
                         ("tiling_factor", np.ceil(400 / np.maximum(d["n_frames"], 1)))):
            eer, _ = metrics.eer_from_labels(y, sc)
            rows.append({"partition": part, "cue": name, "eer_as_score": eer,
                         "mean_bonafide": float(sc[y == 1].mean()),
                         "mean_spoof": float(sc[y == 0].mean())})
    return pd.DataFrame(rows)


def predictions_verdict(table: pd.DataFrame) -> pd.DataFrame:
    """Score the three predictions registered in PROJECT_PLAN.md before 2021 was
    touched. Each compares against a MATCHED control differing in exactly one
    variable, which is why T400 (timepool, unit) was added to the registration."""
    eer = table.set_index("system")["eer"].to_dict()

    def cmp(name, a, b, claim):
        if a not in eer or b not in eer:
            return None
        return {"prediction": name, "system": a, "control": b,
                "eer_system": eer[a], "eer_control": eer[b],
                "delta_pp": (eer[a] - eer[b]) * 100,
                "claim": claim, "supported": bool(eer[a] < eer[b])}

    out = [
        cmp("1: shorter T transfers better (less 2019->2021 padding mismatch)",
            "baseline_T250", "T400", "T=250 beats T=400 on 2021 despite losing on dev"),
        cmp("2a: mild waveform augmentation helps out of domain",
            "flatten_T400_aug1", "flatten_T400", "aug1 beats clean on 2021"),
        cmp("2b: aggressive waveform augmentation helps out of domain",
            "flatten_T400_aug", "flatten_T400", "aug3 beats clean on 2021"),
        cmp("3: CMVN transfers better despite worse in-domain",
            "cmvn_T400", "T400", "CMVN beats matched non-CMVN control on 2021"),
    ]
    return pd.DataFrame([r for r in out if r])


# --- figures -------------------------------------------------------------------

def plot_det(df: pd.DataFrame, systems: list[str], official: dict,
             out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    ticks = np.array([0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8])

    def draw(name, y, sc, **kw):
        fpr, tpr, _ = roc_curve(y, sc)
        fnr = 1 - tpr
        ok = (fpr > 0) & (fnr > 0)
        ax.plot(norm.ppf(fpr[ok]), norm.ppf(fnr[ok]), label=name, **kw)

    for tag in systems:
        m = df[tag].notna()
        draw(tag, df.loc[m, "y"], df.loc[m, tag], lw=1.8)
    for name, sc in official.items():
        m = ~np.isnan(sc)
        draw(f"{name} (official)", df.loc[m, "y"], sc[m], lw=1.2, ls="--", alpha=.75)

    # Equal-error diagonal: every curve crosses it exactly at its own EER.
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], c="k", lw=.7, alpha=.4, zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xticks(norm.ppf(ticks)); ax.set_xticklabels([f"{t*100:g}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks)); ax.set_yticklabels([f"{t*100:g}" for t in ticks])
    ax.set_xlabel("False acceptance rate — spoof accepted (%)")
    ax.set_ylabel("False rejection rate — bonafide rejected (%)")
    ax.set_title(title)
    ax.grid(alpha=.3); ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_eer_bars(table: pd.DataFrame, out: Path, title: str) -> None:
    t = table.sort_values("eer", ascending=True)
    colors = ["tab:blue" if k == "this project" else "tab:grey" for k in t["kind"]]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(t) + 2))
    ax.barh(t["system"], t["eer"] * 100, color=colors)
    for i, v in enumerate(t["eer"] * 100):
        ax.text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("EER (%) — lower is better")
    ax.set_title(title)
    ax.grid(axis="x", alpha=.3)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def plot_conditions(cond: pd.DataFrame, system: str, out: Path) -> None:
    factors = [f for f in cond["factor"].unique()]
    fig, axes = plt.subplots(1, len(factors), figsize=(3.1 * len(factors), 4.2),
                             squeeze=False)
    for ax, f in zip(axes[0], factors):
        g = cond[cond["factor"] == f]
        c = "tab:blue" if g["convention"].iloc[0] == "within-group" else "tab:orange"
        ax.bar(g["level"].astype(str), g["eer"] * 100, color=c)
        ax.set_title(f"{f}\n({g['convention'].iloc[0]})", fontsize=9)
        ax.set_ylabel("EER (%)"); ax.tick_params(axis="x", rotation=60, labelsize=7)
        ax.grid(axis="y", alpha=.3)
    fig.suptitle(f"2021 PA eval — EER by condition ({system})", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# --- score.txt export ----------------------------------------------------------

def export_score_txt(df: pd.DataFrame, systems: list[str]) -> None:
    """One file per system in the official ASVspoof submission format.

    Lets anyone recompute every reported number from a ~20MB text file, with no
    corpus, no GPU and no part of this pipeline. Covers ALL partitions, as the
    official baseline files do.
    """
    for tag in systems:
        sub = df.loc[df[tag].notna(), ["filename", tag]]
        sub.to_csv(config.PHASE7_PREREG_SCORES_DIR / f"{tag}.score.txt",
                   sep=" ", header=False, index=False, float_format="%.6f")
    print(f"  exported {len(systems)} score.txt files -> {config.PHASE7_PREREG_SCORES_DIR}")


# --- main ----------------------------------------------------------------------

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--primary", default="flatten_T400")
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--no-tdcf", action="store_true",
                   help="skip min t-DCF (avoids reading the 2.5M-row ASV protocol)")
    args = p.parse_args()

    out = config.PHASE7_PREREG_DIR      # frozen Phase 7 deliverable
    df, systems = load_scores()
    print(f"loaded {len(df):,} scored files, {len(systems)} systems: {', '.join(systems)}")
    print(f"partitions present: {df['partition'].value_counts().to_dict()}")

    if not args.no_export:
        export_score_txt(df, systems)

    headline = df[df["partition"] == config.PA2021_REPORTED_PARTITION].reset_index(drop=True)
    official = load_official(headline["filename"])
    print(f"\nHEADLINE partition '{config.PA2021_REPORTED_PARTITION}': {len(headline):,} trials "
          f"({int(headline['y'].sum()):,} bonafide / {int((1-headline['y']).sum()):,} spoof)")

    # min t-DCF: the challenge's primary metric for PA. The ASV operating point is
    # fixed on eval and reused everywhere -- see src/tdcf.py for why, and for the
    # 8/8 validation against the published baseline values.
    asv = None if args.no_tdcf else tdcf.asv_error_rates(tdcf.ASV_OPERATING_PARTITION)
    if asv:
        print(f"ASV operating point: EER {asv['asv_eer']*100:.3f}%, "
              f"floor {tdcf.asv_floor(asv):.4f}")
    table = system_table(headline, systems, official, asv)
    table.to_csv(out / "eer_table_2021.csv", index=False)
    print("\n=== 7.1/7.4  EER on 2021 PA eval ===")
    cols = ["system", "kind", "eer", "dev_eer_2019", "roc_auc", "n_trials"]
    if "min_tdcf" in table.columns:
        cols.insert(3, "min_tdcf")
    show = table[cols].copy()
    show["eer"] = (show["eer"] * 100).round(3)
    show["dev_eer_2019"] = (show["dev_eer_2019"] * 100).round(3)
    print(show.to_string(index=False))

    ph = posthoc_table(asv)
    if ph is not None:
        ph.to_csv(config.PHASE7_POSTHOC_DIR / "posthoc_table_2021.csv", index=False)
        print("\n=== POST-HOC systems (NOT pre-registered — reported separately) ===")
        cols = ["system", "partition", "eer", "roc_auc", "n_trials"]
        if "min_tdcf" in ph.columns:
            cols.insert(3, "min_tdcf")
        show = ph[cols].copy()
        show["eer"] = (show["eer"] * 100).round(3)
        print(show.to_string(index=False))
        print("  (partitions per system are whitelisted in "
              "config.PHASE7_POSTHOC_SYSTEMS:\n"
              "   only the progress-selected winner was ever scored on eval)")

    # Consistency check across the non-headline partitions -- free, since they were
    # extracted in the same pass. Never used to select anything.
    others = []
    for part in sorted(set(df["partition"].dropna()) - {config.PA2021_REPORTED_PARTITION}):
        d = df[df["partition"] == part]
        for tag in systems:
            m = d[tag].notna()
            e, _ = metrics.eer_from_labels(d.loc[m, "y"], d.loc[m, tag])
            others.append({"partition": part, "system": tag, "eer": e, "n": int(m.sum())})
    pd.DataFrame(others).to_csv(out / "eer_other_partitions.csv", index=False)

    cond = condition_breakdown(headline, args.primary)
    cond.to_csv(out / "condition_breakdown.csv", index=False)
    print(f"\n=== 7.5  condition breakdown ({args.primary}) ===")
    print(cond.assign(eer=(cond["eer"] * 100).round(3),
                      far=(cond["far_at_global_thr"] * 100).round(3))
              [["factor", "level", "convention", "eer", "far", "n_spoof"]].to_string(index=False))

    short = short_clip_control()
    short.to_csv(out / "control_short_clips_2019dev.csv", index=False)
    dur = duration_control(headline)
    dur.to_csv(out / "control_duration_cue_2021.csv", index=False)
    print("\n=== controls ===")
    print(short.to_string(index=False))
    print(dur.to_string(index=False))

    verdict = predictions_verdict(table)
    verdict.to_csv(out / "registered_predictions.csv", index=False)
    print("\n=== registered predictions ===")
    print(verdict[["prediction", "eer_system", "eer_control", "delta_pp",
                   "supported"]].to_string(index=False))

    plot_det(headline, systems, official, out / "det_curves_2021.png",
             "DET — 2021 PA eval (partition=eval)")
    plot_eer_bars(table, out / "eer_comparison_2021.png",
                  "EER on 2021 PA eval — this project vs official baselines")
    plot_conditions(cond, args.primary, out / "condition_breakdown.png")

    best = table.iloc[0]
    summary = {
        "partition": config.PA2021_REPORTED_PARTITION,
        "n_trials": int(len(headline)),
        "n_bonafide": int(headline["y"].sum()),
        "n_spoof": int((1 - headline["y"]).sum()),
        "primary_system": args.primary,
        "primary": table[table.system == args.primary].iloc[0].to_dict(),
        "best_overall": best.to_dict(),
        "systems": table.to_dict("records"),
        "predictions": verdict.to_dict("records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float),
                                      encoding="utf-8")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
