"""Does the optimal window length track the TARGET clip duration? (9.2.2)

    python -m src.duration_strata            # writes the table, prints the verdict
    python -m src.duration_strata --bins 5   # number of duration quantiles

P9c proposed that the optimal T is a property of the target corpus's duration
distribution: above the optimum a clip is tiled into synthetic periodicity no real
recording contains, below it the clip is truncated, and T=150 is where both are
minimised because it sits nine frames from 2021's median. That was a reading of
five aggregate points, and it could not be refuted by them.

This is the test that can. Partition `progress` by recording duration and ask which
T wins *inside each stratum*. The account predicts a moving optimum -- short strata
favouring short windows, long strata favouring long ones. A fixed winner refutes it.

No retraining and no rescoring: the five T variants are trained and their progress
scores are cached, so this is a re-read of files already on disk.

Two arms are kept separate so augmentation is never confounded with T, and comparing
models *within* a stratum is what makes the design work -- short clips are harder for
every system, and that common factor cancels when models are compared against each
other rather than against the level.

`progress` only. No eval application is spent.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config, metrics

# Two arms, each spanning T with everything else held fixed. The augmented arm is the
# incumbent's configuration; the unaugmented arm is the Phase 6 sweep.
ARMS = {
    "augmented": {
        75: "timepool_T75_aug",
        100: "timepool_T100_aug",
        150: "timepool_T150_aug",
    },
    "unaugmented": {
        150: "T150",
        250: "baseline_T250",
        400: "T400",
    },
}

OUT = config.PHASE7_POSTHOC_DIR / "duration_strata.csv"
OUT_CI = config.PHASE7_POSTHOC_DIR / "duration_strata_ci.csv"


def load_progress() -> pd.DataFrame:
    """Progress-partition scores for every T variant, with each file's frame count."""
    lcnn = pd.read_parquet(config.PA2021_LCNN_SCORES)
    posthoc = pd.read_parquet(config.PA2021_POSTHOC_SCORES)
    man = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet",
                          columns=["filename", "label", "partition"])
    man = man[man["partition"] == "progress"]

    df = man.merge(lcnn.drop(columns=["duration_s"]), on="filename", how="left")
    df = df.merge(posthoc.drop(columns=["partition"]), on="filename", how="left")
    df["y"] = (df["label"] == "bonafide").astype(int)
    return df


def build(bins: int) -> pd.DataFrame:
    df = load_progress()

    # Edges come from the duration distribution alone -- no score and no label is
    # consulted in choosing them, which is what keeps the strata from being tuned.
    qs = np.linspace(0, 1, bins + 1)
    edges = df["n_frames"].quantile(qs).to_numpy()
    edges[0], edges[-1] = edges[0] - 1, edges[-1] + 1
    df["stratum"] = pd.cut(df["n_frames"], bins=edges, labels=False)

    rows = []
    total = bins * sum(len(a) for a in ARMS.values())
    bar = tqdm(total=total, desc="strata x models", unit="cell")
    for s in range(bins):
        sub = df[df["stratum"] == s]
        med = float(sub["n_frames"].median())
        for arm, models in ARMS.items():
            for t, tag in models.items():
                bar.set_postfix_str(f"stratum {s} {tag}")
                bar.update(1)
                m = sub[tag].notna()
                cell = sub[m]
                if cell["y"].nunique() < 2:
                    continue
                eer, _ = metrics.compute_eer(cell.loc[cell["y"] == 1, tag].to_numpy(),
                                             cell.loc[cell["y"] == 0, tag].to_numpy())
                rows.append({
                    "stratum": s,
                    "n_frames_lo": int(np.ceil(edges[s])),
                    "n_frames_hi": int(np.floor(edges[s + 1])),
                    "median_frames": med,
                    "n_trials": int(len(cell)),
                    "n_bonafide": int((cell["y"] == 1).sum()),
                    "arm": arm,
                    "T": t,
                    "system": tag,
                    "eer": 100.0 * eer,
                })
    bar.close()
    return pd.DataFrame(rows)


def paired_cis(bins: int, B: int, seed: int = 0) -> pd.DataFrame:
    """Speaker-clustered paired CIs on each stratum's T contrasts.

    The comparison that matters is between two models on the *same* trials, so every
    replicate draws one set of speakers and scores every model on it. Shared "was
    this a hard draw?" noise then cancels, exactly as in P3 -- and it must, because
    the differences being tested here are of the order of a single point.
    """
    df = load_progress()
    man = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet",
                          columns=["filename", "speaker_id"])
    df = df.merge(man, on="filename", how="left")

    qs = np.linspace(0, 1, bins + 1)
    edges = df["n_frames"].quantile(qs).to_numpy()
    edges[0], edges[-1] = edges[0] - 1, edges[-1] + 1
    df["stratum"] = pd.cut(df["n_frames"], bins=edges, labels=False)

    speakers = np.sort(df["speaker_id"].unique())
    rng = np.random.default_rng(seed)
    rows = []

    bar = tqdm(total=bins * len(ARMS), desc="paired CIs", unit="cell")
    for s in range(bins):
        sub = df[df["stratum"] == s]
        by_spk = {k: v.index.to_numpy() for k, v in sub.groupby("speaker_id")}
        for arm, models in ARMS.items():
            bar.set_postfix_str(f"stratum {s} {arm}")
            bar.update(1)
            tags = [models[t] for t in sorted(models)]
            ts = sorted(models)
            eers = np.full((B, len(ts)), np.nan)
            for b in range(B):
                drawn = rng.choice(speakers, size=len(speakers), replace=True)
                idx = np.concatenate([by_spk[k] for k in drawn if k in by_spk])
                cell = sub.loc[idx]
                y = cell["y"].to_numpy()
                if len(np.unique(y)) < 2:
                    continue
                for j, tag in enumerate(tags):
                    sc = cell[tag].to_numpy()
                    eers[b, j] = 100.0 * metrics.compute_eer(sc[y == 1], sc[y == 0])[0]
            # Contrast every T against the stratum's incumbent, T=150, which is the
            # value the target-median account says should lose in the short strata.
            ref = ts.index(150)
            for j, t in enumerate(ts):
                if j == ref:
                    continue
                d = eers[:, j] - eers[:, ref]
                d = d[~np.isnan(d)]
                rows.append({
                    "stratum": s, "arm": arm, "T": t, "vs": 150,
                    "median_frames": float(sub["n_frames"].median()),
                    "diff_pp": float(np.mean(d)),
                    "lo": float(np.percentile(d, 2.5)),
                    "hi": float(np.percentile(d, 97.5)),
                    "excludes_zero": bool(np.percentile(d, 2.5) > 0
                                          or np.percentile(d, 97.5) < 0),
                    "sign_consistency": float(max((d > 0).mean(), (d < 0).mean())),
                    "B": B,
                })
    bar.close()
    return pd.DataFrame(rows)


def verdict(res: pd.DataFrame) -> None:
    """Print the winner per stratum per arm, which is the whole result."""
    for arm in ARMS:
        a = res[res["arm"] == arm]
        print(f"\n  {arm} arm")
        print(f"    {'stratum':>8} {'frames':>12} {'median':>7} "
              + " ".join(f"{'T=' + str(t):>8}" for t in sorted(ARMS[arm]))
              + "   best")
        winners = []
        for s in sorted(a["stratum"].unique()):
            row = a[a["stratum"] == s].set_index("T")
            best = int(row["eer"].idxmin())
            winners.append(best)
            span = f"{row['n_frames_lo'].iloc[0]}-{row['n_frames_hi'].iloc[0]}"
            cells = " ".join(f"{row.loc[t, 'eer']:8.2f}" if t in row.index else " " * 8
                             for t in sorted(ARMS[arm]))
            print(f"    {s:>8} {span:>12} {row['median_frames'].iloc[0]:>7.0f} "
                  f"{cells}   T={best}")
        moved = len(set(winners)) > 1
        direction = all(x <= y for x, y in zip(winners, winners[1:]))
        print(f"    winners by increasing duration: {winners}"
              f"  -> {'MOVES' if moved else 'FIXED'}"
              + (", monotone increasing" if moved and direction else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bins", type=int, default=5,
                    help="number of duration quantiles (default 5)")
    ap.add_argument("--boot", type=int, default=1000,
                    help="speaker-clustered bootstrap replicates (0 to skip)")
    args = ap.parse_args()

    res = build(args.bins)
    res.to_csv(OUT, index=False)
    print(f"\n  wrote {OUT}")
    verdict(res)

    if args.boot:
        ci = paired_cis(args.bins, args.boot)
        ci.to_csv(OUT_CI, index=False)
        print(f"\n  wrote {OUT_CI}\n")
        for _, r in ci.sort_values(["arm", "stratum", "T"]).iterrows():
            mark = "significant" if r["excludes_zero"] else "not distinguishable"
            print(f"    {r['arm']:<12} stratum {int(r['stratum'])} "
                  f"(median {r['median_frames']:.0f}f)  T={int(r['T'])} vs T=150: "
                  f"{r['diff_pp']:+6.2f} pp  [{r['lo']:+6.2f}, {r['hi']:+6.2f}]  {mark}")


if __name__ == "__main__":
    main()
