"""Thesis figures, generated from the canonical result files.

    python -m src.plot_thesis                     # every figure
    python -m src.plot_thesis --only dose window  # just those two
    python -m src.plot_thesis --out some/other/dir
    python -m src.plot_thesis --no-thesis-copy    # canonical copy only

Every panel here reads a *persisted* results file -- nothing is transcribed out of
PROGRESS_REPORT.md by hand, which is the whole point: a figure that disagrees with
the canonical table is a bug, and this way it cannot happen silently.

Two results had no file of their own and are persisted first, under
`results/phase7/posthoc/`:

  axis_sweeps.csv              the 9.1 dose sweep and the 9.2 window sweep, whose
                               EERs lived only in the per-run score tables
  hidden_decomposition_all.csv the 7.19 simulated / non-speech decomposition, which
                               was persisted only for the handful of post-hoc systems

Nothing here touches audio, a model or a selection decision -- it re-reads scores
that were computed once, so it is cheap and re-runnable.

Every PNG lands in `results/phase7/posthoc/` by default -- the same tracked
location the pre-registered DET curve and condition breakdown already live in
(`results/phase7/preregistered/`), so these are versioned and reviewable on GitHub
the same way. A copy is placed in `Thesis paper/slike/` as well (unless
`--no-thesis-copy`), since the gitignored write-up references figures by relative
path from there and needs the files physically present to compile.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from . import config, metrics

# Where the thesis keeps its figures. Both write-up folders are gitignored copies of
# the template, so this is a path the repo knows about but does not version.
SLIKE_DIR = config.PROJECT_ROOT / "Thesis paper" / "slike"
SLIKE_SR_DIR = config.PROJECT_ROOT / "Diplomski rad" / "slike"

PREREG = config.PHASE7_PREREG_DIR
POSTHOC = config.PHASE7_POSTHOC_DIR

# --- language -------------------------------------------------------------------
# The submitted thesis is in Serbian Cyrillic, so every figure needs a second set of
# labels. Rather than fork the plotting code, each user-visible string is wrapped in
# _t() and looked up here; anything absent falls through unchanged, which is exactly
# what the terms that stay English need (ASVspoof, EER, min t-DCF, front-end, the
# system tags like flatten_T400, the partition names, the condition codes).
# Serbian files are written as <name>_srp.png, leaving the English originals alone.
LANG = "en"

SR = {
    # axes shared by several figures
    "EER (%)": "EER (%)",
    "EER (%) — out of domain": "EER (%) - ван домена",
    "EER (%) — in domain": "EER (%) - у домену",
    "2019 dev EER (%) — in domain": "2019 dev EER (%) - у домену",
    "2021 eval EER (%) — real replay": "2021 eval EER (%) - стварна репродукција",
    "2021 progress EER (%)": "2021 progress EER (%)",
    "2019 dev EER (%)": "2019 dev EER (%)",
    "window length T (frames)": "дужина прозора T (оквира)",
    "clean fraction of the training pool": "удео чистих података у скупу за обуку",
    "2021 progress (out of domain)": "2021 progress (ван домена)",
    "2019 dev (in domain)": "2019 dev (у домену)",
    "out of domain": "ван домена",
    "in domain": "у домену",
    "augmented": "аугментовано",
    "unaugmented": "неаугментовано",
    "augmented arm": "аугментовани режим",
    "unaugmented arm": "неаугментовани режим",
    "duration stratum": "слој трајања",
    "chance": "случајно",
    " chance": " случајно",
    # confidence intervals
    "EER (%) with 95% confidence interval": "EER (%) са 95% интервалом поверења",
    # Kept short: at full length these ran over the top rows of the CI forest, and
    # the unit of each count is already named by the word before it.
    "speaker-clustered (67 speakers)": "груписано по говорницима (67)",
    "trial-level (721,332 trials)": "на нивоу покушаја (721.332)",
    "paired difference in EER (percentage points), 95% CI":
        "упарена разлика у EER (процентни поени), 95% CI",
    "excludes zero": "искључује нулу",
    "spans zero": "обухвата нулу",
    "pre-registered comparisons": "унапред пријављена поређења",
    "post-hoc comparisons": "накнадна поређења",
    # hidden tracks
    "real replay\n(matched D4/d4)": "стварна репродукција\n(упарено D4/d4)",
    "simulated replay": "симулирана репродукција",
    "non-speech removed": "без ванговорних сегмената",
    # DET
    "False acceptance rate — spoof accepted (%)": "стопа лажног прихватања (%)",
    "False rejection rate — bonafide rejected (%)": "стопа лажног одбијања (%)",
    # condition breakdown: the factor codes stay, the names are spelled out. The
    # mapping was read off the numbers quoted in the results chapter, e.g. s4 is the
    # 47.62% replay device and r3/r5 are the 52.45%/27.95% attacker rooms.
    "room": "просторија снимања (room)",
    "mic": "микрофон верификације (mic)",
    "dist": "растојање нападача од верификације (dist)",
    "r": "нападачева просторија (r)",
    "m": "нападачев микрофон (m)",
    "s": "нападачев уређај за репродукцију (s)",
    "c": "растојање нападача од говорника (c)",
    "within-group": "унутар групе",
    "pooled-bonafide": "обједињени истински",
}


def _t(s: str) -> str:
    """User-visible string, in whichever language this run is drawing."""
    return SR.get(s, s) if LANG == "sr" else s

# One palette across every figure, so a colour means the same thing throughout.
C_OURS = "#2f6f9f"        # systems built for this thesis
C_AUG = "#c0603a"         # the augmented arm, wherever it is contrasted with clean
C_OFFICIAL = "#8a8a8a"    # the four published baselines
C_FUSION = "#4f7a4a"      # fused systems
C_ZERO = "#b03030"        # the "no difference" reference line
GRID = dict(alpha=0.25, linewidth=0.6)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

OFFICIAL = ("CQCC-GMM", "LFCC-GMM", "LFCC-LCNN", "RawNet2")


# --- shared loading -------------------------------------------------------------

def _dev_eer(tag: str) -> float | None:
    """Dev EER as recorded by the training run itself, in percent."""
    p = config.RESULTS_DIR / "phase6" / tag / "summary.json"
    if not p.exists():
        return None
    return 100.0 * json.loads(p.read_text())["eer"]


def _colour(system: str) -> str:
    if system in OFFICIAL:
        return C_OFFICIAL
    if "fusion" in system:
        return C_FUSION
    if "aug" in system or "pc" in system:
        return C_AUG
    return C_OURS


def _save(fig, out: Path, name: str) -> Path:
    if LANG == "sr":
        name = name.replace(".png", "_srp.png")
    path = out / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _declutter(values, gap: float) -> list[float]:
    """Nudge label positions apart while keeping their order.

    Purely cosmetic -- the plotted points never move, only the text beside them,
    which is what keeps a crowded right-hand margin readable.
    """
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


# --- results that were never persisted on their own -----------------------------

# The dose axis is p(clean), not the copy count: the loader draws uniformly among the
# clean blob and N augmented ones, so N copies give a clean fraction of 1/(N+1) and
# the copy count moves the dose hyperbolically. Both are carried so a figure can use
# whichever axis reads better.
DOSE_RUNS = [
    ("T150",                1.0,    0),
    ("timepool_T150_pc50",  0.5,    1),
    ("timepool_T150_aug",   0.25,   3),
    ("timepool_T150_pc12",  0.125,  7),
    ("timepool_T150_pc06",  0.0625, 15),
]
# Two arms, because the window axis was swept unaugmented in Phase 6 and augmented
# afterwards; plotting them together is what shows the U is not an artefact of dose.
WINDOW_RUNS = [
    ("timepool_T75_aug",   75,  True),
    ("timepool_T100_aug", 100,  True),
    ("timepool_T150_aug", 150,  True),
    ("T150",              150,  False),
    ("baseline_T250",     250,  False),
    ("T400",              400,  False),
]


def build_axis_sweeps(force: bool = False) -> pd.DataFrame:
    """Persist the 9.1 and 9.2 sweeps as one table, then return it.

    Pre-registered runs already have a progress EER in `eer_other_partitions.csv`;
    the post-hoc ones are scored straight from `posthoc_scores.parquet`, by the same
    metrics module every other number in the project comes through.
    """
    out = POSTHOC / "axis_sweeps.csv"
    if out.exists() and not force:
        return pd.read_csv(out)

    prereg = pd.read_csv(PREREG / "eer_other_partitions.csv")
    prereg = prereg[prereg["partition"] == "progress"].set_index("system")["eer"]

    ph = None
    if config.PA2021_POSTHOC_SCORES.exists():
        d = pd.read_parquet(config.PA2021_POSTHOC_SCORES)
        man = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet",
                              columns=["filename", "label"])
        d = d.merge(man, on="filename")
        d = d[d["partition"] == "progress"]
        d["y"] = (d["label"] == "bonafide").astype(int)
        ph = d

    def progress_eer(tag: str) -> float | None:
        if tag in prereg.index:
            return 100.0 * float(prereg.loc[tag])
        if ph is None or tag not in ph.columns:
            return None
        sub = ph[ph[tag].notna()]
        if not len(sub):
            return None
        eer, _ = metrics.compute_eer(sub.loc[sub["y"] == 1, tag].to_numpy(),
                                     sub.loc[sub["y"] == 0, tag].to_numpy())
        return 100.0 * eer

    rows = []
    for tag, p_clean, copies in DOSE_RUNS:
        rows.append({"axis": "dose", "tag": tag, "p_clean": p_clean,
                     "copies": copies, "n_frames": 150, "augmented": p_clean < 1.0,
                     "progress_eer": progress_eer(tag), "dev_eer": _dev_eer(tag)})
    for tag, n_frames, aug in WINDOW_RUNS:
        rows.append({"axis": "window", "tag": tag,
                     "p_clean": 0.25 if aug else 1.0,
                     "copies": 3 if aug else 0, "n_frames": n_frames,
                     "augmented": aug, "progress_eer": progress_eer(tag),
                     "dev_eer": _dev_eer(tag)})

    df = pd.DataFrame(rows)
    missing = df[df["progress_eer"].isna()]["tag"].tolist()
    if missing:
        print(f"  ! no progress score for: {', '.join(sorted(set(missing)))}")
    df.to_csv(out, index=False)
    print(f"  wrote {out}")
    return df


def build_hidden_decomposition(force: bool = False) -> pd.DataFrame:
    """Persist the 7.19 decomposition for every system, baselines included.

    `hidden` is two tracks, not one: `notrim` is simulated replay and `trim` is real
    replay with non-speech removed, both restricted to the D4/d4 positions. The
    matched reference is therefore `eval` restricted to those same positions --
    without that restriction the comparison would confound track with geometry.
    """
    out = POSTHOC / "hidden_decomposition_all.csv"
    if out.exists() and not force:
        return pd.read_csv(out, index_col=0)

    # Filter on the manifest FIRST and carry only the ~255k rows these three subsets
    # need. Merging all 943,110 rows of every score source before subsetting is what
    # makes this the one memory-hungry step in an otherwise trivial module.
    man = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet",
                          columns=["filename", "label", "partition", "dist", "trim_flag"])
    d4 = man["dist"].isin(["D4", "d4"])
    man = man[((man["partition"] == "eval") & d4) | (man["partition"] == "hidden")]
    man = man.reset_index(drop=True)

    df = pd.read_parquet(config.PA2021_LCNN_SCORES).merge(man, on="filename")
    if config.PA2021_CLASSICAL_SCORES.exists():
        df = df.merge(pd.read_parquet(config.PA2021_CLASSICAL_SCORES),
                      on="filename", how="left")
    df["y"] = (df["label"] == "bonafide").astype(int)

    for name, path in config.PA2021_BASELINE_SCORE_FILES.items():
        if not path.exists():
            print(f"  ! missing official baseline: {path}")
            continue
        # Whitespace-regex, not a literal space: RawNet2's published file carries a
        # trailing space on every line, which a single-character separator turns into
        # a third field and silently shifts the whole table by one column.
        s = pd.read_csv(path, sep=r"\s+", names=["filename", "score"],
                        dtype={"score": "float32"})
        assert s["filename"].is_unique, f"{name}: duplicate filenames in score.txt"
        df[name] = s.set_index("filename")["score"].reindex(df["filename"]).to_numpy()
        del s

    subsets = {
        "real": df["partition"] == "eval",
        "simulated": (df["partition"] == "hidden") & (df["trim_flag"] == "notrim"),
        "no non-speech": (df["partition"] == "hidden") & (df["trim_flag"] == "trim"),
    }

    systems = [c for c in list(config.PHASE7_LCNN_SYSTEMS)
               + list(config.PHASE7_CLASSICAL_SYSTEMS) + list(OFFICIAL)
               if c in df.columns]
    rows = {}
    for tag in systems:
        row = {}
        for label, mask in subsets.items():
            sub = df[mask & df[tag].notna()]
            if not len(sub) or sub["y"].nunique() < 2:
                continue
            eer, _ = metrics.compute_eer(sub.loc[sub["y"] == 1, tag].to_numpy(),
                                         sub.loc[sub["y"] == 0, tag].to_numpy())
            row[label] = 100.0 * eer
        rows[tag] = row

    res = pd.DataFrame(rows).T
    res.to_csv(out)
    print(f"  wrote {out}")
    return res


# --- figures --------------------------------------------------------------------

def fig_dev_vs_target(out: Path) -> Path:
    """The selection-criterion result: in-domain rank against out-of-domain rank."""
    t = pd.read_csv(PREREG / "eer_table_2021.csv")
    t = t[t["dev_eer_2019"].notna()].copy()
    t["dev"] = 100.0 * t["dev_eer_2019"]
    t["target"] = 100.0 * t["eer"]
    lcnn = t[~t["system"].str.startswith("MFCC")]

    rho = lcnn["dev"].rank().corr(lcnn["target"].rank())

    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for _, r in t.iterrows():
        is_lcnn = not r["system"].startswith("MFCC")
        ax.scatter(r["dev"], r["target"], s=52 if is_lcnn else 40,
                   color=_colour(r["system"]) if is_lcnn else C_OFFICIAL,
                   marker="o" if is_lcnn else "s", zorder=3,
                   edgecolor="white", linewidth=0.8)
        ax.annotate(r["system"], (r["dev"], r["target"]),
                    textcoords="offset points", xytext=(6, -1),
                    fontsize=7, color="#333333")
    # The trend over the seven networks only -- the classical pair sit an order of
    # magnitude away on the dev axis and would dominate any fit.
    # Fitted in log space because the axis is logarithmic: the dev EERs span an order
    # of magnitude, and a straight line in linear x would bend across the plot.
    z = np.polyfit(np.log10(lcnn["dev"]), lcnn["target"], 1)
    xs = np.logspace(np.log10(lcnn["dev"].min()), np.log10(lcnn["dev"].max()), 50)
    # Short label on purpose: the full statistic is in the prose, and the long form
    # used to run across the frame and collide with the MFCC-SVM point label.
    ax.plot(xs, np.polyval(z, np.log10(xs)), color=C_ZERO, linewidth=1.0,
            linestyle="--", zorder=2, label=f"CQT-LCNN  ρ = {rho:.2f}")

    ax.set_xlabel(_t("2019 dev EER (%) — in domain"))
    ax.set_ylabel(_t("2021 eval EER (%) — real replay"))
    ax.set_xscale("log")
    ax.set_xlim(0.6, 30)
    ax.grid(**GRID)
    ax.legend(loc="lower left", frameon=False)
    return _save(fig, out, "10_dev_vs_2021.png")


# The fused systems' internal tags mean nothing to a reader of the Serbian text, so
# they take the names the fusion table in the results chapter uses. Every other
# system tag stays as it is.
SR_SYSTEM_NAMES = {
    "inhouse_fusion_progress": "сопствена фузија, на ознакама",
    "inhouse_fusion_dev": "сопствена фузија, без ознака",
    "fusion_ours+2GMM": "фузија са два званична референтна система",
}


def _sysname(tag: str) -> str:
    return SR_SYSTEM_NAMES.get(tag, tag) if LANG == "sr" else tag


def fig_ci_forest(out: Path) -> Path:
    """Speaker-clustered intervals against the conventional trial-level ones."""
    d = pd.read_csv(POSTHOC / "bootstrap_ci_systems.csv")
    sp = d[d["scheme"] == "speaker-clustered"].set_index("system")
    tr = d[d["scheme"] == "trial-level"].set_index("system")
    order = sp.sort_values("eer").index.tolist()
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(6.2, 0.34 * len(order) + 1.4))
    for i, s in enumerate(order):
        r = sp.loc[s]
        ax.plot([r["eer_lo"], r["eer_hi"]], [i, i], color=_colour(s),
                linewidth=2.6, solid_capstyle="round", zorder=2)
        if s in tr.index:
            q = tr.loc[s]
            ax.plot([q["eer_lo"], q["eer_hi"]], [i, i], color="black",
                    linewidth=5.0, alpha=0.75, zorder=3)
        ax.scatter(r["eer"], i, color="white", edgecolor="black", s=16,
                   linewidth=0.7, zorder=4)

    ax.axvline(50, color=C_ZERO, linestyle=":", linewidth=1.0)
    ax.text(50, -0.75, _t(" chance"), color=C_ZERO, fontsize=7, va="center")
    ax.set_yticks(y, [_sysname(s) for s in order], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(_t("EER (%) with 95% confidence interval"))
    ax.grid(axis="x", **GRID)
    ax.plot([], [], color=C_OURS, linewidth=2.6, label=_t("speaker-clustered (67 speakers)"))
    ax.plot([], [], color="black", linewidth=5.0, alpha=0.75,
            label=_t("trial-level (721,332 trials)"))
    ax.legend(loc="upper right", frameon=False)
    return _save(fig, out, "11_ci_forest.png")


# Every paired comparison, each with a Serbian label. The nine fixed in writing before
# the held-out set was scored are drawn first under their own header and the post-hoc
# ones below a rule, which keeps all of them on one axis without blurring the
# separation the evaluation protocol rests on. Post-hoc labels name systems by tag
# where the tag is exact: timepool_T150_aug is the best post-hoc single system,
# flatten_T400 the pre-registered primary, CQCC-GMM the strongest official baseline.
REGISTERED_COMPARISONS = {
    "central claim: CQT-LCNN vs MFCC-SVM":
        "средишња тврдња: CQT-LCNN и MFCC-SVM",
    "headline: best vs best official baseline":
        "главно: најбољи и најјачи референтни",
    "front-end: CQT vs LFCC (both unaugmented)":
        "front-end: CQT и LFCC (без аугментације)",
    "T axis: 150 vs 400":
        "оса T: 150 и 400",
    "head: flatten vs timepool at T=400":
        "излазни степен: flatten и timepool (T=400)",
    "pred 1: shorter T transfers better":
        "предвиђање 1: краћи прозор",
    "pred 2a: mild waveform augmentation":
        "предвиђање 2a: блага аугментација",
    "pred 2b: aggressive waveform augmentation":
        "предвиђање 2b: јака аугментација",
    "pred 3: CMVN transfers better":
        "предвиђање 3: CMVN",
}
POSTHOC_COMPARISONS = {
    "in-house zero-shot fusion vs best official baseline (NOT circular)":
        "сопствена фузија без ознака и CQCC-GMM",
    "post-hoc: vs pre-registered primary":
        "timepool_T150_aug и flatten_T400",
    "post-hoc: vs best official baseline":
        "timepool_T150_aug и CQCC-GMM",
    "post-hoc: vs parent (T150, no augmentation)":
        "timepool_T150_aug и T150",
    "in-house fusion (label-fitted): vs the single system":
        "сопствена фузија на ознакама и појединачни",
    "in-house ZERO-SHOT fusion: vs the single system":
        "сопствена фузија без ознака и појединачни",
    "post-hoc fusion: vs the single system it contains":
        "фузија са два званична референтна и појединачни",
    "post-hoc: new best vs previous best":
        "timepool_T150_aug и flatten_T400_aug",
    "in-house zero-shot vs the borrowed-partner fusion":
        "сопствена и позајмљена фузија",
    "the price of 87,048 labelled target-domain trials":
        "цена 87.048 означених циљних покушаја",
}


def fig_paired_differences(out: Path) -> Path:
    """Every declared comparison as the CI of the paired difference."""
    d = pd.read_csv(POSTHOC / "bootstrap_ci_comparisons.csv")
    d = d[d["scheme"] == "speaker-clustered"]
    names = {**REGISTERED_COMPARISONS, **POSTHOC_COMPARISONS}
    unknown = set(d["comparison"]) - set(names)
    assert not unknown, f"comparison(s) without a label: {sorted(unknown)}"

    reg = d[d["comparison"].isin(REGISTERED_COMPARISONS)].sort_values("eer_diff")
    post = d[d["comparison"].isin(POSTHOC_COMPARISONS)].sort_values("eer_diff")
    rows = ([("head", _t("pre-registered comparisons"))]
            + [("cmp", r) for _, r in reg.iterrows()]
            + [("head", _t("post-hoc comparisons"))]
            + [("cmp", r) for _, r in post.iterrows()])

    fig, ax = plt.subplots(figsize=(6.8, 0.30 * len(rows) + 1.2))
    labels = []
    for i, (kind, r) in enumerate(rows):
        if kind == "head":
            labels.append(r)
            if i:
                ax.axhline(i - 0.5, color="#b5b5b5", linewidth=0.7, zorder=1)
            continue
        sig = bool(r["eer_excludes_zero"])
        col = C_OURS if sig else C_OFFICIAL
        ax.plot([r["eer_lo"], r["eer_hi"]], [i, i], color=col, linewidth=2.4,
                solid_capstyle="round", zorder=2)
        ax.scatter(r["eer_diff"], i, color="white", edgecolor=col, s=22,
                   linewidth=1.1, zorder=3)
        labels.append(names[r["comparison"]] if LANG == "sr" else r["comparison"])
    ax.axvline(0, color=C_ZERO, linewidth=1.1, zorder=1)

    ax.set_yticks(np.arange(len(rows)), labels, fontsize=7.5)
    for tick, (kind, _) in zip(ax.get_yticklabels(), rows):
        if kind == "head":
            tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel(_t("paired difference in EER (percentage points), 95% CI"))
    ax.grid(axis="x", **GRID)
    ax.plot([], [], color=C_OURS, linewidth=2.4, label=_t("excludes zero"))
    ax.plot([], [], color=C_OFFICIAL, linewidth=2.4, label=_t("spans zero"))
    ax.legend(loc="lower left", frameon=False)
    return _save(fig, out, "12_paired_differences.png")


def fig_hidden_tracks(out: Path) -> Path:
    """Real replay against the two hidden tracks, on matched positions."""
    d = build_hidden_decomposition()
    cols = ["real", "simulated", "no non-speech"]
    # The post-hoc systems were decomposed by score_posthoc into their own file, and
    # the best system in the project is among them -- it belongs on this figure.
    ph = POSTHOC / "posthoc_hidden_decomposition.csv"
    if ph.exists():
        p = pd.read_csv(ph, index_col=0).drop(index="n", errors="ignore")
        p.columns = cols[:p.shape[1]]
        d = pd.concat([d, p[~p.index.isin(d.index)]])
    d = d.dropna(subset=cols).sort_values("real")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    xs = np.arange(len(cols))
    tags = list(d.index)
    for tag in tags:
        aug = "aug" in tag
        ax.plot(xs, [d.loc[tag, c] for c in cols], marker="o", markersize=4.5,
                linewidth=2.2 if aug else 1.2, color=_colour(tag),
                alpha=1.0 if aug else 0.7, zorder=3 if aug else 2)

    # Several systems land within a point of each other on the right-hand edge, so
    # the labels are spread apart and joined back to their line by a hairline.
    ends = [float(d.loc[t, cols[-1]]) for t in tags]
    span = max(ends) - min(ends)
    # 0.075 rather than 0.055: at the tighter gap the five systems bunched in the
    # 50-54% band still overprinted each other.
    for tag, y_pt, y_lab in zip(tags, ends, _declutter(ends, span * 0.075)):
        col = _colour(tag)
        ax.plot([xs[-1], xs[-1] + 0.10], [y_pt, y_lab], color=col,
                linewidth=0.6, alpha=0.6, zorder=2)
        ax.text(xs[-1] + 0.13, y_lab, tag, fontsize=7, color=col, va="center")

    ax.axhline(50, color=C_ZERO, linestyle=":", linewidth=1.0)
    ax.text(-0.05, 50.6, _t("chance"), color=C_ZERO, fontsize=7)
    ax.set_xticks(xs, [_t("real replay\n(matched D4/d4)"), _t("simulated replay"),
                       _t("non-speech removed")])
    ax.set_xlim(-0.15, len(cols) - 1 + 0.95)
    ax.set_ylabel(_t("EER (%)"))
    ax.grid(axis="y", **GRID)
    return _save(fig, out, "13_hidden_tracks.png")


def _axis_panel(ax, sub, xcol, xlabel, logx=False):
    ax.plot(sub[xcol], sub["progress_eer"], marker="o", color=C_AUG,
            linewidth=1.8, markersize=5, label=_t("2021 progress (out of domain)"))
    ax.set_xlabel(_t(xlabel))
    ax.set_ylabel(_t("EER (%) — out of domain"), color=C_AUG)
    ax.tick_params(axis="y", labelcolor=C_AUG)
    if logx:
        ax.set_xscale("log")
    ax.grid(**GRID)
    twin = ax.twinx()
    twin.spines["right"].set_visible(True)
    twin.plot(sub[xcol], sub["dev_eer"], marker="s", color=C_OURS,
              linewidth=1.4, markersize=4, linestyle="--",
              label=_t("2019 dev (in domain)"))
    twin.set_ylabel(_t("EER (%) — in domain"), color=C_OURS)
    twin.tick_params(axis="y", labelcolor=C_OURS)
    return twin


def fig_dose_axis(out: Path) -> Path:
    """The dose sweep: one copy is the whole effect, and dev disagrees loudly."""
    d = build_axis_sweeps()
    sub = d[d["axis"] == "dose"].sort_values("p_clean", ascending=False)

    fig, ax = plt.subplots(figsize=(5.8, 3.9))
    twin = _axis_panel(ax, sub, "p_clean", "clean fraction of the training pool")
    ax.invert_xaxis()
    ax.set_xscale("log")
    ax.set_xticks(sub["p_clean"], [f"{v:g}" for v in sub["p_clean"]])

    # No annotation: the 0.103 pp span across the 8x dose range is stated in the
    # results prose, and a sentence drawn on the plot only repeats it.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = twin.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", frameon=False)
    return _save(fig, out, "14_dose_axis.png")


def fig_window_axis(out: Path) -> Path:
    """The window sweep: a bracketed U out of domain, monotone in domain."""
    d = build_axis_sweeps()
    w = d[d["axis"] == "window"].sort_values("n_frames")
    aug = w[w["augmented"]]
    clean = w[~w["augmented"]]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.6))

    ax.plot(aug["n_frames"], aug["progress_eer"], marker="o", color=C_AUG,
            linewidth=1.8, markersize=5, label=_t("augmented"))
    ax.plot(clean["n_frames"], clean["progress_eer"], marker="s", color=C_OURS,
            linewidth=1.4, markersize=4.5, linestyle="--", label=_t("unaugmented"))
    # The circle marks the minimum; that it is bracketed on both sides is the point
    # of the panel and is said in the prose, so no text is drawn here.
    best = aug.loc[aug["progress_eer"].idxmin()]
    ax.scatter([best["n_frames"]], [best["progress_eer"]], s=120, facecolor="none",
               edgecolor=C_ZERO, linewidth=1.2, zorder=4)
    ax.set_xlabel(_t("window length T (frames)"))
    ax.set_ylabel(_t("2021 progress EER (%)"))
    ax.set_title(_t("out of domain"))
    ax.grid(**GRID)
    ax.legend(frameon=False, loc="upper left")

    ax2.plot(aug["n_frames"], aug["dev_eer"], marker="o", color=C_AUG,
             linewidth=1.8, markersize=5, label=_t("augmented"))
    ax2.plot(clean["n_frames"], clean["dev_eer"], marker="s", color=C_OURS,
             linewidth=1.4, markersize=4.5, linestyle="--", label=_t("unaugmented"))
    ax2.set_xlabel(_t("window length T (frames)"))
    ax2.set_ylabel(_t("2019 dev EER (%)"))
    ax2.set_title(_t("in domain"))
    ax2.grid(**GRID)
    ax2.legend(frameon=False)
    return _save(fig, out, "15_window_axis.png")


def fig_fusion(out: Path) -> Path:
    """The fusion result, on both metrics, with what each system is made of."""
    ci = pd.read_csv(POSTHOC / "bootstrap_ci_systems.csv")
    ci = ci[ci["scheme"] == "speaker-clustered"].set_index("system")
    wanted = ["inhouse_fusion_progress", "inhouse_fusion_dev", "fusion_ours+2GMM",
              "timepool_T150_aug", "CQCC-GMM"]
    nice = {
        "inhouse_fusion_progress": "in-house fusion, label-fitted",
        "inhouse_fusion_dev": "in-house fusion, zero-shot",
        "fusion_ours+2GMM": "fusion with two official baselines",
        "timepool_T150_aug": "timepool_T150_aug (single)",
        "CQCC-GMM": "CQCC-GMM (best baseline)",
    }
    rows = [s for s in wanted if s in ci.index]
    y = np.arange(len(rows))

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for i, s in enumerate(rows):
        r = ci.loc[s]
        col = _colour(s)
        ax.barh(i, r["eer"], color=col, height=0.62)
        ax.plot([r["eer_lo"], r["eer_hi"]], [i, i], color="black", linewidth=1.1)
        ax.text(r["eer_hi"] + 0.7, i, f"{r['eer']:.2f}", va="center", fontsize=7.5)
        ax2.barh(i, r["min_tdcf"], color=col, height=0.62)
        ax2.plot([r["tdcf_lo"], r["tdcf_hi"]], [i, i], color="black", linewidth=1.1)
        ax2.text(r["tdcf_hi"] + 0.018, i, f"{r['min_tdcf']:.4f}", va="center",
                 fontsize=7.5)

    ax.set_yticks(y, [nice[s] for s in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("EER (%)")
    ax.set_xlim(0, 48)
    ax.grid(axis="x", **GRID)
    # The ASV floor is the cost a *perfect* countermeasure would still leave behind,
    # so it is the only meaningful zero on this axis.
    ax2.axvline(0.1291, color=C_ZERO, linestyle=":", linewidth=1.0)
    ax2.text(0.1291, -0.75, " ASV floor", color=C_ZERO, fontsize=7, va="center")
    ax2.set_xlabel("min t-DCF")
    ax2.set_xlim(0, 1.14)
    ax2.grid(axis="x", **GRID)
    return _save(fig, out, "16_fusion.png")


def fig_short_clip_control(out: Path) -> Path:
    """How much of the collapse is clip length rather than domain shift."""
    d = pd.read_csv(PREREG / "control_short_clips_2019dev.csv")
    caps = [("dev_eer_all", "all dev"), ("dev_eer_le250f", "≤250"),
            ("dev_eer_le200f", "≤200"), ("dev_eer_le150f", "≤150")]
    xs = np.arange(len(caps))

    # The two systems the argument turns on: the classical baseline, whose collapse
    # is being decomposed, and the pre-registered primary as the neural reference.
    focus = {"MFCC-SVM", "flatten_T400"}

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ends = []
    for _, r in d.iterrows():
        ys = [100.0 * r[c] for c, _ in caps]
        key = r["system"] in focus
        ax.plot(xs, ys, marker="o", markersize=4.5 if key else 3,
                linewidth=2.0 if key else 1.0, color=_colour(r["system"]),
                alpha=1.0 if key else 0.45, zorder=3 if key else 2)
        ends.append((r["system"], ys[-1], key))

    span = max(e[1] for e in ends) - min(e[1] for e in ends)
    for (tag, y_pt, key), y_lab in zip(ends, _declutter([e[1] for e in ends],
                                                        span * 0.055)):
        col = _colour(tag)
        ax.plot([xs[-1], xs[-1] + 0.06], [y_pt, y_lab], color=col, linewidth=0.6,
                alpha=0.6)
        ax.text(xs[-1] + 0.09, y_lab, f"{tag}  {y_pt:.1f}", fontsize=7, color=col,
                va="center", alpha=1.0 if key else 0.7)

    ax.set_xticks(xs, [lab for _, lab in caps])
    ax.set_xlim(-0.1, len(caps) - 1 + 0.75)
    ax.set_xlabel("2019 dev restricted to clips of at most this many frames")
    ax.set_ylabel("dev EER (%)")
    ax.grid(**GRID)
    return _save(fig, out, "17_short_clip_control.png")


def fig_registered_predictions(out: Path) -> Path:
    """Each registered prediction against the control that isolates its variable."""
    d = pd.read_csv(PREREG / "registered_predictions.csv")
    # Keep the registered wording, minus the parenthetical reasoning, which is long
    # and belongs in the text rather than on an axis.
    labels = [p.split("(")[0].strip() for p in d["prediction"]]
    y = np.arange(len(d))
    h = 0.34

    fig, ax = plt.subplots(figsize=(6.2, 0.78 * len(d) + 1.4))
    for i, r in d.iterrows():
        ax.barh(i - h / 2, 100 * r["eer_control"], height=h, color=C_OFFICIAL,
                label="control" if i == 0 else None)
        ax.barh(i + h / 2, 100 * r["eer_system"], height=h,
                color=C_OURS if r["supported"] else C_ZERO,
                label="registered system" if i == 0 else None)
        ax.text(100 * r["eer_control"] + 0.4, i - h / 2, r["control"],
                va="center", fontsize=7, color="#333333")
        ax.text(100 * r["eer_system"] + 0.4, i + h / 2,
                f"{r['system']}  ({r['delta_pp']:+.2f} pp, "
                f"{'supported' if r['supported'] else 'refuted'})",
                va="center", fontsize=7, color="#333333")

    ax.set_yticks(y, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("2021 eval EER (%)")
    ax.set_xlim(0, 66)
    ax.grid(axis="x", **GRID)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2)
    return _save(fig, out, "18_registered_predictions.png")


def fig_duration_strata(out: Path) -> Path:
    """Does the winning T move with the target clip duration? It does not."""
    d = pd.read_csv(POSTHOC / "duration_strata.csv")

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7))
    cmap = plt.get_cmap("viridis")
    for ax, arm in zip(axes, ["augmented", "unaugmented"]):
        a = d[d["arm"] == arm]
        strata = sorted(a["stratum"].unique())
        for s in strata:
            r = a[a["stratum"] == s].sort_values("T")
            med = r["median_frames"].iloc[0]
            col = cmap(0.12 + 0.76 * s / max(1, len(strata) - 1))
            lab = (f"медијана {med:.0f} окв." if LANG == "sr"
                   else f"median {med:.0f} f")
            ax.plot(r["T"], r["eer"], marker="o", markersize=4, linewidth=1.5,
                    color=col, label=lab)
            best = r.loc[r["eer"].idxmin()]
            ax.scatter([best["T"]], [best["eer"]], s=90, facecolor="none",
                       edgecolor=col, linewidth=1.4, zorder=4)
        ax.set_xlabel(_t("window length T (frames)"))
        ax.set_title(_t(f"{arm} arm"))
        ax.set_xticks(sorted(a["T"].unique()))
        ax.grid(**GRID)
    axes[0].set_ylabel(_t("2021 progress EER (%)"))
    axes[1].legend(frameon=False, fontsize=7, title=_t("duration stratum"),
                   title_fontsize=7, loc="upper left")
    # The circled point on each line is that stratum's best T. The account under test
    # predicts those circles marching rightwards as the strata get longer.
    return _save(fig, out, "19_duration_strata.png")


def fig_det_curves(out: Path) -> Path:
    """DET curves for the pre-registered pass.

    Rebuilt here rather than in report_2021, which owns the frozen pre-registered
    directory and would have to re-read the corpus-side parquets to redraw one
    picture. Everything needed is already on disk: the exported per-system
    score.txt files (the format published precisely so any number can be recomputed
    without the corpus) plus the manifest for the labels.
    """
    from scipy.stats import norm
    from sklearn.metrics import roc_curve

    man = pd.read_parquet(config.MANIFESTS_DIR / "pa2021_cm.parquet",
                          columns=["filename", "label", "partition"])
    man = man[man["partition"] == config.PA2021_REPORTED_PARTITION]
    y = (man["label"] == "bonafide").to_numpy().astype(int)
    idx = man["filename"]

    def read_scores(path: Path) -> np.ndarray:
        s = pd.read_csv(path, sep=r"\s+", names=["filename", "score"])
        return s.set_index("filename")["score"].reindex(idx).to_numpy()

    ours = [p for p in sorted((PREREG / "scores").glob("*.score.txt"))]
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ticks = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8])

    def draw(name, sc, **kw):
        ok = ~np.isnan(sc)
        fpr, tpr, _ = roc_curve(y[ok], sc[ok])
        fnr = 1 - tpr
        m = (fpr > 0) & (fnr > 0)
        ax.plot(norm.ppf(fpr[m]), norm.ppf(fnr[m]), label=name, **kw)

    for p in tqdm(ours, desc="DET", unit="sys", leave=False):
        tag = p.name.replace(".score.txt", "")
        draw(tag, read_scores(p), lw=1.6, color=_colour(tag), alpha=0.9)
    for name, path in config.PA2021_BASELINE_SCORE_FILES.items():
        if path.exists():
            draw(f"{name} (official)", read_scores(path), lw=1.2, ls="--",
                 color=C_OFFICIAL, alpha=0.75)

    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], c="k", lw=0.7, alpha=0.35, zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xticks(norm.ppf(ticks), [f"{t*100:g}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks), [f"{t*100:g}" for t in ticks])
    ax.set_xlabel(_t("False acceptance rate — spoof accepted (%)"))
    ax.set_ylabel(_t("False rejection rate — bonafide rejected (%)"))
    ax.grid(**GRID)
    ax.legend(fontsize=6.5, loc="upper right", frameon=False, ncol=1)
    return _save(fig, out, "08_det_curves_2021.png")


def fig_condition_breakdown(out: Path) -> Path:
    """Error rate per physical attack condition, from the frozen breakdown table.

    Laid out 2x4 rather than the original 1x7: across an A4 text block seven panels
    in a row leave each about 2.4 cm wide, which puts the level ticks below legible
    size. The factor codes stay as they are in the metadata, with the name spelled
    out beside them.
    """
    d = pd.read_csv(PREREG / "condition_breakdown.csv")
    factors = list(dict.fromkeys(d["factor"]))

    ncol = 4
    nrow = int(np.ceil(len(factors) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 3.0 * nrow),
                             squeeze=False)
    flat = [a for row in axes for a in row]
    for ax, f in zip(flat, factors):
        g = d[d["factor"] == f]
        within = g["convention"].iloc[0] == "within-group"
        ax.bar(g["level"].astype(str), g["eer"] * 100,
               color=C_OURS if within else C_AUG)
        ax.set_title(f"{_t(f)}\n({_t(g['convention'].iloc[0])})", fontsize=8.5)
        ax.set_ylabel(_t("EER (%)"))
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="y", **GRID)
    for ax in flat[len(factors):]:
        ax.axis("off")
    return _save(fig, out, "09_condition_breakdown.png")


FIGURES = {
    "det": fig_det_curves,
    "conditions": fig_condition_breakdown,
    "dev-vs-target": fig_dev_vs_target,
    "ci-forest": fig_ci_forest,
    "paired-diffs": fig_paired_differences,
    "hidden": fig_hidden_tracks,
    "dose": fig_dose_axis,
    "window": fig_window_axis,
    "fusion": fig_fusion,
    "short-clips": fig_short_clip_control,
    "predictions": fig_registered_predictions,
    "duration-strata": fig_duration_strata,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=POSTHOC,
                    help="canonical directory to write the .png files into "
                         "(default: results/phase7/posthoc/, tracked in git)")
    ap.add_argument("--only", nargs="+", choices=sorted(FIGURES),
                    help="generate only these figures")
    ap.add_argument("--force", action="store_true",
                    help="recompute the persisted intermediate tables")
    ap.add_argument("--no-thesis-copy", action="store_true",
                    help="skip mirroring the PNGs into the write-up's slike/")
    ap.add_argument("--lang", choices=("en", "sr"), default="en",
                    help="label language; 'sr' writes <name>_srp.png and mirrors "
                         "into Diplomski rad/slike/ instead")
    args = ap.parse_args()

    global LANG
    LANG = args.lang
    slike = SLIKE_SR_DIR if LANG == "sr" else SLIKE_DIR

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"figures ({LANG}) -> {args.out}")
    if not args.no_thesis_copy:
        slike.mkdir(parents=True, exist_ok=True)

    if args.force:
        build_axis_sweeps(force=True)
        build_hidden_decomposition(force=True)

    names = args.only or list(FIGURES)
    bar = tqdm(names, desc="figures", unit="fig")
    for name in bar:
        bar.set_postfix_str(name)
        path = FIGURES[name](args.out)
        msg = f"  {name:<14} -> {path.name}"
        if not args.no_thesis_copy and args.out != slike:
            shutil.copyfile(path, slike / path.name)
            msg += f"  (mirrored to {slike.name}/)"
        bar.write(msg)


if __name__ == "__main__":
    main()
