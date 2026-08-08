# PA Replay Countermeasure System

A thesis project on voice biometrics & anti-spoofing: a countermeasure (CM) system that
distinguishes bonafide (live human) speech from **replay-spoofed** speech, trained on
ASVspoof2019 PA and evaluated on the held-out ASVspoof2021 PA eval set (which contains
genuinely re-recorded replay audio, not just simulated attacks).

This README is a placeholder — it'll grow into a proper project README as the thesis
progresses. For now it just orients a new reader. Full planning context and reasoning
lives in [`PROJECT_PLAN.md`](PROJECT_PLAN.md); a detailed narrative of what's actually
been done so far (with numbers, decisions, and bugs found/fixed) lives in
[`PROGRESS_REPORT.md`](PROGRESS_REPORT.md).

## Status

**Phases 0–5 complete** (environment setup, manifest building, speaker-disjoint
resplitting, EDA, DSP feature extraction, classical MFCC baseline). Next up: Phase 6,
the CQT-LCNN main system.

Current best classical baseline: **MFCC-SVM (RBF), 9.216% EER** on the speaker-disjoint
2019 dev split — a *tuning* number, not a generalisation estimate. The held-out
ASVspoof2021 PA eval set remains untouched until Phase 7.

## Repo layout

```
src/
  config.py          -- single source of truth for paths and hyperparameters
  manifest.py        -- Phase 1: raw protocol files -> clean parquet manifests
  resplit.py         -- Phase 2: speaker-disjoint train/dev resplit + ASV-enrollment enrichment
  eda.py             -- Phase 3: exploratory data analysis, writes plots to EDA/
  features.py        -- Phase 4: MFCC + CQT extraction, cached to E:\ASVspoof\features
  metrics.py         -- EER + supplementary metrics (Phase 5, shared with Phase 7)
  train_classical.py -- Phase 5: SVM/RF factorial sweep, writes to results/
EDA/            -- EDA plots and summaries (tracked in git; small, illustrative)
explanations/   -- teaching figures (e.g. how ROC-AUC relates to pairwise ranking)
results/        -- summaries, aggregate tables and figures (tracked); bulky per-file
                   score dumps are gitignored
PROJECT_PLAN.md     -- full thesis plan, reasoning, and dataset breakdown
PROGRESS_REPORT.md  -- detailed log of work completed so far

manifests/, splits/, models/  -- generated artifacts (gitignored; reproducible by
  re-running the src/ scripts against the dataset)
```

## Setup

- Python 3.11 venv, currently a venv shared with another project
  (`OG/.venv`, sibling folder to this one) rather than a fresh one — see
  `PROJECT_PLAN.md` section 7 for why.
- Dataset lives at `E:\ASVspoof\data\` and the Phase 4 feature cache at
  `E:\ASVspoof\features\` — both configured in `src/config.py` (`DATA_ROOT`,
  `FEATURES_DIR`) and currently **absolute local paths** — update those constants if
  running on a different machine.
- Audio is decoded via `ffmpeg` (a portable binary bundled through the
  `imageio-ffmpeg` pip package, not a system install) rather than `soundfile` — see
  `PROGRESS_REPORT.md` Phase 4 for why (`soundfile`'s bundled `libsndfile` fails to
  decode ~46% of this corpus outright).
- Run any phase script as a module from the project root, e.g.:
  ```
  python -m src.manifest
  python -m src.resplit
  python -m src.eda
  python -m src.features
  python -m src.train_classical      # add --force to recompute instead of resuming
  ```
- The long-running phases (`features`, `train_classical`) are **resumable**: completed
  work is cached and skipped on re-run, so an interrupt costs at most the one item in
  flight. `train_classical` writes its sweep CSV after every single fit.
