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

**Phases 0–3 complete** (environment setup, manifest building, speaker-disjoint
resplitting, EDA). Next up: Phase 4, DSP feature extraction (MFCC + CQT).

## Repo layout

```
src/
  config.py     -- single source of truth for paths and hyperparameters
  manifest.py   -- Phase 1: raw protocol files -> clean parquet manifests
  resplit.py    -- Phase 2: speaker-disjoint train/dev resplit + ASV-enrollment enrichment
  eda.py         -- Phase 3: exploratory data analysis, writes plots to EDA/
EDA/            -- EDA plots and summaries (tracked in git; small, illustrative)
PROJECT_PLAN.md     -- full thesis plan, reasoning, and dataset breakdown
PROGRESS_REPORT.md  -- detailed log of work completed so far

manifests/, splits/, features/, models/, results/  -- generated artifacts (gitignored;
  reproducible by re-running the src/ scripts against the dataset)
```

## Setup

- Python 3.11 venv, currently a venv shared with another project
  (`OG/.venv`, sibling folder to this one) rather than a fresh one — see
  `PROJECT_PLAN.md` section 7 for why.
- Dataset paths are configured in `src/config.py` (`DATA_ROOT`) and are currently
  **absolute local paths** — update that constant if running on a different machine.
- Run any phase script as a module from the project root, e.g.:
  ```
  python -m src.manifest
  python -m src.resplit
  python -m src.eda
  ```
