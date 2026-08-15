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

**Phases 0–6 complete** (environment setup, manifest building, speaker-disjoint
resplitting, EDA, DSP feature extraction, classical MFCC baseline, CQT-LCNN main
system). **Phase 7 is built and validated but has not been run** — the held-out 2021
set is still unscored.

| system | dev EER | ROC-AUC |
|---|---|---|
| MFCC-SVM (Phase 5 baseline) | 9.216% | 0.9684 |
| **CQT-LCNN (Phase 6 main system)** | **0.798%** | **0.9996** |

An **11.6x reduction** — the thesis's central claim, that constant-Q features retain
the replay fingerprint that mel-scaled features discard. These are *tuning* numbers on
the speaker-disjoint 2019 dev split, not generalisation estimates: the 2021 PA eval set
remains untouched until Phase 7, where **nine pre-registered systems** are scored in a
single pass against three predictions written down in advance (`PROJECT_PLAN.md`
phase 7). Dev has absorbed heavy selection pressure across Phases 5 and 6; 2021 will be
the only clean number.

## Repo layout

```
src/
  config.py          -- single source of truth for paths and hyperparameters
  manifest.py        -- Phase 1: raw protocol files -> clean parquet manifests
  resplit.py         -- Phase 2: speaker-disjoint train/dev resplit + ASV-enrollment enrichment
  eda.py             -- Phase 3: exploratory data analysis, writes plots to EDA/
  features.py        -- Phase 4: MFCC + CQT extraction, cached to E:\ASVspoof\features
  metrics.py         -- EER + supplementary metrics (Phase 5, shared with Phase 7)
  train_classical.py -- Phase 5: SVM/RF factorial sweep
  pack_features.py   -- Phase 6: pack per-file CQTs into one blob per split (31x faster I/O)
  augment_waveform.py-- Phase 6: pre-computed waveform augmentation copies
  datasets.py        -- Phase 6: torch Dataset + the shared windowing/normalisation
                        functions Phase 7 reuses, so dev and eval are scored identically
  models_lcnn.py     -- Phase 6: LCNN-9 with Max-Feature-Map activations
  train_lcnn.py      -- Phase 6: training loop, EER-scheduled, resumable
  evaluate_2021.py       -- Phase 7 pass 1: stream 2021, score 7 LCNNs, cache CQT+MFCC
  score_classical_2021.py-- Phase 7 pass 2: SVM/RF from the cached MFCC
  report_2021.py         -- Phase 7 pass 3: EER tables, DET curves, condition breakdown,
                            controls, verdicts on the registered predictions
EDA/            -- EDA plots and summaries (tracked in git; small, illustrative)
explanations/   -- teaching figures (e.g. how ROC-AUC relates to pairwise ranking)
results/        -- grouped per phase (phase5/{svm,rf}/, phase6/<run>/); summaries,
                   tables and figures tracked, bulky per-file score dumps gitignored
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
  python -m src.pack_features        # Phase 6 prerequisite
  python -m src.augment_waveform     # optional: waveform-augmented copies
  python -m src.train_lcnn --tag flatten_T400 --n-frames 400 --head flatten
  python -m src.evaluate_2021         # Phase 7 pass 1 (~6.2h, GPU + 8 CPU workers)
  python -m src.score_classical_2021  # Phase 7 pass 2 (~1.1h, CPU only)
  python -m src.report_2021           # Phase 7 pass 3 (seconds, re-runnable)
  ```
- The long-running phases (`features`, `train_classical`, `evaluate_2021`) are
  **resumable**: completed work is cached and skipped on re-run, so an interrupt costs
  at most the one item (or 4,000-file chunk) in flight. `train_classical` writes its
  sweep CSV after every single fit.
- Phase 7's two scoring passes are deliberately separate: libsvm scoring is
  single-threaded CPU work that would contend with pass 1's 8 extraction workers.
  Pass 1 caches the 2021 CQT (~12.7GB) and pooled MFCC (~500MB) under
  `E:\ASVspoof\phase7_2021\`, so re-scoring after any fix costs ~1.5h of GPU rather
  than a full re-extraction — insurance against finding a bug after a 6-hour pass.
