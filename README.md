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

**Phases 0–7 complete.** The held-out ASVspoof2021 PA eval set has been scored once,
against nine systems pre-registered before it was touched, plus the four official
challenge baselines — 943,110 files, zero extraction failures.

### Headline result — 2021 PA eval (721,332 trials)

`min t-DCF` is the ASVspoof 2021 **primary** metric for PA; our implementation
reproduces all eight published baseline values exactly. A normalised t-DCF of 1.0 means
a countermeasure provides no benefit at all over a non-informative one.

| system | min t-DCF | 2021 EER | dev EER |
|---|---|---|---|
| **CQT-LCNN + waveform aug** (`flatten_T400_aug`) | **0.8347** | **32.665%** | 2.353% |
| CQT-LCNN, T=150 | 0.9019 | 34.420% | 6.584% |
| CQCC-GMM *(best official baseline)* | 0.9434 | 38.068% | — |
| LFCC-GMM *(official)* | 0.9724 | 39.540% | — |
| CQT-LCNN, best on dev (`flatten_T400`) | 0.9876 | 39.747% | 0.798% |
| LFCC-LCNN *(official)* | 0.9958 | 44.768% | — |
| RawNet2 *(official)* | 0.9997 | 48.605% | — |
| MFCC-SVM *(this project's classical baseline)* | **1.0000** | 49.635% | 9.216% |

**The measurement chain is verified against the published record.** Scoring the four
official baselines with our own code reproduces Liu et al. (IEEE/ACM TASLP 2023,
doi: 10.1109/TASLP.2023.3285283) Table XV **exactly** — all eight values, evaluation
and progress partitions, to ≤0.005 pp — plus two further exact matches against its
Table X on subsets defined by `dist` and `trim_flag`. Every number here is anchored to
the published challenge results.

Two things are true at once, and both belong in the abstract:

- **Our best system beats all four official baselines** — 5.40 pp / 14.2% relative over
  CQCC-GMM. Five of our seven CQT-LCNNs beat every official baseline. Against
  **LFCC-LCNN**, which shares our backbone and differs mainly in front-end, the margin
  is **12.10 pp / 27% relative** — the cleanest available test of the thesis's claim
  that constant-Q features retain a replay fingerprint mel-scaled features discard.
- **Everything degrades enormously on real replay.** The best in-domain model went from
  0.798% dev EER to 39.747%. At 32.7% EER the best system is not deployable. The 2021 PA
  task defeats the entire published baseline set; ours is defeated less.

### What the held-out set changed

Evaluating out of domain **reversed the project's conclusions**, which is the main
methodological finding:

- The **best model on dev is the second worst on 2021**.
- **Waveform augmentation looked monotonically harmful in-domain** (0.798 → 1.486 →
  2.353%) and is the single most valuable technique out of domain (39.747 → 34.006 →
  32.665%). A dev-only thesis would have discarded it.
- **Three of four pre-registered predictions were supported**, one cleanly refuted
  (CMVN transfers *worse*, not better).
- Only **~20% of the classical baseline's collapse is mechanical** (clip-length pooling
  noise, measured by a control built before the results were known); the rest is genuine
  simulated→real domain shift.
- On the challenge's **simulated** hidden track, every official baseline and every
  non-augmented system degrades — while the augmented models *improve*, reaching
  **26.47% EER**, the best figure this project reaches anywhere on 2021. The challenge
  organisers observed this effect correlationally across whole systems; here it is a
  controlled experiment with augmentation as the only variable.

Placed against the challenge itself, the best system would rank **11th of 24** entries —
**the same placement on min t-DCF as on EER**, ahead of all four baselines (best in the
PA track was 0.6824 / 24.25%). One caveat travels with that claim: these systems trained
on ~3.3× the permitted data, including a partition the rules excluded, so they are **not
challenge-compliant**.

Full analysis in [`PROGRESS_REPORT.md`](PROGRESS_REPORT.md) §7.10–7.20; options for what
to do next in [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §9.

Post-Phase-7 follow-up work is under way and is recorded **separately** from the
pre-registered results, because it was designed after 2021 numbers were seen and carries
no such guarantee — see the "Post-Phase 7" section of the progress report and
[`PROJECT_PLAN.md`](PROJECT_PLAN.md) §9.3.1 for the declared decision rule.

**2021 eval is now spent.** It is a clean generalisation estimate precisely because
nothing was tuned on it; future work develops against the `progress` partition instead
(`PROJECT_PLAN.md` §9.0).

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
results/        -- grouped per phase (phase5/{svm,rf}/, phase6/<run>/, phase7/);
                   summaries, tables and figures tracked, bulky per-file score dumps
                   gitignored. phase7/ holds the final EER table, DET curves, condition
                   breakdown, both controls and the verdicts on the registered
                   predictions
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
