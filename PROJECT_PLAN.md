# PA Replay Countermeasure System — Full Thesis Project Plan & Context

This file is a detailed carry-over of a planning conversation. It's written to be read
cold in a new chat, so it includes reasoning and numbers, not just conclusions — the
point is to lose as little context as possible when switching chats.

---

## 1. The project decision, and how we got here

Thesis topic: **Voice Biometrics & Anti-Spoofing**, specifically a countermeasure (CM)
system that distinguishes bonafide (live human) speech from replay-spoofed speech — i.e.
detecting when a recording of someone's voice is played back through a speaker to try to
fool a voice-based system.

Two other directions were explicitly considered and rejected:
- **ASVspoof 5 (2024) / synthetic-speech-deepfake framing**: rejected because ASVspoof 5
  has no dedicated physical-replay track anymore (LA and DF got merged into a single
  TTS/VC/adversarial-attack track). Pursuing it would mean reframing the whole thesis
  around synthesis-artifact detection (SSL front-ends like wav2vec2/XLS-R + AASIST
  backend) instead of the "speaker/mic fingerprint" story, which is heavier
  compute-wise and less demo-friendly (no "play my own recorded voice into the mic and
  watch it get flagged" moment).
- **Training only on ASVspoof 2019 PA and evaluating only on it too**: rejected in favor
  of using 2019 for train/dev and reserving **2021 PA eval** as the real held-out test,
  because 2021 contains genuinely real re-recorded replay audio (see section 3), which is
  a much stronger empirical claim than reporting results only on simulated data.

Also discussed and partly planned as bonus/extension features (not the core thesis, add
only if time allows): a live demo combining anti-spoofing + keyword spotting ("Unlock"
passphrase via pretrained ASR) + lightweight speaker verification (self-enrolled voice
print + cosine similarity), wrapped in a Gradio/Streamlit UI with `sounddevice` for mic
capture. Full detail in section 6, phase 9.

---

## 2. Where everything lives

- Dataset root: `E:\ASVspoof data\` (kept separate from code/venv because C: drive was
  nearly full — see section 7 for the disk situation and how it was resolved).
  - `E:\ASVspoof data\ASVspoof2019_PA\`
  - `E:\ASVspoof data\ASVspoof2021_PA_eval\`
- Code/project root: `C:\Users\Luka\OneDrive\Радна површина\PythonProject\PA Replay Countermeasure System\` (this file's location).
- Python environment: `C:\Users\Luka\OneDrive\Радна површина\PythonProject\OG\.venv`
  (Python 3.11) — this is the venv being reused for this thesis rather than creating a
  fresh one, since it already had numpy/scipy/pandas/matplotlib/torch etc. See section 7
  for exactly what's installed and a critical gotcha with this venv's `pip.exe`.

---

## 3. Dataset structure — full breakdown

### 3.1 ASVspoof2019_PA

Top-level layout (from `README.txt` / `README.PA.txt` inside the dataset):
```
ASVspoof2019_PA/
  asvspoof2019_evaluation_plan.pdf
  README.txt
  PA/
    README.PA.txt
    ASVspoof2019_PA_train/flac/          -- 54,000 files (PA_T_*.flac)
    ASVspoof2019_PA_dev/flac/            -- 33,534 files (PA_D_*.flac)
    ASVspoof2019_PA_eval/flac/           -- 153,522 files (PA_E_*.flac)
    ASVspoof2019_PA_cm_protocols/        -- CM (countermeasure) labels
      ASVspoof2019.PA.cm.train.trn.txt   -- 54,000 lines
      ASVspoof2019.PA.cm.dev.trl.txt     -- 29,700 lines
      ASVspoof2019.PA.cm.eval.trl.txt    -- 134,730 lines
    ASVspoof2019_PA_asv_protocols/       -- ASV (speaker verification) protocols
      ASVspoof2019.PA.asv.{dev,eval}.{male,female}.{trn,trl}.txt
    ASVspoof2019_PA_asv_scores/          -- baseline ASV system scores (t-DCF eval)
```

**Why dev/eval flac counts don't match CM protocol line counts** — this took some
digging and matters a lot: `PA_dev/flac` has 33,534 files but the CM dev protocol only
lists 29,700. Same story for eval: 153,522 files vs 134,730 CM lines. The difference
(3,834 for dev, 18,792 for eval — 22,626 total) is **ASV-enrollment-only recordings**:
extra bonafide utterances used to build speaker voiceprints for the ASV task, which are
listed in the `.trn` enrollment files, not the CM protocol at all. This was verified
directly: summing the comma-separated file lists inside
`ASVspoof2019.PA.asv.{dev,eval}.{male,female}.trn.txt` gives exactly 3,834 and 18,792,
which reconciles precisely with the flac-count vs CM-protocol-line-count gap. Train has
no such gap (54,000 flac = 54,000 CM lines) because train has no separate ASV
enrollment set.

**CM protocol format** (`SPEAKER_ID AUDIO_FILE_NAME ENVIRONMENT_ID ATTACK_ID KEY`):
- `SPEAKER_ID`: e.g. `PA_0079`
- `ENVIRONMENT_ID`: a triplet like `aaa`, encoding (S, R, D_s) each as a/b/c:
  - S = room size: a=2-5m², b=5-10m², c=10-20m²
  - R = T60 reverberation time: a=50-200ms, b=200-600ms, c=600-1000ms
  - D_s = talker-to-ASV-mic distance: a=10-50cm, b=50-100cm, c=100-150cm
- `ATTACK_ID`: a duple like `AA`, encoding (attacker-to-talker distance, replay device
  quality), or `-` for bonafide. Replay device quality bins (perfect/high/low) are
  defined by occupied bandwidth / lower-bound frequency / linearity specs in the README.
- `KEY`: `bonafide` or `spoof`.
- Checked and confirmed: the **same 9 attack-ID combinations (AA through CC) appear in
  train, dev, AND eval** — unlike the LA track (TTS/VC), PA doesn't have an
  unknown-attack-algorithm generalization test built in, because the attack space here is
  a physical parameter grid (distance × device quality), not different synthesis
  algorithms. All combinations were simulated in every split.
- Bonafide/spoof counts confirmed by direct count:
  - train: 5,400 bonafide / 48,600 spoof
  - dev: 5,400 bonafide / 24,300 spoof
  - eval: 18,090 bonafide / 116,640 spoof
  - Combined CM pool: 28,890 bonafide / 189,540 spoof → **~6.6:1 spoof:bonafide
    imbalance**.

**ASV protocol format**:
- Enrollment (`.trn`): `SPEAKER_ID_ENV FILE1,FILE2,...` (e.g.
  `PA_0073_aaa PA_D_A000001,PA_D_A000002,...`) — a speaker enrolled per environment
  condition, using ~19 files each.
- Trial (`.trl`): `CLAIMED_SPEAKER_ID TEST_FILE ENV_ID ATTACK_ID KEY` where KEY is
  `target` (genuine claimed speaker), `nontarget` (different real speaker, impostor), or
  `spoof` (a spoofing attack played against that speaker's enrollment).
- Baseline ASV scores in `ASVspoof2019_PA_asv_scores/*.scores.txt`: format
  `CM_KEY ASV_KEY SCORE` — precomputed similarity scores from a baseline ASV system, for
  t-DCF computation if the ASV extension gets built.

**CM vs ASV — the conceptual distinction** (came up as its own question, worth keeping
verbatim):
- CM = "is this audio bonafide or spoofed", full stop, no identity involved. This is the
  actual thesis task.
- ASV = "does this voice match the identity claiming to speak", which needs enrollment
  (a voiceprint built from known-genuine recordings) then trial scoring against it.
- Both exist together in this dataset because the real-world threat model the challenge
  targets is a full voice-login pipeline (CM gate + ASV match), scored jointly via
  t-DCF. For this thesis: CM first, ASV only as an optional later extension (reusing the
  enrollment-file concept for the bonus self-enrolled speaker verification feature too —
  same idea, just self-recorded instead of dataset-provided).

Audio format for all of 2019 PA: FLAC, 16kHz, 16-bit.

Sizes on disk (measured): `ASVspoof2019_PA` folder = **18GB**.

### 3.2 ASVspoof2021_PA_eval

Layout — split across 7 parts because of file-size limits on the distribution:
```
ASVspoof2021_PA_eval/
  ASVspoof2021_PA_eval_part00/ASVspoof2021_PA_eval/flac/   -- 150,000 files
  ASVspoof2021_PA_eval_part01/ASVspoof2021_PA_eval/flac/   -- 150,000 files
  ASVspoof2021_PA_eval_part02/ASVspoof2021_PA_eval/flac/   -- 150,000 files
  ASVspoof2021_PA_eval_part03/ASVspoof2021_PA_eval/flac/   -- 150,000 files
  ASVspoof2021_PA_eval_part04/ASVspoof2021_PA_eval/flac/   -- 150,000 files
  ASVspoof2021_PA_eval_part05/ASVspoof2021_PA_eval/flac/   -- 125,745 files
  ASVspoof2021_PA_eval_part06/ASVspoof2021_PA_eval/
    flac/                                                  -- 67,365 files
    ASVspoof2021.PA.cm.eval.trl.txt   -- filenames ONLY, no labels (943,110 lines)
    README.PA.txt
    LICENSE.txt
  PA-keys-full/keys/PA/                -- downloaded SEPARATELY from Zenodo audio
    README.txt
    CM/
      trial_metadata.txt               -- the REAL labels, 943,110 lines
      CQCC-GMM/score.txt                -- official baseline scores (943,110 lines)
      LFCC-GMM/score.txt
      LFCC-LCNN/score.txt
      RawNet2/score.txt
    ASV/
      trial_metadata.txt                -- 2,508,570 lines (ASV trial protocol)
      ASVTorch_Kaldi/score.txt          -- official ASV baseline scores
    PA-C012-{eval,hidden1,hidden2,prog}.npy   -- precomputed C0/C1/C2 cepstral coeffs
```
Total flac files across all 7 parts = **943,110**, exactly matching the 943,110 lines in
both the bare-filename protocol file and the real `trial_metadata.txt` — confirmed by
direct count, nothing missing or extra.

**Why the audio-side protocol file has no labels**: `ASVspoof2021.PA.cm.eval.trl.txt`
(inside part06) is deliberately just a filename list, so nobody training a model can
accidentally peek at the answer key. The actual ground truth (`bonafide`/`spoof`) lives
only in `PA-keys-full/keys/PA/CM/trial_metadata.txt`, which was released separately
(from `asvspoof.org`, not the Zenodo audio host) for challenge-integrity reasons. Always
join on filename against this file to get real labels.

**`trial_metadata.txt` format**: `SPEAKER FILE R{room} M{mic} d{dist} r m s c LABEL
TRIM_FLAG PARTITION` — e.g. `PA_0010 PA_E_1000001 R3 M3 d4 r1 m1 s4 c4 spoof notrim eval`.
This is a genuinely different/richer code scheme than 2019's `aaa`-style triplets,
because it's describing **real recording setups** (real rooms, real mics, real replay
devices), not simulation parameters.

**Partition column matters** — checked the actual distribution, it's not homogeneous:
- `eval` = 721,332 rows — the official scored/leaderboard subset. **Use this one for the
  headline reported EER.**
- `progress` = 87,048 rows — an early subset released during the live challenge.
- `hidden` = 134,730 rows — a held-out portion not used in main scoring.
- Confirmed all four official baseline `score.txt` files (CQCC-GMM, LFCC-GMM, LFCC-LCNN,
  RawNet2) cover the full 943,110 rows (all partitions), so filtering to `partition ==
  "eval"` afterward is safe and stays apples-to-apples with the baselines.
- Trim-flag column: `notrim` = 875,745, `trim` = 67,365 (minor detail, not critical).
- ASV `trial_metadata.txt` (separate file, 2,508,570 rows) has its own partition split:
  `eval` = 1,975,104, `hidden` = 253,530, `progress` = 279,936.

**What the baseline `score.txt` files and folder names actually mean** (this was asked
explicitly, keeping the full answer):
| | front-end | back-end |
|---|---|---|
| **B01 CQCC-GMM** | **CQCC** — Constant-Q Cepstral Coefficients: take the CQT, log it, then DCT to decorrelate into cepstral coefficients (+ deltas) | **GMM**, generative — one Gaussian mixture fitted to bonafide frames, one to spoof; score is the log-likelihood ratio. Frame-level, no temporal modelling |
| **B02 LFCC-GMM** | **LFCC** — like MFCC but with *linearly*-spaced filters instead of mel. Keeps the high-frequency resolution mel compresses away, which is exactly why it beats MFCC on replay | GMM, as above |
| **B03 LFCC-LCNN** | LFCC | **LCNN-LSTM** — Light CNN plus a recurrent stage. *Not* a plain LCNN; see 9.4 for what that costs the front-end comparison |
| **B04 RawNet2** | **none** — raw waveform, sinc-filter front-end learned end to end | deep residual net + GRU |

The two GMM baselines are classical and pre-deep-learning; B03 shows the classical→deep
jump on *identical* features; B04 is the most modern of the four. Which of them are worth
rebuilding in-house, and why the cheapest one is not the most useful one, is 9.8c.
- Each `score.txt` is one row per file: `FILENAME SCORE` (a continuous score, higher
  usually meaning more bonafide-like, sign/scale conventions differ slightly per system
  — check when computing EER against them).
- `ASV/ASVTorch_Kaldi/score.txt`: baseline ASV system (Kaldi-based) scores, only needed
  if the ASV/t-DCF extension gets built.
- `PA-C012-*.npy`: precomputed low-order (C0/C1/C2) cepstral coefficients, optional,
  not required.

Audio format: FLAC, 16kHz, 16-bit, same as 2019.

Size on disk (measured): `ASVspoof2021_PA_eval` folder = **45GB**.

### 3.3 The simulated-vs-real distinction, explained mechanically

This came up as its own detailed question and is central to the thesis narrative, so the
full mechanical explanation is worth keeping:

**2019 PA is simulated**: nobody played a real speaker through a real mic in a real
room. Room-acoustics simulation software (README cites Roomsimove / a MATLAB "shoebox"
room model, plus a swept-sine measurement technique for device response) mathematically
convolves clean VCTK studio speech with:
- a simulated room impulse response for a given room size (S) and reverberation time
  (R=T60),
- a simulated loudspeaker frequency response of a given quality tier (Q: perfect/high/
  low, defined by bandwidth/lowest-frequency/linearity numbers),
- specified talker-to-mic and attacker-to-talker distances.

Each parameter is discretized into 3 categorical levels (a/b/c), giving a **controlled,
reproducible factorial grid** — which is exactly why the same attack-ID grid (AA-CC)
shows up identically in train/dev/eval (see 3.1). There's no real acoustic surprise here:
no mic self-noise character, no real room echo flutter/resonances outside the shoebox
model, no real speaker nonlinearity/doppler, etc.

**2021 PA eval is real**: someone physically played the speech through an actual
loudspeaker in an actual room and re-recorded it with an actual microphone, varying real
rooms/mic placements/replay devices. The richer `R{room} M{mic} d{dist} r m s c` coding
scheme (vs 2019's `aaa` triplets) reflects this — it's describing real setups, not
simulation dial positions. This is the first point in the ASVspoof PA lineage where the
"speaker diaphragm leaves a fingerprint that CQT can see" story is backed by genuinely
physical replay rather than a physics-inspired simulation of it.

**Should simulated-only training data be a worry?** Conclusion reached: yes a little,
but it's a known, accepted, and actually *useful* limitation, not a flaw:
- It's literally the setup the ASVspoof organizers designed on purpose (train 2019,
  generalization-test 2021), so "does a model trained on simulated replay generalize to
  real replay" is the intended research question, not something introduced by mistake.
- Honest thesis framing: report EER on 2019 dev (in-domain/simulated) AND on 2021 eval
  (out-of-domain/real) side by side. A gap between the two is itself a legitimate,
  citable, interesting result — not a failure.
- If the 2021 number is meaningfully worse than the 2019 number, that mirrors the
  documented outcome from the actual ASVspoof 2021 challenge, and is discussable/citable
  as such, not something to be defensive about in a thesis defense.

---

## 4. Reshuffling 2019 PA — decision and reasoning

Question raised: since 2021 PA eval is huge (943,110 files) and will serve as the sole
held-out test set, is it safe/useful to fold 2019's own eval split into training, since it
won't be needed as eval anymore?

**Conclusion**: yes, but do it properly (resplit by speaker), and there's a better,
free bonus discovered while investigating this:

1. Concatenate ALL of 2019's CM-labeled data (train+dev+eval, 218,430 rows total) **plus**
   the 22,626 ASV-enrollment-only bonafide files identified in section 3.1 (these were
   sitting completely unused by the CM task — zero evaluation-contamination risk from
   including them, since they were never part of any CM eval to begin with).
   - Before enrichment: 28,890 bonafide / 189,540 spoof (~6.6:1).
   - After adding the 22,626 enrollment files (all bonafide): 51,516 bonafide / 189,540
     spoof (~3.7:1) — meaningfully better balance, for free.
2. Resplit the combined pool by **speaker** (not by row) — e.g.
   `sklearn.model_selection.GroupShuffleSplit(groups=df.speaker_id)`, roughly 75/25 of
   speakers into train/dev. Speaker-disjointness matters: without it the model could
   learn to recognize individual voices instead of replay artifacts, which is what the
   original split was protecting against and shouldn't be thrown away casually.
3. 2019 no longer needs its own separate "eval" split at all — **2021 PA eval becomes the
   only true held-out test set**, touched exactly once, at the very end.
4. Residual imbalance (3.7:1) handled via `class_weight="balanced"` (SVM/RF) and
   `pos_weight` in `BCEWithLogitsLoss` (LCNN) — not via further oversampling/undersampling
   on top of the enrichment, unless dev EER later indicates it's actually necessary.

**What NOT to do**: don't just dump 2019 eval's spoof files into training while leaving
dev as-is — that creates a mismatch where dev no longer represents the same speaker/attack
distribution as what the model trained on, undermining the validity of hyperparameter
decisions made by watching dev EER.

---

## 5. Dev vs eval — how to actually use them

- **Train**: fit model parameters.
- **Dev** ("development"/validation): used *during* development — tune hyperparameters,
  pick decision threshold, decide when to stop training, compare architectures. Can be
  looked at repeatedly and reacted to.
- **Eval** (final test): touched **once**, at the very end, to report the final number.
  Tuning anything based on eval results contaminates it and invalidates it as a
  generalization estimate — flagged as the single most common methodological mistake in
  student ML projects, and something examiners specifically probe for.

Applied to this project: 2019 train (resplit, enriched) → fit models. 2019 dev (resplit)
→ tune feature settings/threshold/epochs/architecture choices. 2021 PA eval
(`partition=="eval"` subset) → the one true final number, computed once.

---

## 6. Full roadmap

### Feature combination decision (with reasoning, not just the pick)

| Track | Front-end | Backend | Role |
|---|---|---|---|
| Baseline | MFCC (mean/std-pooled stats vector) | SVM + Random Forest | classical, cheap, deliberately weak on replay |
| Main system | CQT log-power spectrogram ("CQTgram") | LCNN (Light CNN, Max-Feature-Map activations) | the thesis's real contribution |
| Optional 3rd point (only if time allows) | CQCC | GMM or SVM | exact reproduction of the official `CQCC-GMM` baseline |

Why LCNN over ResNet-18: LCNN (Max-Feature-Map activation instead of ReLU) is the
literature-standard lightweight architecture for this exact task — won ASVspoof 2017 PA,
top-performing on 2019 PA, and is literally one of the four official 2021 baselines
(`LFCC-LCNN`). Using it gives two clean comparison axes for free: same backend/different
front-end (this project's CQT-LCNN vs the official LFCC-LCNN baseline, isolating the
effect of CQT specifically) and same front-end family/different backend (LFCC-GMM vs
LFCC-LCNN, showing the classical→deep jump). ResNet-18 would work too but has no
PA-specific pedigree and is heavier to train on a 4GB-VRAM GPU (see section 7) than LCNN.

Why MFCC as the deliberately weak baseline (not CQCC): MFCC's mel filterbank compresses
exactly the high-frequency region where loudspeaker/microphone electromechanical
artifacts live, so MFCC→SVM is *expected* to underperform — and that gap becomes the
thesis's central empirical argument ("perceptually-motivated features miss the
fingerprint; constant-Q features find it"). CQCC would blur that contrast since it's
already CQT-based, so it's kept as a stretch-goal third data point rather than the main
baseline. Also worth noting practically: `librosa` has no built-in CQCC (unlike MFCC and
CQT, which it does have), so CQCC requires a third-party implementation or a port — a
real time-cost reason to keep it optional.

Explicitly NOT pursued, but worth a paragraph in the Related Work chapter about why not:
raw-waveform end-to-end (RawNet2-style) and SSL front-ends (wav2vec2/XLS-R) are current
state-of-the-art but need much heavier compute than a 2-month solo thesis on a 4GB GPU
can comfortably absorb — natural "future work" material, ties back to the ASVspoof5/SSL
discussion in section 1.

### Phase 0 — Environment & project skeleton
```
project/
├── data/                        # already exists on E: (ASVspoof2019_PA, ASVspoof2021_PA_eval)
├── manifests/                   # csv/parquet file lists — output of Phase 1
├── features/                    # cached .npy/.h5 feature arrays — output of Phase 4
├── splits/                      # resplit train/dev csvs — output of Phase 2
├── models/                      # saved .pkl (SVM/RF) and .pt (LCNN) checkpoints
├── results/                     # EER tables, plots, score files
├── src/
│   ├── config.py                # single source of truth for all paths/params
│   ├── manifest.py               # Phase 1
│   ├── resplit.py                # Phase 2
│   ├── features.py               # Phase 4
│   ├── datasets.py               # torch Dataset classes
│   ├── models_classical.py       # SVM/RF wrappers
│   ├── models_lcnn.py            # LCNN architecture
│   ├── train.py
│   ├── evaluate.py               # EER, DET, confusion matrix, F1 etc
│   └── demo/                     # Phase 9
└── notebooks/                    # EDA, plotting, scratch work
```
`config.py` should hold `DATA_ROOT = Path("E:/ASVspoof data")` and every
feature/hyperparameter (sample rate, n_mfcc, CQT bins, fixed frame length, batch size) —
never hardcode these inline.

Libraries needed in `OG/.venv` (see section 7 for exact current state and the install
command to use): `numpy pandas librosa soundfile scikit-learn torch torchaudio matplotlib
seaborn tqdm joblib` now; `faster-whisper sounddevice gradio` later for Phase 9.

### Phase 1 — Data loading & manifest building
Goal: turn the scattered protocol `.txt` files into one clean pandas table per dataset.

1.1 — Parse 2019 PA CM protocols (5 columns, whitespace-separated):
```python
def load_2019_cm_protocol(path, split, flac_dir):
    cols = ["speaker_id", "filename", "env_id", "attack_id", "label"]
    df = pd.read_csv(path, sep=" ", names=cols)
    df["filepath"] = df["filename"].apply(lambda f: flac_dir / f"{f}.flac")
    df["split"] = split          # "train" / "dev" / "eval"
    df["year"] = 2019
    return df
```
Run for train/dev/eval, `pd.concat`.

1.2 — Parse the 22,626 ASV-enrollment-only files (`.trn` files, format
`SPEAKER_ID FILE1,FILE2,...`): explode the comma-separated lists, mark every one
`label="bonafide"`, `source="asv_enrollment"`. Keep as a separate DataFrame until Phase 2.

1.3 — Parse 2021: the audio-side protocol (part06) is filenames only — the real labels
are in `PA-keys-full/keys/PA/CM/trial_metadata.txt` (11 columns as described in section
3.2). Build a filename→path lookup dict once across all 7 parts' `flac/` folders (files
are split across parts, so this lookup is needed before resolving `filepath`).

1.4 — Filter 2021 to `partition == "eval"` for the final reported number (see section 3.2
for why — `progress`/`hidden` are additional material, not the headline subset).

1.5 — Keep `env_id`/`attack_id` (2019) and `R/M/d/r/m/s/c` (2021) columns in the manifest
even though they're not training signal — costs nothing to keep, and becomes the basis
for an EER-by-condition breakdown later (Phase 7) for almost no extra work.

### Phase 2 — Reshuffling (full reasoning already in section 4)
Save result as `splits/train_2019.csv` / `splits/dev_2019.csv`, never touch the raw
protocol files again downstream.

### Phase 3 — Exploratory data analysis
- Class balance bar chart, before/after Phase 2 enrichment.
- Duration histogram (use `soundfile.info()` for cheap duration without full decode).
- Waveform + spectrogram plot, one bonafide vs one spoof example side by side.
- Speaker count/gender balance per split.
- Sanity-check: listen to 5-10 files per label to rule out a label-swap bug (flagged
  early on as an easy place to introduce a bug given the 2019/2021 protocol formats
  differ).

### Phase 4 — DSP feature extraction

4.1 — Shared preprocessing: all files already 16kHz/16-bit, no resampling needed. 20-30ms
frames, 50% overlap, Hamming window (implement once by hand in week 1 to demonstrate DSP
understanding, then switch to library calls for actual pipeline speed via
`librosa.feature.mfcc` / `librosa.cqt`, which handle framing/windowing internally via
`n_fft`/`hop_length`). Fixed-length handling: pick a fixed frame count (e.g. ~400 CQT
frames ≈ 4 seconds), pad by repeating if shorter, crop (random crop while training,
center crop for eval) if longer.

4.2 — MFCC pipeline (for SVM/RF), pooled to a fixed-length stats vector since classical
models need fixed-length input:
```python
mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20, n_fft=512, hop_length=160)
delta1 = librosa.feature.delta(mfcc)
delta2 = librosa.feature.delta(mfcc, order=2)
feat = np.concatenate([mfcc, delta1, delta2], axis=0)   # (60, T)
vec = np.concatenate([feat.mean(axis=1), feat.std(axis=1)])   # fixed 120-dim vector
```

4.3 — CQT pipeline (for LCNN). **Corrected during implementation — see the two notes
below; the original sketch here specified 96 bins and a fixed 400-frame cache, and
neither survived contact with the data:**
```python
C = librosa.cqt(y=y, sr=16000, hop_length=256, n_bins=90, bins_per_octave=12)
cqtgram = librosa.amplitude_to_db(np.abs(C), ref=np.max, top_db=80)  # log-power CQT
# NOTE: cached at NATURAL length. The pad/crop to a fixed T happens in Phase 6's
# Dataset, not here -- see 4.3b.
```

**4.3a — `n_bins` must be 90, not 96.** 96 bins at 12 bins/octave from the default
fmin (~32.7Hz, C1) violates the Nyquist limit at 16kHz and `librosa.cqt` raises
`ParameterError`. This is subtle: the top bin's *centre* frequency (~7.9kHz) looks
safely under 8kHz, but the wavelet's own bandwidth pushes past it. Confirmed
empirically that 96 fails and 90 does not. 90 bins = 7.5 octaves.

**4.3b — cache at natural length; do the fixed-length windowing in Phase 6.** The
plan's "random crop while training, center crop for eval" is incompatible with
baking a single crop in at cache time — the crop would happen once and be reused
every epoch, which is not random and throws away the augmentation benefit. So Phase
4 caches each file's CQT unpadded and uncropped, and Phase 6's `Dataset` pads short
clips / randomly (train) or centrally (dev/eval) crops long ones on every access.
No length cap is applied: a full scan of all 241,056 files found only 0.748% exceed
10s, and capping there would have saved ~17MB out of ~6.1GB.

**4.3c — `CQT_FIXED_FRAMES`: the initial call was ~250; the Phase 6 sweep overturned
it and the project uses 400. Original reasoning and outcome both kept below.**

> **OUTCOME.** Measured on dev: T=150 → 6.584%, T=250 → 2.780%, **T=400 → 0.902%**.
> Context wins by a 7.3x margin. T=150 *underfits* (its train EER is also poor),
> which rules out the competing "T=150 gets more crop diversity" explanation, and the
> padding-asymmetry confound was quantified and dismissed (duration alone scores
> 41.5% EER — near chance). The mismatch concern below is **not refuted**, just
> located on the wrong axis: it is a *transfer* risk, invisible in-domain and
> unmeasurable without touching 2021, so it is handled by pre-registration (see
> Phase 7) rather than by picking a compromise value.

Original reasoning — measured on the real cache:
400 frames is a 6.40s window, but 2019 CQTs have a median of **267** frames and
**90.8% are shorter than 400**; 2021 eval projects to ~149 frames (2.39s mean), its
longest file barely reaching 439. Keeping 400 would mean padding 2019 by ~1.50x and
2021 by ~2.68x — a train/test mismatch of our own making on top of the real domain
shift — and would leave ~91% of training files receiving deterministic padding
instead of the per-epoch random crops that 4.3b exists to enable. **Set it to ~250
frames (4.0s)**, near the 2019 median so roughly half the training set genuinely
gets cropped and 2021's padding drops to ~1.7x. Better still, treat it as a Phase 6
hyperparameter tuned on dev — it is consumed only at Dataset load time, so changing
it costs nothing and requires no re-extraction.

4.4 — Caching strategy, with the disk-space math behind it: extracting features once and
caching to disk is essential given 943,110 files in 2021 alone — decoding FLAC every
epoch would be far too slow. But the storage format matters a lot:
- MFCC pooled vectors are tiny (~480 bytes/file) regardless of float32/uint8 — caching
  the full enriched training pool (241,056 files) costs only ~116MB. Not a concern.
  **(Actual: 171MB as a single consolidated parquet, including identifier columns.)**
- CQT spectrograms, **uint8, quantized dB values** (same idea as storing any grayscale
  image — no meaningful loss for training purposes since the CNN already treats this as
  an image). float32 would be 4x larger for no benefit.
  **Actual cost, at natural variable length rather than a fixed 96×400: 6.3GB for the
  241,056-file pool** — less than the 8.4GB this section originally estimated, because
  most clips are shorter than a 400-frame window rather than needing padding up to it.
- ~~**Do NOT cache CQT features for the 2021 eval set at all.**~~ **REVERSED in Phase 7
  — the 27GB figure was wrong.** The original estimate assumed 96 bins × 400 *padded*
  frames per file. Neither holds: `n_bins` is 90 (the 4.3a Nyquist fix), and features
  are cached at *natural* length (4.3b). Measured mean on 2021 is **149.4 frames**, so
  the real cost is `90 × 149.4 × 943,110 = ~12.7GB` for **all** partitions — against
  31.7GB free on E:. Less than half the number the decision was made on.

  It is now cached, and the justification is **insurance, not speed**. The dominant
  risk in Phase 7 is discovering a bug *after* a ~4h pass (a wrong crop, a
  normalisation mismatch, a model rebuilt at the wrong T). Cached, re-scoring all nine
  systems costs ~1.5h of GPU and no CPU; uncached it is another full extraction. The
  write is nearly free, since the arrays already pass through the main process on
  their way to the GPU. Stored as one blob shard per chunk (a monolithic blob would
  need a ~25GB merge at the end), with a `shard` column in the index, and verified
  byte-for-byte against fresh extractions exactly as `pack_features.py` does.

  Honest caveat: a cached eval set lowers the friction to "just try one more thing on
  2021". That is a discipline problem, not a technical one, and the pre-registration
  above is what guards it.

- **Pooled MFCC for 2021 IS cached** (~500MB for all partitions — 0.05% of the CQT
  figure that drove the original decision). It decouples classical scoring from the
  extraction pass, so libsvm does not contend with the 8 extraction workers for cores,
  and makes the classical scores re-runnable without re-extraction.
- **Phase 7 must extract once and score BOTH models in the same pass.** Measured in
  Phase 5: scoring all 721,332 eval files with the tuned SVM costs only ~0.7h
  (3.56 ms/file at 37,943 support vectors), whereas *extracting* their features costs
  ~4h at Phase 4's rates. Running separate streaming passes for the SVM and the LCNN
  would pay that ~4h twice for no reason.

4.5 — **Decoding: use ffmpeg for every file, not `soundfile`.** ~46% of the 2019 corpus
fails to decode with `soundfile`'s bundled `libsndfile` 1.2.2 (`flac decoder lost sync`
/ `unknown error in flac decoder`) — confirmed at scale on a 500-file sample, and
root-caused to a decoder limitation, not file corruption (raw headers check out fine).
`librosa.load`'s `audioread` fallback does not help either, since there is no system
ffmpeg. Fixed by installing `imageio-ffmpeg` (a pip package bundling a portable static
ffmpeg binary — no admin rights needed) and decoding everything through it uniformly.
One consistent decode path for the whole corpus is a stronger methods-section claim
than a try-soundfile-then-fall-back hybrid, and avoids numerical inconsistency between
two decoders. Cost ~76ms/file for decode; the full 241,056-file extraction completed
with **zero failures**.

### Phase 5 — Classical ML baseline
1. Load pooled MFCC vectors + labels from `splits/train_2019.csv`.
2. `StandardScaler` fit on train only, applied to dev (avoid leaking dev statistics).
3. `SVC(kernel="rbf", class_weight="balanced")` — grid-search `C`/`gamma` on dev only.
   **Correction to the original sketch**: this said `probability=True`, on the grounds
   that EER "needs a continuous score not a hard label". True, but `decision_function`
   already supplies one. EER is read off the ROC curve and is therefore purely
   *rank-based*, so any monotonically-ordered score works. `probability=True` triggers
   an internal 5-fold Platt-scaling CV at ~5x the fit cost, and since Platt scaling is
   monotonic it cannot change the EER anyway. Use `decision_function`.
   **Implemented result** (see PROGRESS_REPORT.md): a full factorial sweep of 7
   subsample sizes x 4 C x 4 gamma = 112 fits. Best is C=1.0, gamma≈0.008-0.01 on the
   full train split → 9.216% dev EER. RBF SVM cost scales ~O(n^2.2) here, which is why
   size is swept rather than assumed.
4. Also fit `RandomForestClassifier(class_weight="balanced")` — gives feature
   importances nearly for free (nice extra plot: "which MFCC coefficients matter most").
5. Save both with `joblib.dump`.

### Phase 6 — Deep learning main system (highest-risk phase, budget slack)
LCNN architecture — Max-Feature-Map is the non-standard piece, implemented as a custom
activation:
```python
class MFM(nn.Module):
    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        return torch.max(a, b)

class LCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2), MFM(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 96, 3, padding=1), MFM(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2),
            # ... continue per the LCNN paper's block pattern ...
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(C, 1))
    def forward(self, x):
        return self.head(self.net(x)).squeeze(-1)   # raw logit
```
(Follow the original LCNN paper's exact layer sizes rather than improvising — it's a
known-good recipe.) Training essentials: `BCEWithLogitsLoss(pos_weight=...)` set from the
train-split class ratio; Adam, lr ~1e-3 to 1e-4, `ReduceLROnPlateau` scheduled on **dev
EER, not dev loss** (they don't always move together under class imbalance); batch size
limited by the 4GB GPU (start at 32, adjust — see section 7); 10-20 epochs realistic;
checkpoint on best dev EER.

**Augmentation — upgraded from "optional" to important, with a concrete plan.** The
Phase 5 learning curve is direct evidence: an RBF SVM on 120 pooled features never
plateaued, still gaining 0.147 pp at the final step to the *entire* train split. There
is no more 2019 data, and the LCNN has far more parameters to feed.

But quantity is the weaker argument. The stronger one is the **simulation shortcut**.
2019 PA is entirely simulated (section 3.3): every spoof passed through the same small
family of synthetic shoebox RIRs and device-response curves — a highly regular,
low-entropy signature. A high-capacity CNN can reach excellent 2019 dev EER by keying
on "this has shoebox-convolution structure" rather than on the loudspeaker's
electromechanical fingerprint, and that shortcut collapses on 2021, where the replay is
physically real. Augmentation's real job here is to make the simulation-specific cues
**unreliable as discriminators**, forcing the model back onto the physics that actually
transfers. Each perturbation stands in for something the simulator omits:

| augmentation | missing physics it stands in for |
|---|---|
| additive noise (varied SNR) | real mic self-noise |
| extra/varied reverb, real RIRs | room behaviour beyond the shoebox model |
| mild clipping / nonlinear distortion | real loudspeaker nonlinearity |
| codec simulation (MP3/AAC/GSM), RawBoost-style filtering | real recording-chain processing |
| time shift / random crop | position invariance; blocks onset-timing shortcuts |

**The constraint**: Phase 4 caches *CQT spectrograms*, not waveforms, which splits
augmentation into two very different cost classes.

- *Spectrogram-domain* (random crop, SpecAugment time/frequency masking, mixup, gain
  offsets) applies directly to the cached uint8 arrays — essentially free, fresh every
  epoch.
- *Waveform-domain* (noise, reverb, codec, RawBoost) must be applied **before** the CQT,
  so the cache cannot help. Either (a) re-decode and re-CQT every epoch — ~1h per epoch
  at Phase 4's measured 155ms/file on 8 cores, so ~20h per training run — or (b)
  pre-compute a few augmented copies once.

**Decision: spectrogram-domain augmentation + option (b), with (a) held in reserve.**
Rationale: (a)'s cost is paid *per run* and Phase 6 will involve 5-10 runs (tuning
`CQT_FIXED_FRAMES`, lr, architecture, mask widths), whereas (b) is a one-time ~3-4h.
The performance gap is also smaller than it first appears, because the two compose:
under either option the model never sees a byte-identical input twice, since fresh
random crops and SpecAugment masks are applied on top every epoch. The fixed waveform
copies only need to supply *enough distinct samples of the missing physics* to stop any
single one being a reliable cue — and 1→3 samples captures most of that, with 3→∞ on
the flat part of the curve.

Concretely: **3 augmented copies, training split only** (dev/eval are always evaluated
clean, so there is no reason to augment the other 65,097 files). At ~4.6GB per
training-only copy that is ~13.7GB, comfortable against the ~28GB free on E:.

**Decide (a) empirically rather than by guesswork**: train with (b), then watch train
EER versus dev EER. If dev plateaus while train keeps improving, the model is
memorising the fixed copies and more variety would genuinely pay — then add copies
(cheap) or reconsider (a). If they track each other, (a) would have bought nothing.
This is the same evidence-first approach used for the Phase 5 learning curve and the RF
tree count, and it costs one training run that was happening anyway.

**Caveat to test, not assume**: SpecAugment's frequency masking could mask exactly the
high-frequency band this thesis argues carries the replay fingerprint. The literature
does use it successfully for anti-spoofing, but treat mask width as a dev-tuned
hyperparameter rather than copying ASR defaults — and the result is worth reporting
either way.

### Phase 7 — Evaluation
**PRE-REGISTERED SYSTEMS — fixed before 2021 is touched.** 2021 PA eval is scored
ONCE. To keep that honest, the systems and their roles are declared here in advance,
so nothing is selected post-hoc on eval results. All are scored in a **single
extraction pass** (extract → score every model → discard), since the ~4h of feature
extraction dominates and per-model scoring is ~0.7h.

**AMENDED before the eval run, and before any 2021 score existed** (the amendment
is recorded here rather than in the progress report precisely so it cannot be
mistaken for a post-hoc addition). Three systems were added, all of them trained
and frozen in Phase 5/6 — none is a new model, and none was chosen by looking at
2021:

- **`T400` (timepool, unit) — the matched control, and the important one.** As
  originally written this table could not test two of its own three predictions.
  `cmvn_T400` and `baseline_T250` are both *timepool* models while the primary
  `flatten_T400` is *flatten*, so comparing either against the primary varies **two**
  things at once — norm *and* head for prediction 3, T *and* head for prediction 1.
  `T400` is timepool/unit, so `cmvn_T400` differs from it only in normalisation and
  `baseline_T250` only in T. Without it, predictions 1 and 3 are confounded and no
  honest verdict on either is possible.
- **`T150`** completes the T axis (150/250/400, all timepool) for ~5 min of GPU.
- **MFCC-RF**, the Phase 5 Random Forest, scored from the cached MFCC in ~6 min.

| system | role | dev EER |
|---|---|---|
| `flatten_T400` | **primary** — best on dev | 0.798% |
| `T400` (timepool) | **matched control** — makes predictions 1 and 3 testable | 0.902% |
| `cmvn_T400` | hypothesis: may transfer better despite worse in-domain | 1.293% |
| `flatten_T400_aug1` | mild waveform augmentation (50% clean) | 1.486% |
| `flatten_T400_aug` | aggressive waveform augmentation (25% clean) | 2.353% |
| `baseline_T250` | robustness — smallest 2019→2021 padding mismatch | 2.780% |
| `T150` | completes the T axis | 6.584% |
| MFCC-SVM | Phase 5 classical baseline | 9.216% |
| MFCC-RF | Phase 5 classical baseline | 11.736% |

The primary system remains `flatten_T400` regardless of outcome. Adding systems
cannot bias the headline as long as the headline is not later reselected from among
them — and it is fixed here, in advance.

**Disclosure for the defense.** A 600-file smoke test was run through the full
pipeline before the real pass, to validate that it worked end to end. It produced
2021 numbers over ~0.06% of the corpus. Those numbers selected nothing: all nine
systems were already frozen and registered (in `src/config.py`) before it ran, the
primary was already fixed, and no threshold, hyperparameter or model choice depends
on them. Recorded because "did you look at eval before finalising?" deserves a
documented answer rather than a reassurance.

**Predictions stated in advance** (so either outcome is a result, not a rationalisation):
1. **T=250 may beat T=400 on 2021** despite losing on dev, because T=400 tiles 2021
   clips ~2.7x versus ~1.4x on 2019 — the largest train/test padding mismatch.
2. **Augmentation may help on 2021 despite hurting monotonically on dev.** It targets
   the simulation shortcut, and dev is *also* simulated 2019 data, so the in-domain
   cost is expected. This is the only place that hypothesis becomes testable.
3. **CMVN may transfer better** for the same reason it hurt in-domain: it removes
   recording-specific channel offsets, which plausibly differ most between simulated
   and real replay.

**Implemented in `src/metrics.py` (written in Phase 5, shared with Phase 7).** It fixes
one convention project-wide: **higher score = more bonafide**, with 1 = bonafide as the
positive class — matching the official baseline `score.txt` orientation so 7.4's
comparison table joins with no sign flipping.

**Two things to carry into Phase 7 from Phase 5:**
- **Dev EER is a tuning number, not a generalisation estimate.** 112 SVM configurations
  plus 7 RF fits were selected against dev; best-of-112 carries roughly 0.2 pp of
  selection optimism (the top-5 spread). 2021 eval is genuinely untouched and remains
  the only clean number. Phase 6 adds further dev evaluations on top.
- **Expect a mechanical component to any MFCC-SVM degradation on 2021.** Pooled MFCC
  statistics average over ~453 frames on 2019 but only ~239 on 2021, and the standard
  error of a pooled statistic scales as 1/sqrt(T) — so 2021's features are inherently
  ~1.38x noisier before any question of replay realism arises. Testable by restricting
  2019 dev to short clips and watching EER rise; worth separating from the domain-shift
  story rather than attributing all of it to simulated-vs-real.

7.1 — EER (primary metric, computed identically for every system compared):
```python
def compute_eer(bonafide_scores, spoof_scores):
    scores = np.concatenate([bonafide_scores, spoof_scores])
    labels = np.concatenate([np.ones_like(bonafide_scores), np.zeros_like(spoof_scores)])
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fnr[idx] + fpr[idx]) / 2
    return eer, thresholds[idx]
```
Report on: (a) resplit 2019 dev — tuning/sanity number, (b) 2021 PA eval
(`partition=="eval"` only) — the headline number.

7.2 — Confusion matrix at the EER threshold (state explicitly in the thesis that this
specific operating point was chosen deliberately — it's a design decision).

7.3 — Precision/Recall/F1/Accuracy/ROC-AUC at that same threshold, as the more intuitive
supplementary numbers (official baselines are usually only reported in EER, so frame
these as supplementary, EER as primary).

7.4 — Baseline comparison table — the strongest chart available: compute EER the same
way on the four official `score.txt` files, joined against `trial_metadata.txt` labels,
filtered to `partition=="eval"`, alongside this project's own MFCC-SVM / CQT-LCNN
numbers:

| System | EER (2021 PA eval) |
|---|---|
| CQCC-GMM (official) | ... |
| LFCC-GMM (official) | ... |
| LFCC-LCNN (official) | ... |
| RawNet2 (official) | ... |
| MFCC-SVM (this project, baseline) | ... |
| **CQT-LCNN (this project, main)** | ... |

7.5 — Condition breakdown using the metadata kept from Phase 1.5: EER split by room
size, mic distance, replay device quality (a `groupby` + bar chart) — turns "does the
model survive real physical replay conditions" into a specific, evidenced answer instead
of one aggregate number.

**Two conventions are needed, because half these factors are spoof-only.** Verified
directly against `trial_metadata.txt`: `room` (R1-R9) and `mic` (M1-M3) are populated
for bonafide *and* spoof, but `r`/`m`/`s`/`c` are `-` on every bonafide row — they
describe the replay device, and a bonafide recording was never replayed through
anything. `dist` splits by class too (`D1-D6` bonafide, `d1-d6` spoof), so the two
are different quantities and must not share an axis.

- **`room`, `mic` → within-group EER**, using that group's own bonafide and spoof.
  Answers "how does the system perform in room R6", which is the stronger question.
- **replay-device factors → pooled-bonafide EER**: all bonafide against each
  condition's spoof. A group with no bonafide has no FRR curve and therefore no EER
  at all, so some pooling is forced. Pooling *all* of them makes FRR(θ) literally the
  same function in every group and lets only FAR(θ) move, so a difference between
  conditions is attributable purely to how detectable that condition's attacks are,
  and cannot be an artefact of one group holding easier genuine speech.
  **Caveat to state**: the shared bonafide half makes these EERs statistically
  *correlated*, so no test assuming independent groups may be applied to them.
- Alongside both, **FAR at the single global EER threshold** is reported — no pooling
  convention to explain, and it reads directly as "this share of that condition's
  attacks got through at the operating point we report".

### Phase 8 — ASV extension (optional, only after CM is solid)
1. Reuse ASV `.trn` enrollment files (2019) / `ASV/trial_metadata.txt` (2021) already
   parsed in Phase 1.
2. Build speaker voiceprints from enrollment utterances (mean MFCC embedding for
   consistency with the existing pipeline, or a pretrained x-vector/ECAPA embedding for
   stronger separation).
3. Score trials by cosine similarity/PLDA against the claimed speaker's voiceprint.
4. t-DCF (if wanted): combines CM and ASV miss/false-alarm rates with cost/prior weights
   into one scalar. The official baseline ASV scores exist specifically so this project's
   CM scores can be plugged into that formula without building an ASV system from
   scratch. Treat as "if week 7 has slack," not core-path.

### Phase 9 — Bonus real-time features + live demo (last 1-2 weeks, strict timebox)
1. **Keyword spotting**: `faster-whisper` tiny/base model, transcribe a short recorded
   clip, normalize (`lower().strip()`) and compare to the passphrase. No training needed.
2. **Speaker verification (self-enrollment)**: record the passphrase 5-10 times, extract
   MFCC (reuse Phase 4.2 code), average into a voiceprint vector, cosine-similarity
   threshold at inference. Upgrade option if time allows: swap in a pretrained embedding
   (`resemblyzer` or SpeechBrain ECAPA-TDNN) for cleaner separation — still zero training.
3. **Live demo wiring**: `sounddevice.rec()` captures mic audio → CQT-LCNN CM model gate
   (spoof/bonafide) → if bonafide, Whisper KWS → if passphrase matches, cosine-similarity
   speaker check → grant/deny. Wrap in Gradio (`gr.Blocks`, `gr.Audio(source="microphone")`
   input, colored `gr.Textbox` output) — under 50 lines.
4. Calibrate the live threshold separately from the dataset EER threshold — the laptop's
   own mic won't match the dataset's recording conditions, expect to nudge this by ear
   during rehearsal.

### Thesis chapter mapping
Introduction (biometric security, replay threat) → Related Work (ASVspoof
2015/2017/2019/2021/5 evolution, section 1's reasoning) → DSP Theory (framing,
windowing, MFCC vs CQT math, why CQT suits replay) → Data (dataset description, the
simulated-vs-real distinction from section 3.3, the resplit rationale from section 4) →
Methodology (SVM/RF + LCNN architecture, training setup) → Results (EER table vs
official baselines, confusion matrix, F1/precision/recall, condition breakdown, DET
curves) → Discussion (2019→2021 domain-shift gap, what it means) → Extensions (ASV/
t-DCF if done, KWS+SV+demo) → Conclusion & Future Work (SSL front-ends, ASVspoof5,
real-time robustness).

---

## 7. Environment & disk state (as resolved during planning)

### Venv contents (`PythonProject\OG\.venv`, Python 3.11), checked directly
Already installed before this thesis started: `torch`, `torchvision`, `numpy`, `scipy`,
`pandas`, `matplotlib`, `sympy`, `networkx`, `scikit-image`, plus Jupyter/notebook stack,
plus unrelated geo-course packages (`rasterio`, `geopandas`, `pyproj` — leftover from a
geoinformatics course, harmless to ignore).

**Missing, still to install**: `librosa`, `scikit-learn`, `soundfile`, `gradio` or
`streamlit`, `faster-whisper`, `sounddevice`. (`torchaudio` is already present via the
CUDA torch install below.)

**GPU / CUDA — resolved**: laptop GPU is an **NVIDIA GeForce GTX 1650 (4GB VRAM)**,
driver 555.99 supporting up to CUDA 12.5 (confirmed via `nvidia-smi`). Originally the
venv had `torch 2.8.0+cpu` (CPU-only build, `torch.cuda.is_available()` was `False`) —
not a hardware/driver problem, just the wrong wheel had been installed. Fixed by
reinstalling with:
```
python.exe -m pip uninstall -y torch torchvision torchaudio
python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Result, confirmed directly: `torch 2.5.1+cu121`, `torch.cuda.is_available() == True`,
`torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1650"`. **Keep LCNN training batch
sizes modest (≤32) given the 4GB VRAM ceiling.**

**Critical gotcha, confirmed directly — always remember this**: `OG\.venv\Scripts\pip.exe`
is broken. It has a stale hardcoded interpreter path from whenever this venv was
copied/relocated, and silently redirects to a **completely unrelated venv**:
`E:\Fakultet\3. godina RUS\geoinformatika\PythonProject\.venv` (a Geoinformatika course
project). This was discovered the hard way: an initial attempt to reinstall CUDA torch via
`OG\.venv\Scripts\pip.exe` actually uninstalled/reinstalled torch in that unrelated venv
instead, with no error or warning. The fix and the rule going forward: **never invoke
`Scripts\pip.exe` directly in this venv — always use `Scripts\python.exe -m pip ...`**,
since the venv's own `python.exe` binary correctly resolves its own site-packages (this
was verified directly), unlike the separately-generated `pip.exe` launcher script. The
Geoinformatika venv's accidental CUDA upgrade was deliberately left in place (decided not
worth reverting) — its torch there is now also `2.5.1+cu121` (was `2.8.0`, presumably
CPU), with `sympy` correspondingly at `1.13.1` (was `1.14.0`) due to torch's pinned
dependency. Not expected to cause problems for that course project, but noted here in
case it ever does.

### Disk cleanup performed
Starting point: **C: drive had only 6.35GB free (97% full)**, E: had 31.68GB free.
Investigated a user-reported mismatch (Explorer Properties showed the profile at ~66GB,
manually summing visible folders in `C:\Users\Luka` only reached ~20GB) — resolved: it
was `AppData`, which is hidden by default in Explorer and was never included in the
manual folder-by-folder sum. Direct measurement: `C:\Users\Luka` total = 57GB (after the
user had already freed ~7-8GB before this was measured), of which `AppData` alone =
**44-45GB** (Local 35GB + Roaming 8.5GB) — by far the dominant piece.

Cleaned up (all confirmed executed and verified):
- `pip` cache: 5,869MB (1,508 files) purged via `pip cache purge`.
- `AppData\Local\Temp`: cleared (excluding the active Claude session's own temp
  subfolder, so nothing mid-task was disrupted).
- NVIDIA shader cache (`Local\NVIDIA\DXCache`, `GLCache`, and all of `LocalLow\NVIDIA`):
  removed — regenerates automatically on next GPU use.
- `Local\Downloaded Installations`: removed (leftover installer files).
- Dead torch/tensorflow remnants inside a separate, non-venv global install at
  `AppData\Roaming\Python\Python311\site-packages`: removed `~orch` (a folder Windows
  renames to `~name` when it can't fully delete during a failed uninstall — this was the
  remains of a broken `torch` uninstall), `tensorflow`, `tensorflow-2.20.0.dist-info`,
  `torch-2.10.0.dist-info`, `torchgen`, `functorch`. (That whole Roaming\Python\Python311
  folder is a legacy `pip install --user` setup predating the current per-project-venv
  workflow — the rest of it, e.g. `scipy`/`cv2`/`spacy`/`sklearn`/`numpy` sitting there,
  was left alone since it wasn't explicitly flagged for removal.)
- `AppData\Local\MathWorks` (MATLAB per-user data): removed.
- **Docker Desktop and the WSL Ubuntu distro were deliberately left untouched** — actively
  used (one container, in constant recent use), so no pruning/compacting was done despite
  `docker_data.vhdx` being a large (5.7GB) virtual disk that in general doesn't
  auto-shrink.

**Result, measured directly**: C: went from **6.35GB → 27.56GB free** (~21.2GB freed,
more than the itemized estimate suggested, likely because the actual Temp folder
contents were larger than the earlier partial listing had shown). E: unchanged at
31.68GB free (nothing on that drive was touched).

### Other things flagged during the disk audit but NOT acted on (still there, for later judgment)
- `miniconda3` (4.9GB) — a separate conda installation, redundant if the venv-per-project
  workflow is what's actually used going forward.
- `.julia` (1.5GB) — only relevant if Julia is actually used.
- Two system-wide Python installs (3.13.5 and 3.11.9) — explained by the user: 3.11 was
  needed at the time because PyTorch (or possibly TensorFlow) didn't support the newer
  Python version yet. Small on their own (~150-160MB each), not a real space concern.
- Java 5 Runtime Environment (2005-era) — flagged as almost certainly unused by anything
  modern, but not removed.
- Oracle VirtualBox (234MB installed, but `VirtualBox VMs` folder is empty — no actual VM
  disk images exist).
- Various game/app data (Wargaming.net, Steam, Epic Games + Launcher, bakkesmod, Discord,
  Mediatonic) — left entirely alone, personal-use judgment calls, not flagged as
  "definitely unused."
- Bright VPN, WFDownloaderApp, snapdownloader-updater — small, flagged only as
  "worth confirming you still use these," not removed.

### Dataset sizes on disk (measured directly, for reference)
`ASVspoof2019_PA` = 18GB, `ASVspoof2021_PA_eval` = 45GB, both on `E:\ASVspoof data\`.

---

## 8. Immediate next action *(historical — completed in Phase 0)*

Install the remaining libraries into `OG\.venv` — **using `python.exe -m pip install
...`, never `pip.exe` directly** (see the gotcha above) — then begin Phase 1
(manifest-building code): `librosa scikit-learn soundfile gradio faster-whisper
sounddevice` (streamlit as an alternative to gradio if preferred; torchaudio is already
present).

---

## 9. Options moving forward (a menu, not a plan)

Phases 0–7 are complete. This section lists what *could* be done next and why, so that
whatever is actually chosen can be justified against the alternatives that were
considered and declined. **Nothing here is committed.** When a subset is chosen, record
which and why — "we did A and C, not B, because …" is a far stronger thesis sentence
than a list of everything that was possible.

### 9.0 The rule that governs everything below — read first

**2021 PA eval has been spent.** Its numbers are a clean generalisation estimate
*because* nothing was tuned on them. If new models are now selected by looking at
`eval`, that status is destroyed retroactively and the headline result stops being
defensible — this is precisely the failure mode section 5 identifies as the one
examiners probe for.

Protocol from here:

- **Develop against the `progress` partition** (87,048 trials). Phase 7 showed it
  tracks `eval` closely for every system (PROGRESS_REPORT 7.17), so it is a good
  out-of-domain development surrogate.
- **Touch `eval` at most once per declared experiment**, for a single final confirmation
  of whatever that experiment chose.
- **Report post-hoc work in a section clearly separated from the pre-registered
  results.** The Phase 7 table stands as the clean estimate; anything later is
  exploratory and must be labelled as such.

**There is deliberately NO fixed budget of remaining eval applications.** A cap was
considered and rejected: pre-committing to "two more" would mean that if the first two
experiments produced nothing worth reporting and the third did, the third could not be
confirmed — the cap would be binding on the basis of results it was set before seeing,
which is arbitrary rather than principled. What disciplines eval use is not a quota but
the **per-experiment rule above**, which is unaffected by how many experiments there
eventually are: each one fixes its candidates, decision rule and predictions in this
document first, selects on `progress`, and confirms once. An experiment that has not
declared a protocol does not get to touch `eval` at all, and that is the constraint doing
the actual work.

### 9.1 ~~Push the augmentation axis further~~ — **DONE. The axis is saturated.**

**Outcome (PROGRESS_REPORT P8): the first copy is the entire effect.** At T150/timepool,
`p(clean)` 1.0 → 0.5 buys −2.717 pp on `progress`; the whole range 0.5 → 0.0625 then spans
**0.103 pp** and is not monotone. The incumbent is over-augmented — three copies where one
would do — because three was chosen at T400/flatten, where the axis is real, and does not
transfer. All four declared predictions held; the decision rule retained
`timepool_T150_aug` and **no eval application was spent**. The diversity axis is moot,
having been contingent on dose still moving.

The most quotable by-product: across the same sweep **dev EER moves 3.236 pp while
progress EER moves 0.103 pp — 31x more.** Dev measures the in-domain *cost* of
augmentation precisely and is blind to the out-of-domain benefit.

The original reasoning is kept below, since the reframing is what made the experiment
cheap enough to run at all.

#### 9.1-orig The argument as it stood before the runs

Waveform augmentation was the strongest lever found and **the dose-response has not
plateaued**: 0 → 1 → 3 copies gives 39.747 → 34.006 → 32.665% EER, still falling. On
the `hidden` partition the gap is starker still: augmented systems hold ~30% while every
non-augmented one collapses to 48–54%.

**Corrected framing — the dose is the CLEAN FRACTION, not the copy count.** The Dataset
draws uniformly among `{clean, aug1..N}` per access (6.9), so N copies gives a clean
fraction of `1/(N+1)`, and the axis actually measured was 100% → 50% → 25%. Copy count
therefore moves the dose **hyperbolically**, and the increments are shrinking fast:

| copies | clean | 2021 EER | marginal |
|---|---|---|---|
| 0 | 100% | 39.747 | — |
| 1 | 50% | 34.006 | **−5.74** |
| 3 | 25% | 32.665 | **−1.34** |
| 5 | 16.7% | ? | |
| 7 | 12.5% | ? | |

Going 3 → 7 copies shifts the clean fraction only 25% → 12.5%, a **smaller step than
1 → 3**, which itself bought 1.34 pp. So "try 5–7 copies" spends ~5 h of generation and
~18 GB of disk to move the axis less than the previous step moved it.

**The cheap experiment does the same thing for free.** Keep the three existing copies and
change the *sampling weight* on clean: `p(clean) = 0.125` reproduces the dose of seven
copies with no generation and no disk — a one-line Dataset change.

That also separates two factors currently confounded. **Copy count buys perturbation
diversity; clean probability buys dose.** The 2-factor design follows, and it is ordered
so the free axis reports first:

- **Dose (free):** 3 copies, `p(clean) ∈ {0.25, 0.167, 0.125, 0.0625}` — 4 training runs.
- **Diversity (costly):** only if the dose axis is still moving — more copies at a
  matched clean fraction, which is the only way to attribute a further gain to diversity
  rather than dose.

**Decided: run the dose axis first.** If regenerating copies later, fix the latent seed
collision noted in 6.10 (multiplier `1_000_000`).

*Why it might not pay*: the curve must flatten eventually, and the −1.34 pp step is
already close to P1's ±0.5–1.0 pp checkpoint-noise floor. The in-domain cost also grows
monotonically, so dev EER will keep looking worse — which by now is expected rather than
alarming.

#### 9.1.1 Declared protocol — fixed before any run

**The knob.** `datasets.py` draws the source blob uniformly over `{clean, aug1..N}`, so
`p(clean) = 1/(N+1)` *exactly* — copy count and dose are the same parameter. A `p_clean`
argument decouples them: draw clean with that probability, otherwise uniformly among the
augmented copies. When `p_clean` is unset the existing single-`integers` call is left
untouched, so **every Phase 6 run still reproduces bit-for-bit** — that is the control.

**Configuration: T150 + timepool**, the current best single system, *not* the T400/flatten
setting where the original dose curve was measured. P4 showed these axes interact
sub-additively, so the shape measured at T400 does not transfer for free; and T150 runs at
~6.3 min/epoch against T400's ~22, making the sweep 2–3x cheaper.

**Candidates** (fixed here; not chosen from results):

| tag | p(clean) | equivalent copies | status |
|---|---|---|---|
| `timepool_T150_pc50` | 0.500 | 1 | new — confirms the curve behaves at T150 at all |
| **`timepool_T150_aug`** | **0.250** | 3 | **already run**, 29.359% progress |
| `timepool_T150_pc17` | 0.167 | 5 | new |
| `timepool_T150_pc12` | 0.125 | 7 | new |
| `timepool_T150_pc06` | 0.0625 | 15 | new — tests the turnover |

No copies are generated: all four reweight the three `cqt_train_aug{1,2,3}` blobs already
on disk. **Order by information, not cost** (the four are equally expensive): `pc12` first
— does more dose help at all — then `pc06` to test the turnover, then `pc17` and `pc50` to
fill in. An early stop therefore still answers the primary question.

**Development.** All candidates scored on `progress` only. The four new tags enter
`PHASE7_POSTHOC_SYSTEMS` with a `("progress",)` whitelist, so `score_posthoc` filters
before the forward pass and **none of them can reach eval until one is declared** — the
same in-code enforcement P4 used, and for the same reason.

**Decision rule.** Winner is the lowest `progress` EER. If the winner is not the incumbent
`timepool_T150_aug` and beats it by **less than 1.0 pp**, the incumbent is retained: P1
measured checkpoint-choice noise at ±0.5–1.0 pp, so a smaller gap is not attributable to
the dose change. Only a declared winner receives the single eval application.

**Predictions, stated in advance:**

1. **The curve keeps flattening** — the gain from 0.25 → 0.125 is smaller in magnitude
   than the previous step's −1.34 pp.
2. **`pc06` is not the winner.** 6.9's own reasoning: keeping clean data in the pool is
   deliberate, because training almost exclusively on perturbed audio adapts the model to
   a distribution neither dev nor 2021 has. There should be a turnover or a plateau.
3. **Dev EER worsens monotonically as p(clean) falls.** Declared for the reason P4's
   prediction 3 existed — so a good run is not abandoned mid-training on a bad dev number,
   which nearly happened once already.
4. **The winner beats `timepool_T150_aug` by less than 1.5 pp on progress**, i.e. the
   remaining headroom on this axis is small and probably not separately demonstrable
   against the ±1.5–3.6 pp speaker-clustered CI.

**What this separates, for the first time.** Copy count buys perturbation **diversity**;
clean probability buys **dose**. Every run to date confounded them. If the dose axis is
still moving at 0.125, generating more copies becomes worth testing *at a matched clean
fraction* — the only design that can attribute a further gain to diversity. If the dose
axis is flat, §9.1 closes and no copies need generating at all.

**Out of scope, deliberately.** If a new winner emerges it changes the fusion's primary
input, and re-fusing would be worth doing — but that is a **separate declared experiment**
with its own predictions and its own eval application, not a rider on this one. The GMM
partners are unaffected either way.

### 9.2 Extend T downward

On 2021 the T-axis is monotonic in the opposite direction to dev: T150 34.420 < T250
35.816 < T400 38.031. **T=100 and T=75 are untested**, cheap, and train faster than
anything run so far.

*Why it might not pay*: T=150 already underfits in-domain (6.584% dev EER, and 6.6
showed it underfits rather than overfits), so there is a floor below which the model
simply lacks context. The curve may turn over immediately.

### 9.3 Combine 9.1 and 9.2 — **DONE.** New best, honestly bounded

`timepool_T150_aug`: **31.081% EER / 0.8090 min t-DCF** on 2021 PA eval — a new best
point estimate, ranking **10th of 24 by EER and 9th by t-DCF**, up from 11th on both.
All three declared predictions held; the decision rule chose on the 1.62 pp `progress`
gap rather than falling to its tie-break; `eval` was touched once, for one model.

**The limit, stated with the result:** the improvement over the previous best is
**[−3.87, +0.63] — not statistically distinguishable**. It *is* significantly better than
its unaugmented parent, the pre-registered primary, and every official baseline. The two
effects proved **sub-additive** (predicted ≈27.3% if additive, observed 31.081%):
augmentation does most of the work.

Best figures anywhere in the project come from its hidden-track scores — **24.72% EER /
0.6818 t-DCF on the simulated subset** — and it is the first system that barely degrades
when non-speech is removed. Full detail in PROGRESS_REPORT P4.

**§9.3.2's boundary now applies**: the answer to "the headline margin is not significant"
is *not* to try another configuration and report that one instead. Further exploration is
scored on `progress`, and `eval` is not revisited.

Original protocol kept below, unchanged, as the record of what was declared in advance.

### 9.3-orig Combine 9.1 and 9.2 — protocol as declared

**Every model on the augmentation axis was trained at T=400, the *worst* T for
transfer.** `T150 + 3 augmented copies` has never been run. If the two effects are even
partially additive, this is the most likely route to a materially better number.

#### 9.3.1 Declared protocol — written before the runs were scored

This is **post-hoc exploration**, not pre-registration in the Phase 7 sense: the
configuration was chosen after seeing 2021 results, so it carries no such guarantee and
will be reported in a clearly separate section. What *is* fixed in advance, and is
recorded here before any 2021 number exists for these models, is the **decision rule**.

**Candidates** (both trained on the 2019 enriched resplit; 2021 is never trained on):

```
--tag flatten_T150_aug   --n-frames 150 --head flatten   --wav-aug-copies 3 --epochs 45
--tag timepool_T150_aug  --n-frames 150 --head timepool  --wav-aug-copies 3 --epochs 45
```

**Why two, and why the head is the open variable.** The head axis *also* inverts between
domains: at T=400, `flatten` wins on dev (0.798 vs 0.902) but `timepool` wins on **both**
2021 subsets — 38.031 vs 39.747 on eval, 36.364 vs 37.537 on progress. Every augmented
model to date is `flatten` and every short-T model is `timepool`, so the two are
confounded and **augmented × timepool is an empty cell**. Choosing a head by reasoning
from eval would be precisely the contamination §9.0 forbids, so both are trained and the
choice is made on `progress`.

**Why 45 epochs.** Phase 6 recorded that `aug3` hit its 30-epoch cap while dev EER was
still falling, and that `aug1` received 45 against its control's 30 — an unequal-budget
caveat that had to be disclosed. A common budget of 45 removes it; early stopping
(patience 8) cuts either run short if it plateaus first.

**Standard recipe, justified by measurement.** Dev-EER checkpointing is retained. The
control in PROGRESS_REPORT P1 showed epoch selection by dev EER is unbiased for transfer
(mean delta +0.078 pp over 7 systems), so no per-epoch checkpointing is needed and these
runs stay methodologically identical to their Phase 6 parents.

**Decision rule, fixed now:**

1. Score both candidates on **`progress`** (87,048 trials).
2. The lower `progress` EER wins. **If they differ by less than 1.0 pp, declare it a
   tie and prefer `timepool`** — that is the measured scale of checkpoint-choice noise
   (P1), so a smaller gap is not evidence, and `timepool` is the more parsimonious head
   (184k vs 798k parameters at T=400) and the one that transfers better at matched T.
3. Take **only the winner** to `eval`, once.
4. Report both `progress` figures regardless, so the selection is visible rather than
   implied.

**Predictions, stated in advance so either outcome is a result:**

1. **The combination beats both parents on 2021.** `T150+aug3` < both `T150` (34.420%)
   and `flatten_T400_aug` (32.665%) on eval. If it lands *between* them, the two effects
   overlap rather than add — informative either way.
2. **`timepool` beats `flatten` at T=150 with augmentation**, extending the head
   inversion already seen at T=400.
3. **Dev EER will be worse than every Phase 6 system.** Both parents' effects hurt
   in-domain (T150 6.584%, aug3 2.353%), so a combined dev EER above ~7% is expected and
   is *not* a failure signal. This prediction exists to stop a bad dev number being
   misread mid-run.

#### 9.3.2 Boundary that must not be crossed

If the winner's `eval` number is disappointing, **the answer is not to try another
configuration and report that instead.** That converts `eval` into a tuning set
retroactively and destroys the Phase 7 guarantee along with it. Further exploration
after this point is scored on `progress` only, and `eval` is not revisited.

### 9.4 ~~Verify the official-baseline reproduction~~ — **DONE, verified exactly**

Checked against Liu et al., IEEE/ACM TASLP 2023 (doi: 10.1109/TASLP.2023.3285283),
Table XV and Table X. **All eight baseline EERs match to ≤0.005 pp**, on both the
evaluation and progress partitions, plus an exact match on two further subsets defined
by `dist` and `trim_flag`. See PROGRESS_REPORT 7.12. The results chapter can state
that the pipeline reproduces the published baselines exactly.

Two corrections came out of the same reading, both recorded in PROGRESS_REPORT 7.14:
**B03 is LFCC-LCNN-LSTM**, not a plain LCNN (so that comparison is front-end-dominant,
not front-end-only), and **our systems are not challenge-compliant** — participants
were required to train on the 2019 PA train partition alone (54,000 files) whereas we
used the 175,959-file enriched resplit including 2019 PA eval. Must be disclosed
wherever our numbers sit beside challenge numbers.

### 9.5 ~~Identify conditions `r3` and `s4`~~ — **DONE, and they reproduce the paper**

Table III and supplementary Table IX decode every factor. Five of six of our condition
findings independently reproduce the paper's published ones, using a different metric
and a different system; the sixth (attacker-to-ASV distance) conflicts and is reported
as such. Two findings the paper does not report were added: attacker-room identity
matters enormously (24.50 pp spread) even though room *size* does not, and the
room-triples the paper defines in its footnote 7 split cleanly into group means of
46.7 / 31.8 / 39.7% EER. Full detail in PROGRESS_REPORT 7.15.

### 9.5b Follow-ups created by the paper analysis (new)

- ~~**Train on VAD-trimmed audio**~~ — **DISMISSED.** PROGRESS_REPORT 7.19 showed our
  non-augmented models losing 11.5–16.7 pp when non-speech is removed, against 0.3–6.9 pp
  for the official baselines — leaning on a cue the paper (§VI) flags as an undesirable
  database artefact. This item existed to remove that dependence.

  **P4 closed it by accident, and P7 made it moot.** `timepool_T150_aug` does not degrade
  on trimmed audio at all — 27.87 → **26.87**, it *improves* — so the defect is already
  absent from the system that would have been repaired, and augmentation removed it
  without anything being trained on trimmed audio. What remained was the weaker question
  of whether trimmed *training* would remove the residual dependence in the
  **non-augmented** models — but those are no longer on any path the thesis takes: the
  headline is a fusion whose CQT-LCNN component is augmented, and the non-augmented runs
  survive only as controls on the T and augmentation axes.

  So the experiment would cost a full extraction pass plus a training run to improve a
  system nothing else depends on, and to answer a question the augmentation result has
  already answered in the affirmative by another route. Recorded as dismissed **with its
  reasoning**, so it is not rediscovered and re-proposed later.
- **Investigate the attacker-room grouping.** Why is room group 2 (`r4,r5,r6`) ~15 pp
  easier than group 1? Nothing in the published analysis explains it, because rooms
  were excluded there after a size correlation came back null.
- **Resolve the attacker-to-ASV distance conflict.** Recompute that breakdown with
  distance-matched bonafide instead of the pooled convention, to test whether our
  pooling leaks a level cue.

### 9.6 ~~Statistical rigour~~ — **DONE** (`src/bootstrap_ci.py`)

B = 2000 replicates, both metrics, all 13 systems. **All nine registered comparisons
survive.** Headline methodological result: honest speaker-clustered intervals are
**14.3x wider** than the conventional trial-level ones (4.727 pp vs 0.330 pp mean
width), because the effective sample size is **67 speakers**, not 721,332 trials.

Pairing decided two verdicts: for prediction 1 the marginal intervals *overlap*
(suggesting no effect) while the paired interval excludes zero decisively — comparing
marginal CIs is systematically conservative and discards real effects.

Weakest claims, to be phrased carefully rather than asserted flatly: the head comparison
(margin 0.61 pp to zero) and prediction 1 (1.03 pp). The headline-vs-CQCC-GMM interval
is the widest ([−9.37, −1.53]) because the GMM baselines are far more speaker-variable
than our systems. Full detail in PROGRESS_REPORT P3.

Still outstanding from the original item: the **MFCC-RF score-granularity caveat** (236
distinct values from a 300-tree vote) and the dev→2021 **rank-inversion significance**
(Spearman ρ = −0.607, p = 0.148, n = 7) — the bootstrap does not address either, since
both concern quantities other than per-system error rates.

Original reasoning kept below.

### 9.6-orig Statistical rigour

- **Bootstrap confidence intervals on EER.** With 721,332 trials these will be tight and
  cheap, and they are needed before claiming that e.g. the 2.2 pp T250-vs-T400 gap is
  real.
- The dev→2021 rank inversion is Spearman **ρ = −0.607, p = 0.148** over 7 systems —
  suggestive, not significant. Either report it with the p-value as a described pattern,
  or add systems to strengthen it.
- **MFCC-RF has only 236 distinct score values** (300-tree vote granularity), so its DET
  curve is a coarse step function and its EER sits in a large tie block. Either note the
  caveat or refit with more trees if RF stays in the headline table.

### 9.7 ~~min t-DCF~~ — **DONE, validated 8/8** (`src/tdcf.py`)

Implemented from the tandem model in Kinnunen et al. (2020), since no official scoring
code ships with the keys package, and **validated exactly against all eight published
baseline values** (four systems x two partitions). Best system `flatten_T400_aug` scores
**0.8347**, beating all four baselines and ranking **11th of 24 by min t-DCF — the same
placement the EER ordering gave**. Both classical baselines saturate at ≈1.0000, i.e.
they provide no benefit over a non-informative CM. EER and t-DCF disagree on two
orderings, so both are now reported. Full detail in PROGRESS_REPORT P2.

Original reasoning kept below.

### 9.7-orig min t-DCF — promoted after the paper analysis

EER was the metric chosen here, but **min t-DCF was the 2021 challenge's primary
metric**, and the official ASV scores are already on disk (`PA-keys-full/keys/PA/ASV/`).

This moved *up* the list once the paper was read. The full ranked results are in its
Table XV, so computing t-DCF would place our systems directly on the challenge's own
primary axis rather than on a secondary one — turning "11th of 24 **on EER**"
(PROGRESS_REPORT 7.20) into a directly comparable figure. There is also more room to
show separation than the EER numbers suggest: every PA baseline sits at min t-DCF
**0.943–1.000**, effectively saturated against an ASV floor of 0.12, so the metric
discriminates in a region where EER is compressed. Moderate work, high citability,
and it overlaps with the Phase 8 ASV extension already sketched.

### 9.8b Fusion with the official baselines — **PROTOCOL DECLARED BELOW, then run**

#### 9.8b.1 Declared protocol — fixed before eval was touched

Post-hoc, so no pre-registration guarantee; what is fixed in advance is the **method,
the candidate set and the selection rule**, recorded before any eval number exists.

**Method.** Score-level linear fusion: z-normalise each system's scores, then logistic
regression. Weights are needed rather than plain averaging because the partners are
7–17 pp worse than our system, and equal-weight averaging of unequal systems is what
made §9.8's attempt fail.

**Candidates** (fixed here; not chosen from results):

| name | systems |
|---|---|
| `ours` | `timepool_T150_aug` alone — the reference |
| `ours+2GMM` | + CQCC-GMM + LFCC-GMM |
| `ours+4base` | + all four official baselines |
| `ours+all` | + `flatten_T400_aug` + all four baselines |

Also reported: **equal-weight** fusion of the same sets, to show whether the trained
weights are doing real work or the gain is just from averaging.

**Development.** Speaker-disjoint K-fold cross-validation on the **`progress`**
partition (87,048 trials). Folds split by *speaker*, not by trial, for the reason
established in P3: trials from one voice are not independent, so a trial-level split
would leak between fit and test and flatter every candidate.

**Selection: the one-standard-error rule.** Take the candidate with the lowest CV EER;
among all candidates whose CV EER is within **1 standard error** of it, choose the one
with the **fewest systems**. Standard practice in model selection, and it exists to stop
a marginally-better-but-more-complex combination being preferred on noise — which is
exactly the failure mode a fusion sweep invites.

**Confirmation.** Refit the selected candidate on *all* of `progress`, then apply it
**once** to `eval`. z-normalisation statistics come from `progress` only, so no eval
information enters the model. Running the eval step requires an explicit
`--confirm-eval` flag; development is what the module does by default.

**Predictions, stated in advance:**

1. **Fusion beats the single system on eval**, by roughly 2–3 pp EER — the range measured
   on a speaker-disjoint progress split before this protocol was written.
2. **`ours+2GMM` is selected**, because the two GMMs captured 97% of the available gain
   on progress (−2.82 pp of −2.92 pp) and the 1-SE rule prefers parsimony.
3. **Trained weights beat equal-weight fusion**, because the partners are much weaker and
   need down-weighting.

#### 9.8b.1a What fusion costs the thesis — four things it breaks

Written out because it decides how the result is *reported*, not just what it scores.

1. **You cannot beat a baseline by including it.** The Phase 7 headline is "our system
   beats all four official baselines". A system *containing* CQCC-GMM and LFCC-GMM
   cannot make that claim — it is circular. The comparison anchoring the results chapter
   becomes incoherent for the fused system.
2. **The front-end isolation argument collapses.** Section 6.3 chose the LCNN backbone
   *specifically* so "CQT-LCNN vs LFCC-LCNN" varies exactly one thing — the front-end —
   which is what makes the thesis's central claim attributable. A fusion containing
   cepstral front-ends and GMM backends isolates nothing; it is a combination that
   happens to perform well.
3. **The system stops being ours.** Its performance depends on score files we did not
   produce and cannot regenerate. Fine for a challenge entry, awkward as a thesis
   contribution.
4. **It stops being zero-shot — the deepest one.** Every Phase 7 result comes from
   systems trained *only* on 2019, which never saw a single 2021 file; that is what makes
   "generalises to real replay" meaningful. Fitting fusion weights on `progress` means
   the system has seen **87,048 labelled 2021 trials**. The claim silently changes from
   *"a countermeasure trained purely on simulated replay transfers to real replay"* to
   *"with a modest amount of labelled target-domain data, performance improves by ~X"*.
   Both are legitimate — challenge rules permitted `progress` — but they are **different
   scientific statements**, and conflating them would be the most serious methodological
   error available here.

**Therefore: the thesis's primary system remains the single CQT-LCNN**, and fusion is
reported as a clearly separated extension. Framed that way it contributes three things
without damaging anything — it aligns with universal practice (all top-5 systems in all
three 2021 tracks fused), it gives **direct evidence of complementarity** between CQT-LCNN
and cepstral-GMM front-ends, which *supports* the front-end argument rather than
undermining it, and it **bounds the remaining headroom** available from combination as
opposed to better single models.

A more ambitious alternative, if time allows: train **our own** CQCC-GMM and/or LFCC-GMM
and fuse only in-house systems. That preserves the "beats the baselines" claim and turns
fusion into a *direct test of the central premise*. Costs: CQCC is derivable from the
cached CQT with no audio re-decoding (~3–5 h), while LFCC needs a fresh 943k-file
extraction (~6–8 h) despite being the better partner. Two systems suffice — the two GMMs
captured 97% of the available gain on progress (−2.82 pp of −2.92 pp).

#### 9.8b.2 Why fusion is expected to work here — evidence gathered first

Measured on `progress`, per-file, against `timepool_T150_aug`:

| partner | Spearman ρ | rescue | if independent | ratio |
|---|---|---|---|---|
| `flatten_T400_aug` (ours) | 0.797 | 31.2% | 70.2% | **0.44** REDUNDANT |
| CQCC-GMM | 0.284 | 52.3% | 63.7% | 0.82 |
| **LFCC-GMM** | **0.101** | 58.9% | 60.2% | **0.98** near-independent |
| LFCC-LCNN | 0.244 | 46.9% | 57.8% | 0.81 |
| RawNet2 | 0.117 | 48.6% | 54.0% | 0.90 |

**Reading the two columns**, because both were initially got wrong:

*Spearman ρ* is Pearson correlation computed on **ranks** rather than raw values: replace
every score by its position in sorted order, then correlate the positions. Rank
correlation rather than Pearson for two reasons, and both are specific to this table.
The scales are wildly incommensurable — our LCNN emits logits spanning ±40, the GMMs emit
log-likelihood ratios spanning ±5, MFCC-RF emits probabilities in [0.04, 0.93] with only
236 distinct values — so Pearson would report differences in distribution *shape*. And
**EER is itself rank-based**: it is read off the ROC curve, which depends only on the
ordering of scores and never on their magnitudes. Rank correlation therefore asks exactly
the question the metric cares about — do these two systems order the files the same way?

*Rescue* is the share of the files **we** get wrong (at our EER threshold) that the
partner gets right (at its own). Its reference point is the subtle part: an earlier draft
used **50%**, which is wrong and inverted several readings. If B's correctness were
statistically independent of whether A erred, rescue would equal B's own accuracy —
exactly `1 − EER_B`, since at the EER threshold FRR = FAR = EER and so the overall error
rate is EER regardless of class imbalance. Hence the `ratio` column: **≈1.0 means the
partner's errors are independent of ours** (the ideal), below 1.0 means the two tend to
fail on the same files.

**Why each row reads as it does.** Our own `flatten_T400_aug` is redundant on *both*
measures — same architecture, same training data, same front-end, so it agrees on ranking
*and* fails where we fail, far more than chance. **CQCC-GMM is middling for a mechanical
reason that matters below: CQCC is a cepstral summary of the constant-Q spectrogram — the
same analysis our own front-end uses.** Shared front-end family, shared errors. LFCC-GMM
is the standout precisely because it shares nothing: a linear filterbank rather than
constant-Q, and a generative back-end rather than a discriminative one. RawNet2 is the
"necessary but not sufficient" case in a single row — decorrelated (ρ 0.117) but at 46%
EER too weak to contribute.

**Measured single-partner gains on `progress`**, which is where the parsimony argument
comes from:

| fused with | gain |
|---|---|
| LFCC-GMM alone | **−2.04 pp** |
| CQCC-GMM alone | −1.69 pp |
| both GMMs | **−2.82 pp** |
| all four baselines | −2.92 pp |

**The worst system in the table is the best single partner.** LFCC-GMM at 39.8% EER
outperforms CQCC-GMM at 36.3% as a fusion partner, which is the cleanest available
demonstration that *partner difference matters more than partner quality*. The two GMMs
together capture **97%** of the total available gain; the remaining two baselines buy
0.10 pp between them.

**LFCC-GMM is the best partner despite being the worst system in the table (39.8% EER)**
— decorrelation matters more than partner quality, though not without limit: RawNet2 is
also decorrelated (ρ 0.117) but at 46% EER has too little to contribute. Fusion gain is
roughly *decorrelation × partner strength*, and a partner must satisfy both.

This is also why fusing our own systems fails: `flatten_T400_aug` agrees with us on
ranking (ρ 0.797) *and* fails on the same files (ratio 0.44), so §9.8's null result was
not a fluke but the predictable consequence of combining near-duplicates.

Every top-5 system in **all three** ASVspoof 2021 tracks used score fusion (challenge
paper Table V), so a single-system submission is the outlier.

#### 9.8b.3 Original motivation (kept)

§9.8 below measured fusion **among our own systems** and found it worthless (~0.13 pp).
That was the worst case for fusion and should not have been generalised: those systems
correlate **+0.35 to +0.82** across speaker resamples, so they fail on the same voices
and there is nothing to recover.

P4 found the opposite situation. `timepool_T150_aug` vs **CQCC-GMM** correlates
**−0.15 — negative**: when a speaker draw is hard for our CQT-LCNN it is slightly *easy*
for the CQCC-GMM. **Complementary failure modes across speakers is precisely the
condition under which fusion pays**, and the baseline scores are already on disk.

Cheap to test (no GPU, no retraining), and it must be developed on `progress` with at
most one confirmation on `eval` (§9.0). Two honest caveats: what gets reported is then a
*fusion*, not a CQT-LCNN, which changes the thesis's claim; and it inherits the
training-data non-compliance noted in PROGRESS_REPORT 7.14.

#### 9.8b.4 Three follow-ups on the completed fusion run — declared before running

§9.8b.1 was executed and all three of its predictions held (results in PROGRESS_REPORT
P5). Reviewing that run against this protocol exposed three gaps. Each is closed below,
and the protocol for closing them is fixed **here, before the numbers exist**, for the
reason §9.3.1 and §9.8b.1 already earned.

**The boundary first, because it governs all three.** Every one of these is in the
post-hoc lane and **none of them touches `eval` as a decision**. A and B are computed on
`progress` alone. C recomputes confidence intervals over eval scores that are *already on
disk and already spent*, and selects nothing. The single eval application of §9.8b.1
stands as the only one, and is not repeated.

**A. The fitted weights were computed and discarded.** `fuse.py` fits the logistic
regression and returns only `decision_function`, so the coefficients exist nowhere.
§9.8b.1a names *"direct evidence of complementarity between CQT-LCNN and cepstral-GMM
front-ends"* as one of three things fusion contributes **without** damaging the thesis —
and the coefficients **are** that evidence. Because the inputs are z-normed to unit
variance, the coefficients are directly comparable across systems; that property is what
makes them an argument rather than a decoration. Captured for every trained candidate in
CV, not only the selected one, since watching CQCC-GMM's weight change as LFCC-LCNN is
added measures redundancy *among the baselines*, which nothing else here does.

Recoverable with **zero eval contact**: the eval refit's `mu`, `sd` and coefficients all
come from `progress`, and eval enters only at `decision_function`. A `--weights-only`
path therefore reproduces the weights behind the reported number without re-running
`--confirm-eval`.

*Predictions.* (1) All three coefficients positive, `timepool_T150_aug` largest.
(2) **`|w_LFCC-GMM|` > `|w_CQCC-GMM|` despite LFCC-GMM being the worse system** (39.54%
vs 38.07% EER) — because §9.8b.2 measured it as far more decorrelated (ρ 0.101 vs 0.284).
This is a direct falsifiable test of the *gain ≈ decorrelation × strength* model the whole
experiment rests on. (3) Per-fold coefficients stable across the five speaker-disjoint
folds, partner weights within ~±25% of their mean.

A negative partner weight would mean the fusion is using that system *contrarily* and
would require the complementarity story to be rewritten. It must therefore be visible in
the artifact, not smoothed over.

**B. The selection margin is uncharacterised.** `ours+2GMM` cleared the 1-SE threshold by
**0.162 pp** at one fold split. Swept over 20 seeds × K ∈ {5, 10} — K is included because
§9.8b.1 declared "speaker-disjoint K-fold" *without fixing K*, so `N_FOLDS = 5` was an
implementation choice, and K is the parameter the 1-SE band is most sensitive to
(SE ~ 1/√K). **Leave-one-speaker-out is not viable**, and the reason is structural rather
than budgetary: a single speaker's ~1.3k trials give a degenerate per-fold EER, and 19 of
the 67 speakers carry no spoof trials at all.

*Predictions.* (1) `ours` excluded in **100%** of splits — it misses by 2.37 pp against
SE ~0.6, so the fusion-vs-no-fusion half of the decision is not marginal at all.
(2) `ours+2GMM` selected in a majority but **not all** splits; 50–80%. (3) Every flip goes
toward *more* systems, never toward `ours`.

*This is evidence about the robustness of a decision already made; it is **not** a licence
to re-select.* If most splits would have chosen `ours+4base`, that is reported as a
limitation on the existing result. Eval is not re-applied.

**C. The result carries no confidence interval**, while every other headline in this
project does. P3 established the standard: speaker-clustered paired bootstrap, and that
comparing marginal CIs is systematically conservative — the CI of the *difference* is the
test. One comparison is added, `fusion_ours+2GMM` vs `timepool_T150_aug`.

**`fusion_ours+2GMM` vs `CQCC-GMM` is deliberately NOT added, and the refusal is enforced
in code with its reason.** §9.8b.1a point 1: you cannot beat a baseline by including it.
Materialising that row in a results CSV would create exactly the incoherent comparison
this protocol forbids, and a boundary written into the data path holds where one written
in a document depends on whoever reads it next.

*Predictions.* (1) The paired CI **excludes zero**, even though −2.27 pp is *smaller* than
the −1.62 pp that came back "not distinguishable" in P4 — because the fused score
*contains* the single system, so the pair correlates far above P4's 0.19–0.82 range.
Predicted `corr_eer` **> 0.90**, CI approximately **[−3.2, −1.4]**. (2) The marginal CIs
overlap heavily, reproducing P3's central point.

**Three controls, fixed in advance.** A: recovered weights applied to eval *features*
already on disk must reproduce the stored fused scores to float tolerance — no labels, no
EER, no decision. B: seed 42 at K=5 must reproduce `fusion_cv_progress.csv` exactly.
C: **all 13 existing systems and all 13 existing comparisons must reproduce bit-for-bit.**
C's control is load-bearing and should hold by construction — the resample `w` is drawn
once per replicate *before* the system loop and the loop never consumes the rng, so a 14th
system cannot perturb the weight sequence. If any existing row moves, the change is wrong
and is reverted.

**Two reporting decisions, settled here.** The fusion is **excluded from the CI-width
aggregate** in `bootstrap_ci_summary.json`: that statistic backs the published "14.3x
wider" claim about the 13 zero-shot systems, and letting it drift because a
differently-shaped object joined the table is the documentation drift this repo warns
against. And the fusion **stays out of `posthoc_table_2021.csv`** — tabulating a
non-zero-shot system beside zero-shot ones is the conflation §9.8b.1a point 4 calls the
most serious methodological error available here. Its canonical record is
`fusion_eval.json`, which already carries `zero_shot: false`.

### 9.8c Build our own cepstral-GMM partners — the only route that keeps every claim

§9.8b.1a lists four things fusing with the official baselines costs. Building the partners
in-house recovers **three** of them: the "beats every official baseline" claim stops being
circular, the system becomes ours end to end, and fusion turns into a *direct test of the
front-end premise* rather than a combination that happens to work.

**And it is the only route that can recover the fourth, which is the deepest.** The fused
§9.8b system is not zero-shot: its weights were fitted on 87,048 labelled 2021 trials.
That cannot be fixed with the official baselines, because their score files exist **only
for 2021** — there is nowhere else to fit. Our own GMMs can be scored on **2019 dev**, so
the fusion weights can be fitted there and the whole system never sees a 2021 label. The
claim stays *"a countermeasure trained purely on simulated replay transfers to real
replay"*, with fusion inside it rather than beside it.

Whether dev-fitted weights transfer is a genuine risk and must be declared as a
prediction, not assumed. The relevant precedent is P1: dev EER is anti-correlated with
2021 performance at the **configuration** level but sound at the **epoch** level. Fitting
three weights is far closer to the epoch case than the configuration case, so the prior is
favourable — but Phase 7's central lesson is that this is exactly the intuition that
failed before. Fit on `progress` as well, report both, and let the gap between them
*measure* how much labelled target-domain data is worth.

#### The cost ordering and the value ordering are INVERTED

This is the decision's crux and it reverses the obvious plan.

| | cost | why that cost | ρ with ours | ratio |
|---|---|---|---|---|
| CQCC-GMM | **~3–5 h** | derivable from the cached CQT — no audio re-decoded | 0.284 | 0.82 |
| LFCC-GMM | ~6–8 h | needs fresh extraction from waveform: ~1 h for 2019, ~2.5 h for 2021's 943,110 files, plus GMM fitting over ~79 M frames (subsampling required) | **0.101** | **0.98** |

**CQCC is cheap *because* it is redundant.** It reuses our CQT cache — and 9.8b.2 already
identified shared constant-Q analysis as the mechanism behind CQCC-GMM's mediocre ratio of
0.82. The official CQCC-GMM at least used an independent CQT implementation; ours would
read *the same cached array our LCNN reads*, so its decorrelation from our system should
be **worse than 0.284, not equal to it**. LFCC is expensive because it is a genuinely
different front-end, which is the same fact that makes it the ratio-0.98 partner.

So the project's usual cheapest-first ordering is here also **weakest-first**, and the
usual justification for it does not hold: an early interrupt after CQCC would leave us
with the partner least able to demonstrate anything.

#### What we could actually build is not CQCC, and must not be called CQCC

The cached CQT is **uint8-quantised dB, `top_db`-clipped, and peak-normalised per file**
(`ref=np.max`, `features.extract_cqt_uint8`). Three consequences, all of which belong in
the write-up rather than in a footnote:

1. Real CQCC's defining step is **resampling the geometrically-spaced CQT bins onto a
   uniform scale** before the DCT. Our cache has no such resampling, so what we would
   build is *log-CQT + DCT* — a constant-Q cepstral feature of our own design.
2. **C0 carries no absolute level**, since each file is normalised to its own peak.
3. Quantisation is ~0.31 dB per step over the `top_db` range, and everything below the
   floor is already gone.

None of that is disqualifying — it is a legitimate feature, and 6.8's finding that CMVN
destroys the evidence does *not* transfer to peak normalisation, which removes one scalar
per file rather than per-bin statistics over time. But calling it "CQCC" would be false,
and a home-built LFCC-GMM will likewise not reproduce the official one: implementation
details differ, so its fusion contribution may differ from the −2.04 pp measured in 9.8b.2.
**We would be building complementary systems, not reproducing those two.**

#### Settled

**Build both, LFCC-GMM first**, and fit the fusion weights on **both `dev` and
`progress`**, reporting the two side by side.

Both rather than one, for two reasons that are not the 0.78 pp. The 1-SE selection needs
the full candidate set — building only LFCC would prejudge the choice the declared rule
exists to make — and the 28/40 result from 9.8b.4 B is evidence about the **official**
partners, which does not transfer: the in-house CQCC's expected redundancy is *higher*,
so an in-house `ours+2GMM` may not reproduce that margin.

And CQCC earns its 3–5 h as a **falsifiable test of the front-end-family mechanism**
rather than as a marginal gain. 9.8b.2 asserts the official CQCC-GMM is a mediocre partner
*because* it shares constant-Q analysis with us. An in-house version reading the identical
cache should land at ρ well above 0.284; if it does, the mechanism is confirmed, and if it
lands at 0.284 anyway the explanation is wrong. That is worth more than the incremental
gain, which at **−0.78 pp** sits below P1's ±0.5–1.0 pp checkpoint-noise floor and may not
be separately demonstrable even if real.

Fitting on both partitions is not hedging: the dev fit is the *zero-shot* system and the
`progress` fit is the stronger one, and **the gap between them is the measurement** — it
prices what 87,048 labelled target-domain trials are actually worth. Reporting only one
would discard that.

#### 9.8c.1 Declared protocol — fixed before anything was built

**The two systems.** Both are 512-component **diagonal-covariance GMMs, one per class**,
scored as the log-likelihood ratio averaged over frames, trained on the **enriched
resplit** (matching our LCNN, and inheriting the training-data non-compliance already
disclosed in 7.14 — all fused components share it).

| tag | front-end |
|---|---|
| `our-LFCC-GMM` | LFCC, standard ASVspoof recipe: 70 linear-spaced filters, 20 coefficients + Δ + ΔΔ = **60 dims**, 20 ms window / 10 ms hop |
| `our-CQT-DCT-GMM` | dequantised cached log-CQT → DCT → 20 coefficients + Δ + ΔΔ. **Named for what it is.** It is *not* CQCC: no uniform-scale resampling of the geometric bins, C0 carries no absolute level (peak-normalised cache), 0.31 dB quantisation |

**Fitting: exact EM with a chunked E-step, not sklearn.** `GaussianMixture` materialises
2–3 `(n_frames × n_components)` float64 arrays — 500 k frames × 512 components is
**2.05 GB each**, so 4–6 GB live, which does not fit in 5.9 GB. Diagonal-covariance EM
decomposes into additive sufficient statistics (`N_k`, `Σ r·x`, `Σ r·x²`) that accumulate
over chunks, bounding memory at `chunk × components` ≈ **41 MB**. This is **exact batch
EM computed in pieces, not minibatch or online EM** — the same fixed point, so it costs
nothing statistically. Frame count therefore becomes a *runtime* decision, not a memory
one.

**Frame sampling: equal frames per file, within a class.** A GMM scores a file by the
**mean** frame log-likelihood, so every file counts once at test time; pooled random
sampling would instead weight the fit by duration. That matters here specifically because
duration is **not class-neutral** — 6.10 measured bonafide at 323 frames against spoof at
274 in train, and duration alone scores 41.5% EER. Across classes there is no shared fit,
so the quota is set *per class* to reach ~1.5–2 M frames each (≈90 frames/file bonafide,
≈10 spoof at the ~1:9 split), giving both GMMs ~3–4 k frames per component.

**Candidates** (fixed here; not chosen from results):

| name | systems |
|---|---|
| `ours` | `timepool_T150_aug` alone — the reference |
| `ours+LFCC` | + `our-LFCC-GMM` |
| `ours+CQTDCT` | + `our-CQT-DCT-GMM` |
| `ours+2GMM-inhouse` | + both |

**Two fit arms, run identically:** weights fitted on **2019 `dev`** (the system is then
fully zero-shot w.r.t. 2021) and on **2021 `progress`** (not zero-shot, directly
comparable to 9.8b). Selection is the same **1-SE rule** as 9.8b.1 — lowest CV EER, then
fewest systems among those within 1 SE — run *independently per arm* on speaker-disjoint
5-fold CV **of that arm's own fitting partition**. Equal-weight fusion reported alongside
as the same control.

**Confirmation. This experiment spends TWO eval applications**, one per arm, declared
here rather than discovered later — the comparison *between* the arms is the result, so a
single arm would not answer the question. If the two arms select different candidates, the
matched pair (the progress-selected candidate under both fits) is reported as well, so
fit-partition is never confounded with candidate choice.

**Predictions, stated in advance:**

1. **`our-CQT-DCT-GMM` correlates with `timepool_T150_aug` at ρ > 0.284**, the official
   CQCC-GMM's value. This is the experiment's real prize: 9.8b.2 explains CQCC-GMM's
   mediocre partnership by *shared constant-Q analysis*, and ours reads the identical
   cached array, so it should be strictly more redundant. If it lands at 0.284 anyway,
   that explanation is wrong.
2. **`our-LFCC-GMM` is the better single partner** — lower ρ, larger fitted weight, larger
   solo fusion gain than `our-CQT-DCT-GMM`. (Note A2's failure mode: strength must be read
   on the *fitting* partition, not on eval.)
3. **The 1-SE rule selects `ours+LFCC` — two systems — on at least one arm**, unlike
   9.8b's three, because the in-house CQT-DCT partner's marginal contribution should fall
   inside 1 SE once its redundancy is higher.
4. **The dev-fitted fusion beats the single system on eval but loses to the
   progress-fitted one.** The gap between them is the price of 87,048 labelled
   target-domain trials. Risk acknowledged: P1 found dev sound at the *epoch* level but
   Phase 7 found it anti-correlated at the *configuration* level; fitting 3–4 weights is
   nearer the former, but this is exactly the intuition that failed before.
5. Both in-house GMMs are **individually worse on 2021 than their official counterparts**
   (different implementations, no tuning), while the in-house fusion gain lands within
   ~1 pp of the −2.30 pp measured in P5c.

**Controls, fixed in advance.** (a) The chunked EM must reproduce sklearn's
`GaussianMixture` to numerical tolerance on a subset small enough for sklearn to fit
(50 k frames, 32 components) — the optimisation is worthless if it is not the same
algorithm. (b) Streaming per-file LLRs must match a batch computation exactly on a held
subset. (c) Every training file contributes exactly its class quota of frames. (d) Every
new module reports progress continuously and is resumable at fine granularity.

**Estimated cost** ~6–8 h total, ordered **LFCC first** so an interrupt leaves the better
partner finished: LFCC extraction + quota sampling ~1 h, LFCC GMM fits ~0.5–1 h, LFCC
streaming score over 2019 dev + 943,110 2021 files ~2.5–3 h, then CQT-DCT end to end
~1.5–2.5 h (no audio decoded — it reads the existing CQT cache), fusion minutes. Disk:
~1–2 GB of sampled training frames; **no frame-level store for 2021**, which would be
~79 GB.

#### 9.8c.2 Amendment: the normalisation arm — declared before the fusion was run

The GMMs are built and scored (PROGRESS_REPORT P6); the fusion has not run. Inspecting
the **fitting partitions' own score distributions** — dev and `progress`, never `eval` —
exposed a defect in the method 9.8b.1 declared, which is corrected here *before* any
fusion number exists rather than after.

**The defect.** 9.8b.1 fixes z-normalisation statistics to the *fitting* partition. That
was correct there, where fitting and application were both 2021. It is not correct for
the zero-shot arm, because dev and 2021 put these scores on incomparable scales:

| system | dev mean/std | progress mean/std | std ratio |
|---|---|---|---|
| `our-LFCC-GMM` | −11.82 / 11.51 | 0.83 / **0.53** | **21.7x** |
| `our-CQT-DCT-GMM` | −3.82 / 6.81 | −1.31 / 2.76 | 2.5x |

Not a monotone shift but a change of shape, and the mechanism is plain: dev is *in
domain* for a GMM trained on 2019 train, so the log-likelihood ratio spreads widely,
while on 2021 every file is roughly equally unlikely under both models and the ratio
compresses toward zero. Dividing 2021's LFCC scores by 11.51 instead of 0.53 delivers
the partner at ~1/470th of its proper variance, and **the fusion would switch LFCC off**.
The zero-shot arm would then lose for an arithmetic reason that has nothing to do with
whether zero-shot fusion works — the most misleading kind of negative result.

**What the original rule was actually protecting.** Label leakage and eval leakage. A
mean and a standard deviation taken over 2021 scores use **no labels whatsoever**; this
is ordinary test-time score normalisation, the same family as T-norm and Z-norm in
speaker verification. It does introduce a **transductive assumption** — a batch of target
data must be available at inference — and that must be stated wherever the arm is
reported. It does not touch the zero-shot property, which is about labels.

**Therefore both normalisation schemes are run, and the gap between them is the
measurement:**

| scheme | z-norm statistics from |
|---|---|
| `fit-norm` | the fitting partition — exactly as 9.8b.1 declared |
| `apply-norm` | the partition being scored, labels never read |

Running both is nearly free (fusion is minutes) and converts a methodological hazard into
a quantity: **how much does score-scale mismatch cost a zero-shot fusion?** Reporting only
the amended scheme would hide that; reporting only the original would misattribute an
arithmetic failure to the scientific question.

**Prediction, stated in advance:** `fit-norm` and `apply-norm` differ negligibly on the
**progress** arm (fit and application are both 2021, and P5c already bounded that residual
at 0.24 pp), while on the **dev** arm `apply-norm` beats `fit-norm` substantially —
because that is where the 21.7x mismatch lives. If `fit-norm` does *not* lose on the dev
arm, the mechanism above is wrong and the amendment was unnecessary.

#### 9.8c.3 Confidence intervals on the in-house fusion — declared before the run

Both arms have been applied to eval once each, as 9.8c.1 declared (PROGRESS_REPORT P7).
The point estimates are in hand and **carry no error bars**, which by this project's own
standard (P3) means the claims are not yet reportable. Five comparisons are fixed here,
before `bootstrap_ci` is re-run, because two of them are *expected to come back null* and
that is only credible if it is written down first.

**Why this cannot wait until after the write-up.** The gap between the in-house zero-shot
fusion and the 9.8b borrowed-partner fusion is **0.147 pp**. For calibration, P5c's
significant result cleared zero by 0.78 pp and P4's −1.62 pp came back *not
distinguishable*. Writing "beats the official fusion" now would almost certainly require
retracting it — so the framing of the headline is itself waiting on this measurement.

| # | comparison | what it tests | prediction |
|---|---|---|---|
| 1 | `inhouse_fusion_dev` vs `timepool_T150_aug` | the zero-shot fusion gain | **excludes zero** |
| 2 | `inhouse_fusion_progress` vs `timepool_T150_aug` | the label-fitted gain | **excludes zero**, wider margin than 1 |
| 3 | `inhouse_fusion_dev` vs `inhouse_fusion_progress` | price of target labels | **NOT distinguishable** (0.516 pp) |
| 4 | `inhouse_fusion_dev` vs `fusion_ours+2GMM` | in-house vs borrowed partners | **NOT distinguishable** (0.147 pp) |
| 5 | `inhouse_fusion_dev` vs CQCC-GMM | the recovered baseline claim | **excludes zero** |

**Comparison 5 is the one the whole of 9.8c was built for.** §9.8b.1a.1 forbids comparing
a fusion against a baseline it contains. The in-house fusion contains no borrowed system,
so this comparison is legitimate — the circularity is genuinely gone, not merely disclosed.

**Comparisons 3 and 4 are predicted null, and a null is the desired result in both.**
For 3, "labelled target-domain data buys nothing demonstrable" is a *stronger* claim for
the thesis than a measurable gain, because the zero-shot system is the one that keeps
every other claim intact. For 4, "matches the borrowed-partner fusion" is the honest and
more interesting statement — the 0.147 pp was never the point; equivalence *without*
borrowing systems or reading target labels is.

**Registration.** Both systems join `config.PHASE7_FUSION_SYSTEMS`, appended last so the
shared resample stream and `validate`'s `systems[:4]` are untouched, and **excluded from
the CI-width aggregate**, which backs the published 14.3x claim about the 14 zero-shot
single systems. **Control, as in 9.8b.4 C: every pre-existing system and comparison must
reproduce bit-for-bit** — git should show insertions only.

### 9.8 Score fusion — measured among our own systems, and it does not pay

Estimated on `progress` only, leaving `eval` untouched: mean-z fusion of the three best
systems gives **29.668%** against **29.795%** for the best single system — a gain of
**0.13 pp**. Recorded as a *negative* result so the time is not spent again. A trained
fusion (logistic regression on `progress`) might do somewhat better, but the naive
estimate says the headroom is small; the systems' rank correlations (0.375–0.876) are
apparently not decorrelated enough in the way fusion needs.

### 9.9 Things deliberately *not* in this list

- **Retuning anything against `eval`.** See 9.0.
- **A different backbone** (ResNet, AASIST). 6.3 chose LCNN specifically so the
  comparison against the official LFCC-LCNN isolates the front-end; swapping it forfeits
  the cleanest result in the project.
- **SSL front-ends (wav2vec2/XLS-R).** Still the right "future work" paragraph, still
  out of reach on a 4 GB GPU in this timeframe — see section 1.

### 9.10 Where the other directions sit

The extensions already sketched — **Phase 8** (ASV + t-DCF) and **Phase 9** (live demo:
keyword spotting, self-enrolled speaker verification), and any move beyond physical
access into logical access / deepfake detection — are orthogonal to everything above.
They broaden the project; 9.1–9.7 deepen the result that already exists. Both are
legitimate; doing a little of each badly is not.

One practical note for Phase 9 carried over from Phase 7: **this machine's binding
constraint is system commit and kernel resources, never VRAM or CPU.** Three separate
runs died on it (PROGRESS_REPORT 7.4, 7.4b, and the loky pool exhaustion at chunk 117).
A live demo loading a torch model alongside Whisper and an audio stack will meet the
same ceiling.
