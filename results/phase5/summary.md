# Phase 5 — Classical MFCC baseline (SVM + Random Forest)

Primary metric is EER on the speaker-disjoint resplit 2019 dev set. These are
*tuning* numbers, not the final result — 2021 PA eval is still untouched and gets
scored once in Phase 7.

A full factorial sweep of 6 subsample sizes × 4 C × 4 gamma = **112 SVM fits**, plus a Random Forest at every one of those sizes and
on the full train split. Per-size subsamples are independent stratified draws.

## Best configuration

**n_train=175,959, C=1.0, gamma=0.01** → dev EER **9.216%**

> **Boundary warning(s)** — the optimum may lie outside the searched ranges:

> - best n_train=175,959 is the ENTIRE train split -- dev EER was still improving with data, but no more training data exists to add

## Best SVM per training size

| n_train | best C | best gamma | dev EER | support vectors |
|---|---|---|---|---|
| 10,000 | 1.0 | scale | 11.395% | 3,570 |
| 20,000 | 1.0 | scale | 10.628% | 6,315 |
| 50,000 | 1.0 | scale | 9.896% | 13,478 |
| 80,000 | 1.0 | scale | 9.618% | 19,725 |
| 100,000 | 1.0 | scale | 9.423% | 23,634 |
| 150,000 | 1.0 | 0.01 | 9.363% | 33,358 |
| 175,959 | 1.0 | 0.01 | 9.216% | 37,943 |

## Random Forest per training size

| n_train | dev EER | ROC-AUC |
|---|---|---|
| 10,000 | 12.929% | 0.9466 |
| 20,000 | 12.623% | 0.9485 |
| 50,000 | 12.155% | 0.9518 |
| 80,000 | 12.057% | 0.9530 |
| 100,000 | 11.856% | 0.9539 |
| 150,000 | 11.752% | 0.9549 |
| 175,959 (full) | 11.736% | 0.9548 |

## Head-to-head at the winning size

| System | n_train | Dev EER |
|---|---|---|
| MFCC-SVM (tuned) | 175,959 | 9.216% |
| MFCC-RF (size-matched) | 175,959 | 11.736% |
| MFCC-RF (full train) | 175,959 | 11.736% |

## SVM supplementary metrics (at the EER threshold)

| Metric | Value |
|---|---|
| ROC-AUC | 0.9696 |
| Accuracy | 0.9078 |
| Precision (bonafide) | 0.7308 |
| Recall (bonafide) | 0.9079 |
| F1 (bonafide) | 0.8098 |

### Confusion matrix at the EER threshold

| | predicted **bonafide** | predicted **spoof** | total |
|---|---|---|---|
| **actual bonafide** | 12,771 <br>*(TP — correctly accepted)* | 1,296 <br>*(FN — genuine user rejected)* | 14,067 |
| **actual spoof** | 4,704 <br>*(FP — attack got through)* | 46,326 <br>*(TN — correctly rejected)* | 51,030 |
| **total** | 17,475 | 47,622 | 65,097 |

Read row-wise: 9.21% of genuine speech was rejected (FNR) and 9.22% of replay attacks were accepted (FPR) — equal by construction, since this is the EER operating point.

## Artifacts

- `models/svm_mfcc.joblib` — winning Pipeline (scaler + SVC), all 120 features
- `models/rf_mfcc_full.joblib`, `models/rf_mfcc_sub_<n>.joblib`
- `results/phase5_svm_sweep.csv` — every sweep point (the authoritative table)
- `results/phase5_rf_curve.csv`, `results/phase5_dev_scores_*.csv`
- `results/phase5_svm_sweep.png`, `results/phase5_rf_feature_importance.png`