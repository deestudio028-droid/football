# Phase 5A — Dropped V1 Feature Validation Experiment: Results

**Authoritative source:** `data/audit/phase5a_dropped_feature_comparison.json`. Every value in this document is transcribed from that artifact. The experiment was **not** rerun and nothing was recomputed to produce this document.

## 1. Purpose and question

Phase 5 identified that the approved V2 candidate (Model B, 80 features) omits exactly one feature that V1 uses: `league_home_advantage_season`. That omission arose as a side effect of the frozen Phase 4A tier mapping, not from evidence that the feature should be removed. Phase 5A was authorized to supply that evidence.

The question, verbatim from the artifact:

> *Does adding V1's omitted feature 'league_home_advantage_season' to Model B improve or worsen it under the frozen walk-forward validation protocol?*

This was a single-question experiment — not a feature search, not model selection, not hyperparameter tuning.

## 2. Scope

| Property | Value |
|---|---|
| `validation_only` | `true` |
| **`final_test_used`** | **`false`** |
| Artifact note | *"2025/26 is never loaded, evaluated, or referenced by this experiment."* |
| `model_version_unchanged` | `v1.0` |
| `creates_v2` | `false` |
| Labeled rows available | 10,734 |

Environment: scikit-learn `1.9.0`, Python `3.13.14`, `Windows-11-10.0.26200-SP0`.

## 3. The two arms

Verified mechanically before training (`arm_verification`, `verified: true`):

| Arm | Features |
|---|---|
| `model_b` | **80** |
| `model_b_plus_home_advantage` | **81** |

- `B_PLUS \ B` = **`["league_home_advantage_season"]`** — exactly one added feature
- `B \ B_PLUS` = `[]` — nothing removed
- Intersection size = **80**

The only difference between the arms was the single feature under test.

## 4. Frozen configuration

| Item | Value |
|---|---|
| Estimator | `LogisticRegression` |
| `max_iter` | `2000` |
| `C` | `1.0` |
| `random_state` | `0` |
| Calibration | **`none`** |
| Class order | `["H", "D", "A"]` |

No hyperparameter was varied. No calibration was applied. No preprocessing was modified.

## 5. Folds

The frozen three walk-forward folds, unchanged:

| Fold | Train | Validate | n_train | n_val |
|---|---|---|---|---|
| fold_1 | 2020/2021, 2021/2022 | 2022/2023 | 3,652 | 1,827 |
| fold_2 | 2020/2021, 2021/2022, 2022/2023 | 2023/2024 | 5,479 | 1,752 |
| fold_3 | 2020/2021, 2021/2022, 2022/2023, 2023/2024 | 2024/2025 | 7,231 | 1,752 |

2025/26 appears in no fold. All three folds passed every validity check (`all_passed: true`, `failures: []`).

## 6. Fold-level metrics

**Model B (80 features)**

| Fold | Log loss | Brier | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---|---|---|---|
| fold_1 | 1.010986668383243 | 0.6036136927857372 | 0.5095785440613027 | 0.39477031576496896 | 0.44015616277590497 |
| fold_2 | 0.9909502394658608 | 0.5911835324345082 | 0.5228310502283106 | 0.4036798772859716 | 0.4611014174152919 |
| fold_3 | 0.996200409241518 | 0.594707862053424 | 0.5228310502283106 | 0.4120006208960765 | 0.4586200332249111 |

**Model B + `league_home_advantage_season` (81 features)**

| Fold | Log loss | Brier | Accuracy | Macro-F1 | Balanced accuracy |
|---|---|---|---|---|---|
| fold_1 | 1.0121835067733609 | 0.6041604725405312 | 0.5128626163108921 | 0.3962908713178171 | 0.4427593353015277 |
| fold_2 | 0.9905619327811397 | 0.5909432291023855 | 0.5256849315068494 | 0.4061054603300666 | 0.4634916415556463 |
| fold_3 | 0.9963010343658607 | 0.5947495057719464 | 0.5234018264840182 | 0.41217262106662506 | 0.45895012508744265 |

## 7. Per-fold deltas (B_PLUS − B)

For log loss and Brier, negative = B_PLUS better. For accuracy, macro-F1 and balanced accuracy, positive = B_PLUS better.

| Fold | Log loss Δ | B_PLUS better? | Brier Δ | Accuracy Δ | Macro-F1 Δ | Balanced acc. Δ |
|---|---|---|---|---|---|---|
| fold_1 | +0.0011968383901179713 | No | +0.0005467797547940023 | +0.003284072249589487 | +0.0015205555528481352 | +0.0026031725256227545 |
| fold_2 | −0.00038830668472111807 | **Yes** | −0.00024030333212265997 | +0.0028538812785388057 | +0.0024255830440950144 | +0.002390224140354391 |
| fold_3 | +0.00010062512434272364 | No | +0.00004164371852244386 | +0.0005707762557076723 | +0.00017200017054858074 | +0.0003300918625315785 |

## 8. Mean metrics across the three folds

| Metric | Model B | Model B + feature | Mean Δ (B_PLUS − B) | B_PLUS better? |
|---|---|---|---|---|
| **Log loss (primary)** | **0.9993791056968738** | **0.9996821579734537** | **+0.000303052276579896** | **No** |
| Brier | 0.5965016957578898 | 0.5966177358049544 | +0.0001160400470645584 | No |
| Accuracy | 0.5184135481726413 | 0.5206497914339199 | +0.002236243261278581 | Yes |
| Macro-F1 | 0.40348360464900573 | 0.4048563175715029 | +0.001372712922497188 | Yes |
| Balanced accuracy | 0.4532925378053693 | 0.45506703398153886 | +0.0017744961761695377 | Yes |

Standard deviations across folds — Model B log loss `0.008483030288239995`; Model B + feature log loss `0.009145020980936069`.

## 9. Classification

| Field | Value |
|---|---|
| `case` | **`CASE_B_worsens_mean_log_loss`** |
| `conclusion` | **`RETAIN_OMISSION_SUPPORTED`** |
| `mean_log_loss_delta_plus_minus_b` | `+0.000303052276579896` |
| `n_folds_where_b_plus_better` (log loss) | **1** of 3 |
| `direction_consistent_across_folds` | `false` |

Rationale, verbatim from the artifact:

> *"Adding the feature worsened mean validation log loss. The evidence supports keeping it omitted."*

## 10. Honest reading of a mixed picture

The conclusion follows the pre-registered decision rule, which keys on the **primary metric**: mean validation log loss worsened when the feature was added (+0.000303052276579896), which is `CASE_B` and therefore `RETAIN_OMISSION_SUPPORTED`.

The underlying evidence is nonetheless **mixed**, and that must be recorded rather than smoothed over:

- On log loss, B_PLUS was better in only **1 of 3 folds** (fold_2); `direction_consistent_across_folds` is `false`.
- Brier also favoured Model B (mean Δ +0.0001160400470645584).
- But **accuracy, macro-F1 and balanced accuracy all favoured B_PLUS** on the mean (+0.002236243261278581, +0.001372712922497188, +0.0017744961761695377 respectively), and all three favoured B_PLUS in every individual fold.

So the feature did not help the probability-quality metrics (log loss, Brier) that drive selection in this project, while slightly helping the class-assignment metrics. All differences are very small in absolute terms — the largest mean delta on any metric is under 0.003.

The decision rule was fixed before the experiment ran and keys on log loss, so the outcome is `RETAIN_OMISSION_SUPPORTED`. What this experiment established is narrow and should be stated narrowly: **adding `league_home_advantage_season` did not improve the primary metric, so removing it is not an unevidenced deletion.** It did not establish that the feature is harmful or worthless.

## 11. Interpretation guard

Verbatim from the artifact:

> *"Validation-only evidence. No epsilon threshold was defined, so no delta may be called meaningful or significant. No statistical test was performed. No causal claim. This does not select a production model, does not create V2, and does not authorize any change to MODEL_VERSION."*

**No statistical significance claim is made. No causal claim is made.** No epsilon threshold exists, so no delta in this document is described as meaningful. Three folds is a small basis and no variance-based inference was attempted.

## 12. Governance status

| Item | Status |
|---|---|
| `MODEL_VERSION` | **`v1.0`** — unchanged |
| V2 created | **No** |
| Production authorization | **None** |
| Final-test (2025/26) used | **No** (`final_test_used: false`) |
| Candidate feature contract | Unchanged at **80 columns**; `league_home_advantage_season` remains **omitted** |
| Condition 1 | **CLOSED — `RETAIN_OMISSION_SUPPORTED`** |

This experiment resolved Phase 5 Condition 1 only. It conferred no other authorization.
