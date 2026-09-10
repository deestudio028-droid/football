# Phase 4A — Feature Ablation Study Report

## 1. Status and scope

**Status: complete.** The ablation experiment ran to completion on the project's Windows environment (scikit-learn 1.9.0, Python 3.13.14) and produced `data/audit/phase4a_ablation_comparison.json` plus 8 per-fold prediction CSVs. This report is a factual account of that run, audited read-only against the authoritative artifact.

Phase 4A is an **ablation study**, not a model-selection or replacement exercise. Nothing in this report changes Phase 3 V1, which remains frozen: `MODEL_VERSION` v1.0, LogisticRegression, the approved 15-column feature contract, uncalibrated probabilities, and the locked 2025/26 final-test result.

## 2. Objective

Measure the incremental validation-set contribution of successive feature groups, using the frozen Phase 3 walk-forward protocol and an unchanged LogisticRegression configuration, so that any metric difference between tiers is attributable to the feature set alone:

- **A → B**: does shot / shots-on-target information add predictive value?
- **B → C**: does xG add predictive value where coverage actually exists?
- **C → D**: do the remaining eligible Phase 2 feature groups add value?

## 3. Dataset

`data/processed/features.db`, table `feature_rows`, filtered to `label_result IS NOT NULL`: **10,734 labeled rows** (confirmed in the artifact metadata). Target is the 3-class H/D/A result. Fixture `420450481` (ABANDONED) remains excluded. No database was read for anything other than loading; both are byte-identical to their pre-experiment checksums (§13).

## 4. Frozen feature tiers

Verified against the artifact's own recorded `feature_columns` — the tiers are strictly nested, and each step adds exactly the intended group and nothing else:

| Tier | Columns | Groups | Added vs. previous |
|---|---|---|---|
| **A** | 44 | goals_core, form, competition_id, strength | — (baseline) |
| **B** | 80 | A + shots_core, shots_on_core | 36 columns, **all shots / shots-on-target** (verified) |
| **C** | 102 | B + xg_core | 22 columns, **all xG** (verified) |
| **D** | 124 | C + goals_venue, tempo, discipline, league_home_advantage, league_mean_goals, venue_goal_diff, history_flags | 22 columns (verified list) |

A ⊂ B ⊂ C ⊂ D confirmed mechanically. This matches the frozen Phase 4A mapping exactly.

## 5. Walk-forward evaluation protocol

The unmodified Phase 3 boundaries — fold_1: 2020/21–2021/22 → 2022/23 (n_train 3,652 / n_val 1,827); fold_2: 2020/21–2022/23 → 2023/24 (5,479 / 1,752); fold_3: 2020/21–2023/24 → 2024/25 (7,231 / 1,752). Preprocessing (median impute + standardize + one-hot `competition_id`) is fit on each fold's training partition only. **2025/26 was not touched by this experiment** — the runner has no final-test code path at all.

## 6. xG coverage limitation and eligibility rule

xG is structurally absent from the early historical record. In both fold_1's and fold_2's training partitions, **22 of 22 required xG columns have zero observed values**. The recorded reason in the artifact, verbatim for fold_1 (fold_2 identical but for the name):

> `INVALID_FOR_XG: 22 of 22 required xG columns have ZERO observed values in fold_1's training partition (e.g. home_xg_for_per_match_last5). No imputation statistic can be learned from an entirely-empty training column, and filling one would require fabricating a value (forbidden). This fold is structurally unusable for xG-inclusive training; it is recorded as invalid rather than skipped or filled.`

For those folds the artifact records `fold_valid: false`, `status: INVALID_FOR_XG`, `xg_training_observed: false`, all five metrics `null`, and `probability_validity: not_applicable_no_model_trained` — **no model was trained**. Nothing was zero-filled, mean-filled, fabricated, or silently dropped. fold_3 is the only coverage-valid fold (0 of 22 columns empty).

Consequently **no 3-fold mean exists for C or D**, and the artifact refuses to compute one, recording instead: *"2 of 3 folds are invalid for this model (fold_1, fold_2). A 3-fold mean would be fabricated."*

## 7. Experimental results

### 7.1 Primary comparison — A vs B (both 3/3 folds valid)

| Metric | Model A | Model B | Δ (B−A) | Relative |
|---|---|---|---|---|
| **Log loss** ↓ | 1.004013383939 | **0.999379105697** | **−0.004634278243** | −0.4616% |
| Brier ↓ | 0.599919820526 | **0.596501695758** | −0.003418124768 | −0.5698% |
| Macro-F1 ↑ | 0.391737211473 | **0.403483604649** | +0.011746393176 | +2.9985% |
| Balanced accuracy ↑ | 0.446784492367 | **0.453292537805** | +0.006508045438 | +1.4566% |
| Accuracy ↑ | 0.512911664955 | **0.518413548173** | +0.005501883218 | +1.0727% |

B is better than A on **all five metrics**, not just log loss.

### 7.2 xG feasibility observations — C and D (fold_3 only)

**These are single-fold, coverage-valid observations. They are NOT comparable to A/B's 3-fold means and must not be ranked against them.**

| Model (fold_3 only) | Log loss | Brier | Macro-F1 | Balanced acc. | Accuracy |
|---|---|---|---|---|---|
| C (B + xG) | 1.021097 | 0.611678 | 0.430959 | 0.441021 | 0.478311 |
| D (C + remaining groups) | 1.011518 | 0.603897 | 0.449582 | 0.457469 | 0.497146 |

For orientation only — on that *same single fold*, A scored 1.008045 and B scored 0.996200 log loss. Even this same-fold comparison is a one-season observation, not a protocol-level result.

## 8. Detailed per-fold metrics

| Model | Fold | Log loss | Brier | Macro-F1 | Bal. acc. | Accuracy |
|---|---|---|---|---|---|---|
| A | fold_1 | 1.012337 | 0.605445 | 0.387117 | 0.436905 | 0.508484 |
| A | fold_2 | 0.991658 | 0.591820 | 0.399000 | 0.459242 | 0.519977 |
| A | fold_3 | 1.008045 | 0.602495 | 0.389095 | 0.444207 | 0.510274 |
| B | fold_1 | 1.010987 | 0.603614 | 0.394770 | 0.440156 | 0.509579 |
| B | fold_2 | 0.990950 | 0.591184 | 0.403680 | 0.461101 | 0.522831 |
| B | fold_3 | 0.996200 | 0.594708 | 0.412001 | 0.458620 | 0.522831 |
| C | fold_1 | — | — | — | — | INVALID_FOR_XG |
| C | fold_2 | — | — | — | — | INVALID_FOR_XG |
| C | fold_3 | 1.021097 | 0.611678 | 0.430959 | 0.441021 | 0.478311 |
| D | fold_1 | — | — | — | — | INVALID_FOR_XG |
| D | fold_2 | — | — | — | — | INVALID_FOR_XG |
| D | fold_3 | 1.011518 | 0.603897 | 0.449582 | 0.457469 | 0.497146 |

B beats A on log loss in **every one of the three folds** (−0.001350, −0.000708, −0.011845), so the mean improvement is not driven by a single fold. The margin is, however, an order of magnitude larger in fold_3 than in folds 1–2.

## 9. A→B incremental effect

Adding shots + shots-on-target (36 columns) reduced mean validation log loss by **0.004634278243** (−0.4616%), with consistent same-direction movement in Brier, macro-F1, balanced accuracy and accuracy, and a consistent sign across all three folds. The absolute magnitude is small.

## 10. Comparison with the V1 locked reference

V1's locked figures — log loss `1.012643607907126`, Brier `0.6059500599589396`, accuracy `0.500856653340948` — are quoted **as a frozen reference point only**. They were measured on the **2025/26 final test set**; every Phase 4A number above is a **validation-fold** measurement on 2022/23–2024/25. These are different evaluation sets and the numbers are **not directly comparable**. No claim in this report rests on comparing them.

## 11. Interpretation

- **Shots and shots-on-target provide a measurable incremental validation signal under this experiment**: B improves on A across the 3-fold mean for all five metrics, with a consistent per-fold direction. The effect is small in absolute terms.
- This is an associational result under a fixed ablation design with one model family and no tuning. It does not establish causality, and it does not imply the same margin would hold on unseen data or on the locked test season.
- **Nothing here shows xG to be useless.** C and D simply *cannot be evaluated* under the full historical walk-forward protocol, because folds 1–2 have no xG training signal at all. Their fold_3 figures are one coverage-valid season's observation, sitting in a range where a single season's noise is plausibly larger than the differences involved.
- The C/D fold_3 numbers should not be read as "xG hurt." On that fold, xG columns are ~97.8% null even where coverage exists, so their imputed values are near-constant — a model given 22 near-constant columns is mostly absorbing noise, which is a statement about *coverage*, not about xG's intrinsic value.

## 12. Limitations

- **xG is not evaluable under the current protocol.** Two of three folds are structurally invalid; the one valid fold has ~97.8% missing xG in training (min. 162 observed values per column out of 7,231 rows). Any conclusion about xG's value awaits either more post-cutover history or a purpose-designed, coverage-aware evaluation.
- **The A→B improvement is small** (−0.0046 mean log loss). Three folds is a small sample for distinguishing a real effect from favourable noise; no significance test was performed and none is claimed.
- **Single model family.** Only LogisticRegression was used, deliberately (to isolate the feature effect). A different learner could rank these tiers differently.
- **No tuning.** Larger feature sets may be disadvantaged relative to smaller ones at a fixed regularisation strength (C=1.0); B/C/D are not tuned for their dimensionality. This is a deliberate design choice, but it is a real confound when comparing tiers of very different width.
- **Validation-set results only.** No Phase 4A configuration has been evaluated on the locked 2025/26 test set, by design.

## 13. Integrity and reproducibility audit

Full read-only audit performed against the authoritative artifact:

- **Artifact complete and parseable**; metadata verified: scikit-learn 1.9.0, Python 3.13.14, 10,734 labeled rows, `"LogisticRegression (identical configuration to Phase 3 V1; no tuning)"`, primary basis = mean across folds 1–3, xG basis = coverage-valid folds only.
- **All 8 expected prediction CSVs present**, exact filename match, no extras.
- **Every CSV structurally valid**: exact columns; row counts match validation folds (1,827/1,752/1,752); fixture IDs match the fold exactly with no duplicates; no train/validation overlap; **no 2025/26 rows**; labels valid H/D/A; probabilities finite, non-negative, sum to 1, and pass the production `validate_probabilities()`; `y_pred` equals probability argmax.
- **Independent recomputation**: every metric recomputed from the raw CSVs matches the JSON to ≤3.3e-16 (floating-point noise). **No discrepancies.**
- **Mean arithmetic verified**: A and B means recomputed from per-fold records match the stored means to <1e-12.
- **C/D invalidity correctly recorded** for folds 1–2 (22/22 zero-observation columns, no model trained, null metrics, probability validity not applicable).
- **Reproducibility finding**: the new run's 8 CSVs are **byte-identical (MD5)** to the archived incomplete run's CSVs. The pipeline is fully deterministic — same fold boundaries, same fixed `random_state=0`, no random splitting anywhere. The earlier run was incomplete (no summary JSON), not incorrect.
- **Repository integrity**: `features.db` = `e7ebe7fc07040a5927683c35b6371e63`, `matches.db` = `fdeed042096fa1c851aaee6c84995247` — both unchanged. All six Phase 3 audit JSONs unchanged (checksums match the pre-experiment forensic record). Archived incomplete run untouched (8 files, original 09:59 timestamps, checksums intact). No stray experiment outputs anywhere else.
- **Test suite**: 304 tests, 304 passed, 0 failed, 0 errors, 32 skipped.

## 14. Conclusion

Under this experiment, **shots + shots-on-target provide a measurable incremental validation signal** over the goals+form+strength baseline: Model B improves Model A's 3-fold mean log loss by 0.004634278243 (−0.4616%) and improves Brier, macro-F1, balanced accuracy and accuracy as well, with a consistent direction across all three folds. The magnitude is small, and this is an associational finding under one fixed design — no causal claim is made.

**xG could not be evaluated across the full historical walk-forward protocol**, because folds 1 and 2 have zero observed xG values in their training partitions. Models C and D are not comparable to A/B on the 3-fold mean, and their fold_3 figures are single-season, coverage-valid observations only. **No conclusion that xG lacks value is drawn or supported by this study.**

**No V1 production contract was changed.** Phase 4A does not authorize replacing V1, and Model B is not designated a production model. Any such change would require its own evaluation, decision, and version bump.

## 15. Phase 4B / deferred-work boundary

Not started and out of scope: Phase 4B, hyperparameter tuning, xG redesign or coverage-aware re-evaluation, recalibration of any ablation tier, final-test evaluation of any Phase 4A configuration, Poisson/Monte Carlo simulation, betting logic, production prediction API, and UI. Each requires explicit authorization.

---

**Source artifacts:** `data/audit/phase4a_ablation_comparison.json` (authoritative), `data/audit/phase4a_predictions/` (8 CSVs), `data/audit/phase4a_feature_group_diagnostic.json` (design-stage coverage diagnostic), `data/audit/_incomplete_phase4a_run_20260817_0959/` (archived incomplete prior run — historical evidence only, not used for any result in this report).
