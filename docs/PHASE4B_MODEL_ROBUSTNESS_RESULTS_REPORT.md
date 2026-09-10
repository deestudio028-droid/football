# Phase 4B — Model Robustness / Sensitivity Study: Results Report

## 1. Status and scope

**Complete.** The regularization-sensitivity experiment ran on the project's Windows environment (scikit-learn 1.9.0, Python 3.13.14) and produced `data/audit/phase4b_robustness_comparison.json`, the authoritative artifact this report is built from and audited against. Every number below traces to that file.

This is a **robustness study**, not model selection and not a production decision. V1 remains frozen. Phase 4A remains frozen. No V2 is defined, and nothing here authorizes replacing V1.

## 2. Objective

Determine whether Model B's Phase 4A advantage over Model A (mean validation log loss −0.004634278243 at the default C=1.0) is robust to reasonable LogisticRegression regularization choices, or whether it depends on that single arbitrary C value.

## 3. Frozen protocol

Two feature sets — Model A (44 columns) and Model B (80 columns = A + shots + shots-on-target) — swept across a pre-declared 9-value C grid, evaluated on the three approved walk-forward folds. Every other setting is V1's, unchanged: preprocessing, median imputation, standardization, `competition_id` one-hot encoding, `CLASS_ORDER` [H,D,A], solver, `random_state=0`, and probability validation. No calibration, no random splitting, no xG, no feature engineering, no final-test evaluation. 2 × 9 × 3 = **54 training runs**, all present and all with 3/3 valid folds. Basis metric: mean validation log loss across folds 1–3.

**2025/26 was not used.** The artifact's evaluation data spans only validation seasons 2022/23, 2023/24, 2024/25, and training seasons 2020/21–2023/24; `2025/2026` appears in neither.

## 4. C grid

`0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0` — exactly the frozen grid, verified against the artifact. V1's C=1.0 sits inside it.

## 5. Phase 4A equivalence at C=1.0

`phase4a_equivalence_at_C1.all_match == true`. At C=1.0 the Phase 4B path reproduces Phase 4A's recorded means **exactly** (absolute difference 0.00e+00 for both models):

| Model | Phase 4A recorded | Phase 4B at C=1.0 | |Δ| |
|---|---|---|---|
| A | 1.004013383939449 | 1.004013383939449 | 0 |
| B | 0.999379105697 (0.9993791056968738) | 0.9993791056968738 | 0 |

This confirms Phase 4B is genuinely V1's estimator with C as the only varied parameter — the comparison is valid.

## 6. Model A results across C (mean validation log loss)

| C | Mean log loss |
|---|---|
| 0.01 | 1.002237508550 |
| 0.1 | 1.003933725652 |
| 0.25 | 1.004148711352 |
| 0.5 | 1.004133853189 |
| 1.0 | 1.004013383939 |
| 2.0 | 1.003846842191 |
| 4.0 | 1.003741514882 |
| 10.0 | 1.003571137906 |
| 100.0 | 1.003413535943 |

Model A's mean log loss varies only within a narrow band (~1.00224 to ~1.00415) across four orders of magnitude of C — it is largely insensitive to regularization strength, with a shallow best at the lowest grid value.

## 7. Model B results across C (mean validation log loss)

| C | Mean log loss |
|---|---|
| 0.01 | 0.995716619851 |
| 0.1 | 0.998962467849 |
| 0.25 | 0.999330913562 |
| 0.5 | 0.999414956774 |
| 1.0 | 0.999379105697 |
| 2.0 | 0.999402198260 |
| 4.0 | 0.999115086946 |
| 10.0 | 0.999108186454 |
| 100.0 | 0.999186312655 |

Model B's mean log loss is similarly stable (~0.99572 to ~0.99942), also with its best at the lowest grid value.

## 8. A vs B robustness comparison

**B improves on A's mean validation log loss at every one of the 9 C values** (B−A negative throughout):

| C | B − A (mean log loss) |
|---|---|
| 0.01 | −0.006520888699 |
| 0.1 | −0.004971257803 |
| 0.25 | −0.004817797790 |
| 0.5 | −0.004718896415 |
| 1.0 | −0.004634278243 |
| 2.0 | −0.004444643931 |
| 4.0 | −0.004626427936 |
| 10.0 | −0.004462951452 |
| 100.0 | −0.004227223288 |

Delta range: **−0.006520888699** (largest advantage, at C=0.01) to **−0.004227223288** (smallest, at C=100.0). The advantage never disappears, never reverses, and stays within a tight band; it does not depend on any single grid point. Leave-one-C-out: dropping any one C value, B still wins the mean at all 8 remaining C values.

At V1's C=1.0 the delta is −0.004634278242575118, reproducing the Phase 4A figure exactly.

**B's advantage over A is robust across the tested C grid.**

## 9. Per-fold consistency

At **8 of the 9** C values, B beats A in **all three folds individually**. The one exception is **C=100.0, fold_2**, where B is marginally worse than A (B 0.991884354, A 0.991682932, B−A = +0.000201421) while still winning folds 1 and 3 decisively (−0.001329 and −0.011554) and therefore winning the C=100.0 mean. So B wins the mean at 9/9 C values and wins every individual fold at 8/9 C values. The single per-fold reversal occurs only at the highest (least-regularized) grid point and is very small.

## 10. Secondary metrics

Directional consistency is complete: **B is better than A at all 9 C values on Brier, macro-F1, balanced accuracy, and accuracy** (9/9 each, recomputed and matching the artifact). The improvement is not a log-loss-only artifact — every recorded metric moves the same way.

## 11. Best observed C values

- Model A: **best observed value within the frozen grid was C=0.01** (mean log loss 1.002237508550).
- Model B: **best observed value within the frozen grid was C=0.01** (mean log loss 0.995716619851).

These are the best *observed* values within the frozen grid. They are **not** claimed to be optimal C values.

## 12. Edge-of-grid limitation

Both models achieved their best mean log loss at **C=0.01, the lower boundary of the grid** (`q10_edge_of_grid_flags`: both `true`). This means the true optimum may lie **below 0.01** — i.e. at stronger regularization than any value tested. This is reported as a limitation, not acted upon. **No larger or extended grid is run or recommended in this step.**

## 13. Interpretation

Model B's advantage over Model A is stable across a C sweep spanning four orders of magnitude: B wins the 3-fold mean at every C, wins every individual fold at 8 of 9 C values, and improves all four secondary metrics at every C. The advantage therefore does not depend on the arbitrary default C=1.0. Both models are relatively insensitive to C in absolute terms, and both prefer the strongest regularization tested.

The artifact's embedded interpretation guard is preserved verbatim: *"Robustness is descriptive only. No statistical test was performed, so no significance claim may be made. No causal claim. This does not select a production model and does not authorize changing V1."* This report makes no significance claim, no causal claim, no model-selection claim, and no production-deployment claim.

## 14. Limitations

- Three folds is a small basis; consistent sign across C is suggestive of robustness, not proof, and no significance test was performed or is claimed.
- Both models are best at the lower grid boundary (C=0.01), so the optimum may lie below the tested range; this is not investigated here.
- Only regularization strength was swept; solver, penalty and class weighting were held at V1 values, so this measures sensitivity to C alone.
- The A→B advantage remains **small in absolute terms** (roughly 0.004–0.007 log loss) at every C — robustness of the difference does not make the difference large.
- Validation-only. Nothing here was evaluated on the locked 2025/26 test set, so no statement about generalization to the final test season is possible, and Phase 4B validation metrics are **not** compared against V1's 2025/26 final-test metrics.

## 15. Integrity / reproducibility audit

- **Artifact** complete, valid JSON, 54 cell records, frozen 9-value C grid, both models 44/80 columns with 3/3 valid folds at every C.
- **Independent recomputation**: every mean (all metrics, all model/C combinations) recomputed from the per-fold records matches the stored summary to **0.00e+00**. Every stored B−A delta matches recomputation to <1e-12. Every per-fold win flag, secondary-metric consistency count, leave-one-C-out count, best-C, and edge-of-grid flag independently reproduced.
- **Phase 4A equivalence** at C=1.0 exact (Δ=0), cross-checked against `phase4a_ablation_comparison.json` directly.
- **2025/26** absent from all Phase 4B training and validation data.
- **V1 / Phase 4A integrity**: `features.db` = `e7ebe7fc07040a5927683c35b6371e63`, `matches.db` = `fdeed042096fa1c851aaee6c84995247`, `phase4a_ablation_comparison.json` = `075b0686bce20bfce9f7289fa37076c0`, all Phase 3 audit artifacts, and `config.py`/`train.py`/`run_experiments.py`/`ablation.py`/`run_ablation.py` all unchanged from the pre-Phase-4B baseline.
- **Test suite**: 351 tests, 351 passed, 0 failed, 0 errors, 34 skipped.

## 16. Conclusion

**Model B's advantage over Model A is robust across the tested C grid.** B improves the 3-fold mean validation log loss over A at all 9 C values (delta range −0.006521 to −0.004227), improves every secondary metric at all 9 C values, and beats A in every individual fold at 8 of the 9 C values (the sole exception being a marginal fold_2 reversal at the extreme C=100.0). The Phase 4A A→B result is therefore not an artifact of the default C=1.0. The advantage remains small in absolute terms. Both models were best at C=0.01, the lower grid boundary, so the true optimum may lie below the tested range — reported as a limitation, not pursued here.

## 17. Explicit boundary: no V1 replacement authorized

Phase 4B is a validation-only robustness study. It does not select a production model, does not define V2, and does not authorize replacing V1 or changing `MODEL_VERSION`. Model B is **not** the production model. Any V1 replacement would require its own scope, a final-test protocol, and explicit authorization. Not started and out of scope: Phase 4C, any extended C grid, calibration, xG redesign, final-test evaluation, Poisson/Monte Carlo, betting logic, API, UI.
