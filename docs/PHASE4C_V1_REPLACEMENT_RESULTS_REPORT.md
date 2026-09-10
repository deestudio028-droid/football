# Phase 4C — V1 vs Model B Final-Test Comparison: Results Report

## 1. Status and scope

The pre-registered final-test comparison authorized under `docs/PHASE4C_V1_REPLACEMENT_DECISION_PROTOCOL.md` has been executed **once** on the project's Windows environment (scikit-learn 1.9.0, Python 3.13.14, Windows-11). The authoritative artifact is `data/audit/phase4c_final_comparison.json`, supported by `data/audit/phase4c_prerun_manifest.json` and two per-fixture prediction CSVs.

This report is an **audit-based factual record**. It makes no promotion decision. `MODEL_VERSION` remains `v1.0`, no V2 exists, and no production change has been made.

Authorized frozen decisions, confirmed present in both the manifest and the result artifact: **C = 1.0** (Option C-1), **no epsilon tie-threshold authorized** (`epsilon_authorized: false`).

## 2. Independent audit methodology

Every number below was **recomputed from the raw per-fixture prediction CSVs** and compared against the artifact — not read from it and restated. Recomputation used the project's own `evaluate.py` implementations plus independently coded per-class precision/recall/F1.

**Result: maximum |recomputed − artifact| across every metric, both arms, was 0.00e+00.** Confusion matrices matched exactly. All per-class figures matched. All paired deltas matched to 0.00e+00. No discrepancy of any kind was found.

## 3. Structural and validity verification (25 independent checks, all passed)

| Check | Result |
|---|---|
| 1,751 predictions per arm | PASS (both) |
| Identical `fixture_id` sets between arms | PASS |
| Row-order aligned (paired per fixture) | PASS |
| No duplicate fixtures | PASS (both) |
| No missing rows vs. the dataset's 2025/26 labeled set | PASS |
| Identical `y_true` across arms | PASS |
| `y_true` matches the source dataset | PASS |
| Abandoned fixture `420450481` absent | PASS |
| No train/test fixture overlap | PASS |
| Temporal safety verified | PASS |
| Class order `["H","D","A"]` | PASS |
| Probabilities finite, non-negative, sum to 1, pass `validate_probabilities` | PASS (both) |
| `y_pred` equals probability argmax | PASS (both) |
| n_train = 8,983 / n_test = 1,751 match dataset | PASS |
| Test class distribution H 771 / A 535 / D 445 | Matches artifact |

The artifact's own `validity_checks.all_passed` is `true` with an empty `failures` list, consistent with my independent reproduction.

## 4. V1 reproduction check

The V1 arm reproduces the locked Phase 3 final-test metrics **exactly** — absolute difference `0.00e+00` on log loss, Brier, accuracy, macro-F1, balanced accuracy and n, well inside the 1e-9 tolerance, with an identical confusion matrix. This confirms the Phase 4C harness is the same evaluation path that produced V1's locked numbers, so the two arms are genuinely comparable.

**V1 reproduction: PASS.**

## 5. Headline results (2025/26, identical 1,751 rows, paired by fixture_id)

| Metric | V1 (15 features) | Model B (80 features) | Δ (B − V1) | Direction |
|---|---|---|---|---|
| **Log loss** ↓ | 1.012643607907126 | **0.996048764906360** | **−0.016594843000765858** | B better |
| Brier ↓ | 0.605950059958940 | **0.594718631312907** | −0.011231428646032882 | B better |
| Accuracy ↑ | 0.500856653340948 | **0.515134209023415** | +0.014277555682467247 | B better |
| Macro-F1 ↑ | 0.371275391581008 | **0.393849150581964** | +0.022573759000955207 | B better |
| Balanced accuracy ↑ | 0.432179756031532 | **0.448954879712117** | +0.016775123680584203 | B better |

**Model B improved final-test log loss by 0.016594843000765858.** Brier, accuracy, macro-F1 and balanced accuracy all improved as well — every recorded metric moved in Model B's favour.

## 6. Confusion matrices and per-class performance

Rows = true H/D/A, columns = predicted H/D/A.

- **V1:** `[[599, 2, 170], [293, 0, 152], [257, 0, 278]]`
- **Model B:** `[[598, 4, 169], [273, 8, 164], [231, 8, 296]]`

| Class | Arm | Precision | Recall | F1 | TP / support |
|---|---|---|---|---|---|
| H | V1 | 0.521323 | 0.776913 | 0.623958 | 599 / 771 |
| H | Model B | 0.542650 | 0.775616 | 0.638548 | 598 / 771 |
| **D** | **V1** | **0.000000** | **0.000000** | **0.000000** | **0 / 445** |
| **D** | **Model B** | **0.400000** | **0.017978** | **0.034409** | **8 / 445** |
| A | V1 | 0.463333 | 0.519626 | 0.489868 | 278 / 535 |
| A | Model B | 0.470588 | 0.553271 | 0.508591 | 296 / 535 |

**Draw-class performance remains weak despite improvement.** V1 predicted Draw correctly zero times out of 445 actual draws (F1 = 0.000). Model B predicted Draw correctly 8 times out of 445 — a recall of 1.8% and F1 of 0.034. This is an improvement from literally nothing to almost nothing. Model B has not solved the draw problem; it has barely moved it. Any downstream use that depends on identifying draws should treat both models as effectively unable to do so.

Model B's gains come mainly from the Away class (recall 0.520 → 0.553, 18 more correct) and slightly better Home precision, alongside better-calibrated probabilities overall (the log-loss and Brier improvements).

## 7. Governance distinctions

**1. Numerical improvement — YES, unambiguous.** Model B improves every recorded metric on the final test set, with the primary metric (log loss) improving by 0.0166. All deltas were independently reproduced to 0.00e+00.

**2. Practical improvement — NOT ESTABLISHED HERE.** Whether a 0.0166 log-loss reduction (and ~1.4 percentage points of accuracy) is *practically* meaningful for the client's use case is a judgment this report cannot make. For context, both models remain near 50% accuracy on a 3-class problem and both are effectively unable to predict draws. The improvement is real but modest, and it does not change the qualitative character of the system's output.

**3. Statistical uncertainty — NOT ESTIMATED.** **Only one final-test season exists (n = 1,751 matches, a single realization of "the future").** There is no distribution of held-out seasons from which to estimate variance, so no confidence interval, no standard error, and no significance test is available or claimed. The observed deltas are point estimates of unknown precision. **No epsilon tie-threshold was authorized**, and none has been invented here; consequently this report does not and cannot assert that the improvement exceeds any pre-registered bar of practical or statistical size — it asserts only that the improvement is *numerically* present and consistent in direction.

**4. Model-risk — MATERIAL, must be weighed.** Three specific risks:
- **Model B uses 80 features versus V1's 15** — a 5.3× increase in feature-surface, with correspondingly greater exposure to upstream data-quality issues, coverage gaps, and maintenance burden.
- **Model B is NOT a strict superset of V1**: it omits `league_home_advantage_season`, a V1-approved feature that the frozen Phase 4A tier mapping placed in Tier D. Verified directly — the set difference V1 \ B is exactly `{league_home_advantage_season}`. So this is not "V1 plus shots"; it is a different, overlapping feature set. Adopting Model B would mean *dropping* a feature V1 currently relies on, which was never separately evaluated.
- **Shots / shots-on-target coverage dependence**: Model B's added families carry the coverage characteristics documented in Phase 2/4A. Their availability for future fixtures is a live operational dependency V1 does not have.

**5. V2 eligibility — see §8.** **6. Production authorization — NOT GRANTED; out of scope for this phase entirely.**

## 8. Decision zone (per pre-registered protocol §6/§8)

Evaluating strictly against the pre-registered rules:

- **Zone E (protocol violation)** — ruled out. All locked checksums matched before and after execution; V1 reproduction passed exactly; all validity checks passed; the runner was write-once and wrote only to `phase4c_*` paths; no tuning occurred after results were seen; the comparison ran once.
- **Zone A (clearly worse)** — ruled out. Δ < 0 on the primary metric and every secondary metric favours B.
- **Zone B (effectively tied)** — this zone is defined by an epsilon that was **not authorized**. It cannot be objectively applied, and the observed improvement is not zero.
- **Zone C (meaningful but insufficient)** — the protocol reserves this for cases where secondary metrics disagree in direction or the improvement rests on an unstable subset. Here **all** secondary metrics agree in direction, and the direction is consistent with Phase 4A/4B validation findings.
- **Zone D (eligible for explicit V2 review)** — the protocol's conditions are: Δ clearly negative on log loss, all secondary metrics directionally consistent with the Phase 4B validation picture, all §5 validity checks passed, and every §4 lock verified intact. **All four conditions are satisfied.**

### **DECISION ZONE: D — ELIGIBLE FOR EXPLICIT V2 REVIEW**

**Zone D means eligible for review, NOT automatic promotion.** Per protocol §8 and §9, reaching Zone D unlocks a human V2 review and nothing else. It does not make Model B a V2 candidate, does not make it production, and does not authorize any version change. The reviewer must weigh the numerical improvement against the practical, uncertainty and model-risk considerations in §7 — in particular the single-season uncertainty, the 80-vs-15 feature expansion, the dropped `league_home_advantage_season`, and the still-broken draw class.

## 9. Integrity and reproducibility audit

- **All 13 locked inputs** (§11 of the protocol) verified byte-identical **before** execution (recorded in the pre-run manifest) and **again after** the audit: `features.db` `e7ebe7fc07040a5927683c35b6371e63`, `matches.db` `fdeed042096fa1c851aaee6c84995247`, `phase3_model_comparison.json` `616279914b2730749d52eae15b5f96b9`, `phase4a_ablation_comparison.json` `075b0686bce20bfce9f7289fa37076c0`, `phase4b_robustness_comparison.json` `97d0f8b8674c9bf8598b6c3d6b7c825c`, `config.py` `c2ed32cb53ec34199fd245624afea4dd`, `train.py` `21425459195311492f49e73f5ae38fe0`, plus `splits.py`, `evaluate.py`, `data.py`, `baselines.py`, `run_experiments.py`.
- **Pre-run manifest** confirms the frozen spec was recorded *before* 2025/26 was opened: C=1.0, no epsilon, estimator kwargs `{max_iter: 2000, C: 1.0, random_state: 0}`, calibration `none`, class order `[H, D, A]`, V1 15 features / Model B 80 features, `model_version_unchanged: v1.0`.
- **Independent recomputation**: 0.00e+00 maximum deviation across all metrics, both arms, plus per-class figures, confusion matrices and paired deltas.
- **No artifact overwritten**; no V2 artifact, no unauthorized runner, no Phase 4D output exists.
- **V1 contract unchanged**: `MODEL_VERSION` v1.0, 15 approved columns, `FINAL_TEST_SEASONS` ('2025/2026',).

## 10. Conclusion

Model B improved final-test log loss by **0.016594843000765858**, and improved Brier, accuracy, macro-F1 and balanced accuracy as well. The V1 arm reproduced its locked Phase 3 metrics exactly, all validity checks passed, and every figure was independently reproduced to 0.00e+00. The comparison is valid and protocol-compliant.

Draw-class performance remains weak despite improvement (V1 F1 0.000 → Model B F1 0.034, recall 1.8%). Model B uses 80 features versus V1's 15 and is **not** a strict superset of V1, because it omits `league_home_advantage_season`. Only one final-test season exists, so uncertainty is not estimated from multiple held-out seasons. No epsilon was authorized, so no claim of statistical significance or of a pre-registered "meaningful" threshold is made.

The pre-registered outcome is **Zone D — eligible for explicit V2 review**. `MODEL_VERSION` remains **v1.0**. No V2 is created. No production change is made. The next step is a separate, explicit human decision.

---

## Audit verdict

```
AUDIT_STATUS:          PASS — no discrepancies; all figures independently reproduced to 0.00e+00
DECISION_ZONE:         D — ELIGIBLE FOR EXPLICIT V2 REVIEW
V2_ELIGIBLE:           YES (eligible for review only — not a V2 candidate, not promoted)
PRODUCTION_AUTHORIZED: NO
MODEL_VERSION:         v1.0 (unchanged)
```
