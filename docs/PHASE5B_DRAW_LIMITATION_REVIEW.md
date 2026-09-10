# Phase 5B — Draw Limitation Governance Review (Condition 2)

## 1. Status

**Governance review only.** No model was trained, no threshold or class weight tuned, no calibration run, no experiment executed, no candidate modified, and 2025/26 was **not** re-evaluated — the existing frozen Phase 4C prediction artifacts were read only. `MODEL_VERSION` remains `v1.0`. No V2 exists. No production authorization exists.

**Condition 2 outcome: `DRAW_LIMITATION_ACCEPTED`** — scoped strictly as set out in §6–§8.

## 2. Integrity verification

All 17 locked checksums matched: `features.db` `e7ebe7fc07040a5927683c35b6371e63`, `matches.db` `fdeed042096fa1c851aaee6c84995247`, Phase 3 artifacts, `phase4a_ablation_comparison.json` `075b0686bce20bfce9f7289fa37076c0`, `phase4b_robustness_comparison.json` `97d0f8b8674c9bf8598b6c3d6b7c825c`, `phase4c_final_comparison.json` `effd9e54130b2bc5aaf51cd643962396`, `phase4c_prerun_manifest.json` `4f166d1826edbf7bfe1e022639a8b1f4`, both Phase 4C prediction CSVs, and all V1/production source files.

`MODEL_VERSION` = `v1.0`. No V2 module or artifact exists. No production code changed. No new final-test result generated — the Phase 4C artifact set is unchanged (`phase4c_final_comparison.json`, `phase4c_predictions`, `phase4c_prerun_manifest.json`).

**Phase 5A artifact verified present and independently confirmed:** `data/audit/phase5a_dropped_feature_comparison.json` (md5 `3ceb7090f9e19c106d88eaa7a8848818`) reports `conclusion = RETAIN_OMISSION_SUPPORTED`, `case = CASE_B_worsens_mean_log_loss`, Model B mean validation log loss `0.9993791056968738`, Model B + feature `0.9996821579734537`, delta `+0.000303052276579896`, and `final_test_used: false`.

## 3. Existing draw evidence (recomputed from the frozen Phase 4C prediction CSVs)

Every figure below was recomputed from the raw probabilities and matched the stored Phase 4C artifact exactly.

| Measure | V1 | Model B |
|---|---|---|
| Draw support | 445 of 1,751 | 445 of 1,751 |
| Actual draw rate | 0.254140491147915 | 0.254140491147915 |
| Correct draws (TP) | **0** | **8** |
| Draw recall | 0.000000 | **0.017977528089887642** |
| Draw precision | 0.000000 | 0.400000 |
| Draw F1 | 0.000000 | **0.03440860215053764** |
| Draw is argmax | 2 / 1,751 | 20 / 1,751 |
| Mean p_draw | 0.258630595987114 | 0.255008484107616 |
| Max p_draw | 0.362594107324726 | 0.396027912873153 |
| Min p_draw | 0.061921618950657 | 0.070758542141984 |
| Mean p_draw − actual rate | +0.004490104839199 | **+0.000867992959701** |
| p_draw ranked 1st / 2nd / 3rd | 0.001142 / 0.598515 / 0.400343 | 0.011422 / 0.649914 / 0.338664 |
| p_draw > 1/3 · > 0.40 · > 0.50 | 15 · 0 · 0 | 101 · 0 · 0 |

## 4. Probability-mass versus hard-class behaviour

This is the decisive mechanical finding, and it is supported directly by the artifacts:

| Arm | Σ p_draw (expected draws by probability mass) | Actual draws | Draws predicted by argmax |
|---|---|---|---|
| V1 | 452.86 | 445 | **2** |
| Model B | **446.52** | **445** | **20** |

Model B's summed draw probability over the 1,751 fixtures implies 446.52 draws against 445 actually observed — a discrepancy of about 1.5 draws across a full season. **As a probability estimator for draws in aggregate, Model B is close to the observed frequency.** As a hard-class assigner, it identifies 20 of 445.

The mechanism is visible in the distribution: `p_draw` never exceeds 0.396 in any of the 1,751 predictions (and never exceeds 0.40 for either arm). Draw is ranked second of three in 65.0% of Model B's predictions and third in 33.9% — it is almost never the single largest of three probabilities, because it would have to exceed both H and A simultaneously to become the argmax.

**Both statements are true simultaneously and must be reported together:** Model B's aggregate draw probability mass is close to the observed draw frequency, while hard Draw argmax classification remains very weak. Model B does **not** solve draw classification, and neither model should be described as a reliable Draw-class predictor.

## 5. Downstream requirement evidence

Searched the repository documentation and source for any stated requirement bearing on H/D/A probabilities, draw probability, a predicted result class, winner prediction, scorelines, Poisson, betting/decision logic, API consumers, and UI consumers.

**What the repository does establish:**

1. **The V1 output contract is probabilities, not a class.** `docs/PHASE3_MODEL_SPEC.md` §Output contract: *"`predict_proba`-style output: an array/row of exactly 3 floats, in `CLASS_ORDER = ["H", "D", "A"]` order, each finite, non-negative, summing to ~1"*. A predicted class is not part of the output contract at all.
2. **The client-requirement record asks for probabilities.** `docs/ODD_ALERTS_V1_DATA_SUFFICIENCY.md` Phase 7 capability table lists **B — "Win/Draw/Loss probability"** and **J — "Uncertainty / confidence estimates"**. `docs/PHASE3_MODEL_DEVELOPMENT_DESIGN_REPORT.md` records the client-facing requirement as *"which team is more likely to win," "draw probability"* — both comparative/probabilistic formulations.
3. **No stated requirement for hard Draw classification exists anywhere** in the repository. Every `argmax` occurrence in the codebase is internal evaluation or diagnostic machinery (`evaluate.py`, `calibration.py`, the experiment runners) — none is an output-contract or consumer requirement.
4. **No consumer exists.** There is no API, serving layer, or UI in the repository (`src/` contains only `ingestion`, `features`, `models`). No code consumes a predicted class.

**What the repository does not establish:**

- The **Poisson / Monte Carlo scoreline layer is deferred and unbuilt** (capabilities A, C, D and J in the sufficiency table are marked "DERIVED" via a simulation that does not exist). What that future layer would require of the H/D/A model is **not established by current evidence**.
- What a future UI or API would display as a "predicted result" is **not established by current evidence**.

**Classification: (B)** — repository evidence establishes that consumers require calibrated H/D/A probabilities, and hard Draw classification is not a stated requirement.

## 6. Governance classification

## **`DRAW_LIMITATION_ACCEPTED`**

## 7. Exact rationale

Decision rule 2 of the Phase 5B protocol applies: repository evidence clearly establishes that the required output is H/D/A probabilities, and hard Draw classification is nowhere stated as a requirement.

Specifically:
- The **only** documented output contract for the model (`PHASE3_MODEL_SPEC.md`) is three probabilities in `CLASS_ORDER`. The draw limitation is a property of `argmax(P)`, which the contract does not produce or promise.
- The documented client requirement is *"draw probability"* and *"Win/Draw/Loss probability"* — a requirement Model B meets more closely than V1 does (mean p_draw within +0.00087 of the observed rate; Σ p_draw 446.52 vs 445 actual).
- Against that contract, **Model B is not worse than V1 on draws — it is better on every draw measure**: probability-mass accuracy (+0.00087 vs +0.00449 gap), correct draws (8 vs 0), recall, precision and F1. Accepting the limitation therefore does not accept a regression; the candidate improves on the incumbent's already-accepted behaviour.
- No consumer exists that could impose a hard-class requirement, and none is documented.

**This acceptance is scoped, not general.** It accepts the limitation **for the probability-output contract as currently specified**. It does not accept it for any future consumer that derives a displayed or actioned "predicted result" from `argmax`, because under such a consumer draws would be surfaced in roughly 1% of fixtures against a ~25% base rate — a material product defect that this review does not authorize anyone to ignore.

## 8. Explicit limitations

- **One final-test season** (n = 1,751). No variance estimate, no confidence interval, no significance test — and none is claimed.
- **No epsilon threshold exists** and none was invented. No delta in this review is described as meaningful or significant.
- **No causal claim** is made about why draw discrimination is weak.
- Draw performance is **not solved**: recall 0.018, F1 0.034. Model B must never be presented as a reliable Draw-class predictor.
- The acceptance rests on the *current* output contract. **If the contract changes to include a predicted class, or if the deferred Poisson/scoreline layer or any UI/API consumer introduces a hard-class requirement, Condition 2 must be reopened.**
- Requirements of the unbuilt Poisson/scoreline layer are **not established by current evidence**.
- Aggregate calibration of draw probability does not imply per-fixture discrimination; these are different properties, and only the former is evidenced here.

## 9. Non-authorizations

This review does not authorize: creating V2, bumping `MODEL_VERSION`, modifying `config.py` or any V1/production source, retraining or tuning anything (including class weights or decision thresholds), calibration, re-evaluating 2025/26, modifying any Phase 3/4/5/5A artifact, production deployment, or API/UI work.

## 10. Next step

Condition 1 (`RETAIN_OMISSION_SUPPORTED`) and Condition 2 (`DRAW_LIMITATION_ACCEPTED`) are both now closed. Per the Phase 5 governance language, the V2 candidate status advances to **`APPROVED_V2_CANDIDATE_FOR_IMPLEMENTATION`**, which authorizes **only** the next engineering/specification step and explicitly **not** production deployment. The next action is a human decision to authorize (or decline) implementation work against the ten acceptance gates in `docs/PHASE5_V2_MODEL_SPEC_DRAFT.md`.
