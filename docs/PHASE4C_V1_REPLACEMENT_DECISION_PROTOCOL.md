# Phase 4C — V1 Replacement Decision Protocol / V2 Candidate Evaluation Protocol

## 1. Status and scope

**Phase 4C is DESIGN ONLY.** This document defines, in advance and before any final-test result is seen, exactly what evidence would make Model B *eligible to be considered* as a challenger to V1. It is a model-governance checkpoint, not a modeling step.

In this phase:
- No model is trained.
- No final-test evaluation is executed.
- 2025/26 is not loaded, read, or evaluated.
- No production change is made.
- No V2 is created and no winner is selected.
- V1 remains frozen exactly as specified in §2.

Writing this protocol does not authorize the final-test comparison it describes. The only permitted next action is a human decision to authorize or reject that comparison (§13).

**Material precondition discovered during this design (stated, not assumed):** the "final-test candidate Model B" **does not exist yet.** Phase 4A/4B only ever trained Model B on the three walk-forward *validation* folds. No Model B has been trained on `FINAL_TRAIN_SEASONS` (2020/21–2024/25). Building that estimator is itself a pre-lock construction step (§4), and it must be frozen before 2025/26 is opened.

## 2. Current baseline (V1 — locked, do not alter)

Recorded verbatim from the authoritative sources (`src/models/config.py`, `data/audit/phase3_model_comparison.json`, `data/audit/phase3_calibration_comparison.json`):

| Item | Value |
|---|---|
| `MODEL_VERSION` | `v1.0` |
| `REQUIRED_FEATURE_VERSION` | `v1.0` |
| `APPROVED_FEATURE_COLUMNS_V1` | 15 columns: `home_goals_for_per_match_season`, `home_goals_against_per_match_season`, `away_goals_for_per_match_season`, `away_goals_against_per_match_season`, `home_goals_for_per_match_last5`, `away_goals_for_per_match_last5`, `home_points_last5`, `away_points_last5`, `home_attack_strength_score`, `home_defence_strength_score`, `away_attack_strength_score`, `away_defence_strength_score`, `strength_diff`, `league_home_advantage_season`, `competition_id` |
| `CLASS_ORDER` | `["H", "D", "A"]` |
| `WALK_FORWARD_FOLDS` | fold_1: 2020/21–2021/22 → 2022/23; fold_2: 2020/21–2022/23 → 2023/24; fold_3: 2020/21–2023/24 → 2024/25 |
| `FINAL_TRAIN_SEASONS` | 2020/21, 2021/22, 2022/23, 2023/24, 2024/25 |
| `FINAL_TEST_SEASONS` | `('2025/2026',)` |
| Selected model | LogisticRegression (`max_iter=2000, C=1.0, random_state=0`) |
| Calibration decision | uncalibrated (raw `predict_proba`) |
| **Locked final-test (2025/26)** | log loss **1.012643607907126**, Brier **0.6059500599589396**, accuracy **0.500856653340948**, macro-F1 **0.3712753915810083**, balanced accuracy **0.4321797560315324**, n_test **1751**, n_train **8983** |
| Locked confusion matrix (rows=true H/D/A, cols=pred) | `[[599, 2, 170], [293, 0, 152], [257, 0, 278]]` |

None of these may be changed by Phase 4C or by any subsequent phase without an explicit versioned re-authorization.

## 3. Candidate definition — "Model B"

Model B is the Phase 4A Tier B feature set under V1's exact estimator. Frozen definition:

| Attribute | Value | Source of truth |
|---|---|---|
| Feature count | 80 columns | `ablation.MODEL_B_COLUMNS` |
| Composition | Model A (44: goals_core + form + competition_id + strength) **+ shots_core (18) + shots_on_core (18)** | Phase 4A frozen mapping |
| Additional families vs V1 | shots and shots-on-target rolling/season features (V1's 15 columns are a subset of Model A, except `league_home_advantage_season` which sits in Tier D — see note below) | Phase 4A |
| Estimator | `sklearn.linear_model.LogisticRegression` | `train.py` |
| Preprocessing | V1's `LogisticRegressionPreprocessor` (median impute + standardize, fit on training partition only) + one-hot `competition_id` against training categories | `train.py` |
| **C value** | **Requires an explicit decision — see below** | — |
| Class ordering | `["H", "D", "A"]` | `baselines.CLASS_ORDER` |
| `random_state` | `0` | `train.py` |
| `max_iter` | `2000` | `train.py` |
| Calibration | none (raw `predict_proba`), to match V1's calibration decision | Phase 3 Step 4 |
| Training/validation protocol so far | 3 walk-forward validation folds only; **never trained on FINAL_TRAIN_SEASONS** | Phase 4A/4B |

**C-value decision (must not be silently optimized):** Phase 4B observed that C=0.01 gave the best validation mean log loss for both A and B, but this was an *edge-of-grid* observation and Phase 4B explicitly forbade acting on it. Two mutually exclusive options exist, and **choosing between them is a governance decision, not a default:**

- **Option C-1 (recommended default): retain C=1.0.** Rationale: it is V1's exact value, so the *only* thing differing between V1 and the candidate is the feature set — the cleanest, most interpretable apples-to-apples comparison, and it avoids selecting a hyperparameter on validation data (which would be a soft form of tuning-on-the-thing-being-measured). Under this option the candidate is fully specified with no open hyperparameter.
- **Option C-2: adopt C=0.01 (or any other grid value).** This would be selecting C on validation performance. It is a *separate, explicit selection decision* that (a) must be justified, (b) must be frozen before the final test, and (c) inherits Phase 4B's edge-of-grid caveat (the true optimum may lie outside the grid, so C=0.01 is not established as optimal). This protocol does not adopt C-2 by default and does not recommend extending the grid to investigate it.

Until a human explicitly selects an option, **the candidate is defined with C=1.0 (Option C-1)** for the remainder of this protocol, because that is the choice that introduces no new tuning.

**Note on `league_home_advantage_season`:** this V1-approved column is placed in Tier D (not A/B) under the frozen Phase 4A mapping, so it is **absent from Model B**. Consequently Model B is *not* a strict superset of V1's feature set — it adds shots/shots-on-target but omits one V1 column. This is a real, documented consequence of the frozen tier mapping and must be stated in any final-test comparison, because it means "V1 vs Model B" is not a pure "add features to V1" test; it is "V1 vs a different but overlapping feature set." Whether this asymmetry is acceptable for a replacement decision is itself a point requiring human awareness.

## 4. Pre-final-test lock

Before 2025/26 is opened, **every one of the following must be frozen, recorded, and checksum-pinned.** Any change after the lock invalidates the comparison (§12).

1. **Candidate feature columns** — the exact 80 `MODEL_B_COLUMNS`, in order.
2. **Preprocessing** — V1's `LogisticRegressionPreprocessor`, fit on `FINAL_TRAIN_SEASONS` only; imputation medians and scaler statistics derived from training rows only; `competition_id` one-hot categories from training only.
3. **Estimator + hyperparameters** — `LogisticRegression(max_iter=2000, C=<locked C>, random_state=0)`.
4. **C** — the value chosen in §3 (default 1.0), locked as a literal.
5. **Calibration** — none (raw `predict_proba`), matching V1.
6. **Probability handling** — `evaluate.validate_probabilities` (finite, non-negative, 3-class, rows sum to 1 within 1e-6); columns in `CLASS_ORDER`.
7. **Class order** — `["H", "D", "A"]`.
8. **Fold / split definitions** — `FINAL_TRAIN_SEASONS` → `FINAL_TEST_SEASONS`, unchanged; the same `final_split` boundary V1 used.
9. **Metrics** — the exact set in §5, with their exact implementations (`evaluate.py`), pinned by checksum.
10. **Comparison rules** — §5 (paired, same rows, same conditions).
11. **Acceptance thresholds** — §6, pre-registered before results are seen.
12. **Artifact format** — §10 schema, fixed before execution.
13. **Checksums** — a manifest (§10/§11) capturing the state of all locked inputs *before* the final test runs.
14. **Code version** — the git commit / file checksums of `train.py`, `evaluate.py`, `data.py`, `splits.py`, `config.py`, `baselines.py`, and the (future) final-test runner, recorded in the manifest.

The **construction of the final-test Model B** (training it once on `FINAL_TRAIN_SEASONS`) happens *inside* the authorized final-test execution, using the locked spec above — it is not done during Phase 4C design.

## 5. Final-test comparison design (paired, apples-to-apples)

The comparison must be a **paired evaluation on the identical 2025/26 rows** under identical conditions. Both models:
- are trained once on the identical `FINAL_TRAIN_SEASONS` rows (the same 8,983-row labeled training set V1 used),
- predict on the identical `FINAL_TEST_SEASONS` rows (the same 1,751-row labeled test set),
- with per-fixture predictions aligned by `fixture_id` so metrics are computed over exactly the same events.

**Primary metric:** multiclass **log loss** (lower is better) — the same proper scoring rule that drove V1 selection. Reported to full float precision.

**Secondary metrics:** multiclass Brier score, accuracy, macro-F1, balanced accuracy, per-class precision/recall/F1, and confusion matrices for both models.

**Mandatory sanity / validity checks (all must pass or the run is invalid, §12):**
- Both prediction sets pass `validate_probabilities` (finite, non-negative, 3-class, rows sum to ~1).
- Identical `n_test` for both models, equal to V1's locked `n_test = 1751`.
- Prediction-level pairing: the set of `fixture_id`s is identical between the two models and equal to the 2025/26 labeled set; no duplicates; no missing.
- No `fixture_id` from `FINAL_TRAIN_SEASONS` appears in the test set (temporal-safety re-verification via the existing `verify_temporal_safety`).
- Fixture `420450481` (ABANDONED) absent from both training and test.
- Class distribution of the 2025/26 test labels recorded and identical for both models (it is the same rows, so it must be).
- **V1 reproduction check:** the V1 arm of this comparison must reproduce the locked V1 final-test metrics (§2) exactly (to <1e-9). If it does not, the harness itself is wrong and the comparison is invalid — this protects against a subtly different evaluation path silently changing V1's number.

## 6. Acceptance criteria (pre-registered)

The core rule, stated first: **better validation performance is not, by itself, grounds for replacement, and neither is a better final-test number alone.** Replacement requires clearing pre-registered bars *and* explicit human authorization. This section defines eligibility zones, not an automatic decision.

Let `Δ = logloss(B) − logloss(V1)` on the 2025/26 test set (negative = B better).

Because there is **exactly one** final-test season (n=1751, a single realization of "the future"), the project has **no established, statistically-justified minimum-effect threshold** — there is no held-out distribution of season-to-season final-test deltas from which to derive one. This is a real limitation, and the protocol refuses to invent a number that looks rigorous but is not. The zones below therefore combine an objective sign/magnitude reading with an explicit **"requires human authorization"** marker wherever a threshold cannot be justified from existing evidence.

| Zone | Objective condition | Meaning | Threshold justified from evidence? |
|---|---|---|---|
| **Clearly worse** | Δ ≥ 0 (B's log loss ≥ V1's) **or** B fails any §5 validity check | B does not improve on V1 | Yes — sign is objective |
| **Effectively tied** | −ε < Δ < 0 for a small ε that **cannot currently be justified** | Numerical improvement too small to be distinguishable from noise given n=1 season | **No — ε requires human authorization**; do not invent it |
| **Meaningful-but-insufficient** | Δ ≤ −ε (if ε is ever authorized) but secondary metrics disagree in direction, or the improvement rests on a small/unstable subset | Direction promising, evidence incomplete | Partly — requires investigation (§8 Zone C) |
| **Eligible for V2 review** | Δ clearly negative, **all** secondary metrics directionally consistent with the Phase 4B validation picture, all §5 checks pass, and every §4 lock verified intact | Candidate has earned a *review*, not a promotion | **No automatic promotion** — see §8/§9 |

Distinctions the reviewer must hold separately:
- **Numerical improvement** — Δ < 0. Objective, but on one season.
- **Practical improvement** — whether a log-loss change of the observed magnitude matters for the client's use. The Phase 4B/4A A→B effect is *small in absolute terms* (~0.004–0.007 validation log loss); a final-test effect of similar size would likely be practically marginal. This judgment is the client's/human's, not the harness's.
- **Uncertainty** — with one test season there is no confidence interval; the result is a point estimate with unknown variance. This must be stated wherever the number is quoted.
- **Model-risk** — Model B is larger (80 vs 15 columns), depends on shots/shots-on-target coverage, and omits one V1 feature (§3 note). A near-tie does not justify taking on that additional complexity and coverage dependence.

**No threshold ε is set in this document.** If the eventual decision needs one, it must be authorized by a human *before* the final test is opened (to preserve pre-registration), with a stated rationale, and recorded in the manifest.

## 7. No-peek / anti-tuning rules

Once 2025/26 is opened, all of the following are prohibited; each is also a hard stop (§12):
- Tuning **anything** (features, C, preprocessing, calibration, class handling) after seeing any 2025/26 result.
- Changing the metric set, the primary metric, or the acceptance zones after seeing results.
- Re-running the final test "to get a better number," or running it more than once for a fixed candidate spec.
- Using 2025/26 in any way to build, select, or tune a V2.
- Selecting the C value (or any hyperparameter) using final-test performance.
- Adding or removing feature columns after the lock.
- Introducing calibration after the lock.

The candidate specification and the acceptance criteria are frozen and checksum-pinned (§4) **before** the test set is touched; the manifest timestamp/commit is the evidence that they preceded the result.

## 8. Decision matrix

| Outcome | Objective trigger | Action |
|---|---|---|
| **A. B clearly loses** | Δ ≥ 0, or B is worse on the primary metric | **Reject candidate. V1 remains.** No further action. |
| **B. B effectively tied** | Improvement smaller than an authorized ε (or ε not authorized and Δ near 0) | **V1 remains** unless explicit human authorization, weighing model-risk (§6) against a marginal gain. Default is *keep V1*. |
| **C. Meaningful improvement, evidence insufficient** | Δ clearly negative but secondary metrics inconsistent, or concerns about coverage/feature-asymmetry/stability | **Investigation required.** No promotion. Document what additional evidence (e.g. more test seasons when available) would be needed. |
| **D. B satisfies all pre-registered criteria** | Δ clearly negative, secondary metrics consistent, all validity + lock checks pass | **B becomes eligible for an explicit V2 review** (§9). Eligibility ≠ promotion. |
| **E. Any protocol violation** | Final-test data modified, leakage, row/class mismatch, invalid probabilities, artifact/checksum mismatch, unauthorized tuning, V1 reproduction check fails | **Invalidate the comparison.** Discard the result; it may not inform any decision. Re-authorization required to retry. |

It is deliberately impossible, under this matrix, for "better validation performance" or even "a lower final-test log loss" to *automatically* replace V1. The best a candidate can achieve from the harness is **Zone D eligibility**, which only unlocks a human review.

## 9. V2 authorization boundary

Four distinct statuses, never to be conflated:
1. **Validation candidate** — what Model B is *now*. Beats Model A on validation across the C grid. No final-test evidence. (Current status.)
2. **Final-test candidate** — a validation candidate whose full spec has been frozen (§4) and which has been evaluated once on 2025/26 under this protocol. *Does not exist yet.*
3. **V2 candidate** — a final-test candidate that reached Zone D and passed an explicit human V2 review.
4. **Production model** — a V2 candidate that has been explicitly authorized, versioned (`MODEL_VERSION` bump), and released, with its own spec document mirroring `PHASE3_MODEL_SPEC.md`.

**Model B must not advance between these statuses automatically.** Each transition requires an explicit human decision. Reaching Zone D authorizes *review*, not deployment.

## 10. Artifact requirements (for the future authorized final-test phase)

If and when the final-test comparison is authorized, it must emit (to Phase-4C-specific paths, never overwriting any existing artifact):
- **Comparison JSON** — both models' full metric sets, per-class metrics, confusion matrices, the paired Δ for every metric, the V1-reproduction check result, and every §5 validity-check result.
- **Prediction CSVs** (if permitted at authorization time) — per-fixture H/D/A probabilities for V1 and for B on the 2025/26 rows, aligned by `fixture_id`, in `CLASS_ORDER`.
- **Audit report** — a factual results report mirroring the Phase 4A/4B report structure.
- **Checksum manifest** — pre-run checksums of all locked inputs (§4/§11) and post-run checksums of all generated artifacts.
- **Test log** — full test-suite result captured at execution time.
- **Environment/version record** — scikit-learn version, Python version, OS, and the source-file checksums/commit used.
- **Decision record** — the zone reached, the pre-registered criteria, and the human decision (or "pending review").

## 11. Integrity requirements (must remain byte-identical unless explicitly authorized)

Baseline checksums recorded at Phase 4C design time (to be re-verified before any authorized execution):

| Artifact | MD5 |
|---|---|
| `data/processed/features.db` | `e7ebe7fc07040a5927683c35b6371e63` |
| `data/processed/matches.db` | `fdeed042096fa1c851aaee6c84995247` |
| `data/audit/phase3_model_comparison.json` | `616279914b2730749d52eae15b5f96b9` |
| `data/audit/phase3_calibration_comparison.json` | `d94ed430ab13337797f0592bf82878cb` |
| `data/audit/phase4a_ablation_comparison.json` | `075b0686bce20bfce9f7289fa37076c0` |
| `data/audit/phase4b_robustness_comparison.json` | `97d0f8b8674c9bf8598b6c3d6b7c825c` |
| `src/models/config.py` | `c2ed32cb53ec34199fd245624afea4dd` |
| `src/models/train.py` | `21425459195311492f49e73f5ae38fe0` |
| `src/models/splits.py` | `8b7991ab3739c7d4daa2bf1998163da4` |
| `src/models/evaluate.py` | `4e9d9313867d47a19001a383a301c2fe` |
| `src/models/data.py` | `b78e30eb45dbc46621c0160a188ce981` |
| `src/models/baselines.py` | `42e64e3a0ba8c4bf8cf264f80cdcd208` |
| `src/models/run_experiments.py` | `546ea0105ca8b235ab80b393a58f2831` |

The 2025/26 rows inside `features.db` are the locked test data; `features.db`'s checksum above pins them. Phase 3 prediction CSVs and Phase 4A prediction CSVs must also remain unchanged. Any deviation without written authorization is a stop condition (§12).

## 12. Stop conditions (halt immediately, do not proceed, report)

- Any locked-input checksum (§11) differs from its recorded value.
- 2025/26 / final-test data modified in any way.
- Feature leakage detected (a training-fold statistic derived from test rows; a test fixture in training; a post-cutover value where none should exist).
- Row mismatch — `n_test` ≠ 1751, or the two models' fixture sets differ, or duplicates/missing appear.
- Class-order mismatch — probabilities not in `["H","D","A"]` order for either model.
- Invalid probabilities — either model fails `validate_probabilities`.
- Artifact mismatch — a generated artifact does not match its declared schema, or an existing artifact would be overwritten.
- Unauthorized tuning — any post-peek change to features, C, preprocessing, calibration, metrics, or thresholds.
- V1 integrity failure — the V1 reproduction check (§5) does not match the locked metrics to <1e-9.
- Unexpected files — any new experiment output outside declared Phase 4C paths.
- Any ambiguity that could bias the comparison — stop and report rather than resolve it silently.

## 13. Recommended next action

The **only** permitted next action after this document is a human decision:

> **Review this protocol and explicitly authorize or reject Phase 4C final-test execution** — and, if authorizing, explicitly choose the C option (§3: default C-1 = retain C=1.0) and state whether an ε tie-threshold (§6) is being authorized and at what value.

No final-test evaluation, no training, and no candidate construction may occur until that authorization is given. This document does not itself authorize any of them.
