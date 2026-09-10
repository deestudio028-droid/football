# Phase 5 — V2 Candidate Review

**Status: REVIEW AND SPECIFICATION ONLY.** No model was trained, no configuration modified, no 2025/26 re-evaluation performed, `MODEL_VERSION` remains `v1.0`, and no V2 exists. Every figure below is quoted from, or mechanically recomputed against, the authoritative repository artifacts.

## 0. Integrity gate (performed before any review work)

All 13 locked inputs verified byte-identical: `features.db` `e7ebe7fc07040a5927683c35b6371e63`, `matches.db` `fdeed042096fa1c851aaee6c84995247`, Phase 3 artifacts, `phase4a_ablation_comparison.json` `075b0686bce20bfce9f7289fa37076c0`, `phase4b_robustness_comparison.json` `97d0f8b8674c9bf8598b6c3d6b7c825c`, and all V1/production source files. `MODEL_VERSION` = `v1.0`, 15 approved V1 columns. **No V2 model, artifact, or module exists.** No production code changed. No final-test rerun performed.

Phase 4C artifact checksums recorded as this phase's baseline: `phase4c_final_comparison.json` `effd9e54130b2bc5aaf51cd643962396`, `phase4c_prerun_manifest.json` `4f166d1826edbf7bfe1e022639a8b1f4`, V1 predictions `e32c5928a2593a5c5cd804c81134f17d`, Model B predictions `a8bb4493f1e9ff4fd1ae1b504d2319cf`.

---

# PART 1 — Evidence synthesis

## A. Phase 4A evidence (feature ablation)

**A vs B difference:** Model A = 44 columns (goals_core 18, form 20, strength 5, competition_id 1). Model B = 80 columns = A + shots_core (18) + shots_on_core (18). Verified mechanically; strict nesting A ⊂ B confirmed.

**Mean validation log loss:** A `1.004013383939449`, B `0.9993791056968738`, Δ = **−0.004634278243**.

**Secondary metrics:** B improved Brier, macro-F1, balanced accuracy and accuracy — all four, at the tested setting.

**Per-fold consistency:** B beat A on log loss in all three folds individually (fold_1 −0.001350, fold_2 −0.000708, fold_3 −0.011845). The margin is roughly an order of magnitude larger in fold_3.

**xG limitation:** Models C and D could not be evaluated under the 3-fold protocol — all 22 xG columns have **zero observed values** in fold_1's and fold_2's training partitions, making those folds structurally invalid for xG-inclusive training. C/D were run only on fold_3 and no 3-fold mean was computed. **This is not evidence that xG lacks value**; it is evidence that xG cannot currently be evaluated.

**Why Model B became the Phase 4C candidate:** it was the only tier with (a) a complete, valid 3-fold mean, (b) a consistent improvement over the A baseline on every metric and every fold, and (c) no dependence on the structurally-unavailable xG group.

**Limitations:** three folds is a small basis; the improvement is small in absolute terms; only one model family (LogisticRegression) was tested; larger tiers were not tuned for their dimensionality.

## B. Phase 4B evidence (regularization robustness)

**Grid (frozen, 9 values):** 0.01, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 100.0.

**B vs A across all C:** B improved mean validation log loss at **all 9 C values**. Delta range: **−0.006520888699** (C=0.01) to **−0.004227223288** (C=100.0). The advantage never reversed and never vanished.

**C=1.0 equivalence:** at C=1.0 Phase 4B reproduced Phase 4A's means **exactly** (|Δ| = 0.00e+00 for both models), confirming the sweep path was V1's estimator with C as the only varied parameter.

**Per-fold consistency:** B beat A in all three folds at **8 of 9** C values. The single exception is C=100.0 fold_2 (B−A = +0.000201421), where B still won the mean.

**Secondary metric consistency:** B was better than A on Brier, macro-F1, balanced accuracy and accuracy at **all 9** C values (9/9 each).

**Edge-of-grid finding:** both A and B achieved their best mean log loss at **C=0.01, the lower grid boundary**, meaning the true optimum may lie below the tested range. Phase 4B explicitly reported this without acting on it.

**Why C=1.0 remains the correct frozen choice for V2:** C=1.0 is V1's own value, so the feature set is the *only* difference between the arms — the cleanest attributable comparison. Phase 4C was explicitly authorized and executed at C=1.0, and the resulting final-test evidence is the only final-test evidence that exists. **C=0.01 is not retrospectively selected here.** Adopting a validation-selected C would be a new hyperparameter-selection decision, would invalidate the pre-registered Phase 4C evidence, and inherits the unresolved edge-of-grid caveat.

## C. Phase 4C evidence (final test, 2025/26, paired on 1,751 identical rows)

| Metric | V1 (15 features) | Model B (80 features) | Δ (B − V1) |
|---|---|---|---|
| Log loss ↓ | 1.012643607907126 | **0.9960487649063601** | **−0.016594843000765858** |
| Brier ↓ | 0.6059500599589396 | **0.5947186313129067** | −0.011231428646032882 |
| Accuracy ↑ | 0.500856653340948 | **0.5151342090234152** | +0.014277555682467247 |
| Macro-F1 ↑ | 0.3712753915810083 | **0.3938491505819635** | +0.022573759000955207 |
| Balanced accuracy ↑ | 0.4321797560315324 | **0.4489548797121166** | +0.016775123680584203 |

**V1 reproduction check:** PASS — V1 reproduced its locked Phase 3 metrics to |Δ| = 0.00e+00 (tolerance 1e-9), with an identical confusion matrix, confirming the harness is the same evaluation path.

**Structural validity:** all checks passed — 1,751 rows per arm, identical fixture sets, row-aligned pairing, no duplicates, no missing rows, identical `y_true`, abandoned fixture `420450481` absent, no train/test overlap, temporal safety verified, class order H/D/A, probabilities valid for both arms.

**Protocol violations:** none. All locked checksums matched before and after; the run was write-once and single-execution; no post-hoc tuning.

**Draw performance:** V1 recall `0`, F1 `0` (0 of 445 draws). Model B recall `0.017977528089887642`, F1 `0.03440860215053764` (8 of 445). **This is not solved** — it is an improvement from zero to near-zero.

**Outcome:** ZONE D — eligible for explicit V2 review.

---

# PART 2 — V2 candidate definition

**Question: should Model B become the V2 candidate?** Assessed across ten dimensions, not performance alone.

| # | Dimension | Assessment | Evidence / judgment |
|---|---|---|---|
| 1 | Predictive improvement | **Positive** | All five metrics improved on the held-out final test; log loss −0.0166. *Evidence.* |
| 2 | Robustness | **Positive** | B beat A at 9/9 C values on the mean, 8/9 on every fold. *Evidence.* |
| 3 | Feature complexity | **Negative** | 80 vs 15 features (5.3×). *Evidence (counts); judgment on impact.* |
| 4 | Feature availability | **Acceptable** | shots/shots-on-target max null rate 5.5% on final-train and 5.5% on the 2025/26 test partition; no all-null or constant column in any partition. *Evidence.* |
| 5 | Maintainability | **Negative** | 66 additional columns to validate, monitor and keep supplied. *Judgment.* |
| 6 | Leakage risk | **Low** | Static audit found no leakage path; Phase 2 leakage tests (10/10) still pass. *Evidence — see Part 5.* |
| 7 | Operational dependency | **Elevated** | Adds a hard runtime dependency on shots and shots-on-target feeds that V1 does not have. *Evidence (feature list); judgment on risk.* |
| 8 | Draw weakness | **Unresolved** | Draw recall 1.8%, F1 0.034; p_draw never exceeds 0.396 in 1,751 predictions. *Evidence.* |
| 9 | Difference from V1 contract | **Unresolved concern** | B is **not** a superset — it drops `league_home_advantage_season`, never separately evaluated. *Evidence — see Part 3.* |
| 10 | Rollback feasibility | **Positive** | V1 is fully frozen, reproducible bit-for-bit (reproduction check PASS), and its artifacts are checksum-pinned. Rollback is a config revert. *Evidence.* |

## STATUS: `APPROVED_V2_CANDIDATE_FOR_IMPLEMENTATION`

> **Status history.** This review originally recorded `CONDITIONAL_V2_CANDIDATE` pending two conditions. Both have since been closed by dedicated evidence-gathering phases:
>
> - **Condition 1 — RESOLVED: `RETAIN_OMISSION_SUPPORTED`.** Phase 5A (`data/audit/phase5a_dropped_feature_comparison.json`, `docs/PHASE5A_DROPPED_FEATURE_RESULTS.md`) ran the authorized validation-only experiment. Adding `league_home_advantage_season` to Model B *worsened* mean validation log loss: Model B `0.9993791056968738` vs Model B + feature `0.9996821579734537`, delta `+0.000303052276579896` → `CASE_B_worsens_mean_log_loss`. The evidence supports keeping the feature omitted.
> - **Condition 2 — RESOLVED: `DRAW_LIMITATION_ACCEPTED`** (scoped). Phase 5B (`docs/PHASE5B_DRAW_LIMITATION_REVIEW.md`) found that the only documented output contract is H/D/A probabilities (`PHASE3_MODEL_SPEC.md`), that the client-requirement record asks for *"draw probability"* / *"Win/Draw/Loss probability"*, that no hard-Draw-class requirement is stated anywhere, and that no consumer exists. Model B is better than V1 on every draw measure. **The acceptance is scoped to the current probability-output contract and must be reopened if a predicted class enters the contract, or if the deferred Poisson/scoreline layer or any UI/API consumer introduces a hard-class requirement.**
>
> With both conditions closed, the status advances to `APPROVED_V2_CANDIDATE_FOR_IMPLEMENTATION`. **This authorizes only the next engineering/specification step. It does not authorize production deployment, does not create V2, and does not change `MODEL_VERSION` (still `v1.0`).**

Model B is **not rejected** — the predictive and robustness evidence is genuine, independently reproduced, and protocol-compliant. The two questions that originally held it at conditional status were:

**Condition 1 — resolve the dropped V1 feature.** Model B omits `league_home_advantage_season`, a feature V1 relies on. No evidence exists on whether Model B *plus* that column performs better, worse, or the same. Adopting Model B as specified means silently discarding a V1 feature on no evidence. Resolving this requires a validation-only comparison (Model B vs Model B + `league_home_advantage_season` across folds 1–3), which is **new training and therefore requires separate explicit authorization**. It must not be run under this phase.

**Condition 2 — accept or reject the draw limitation in writing.** Model B's draw class is effectively unusable (1.8% recall). If any downstream requirement — the client's stated "draw probability" need, or the planned Poisson/scoreline work — depends on identifying draws as a *class*, that requirement is not met by either model. A named human must record whether this is acceptable for V2, or whether V2 must wait for a modeling approach that addresses it.

**Note on Condition 2:** the *probability* for draws is well-calibrated in aggregate (mean p_draw 0.2550 vs actual draw rate 0.2541, gap +0.00087). The failure is specifically at the **argmax/class-assignment** level, not in the probability mass. If downstream consumers use probabilities rather than the predicted class, the limitation is far less severe. This distinction should drive the Condition 2 decision.

**Even if these conditions are met and the status is later raised to `APPROVED_V2_CANDIDATE_FOR_IMPLEMENTATION`, that authorizes only the next engineering/specification step — never production deployment.**

---

# PART 3 — Critical feature-asymmetry review (mechanically verified)

Computed directly from `config.APPROVED_FEATURE_COLUMNS_V1` and `ablation.MODEL_B_COLUMNS`, not from documentation:

- **|V1| = 15, |Model B| = 80, |intersection| = 14**
- **B is a strict superset of V1: FALSE**
- **V1 \ B (1 column):** `league_home_advantage_season`
- **B \ V1 (66 columns):** shots_core 18, shots_on_core 18, form 18, goals_core 12
- **Model B family composition (80/80 accounted, 0 unclassified):** goals_core 18, form 20, strength 5, competition_id 1, shots_core 18, shots_on_core 18
- **xG columns in Model B:** none

**Documentation cross-check: no discrepancy found.** The Phase 4C protocol's claims (V1 \ B = {`league_home_advantage_season`}, |B| = 80) both match the actual code exactly.

**Is the asymmetry acceptable for V2?** **Not yet — this is Condition 1.** The 66 added columns are the intended, evaluated change. The single *dropped* column is not: it entered Model B's definition only as a side effect of the frozen Phase 4A tier mapping (which placed `league_home_advantage_season` in Tier D), never as a deliberate finding that it should be removed. Adopting Model B as-is would remove a V1 feature on no evidence whatsoever. This is a specification defect, not a performance concern, and it should be resolved before the feature contract is frozen.

---

# PART 4 — Feature coverage / operational risk review

No training performed. Missingness computed from the existing dataset and split definitions.

**Max null rate per family, per partition:**

| Partition | goals_core | form | strength | competition_id | shots_core | shots_on_core |
|---|---|---|---|---|---|---|
| fold_1 train | 0.0764 | 0.0764 | 0.0027 | 0.0000 | 0.0772 | 0.0772 |
| fold_2 train | 0.0555 | 0.0555 | 0.0027 | 0.0000 | 0.0560 | 0.0560 |
| fold_3 train | 0.0542 | 0.0444 | 0.0028 | 0.0000 | 0.0550 | 0.0550 |
| final train (2020/21–2024/25) | 0.0543 | 0.0374 | 0.0028 | 0.0000 | 0.0551 | 0.0550 |
| **final test (2025/26)** | 0.0548 | 0.0074 | 0.0029 | 0.0000 | **0.0548** | **0.0548** |

**No Model B column is all-null or constant in any partition.**

| Family | Source | Historical availability | Missingness | Pre-prediction-time? | Depends on post-match info? | Known coverage gap | Operational risk |
|---|---|---|---|---|---|---|---|
| goals_core | OddAlerts fixture results, Phase 2 rolling/season aggregates | All 6 seasons | ≤7.6% (early-season history) | Yes | No — prior fixtures only | None | Low |
| form | Same, derived from prior results | All 6 seasons | ≤7.6% | Yes | No | None | Low |
| strength | Phase 2 shrinkage scores from prior results | All 6 seasons | ≤0.29% | Yes | No | None | Low |
| competition_id | Fixture metadata | All 6 seasons | 0.0% | Yes | No | None (5 leagues: 200, 419, 423, 477, 499) | Low |
| shots_core | OddAlerts match statistics, Phase 2 aggregates | All 6 seasons incl. 2025/26 | ≤7.7%, 5.5% on 2025/26 | Yes — prior matches only | No | None observed in the modelled seasons | **Moderate** — new external feed dependency |
| shots_on_core | Same | All 6 seasons incl. 2025/26 | ≤7.7%, 5.5% on 2025/26 | Yes | No | None observed | **Moderate** — same |

**Special attention items:**
- **Shots / shots-on-target:** coverage is complete enough across every modelled season to train and predict without any structurally-empty column — materially unlike xG. Residual missingness (~5.5%) is consistent with the same early-season insufficient-history pattern affecting goals and form, not a feed gap.
- **Rolling windows and season aggregates:** all are computed from *prior* fixtures only (Part 5).
- **Newly introduced competitions:** **Not established by current evidence.** Only the 5 existing leagues appear in the data; behaviour for a league added in future is untested. `LogisticRegressionPreprocessor` maps an unseen `competition_id` to an all-zero one-hot row rather than erroring, so it degrades rather than crashes, but predictive quality in that case is unknown.
- **Missing-feature behaviour at inference time:** median imputation uses statistics fit on the training partition. Behaviour if an entire feed (e.g. shots) were to go missing at prediction time is **not established by current evidence** and is a monitoring requirement, not a demonstrated capability.

---

# PART 5 — Leakage review (static, no training)

**Phase 2 leakage tests re-run: 10/10 PASS.**

| Check | Finding | Basis |
|---|---|---|
| No current-match result used | **PASS** | `history.py::MatchHistory.record()` appends a fixture's result to team histories and the league accumulator **only after** its features are computed; `history_before()` returns strictly prior matches. This ordering is the architectural leakage guarantee. |
| No future fixture used | **PASS** | Fixtures processed in strict chronological order; accumulators are append-only forward in time. |
| No final-test statistic leaks into training | **PASS** | `final_split` partitions by season; `verify_temporal_safety` enforces `max(train.unix) < min(test.unix)` and zero fixture overlap — re-verified in the Phase 4C run. |
| No post-match shots leak into pre-match prediction | **PASS** | shots/shots-on-target features are the same rolling/season aggregates over *prior* fixtures, built through the identical `history_before()` path as goals; no separate code path exists that reads the target fixture's own statistics. |
| No label-derived feature used improperly | **PASS** | `LABEL_COLUMNS` (`label_home_goals`, `label_away_goals`, `label_result`) are all in `X_EXCLUDED_COLUMNS`; **zero** label or bookkeeping columns appear in Model B's 80. |
| Preprocessing fit isolation | **PASS** | `LogisticRegressionPreprocessor.fit(X_train)` only; `.transform()` never refits. Verified in code and by existing leakage tests. |
| Competition encoding | **PASS** | One-hot categories learned from the training partition only. |
| ABANDONED fixture handling | **PASS** | Fixtures without a played status never enter any team's history. |

**No Model B feature was classified UNKNOWN.** Every one of the 80 columns is produced by the same Phase 2 temporal machinery already audited in Phase 2 and re-verified here; none required an assumption of safety.

---

# PART 6 — Draw-class risk review

**Observed problem:** both models are effectively unable to *predict* the Draw class. V1: 0 of 445 draws (recall 0.000, F1 0.000). Model B: 8 of 445 (recall 0.017978, F1 0.034409, precision 0.400).

**Evidence (from existing Phase 4C prediction artifacts, no new computation on the model):**
- Draw is argmax in **2/1,751** V1 predictions and **20/1,751** Model B predictions.
- **Maximum p_draw across all 1,751 predictions: V1 0.362594, Model B 0.396028.** Neither model ever assigns draw a probability above 0.40.
- Draw is ranked 2nd of 3 in 59.9% (V1) / 65.0% (B) of predictions, and 3rd in 40.0% / 33.9%.
- **Mean p_draw is well calibrated:** V1 0.258631 vs actual draw rate 0.254140 (gap +0.00449); Model B 0.255008 vs 0.254140 (gap +0.00087).

**What is known:** the failure is at the **class-assignment (argmax)** level, not in the probability mass. Both models assign draws approximately the correct aggregate probability but essentially never make draw the single most likely outcome, because p_draw's ceiling (~0.40) sits below what is needed to beat both H and A simultaneously. Model B raises the ceiling slightly (0.363 → 0.396) and roughly triples the argmax count, which is where its macro-F1 and balanced-accuracy gains partly originate.

**Plausible hypotheses (not established, no causal claim):** draws may be intrinsically less separable from the available pre-match features; a 3-class model optimizing log loss has little incentive to push a middle class above two flanking classes; the feature set may lack signals that specifically discriminate draws (e.g. defensive-stalemate or motivation indicators).

**What is unknown:** whether any feature set available in this project could raise draw discrimination; whether a different objective, class weighting, or a goals-based/Poisson formulation would help; whether the client's requirements depend on draw *class* prediction or draw *probability*.

**Does this block V2 candidacy?** **It does not block candidacy, but it is Condition 2.** Model B is not worse than V1 here — it is marginally better. If downstream consumers use probabilities (which are well-calibrated), the limitation is tolerable and should be documented. If any consumer requires a predicted draw *label*, neither model meets that requirement and V2 should not be presented as doing so.

---

# PART 7 — V2 risk register

Evidence and judgment are labelled separately.

| # | Risk | Severity | Evidence | Mitigation | Blocks V2? |
|---|---|---|---|---|---|
| 1 | 80-feature complexity | Medium | *Evidence:* 80 vs 15 columns (5.3×), verified. *Judgment:* larger surface to validate/monitor. | Freeze exact column list + order in spec; contract test asserting all 80 present | No |
| 2 | Feature coverage | Low | *Evidence:* no all-null/constant column in any partition; max null ≤7.7%, 5.5% on 2025/26. | Pre-inference coverage check; alert on null-rate drift | No |
| 3 | Shots availability | Medium | *Evidence:* complete across all 6 seasons, 5.5% null on 2025/26. *Judgment:* new external feed dependency V1 lacks. | Monitor feed; define documented behaviour if feed drops | No |
| 4 | Shots-on-target availability | Medium | *Evidence:* same as above. *Judgment:* same. | Same | No |
| 5 | **Dropped V1 feature** (`league_home_advantage_season`) | **High** | *Evidence:* V1 \ B = exactly this column; never separately evaluated. *Judgment:* removing a V1 feature on zero evidence is a specification defect. | Authorize a validation-only B vs B+column comparison **before** freezing the contract | **YES — Condition 1** |
| 6 | Draw weakness | **High** | *Evidence:* recall 0.018, F1 0.034, max p_draw 0.396. *Judgment:* unusable as a class predictor. | Written accept/reject; document probability-vs-class distinction for consumers | **YES — Condition 2** |
| 7 | Data drift | Medium | *Evidence:* none — no drift monitoring exists. *Judgment:* single test season gives no drift estimate. | Monitoring gate (Gate 10) | No |
| 8 | Leakage risk | Low | *Evidence:* full static audit PASS; Phase 2 leakage tests 10/10 PASS; no UNKNOWN features. | Retain leakage tests in CI | No |
| 9 | Maintainability | Medium | *Judgment:* 66 extra columns to keep supplied and validated. | Spec + contract tests | No |
| 10 | Rollback | Low | *Evidence:* V1 frozen, reproduced bit-for-bit, checksum-pinned. | Gate 9 rollback rehearsal | No |
| 11 | Reproducibility | Low | *Evidence:* `random_state=0`, no random splitting; Phase 4B reproduced Phase 4A exactly; Phase 4A predictions were byte-identical across two runs. | Gate 5 determinism check | No |
| 12 | Future competition coverage | Medium | *Evidence:* **Not established by current evidence** — only 5 leagues present. *Judgment:* unseen `competition_id` degrades to all-zero encoding rather than erroring. | Document; add unseen-league monitoring | No |
| 13 | Single-season uncertainty | **High** | *Evidence:* n=1 final-test season (1,751 matches); no variance estimate possible. | Re-evaluate when a further season completes | No — but caps confidence permanently |

---

# PART 8 — V2 specification draft

Because the status is not `REJECT_V2_CANDIDATE`, a draft specification has been created: **`docs/PHASE5_V2_MODEL_SPEC_DRAFT.md`**, clearly marked **DRAFT — NOT PRODUCTION AUTHORIZED**. It freezes the candidate definition subject to Conditions 1 and 2 being resolved. `src/models/config.py` was **not** modified and `MODEL_VERSION` remains `v1.0`.

---

# PART 9 — V2 acceptance gates

See `docs/PHASE5_V2_MODEL_SPEC_DRAFT.md` §Acceptance Gates for the full ten-gate definition (requirement / verification method / pass-fail condition / stop condition for each).

---

# PART 10 — Compliance statement

This phase trained no model, modified no configuration, modified no database, did not re-evaluate 2025/26, tuned no hyperparameter, ran no calibration, created no production artifact, did not bump `MODEL_VERSION`, deployed nothing, and made no API change. It is review and specification only.
