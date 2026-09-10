# Phase 5 — V2 Model Specification

# ⚠️ DRAFT — NOT PRODUCTION AUTHORIZED

This is a **draft specification only**. It does not create V2, does not modify `src/models/config.py`, does not change `MODEL_VERSION` (which remains **`v1.0`**), and does not authorize implementation or deployment.

Its status is **`APPROVED_V2_CANDIDATE_FOR_IMPLEMENTATION`**. Both conditions that previously held it at `CONDITIONAL_V2_CANDIDATE` are now **closed**:

- **Condition 1 — dropped V1 feature: CLOSED — `RETAIN_OMISSION_SUPPORTED`.** Resolved by Phase 5A (`data/audit/phase5a_dropped_feature_comparison.json`, `docs/PHASE5A_DROPPED_FEATURE_RESULTS.md`). Adding `league_home_advantage_season` back to the candidate *worsened* mean validation log loss (0.9993791056968738 → 0.9996821579734537, delta +0.000303052276579896 → `CASE_B_worsens_mean_log_loss`). **`league_home_advantage_season` remains omitted and must not be added back.** Note the underlying evidence was mixed: B_PLUS was better on log loss in only 1 of 3 folds, and accuracy/macro-F1/balanced accuracy slightly favoured B_PLUS; the pre-registered rule keys on the primary metric (log loss). No significance or causal claim is made.
- **Condition 2 — draw-class limitation: CLOSED — `DRAW_LIMITATION_ACCEPTED` (scoped).** Resolved by Phase 5B (`docs/PHASE5B_DRAW_LIMITATION_REVIEW.md`). The only documented output contract is H/D/A probabilities, the client-requirement record asks for *"draw probability"*, no hard-Draw-class requirement is stated anywhere, and no consumer exists. **The acceptance is scoped to the current probability-output contract and must be reopened if a predicted class enters the contract, or if the deferred Poisson/scoreline layer or any UI/API consumer introduces a hard-class requirement.** Draw performance is not solved (recall 0.018, F1 0.034) and the candidate must never be presented as a reliable Draw-class predictor.

With both conditions closed, the feature contract below is **frozen** (it was provisional while the conditions were open).

**`APPROVED_V2_CANDIDATE_FOR_IMPLEMENTATION` authorizes only engineering/specification work against the acceptance gates below. It does not create V2, does not change `MODEL_VERSION` (still `v1.0`), and does not authorize production deployment.**

---

## 1. Candidate identity

| Item | Value |
|---|---|
| Candidate name | `model_b` (Phase 4A Tier B) |
| Proposed version if promoted | **not assigned** — `MODEL_VERSION` stays `v1.0` until explicit authorization |
| Lineage | Phase 4A ablation → Phase 4B robustness → Phase 4C final test (Zone D) |
| Required feature version | `v1.0` (`REQUIRED_FEATURE_VERSION`, unchanged) |

## 2. Feature contract (frozen — 80 columns)

Exactly the 80 columns in `ablation.MODEL_B_COLUMNS`, **in that stored order**, verified present in `features.db`:

| Family | Count |
|---|---|
| `goals_core` | 18 |
| `form` | 20 |
| `strength` | 5 |
| `competition_id` | 1 |
| `shots_core` | 18 |
| `shots_on_core` | 18 |
| **Total** | **80** |

- **Feature ordering:** the tuple order of `MODEL_B_COLUMNS` is part of the contract; column selection must preserve it.
- **Explicitly excluded:** all 22 xG columns, all Tier D groups (`goals_venue`, `tempo`, `discipline`, `league_home_advantage`, `league_mean_goals`, `venue_goal_diff`, `history_flags`), all label columns (`label_home_goals`, `label_away_goals`, `label_result`), and all bookkeeping columns in `X_EXCLUDED_COLUMNS`.
- **Known asymmetry:** the candidate is **not** a strict superset of V1 — it omits `league_home_advantage_season`. **Condition 1 is CLOSED (`RETAIN_OMISSION_SUPPORTED`, Phase 5A): the omission is evidence-backed and the feature must not be added back.**

## 3. Preprocessing

Identical to V1's, reused unmodified (`train.LogisticRegressionPreprocessor`):
- `SimpleImputer(strategy="median")` — fit on the training partition only.
- `StandardScaler` — fit on the training partition only.
- `competition_id` one-hot encoded against categories observed in training only; an unseen category yields an all-zero encoding (degrades, does not raise).
- **No zero-filling. No blanket `dropna()`. No preprocessing statistic may be derived from validation or test rows.**

## 4. Estimator

| Parameter | Value |
|---|---|
| Estimator | `sklearn.linear_model.LogisticRegression` |
| `C` | **1.0** |
| `max_iter` | **2000** |
| `random_state` | **0** |
| Solver / penalty / tol | scikit-learn defaults, as used by V1 |

**C = 1.0 is frozen.** Phase 4B observed C=0.01 as the best *observed* grid value for both A and B, but that was an edge-of-grid observation which Phase 4B explicitly declined to act on, and Phase 4C's authorized final-test evidence exists only at C=1.0. Adopting any other C would be a new hyperparameter-selection decision requiring its own authorization and its own final-test evidence.

## 5. Target and output

- **Class order:** `["H", "D", "A"]` — fixed everywhere probabilities are produced, stored or consumed.
- **Output:** `predict_proba` — exactly 3 floats per row, in `CLASS_ORDER`.
- **Calibration:** **none.** Raw `predict_proba`, matching V1's Phase 3 Step 4 decision.

## 6. Training protocol

- **Training seasons:** `FINAL_TRAIN_SEASONS` = 2020/21, 2021/22, 2022/23, 2023/24, 2024/25 (8,983 labeled rows).
- **Row filter:** `label_result IS NOT NULL`. Fixture `420450481` (ABANDONED) permanently excluded.
- **Validation protocol (for any future re-evaluation):** the three frozen walk-forward folds; **2025/26 must not be reused for any selection or tuning.**
- Single deterministic fit; no random splitting, no K-fold, no resampling.

## 7. Prediction interface

- **Input:** a feature frame containing all 80 contract columns, in contract order, for fixtures whose kickoff is strictly after every fixture used to build their features.
- **Output:** array of shape `(n, 3)` in `CLASS_ORDER`, plus `fixture_id` alignment.
- **Failure mode:** a missing contract column must raise loudly (never substitute, never silently drop) — the `select_feature_columns` pattern.

## 8. Probability validation

Every emitted probability set must pass `evaluate.validate_probabilities`: exactly 3 classes, all finite, all non-negative, each row summing to 1 within `atol=1e-6`. A failure is a hard stop, not a warning.

**Tolerance disclosure (documentation-only; no code change).** The *intended/documented* tolerance is `atol=1e-6`. The *actual* behaviour of the current V1 implementation is `np.allclose(row_sums, 1.0, atol=1e-6)`, and NumPy's default `rtol=1e-5` makes the effective tolerance approximately `1e-6 + 1e-5·1.0 = 1.1e-5` at row sums near 1 — about 11× looser than the documented figure. This is recorded as a known, disclosed deviation. **`src/models/evaluate.py` is NOT modified**: it is one of the seven frozen V1 files pinned in `data/audit/phase4c_prerun_manifest.json` and verified byte-identical under Gate 9, so altering it would break the Gate 9 checksum verification and the §11 rollback evidence base. Whether to tighten the implementation to match the documented figure remains an **OPEN technical-governance item** requiring explicit authorization (see the Gate 10 decision matrix, §5.2). Practical note: probability rows produced by the `predict_proba` path deviate from 1.0 at floating-point-residue scale (~1e-16), roughly ten orders of magnitude below either candidate tolerance, so the distinction is near-certainly immaterial in practice — an observation, not a justification for leaving it unresolved. The validity rule itself is unchanged and remains zero-tolerance.

## 9. Missing-data policy

- Training/inference imputation uses **training-partition medians only**.
- No zero-fill, no global-mean fill, no fabricated constants, no value derived from validation/test data.
- Structurally-absent feature groups (as xG is in early seasons) must be **excluded from the contract**, never imputed into existence.

## 10. Feature availability requirements

Before any inference run:
- All 80 contract columns present.
- No contract column entirely null for the batch.
- Null rate per family within documented historical bounds. **Historical maximum observed null rate: 0.077218. Operational availability bound: 0.078.** (The 0.077218 figure is the exact per-family maximum observed — `fold_1` train, `shots_core` / `shots_on_core`; the earlier wording "≤ ~7.7%" was approximate and must **not** be read as an exact 0.0770 ceiling, which would have excluded the very observation it described by 0.000218. The corresponding final-season figure is 0.0548 observed on 2025/26, with an operational bound of 0.056.) The bound is an **absolute availability ceiling**; it is **not** a drift threshold — see §12, Signal 1.
- `competition_id` within the known set {200, 419, 423, 477, 499}; an unseen league must be flagged (behaviour is degraded-not-failed, and predictive quality for it is **not established by current evidence**).

## 11. Rollback requirement

V1 must remain fully restorable at all times: its configuration, artifacts and checksums stay frozen and pinned, and V1's final-test metrics must remain bit-for-bit reproducible (verified PASS in Phase 4C). Rollback must be a configuration revert requiring no data regeneration or retraining, and must be rehearsed before any promotion (Gate 9).

## 12. Monitoring requirements

> **GATE 10 SPECIFICATION: DECISIONS LOCKED (thresholds now included)**
> **GATE 10 IMPLEMENTATION: ADDITIVE MONITORING MODULE IMPLEMENTED (`src/monitoring/`)**
> **GATE 10 ACCEPTANCE: see the Gate 10 status note under the acceptance-gate table**
>
> No dashboard, web API, scheduler, alert integration, or deployment exists. Nothing in this section changes V1 behaviour. Full decision record: `docs/GATE10_MONITORING_SPECIFICATION_DECISION_MATRIX.md`.

**Original requirement (retained verbatim, unchanged, and still binding):** *Post-promotion (if ever authorized), at minimum: per-family null-rate drift, unseen-`competition_id` occurrences, predicted class distribution vs. historical base rates, mean predicted probability per class vs. realized outcome frequency, probability-validity failures, and draw-class behaviour specifically (given §Condition 2).*

The subsections below refine that sentence into signal definitions, baselines, thresholds and timing. They **narrow ambiguity only** — they do not add, remove or weaken any required signal.

### 12.0 Timing and promotion relationship (D-01 — LOCKED)

- **No production exposure currently exists.**
- Gate 10 monitoring implementation is a **mandatory pre-promotion requirement**.
- Operational monitoring **does not need to have been exercised** before a production exposure exists.
- Monitoring **specification** is finalized; the **additive implementation** now exists and is exercised by tests.
- **Gate 10 remains a promotion precondition.** It **cannot be marked PASS** on the basis that code exists — its acceptance tests must execute and pass.
- **Any future production exposure requires Gate 10 PASS first.**

**The absence of production exposure does NOT satisfy Gate 10 and must never be recorded as satisfying it.**

### 12.0.1 Family mapping (derived from the frozen 80-column contract — LOCKED)

The six families are **derived**, not inferred: each is an existing frozen tuple in `models.ablation`, and their concatenation is byte-for-byte `MODEL_B_COLUMNS` in contract order. Verified: exact set equality, pairwise disjoint, 80 total, concatenation order identical to the contract.

| Family | Source tuple (frozen) | Columns |
|---|---|---|
| `goals_core` | `ablation.GOALS_CORE_COLUMNS` | 18 |
| `form` | `ablation.FORM_COLUMNS` | 20 |
| `strength` | `ablation.STRENGTH_COLUMNS` | 5 |
| `competition_id` | `ablation.COMPETITION_ID_COLUMN` | 1 |
| `shots_core` | `ablation.SHOTS_CORE_COLUMNS` | 18 |
| `shots_on_core` | `ablation.SHOTS_ON_CORE_COLUMNS` | 18 |
| **Total** | — | **80** |

No unrelated feature group is included: the 22 xG columns, all Tier D groups, `league_home_advantage_season`, `league_mean_goals_per_team_match_season`, all label columns and all `X_EXCLUDED_COLUMNS` are excluded, consistent with §2.

### 12.1 Signal 1 — per-family null-rate drift

| Field | Status |
|---|---|
| **Signal definition** | **LOCKED.** Per-family null rate of an inference batch, compared against a historical baseline. Reported as **two distinct, unmerged signals**: (a) **absolute ceiling violation**, (b) **drift alert**. |
| **Metric** | **LOCKED.** Absolute **percentage-point** deviation between the current family null rate and the selected historical baseline. |
| **Drift alert threshold** | **LOCKED: +2 percentage points.** Deviation `> +2.0 pp` above baseline raises `ALERT`. |
| **Absolute availability ceilings** | **UNCHANGED and SEPARATE** — historical operational bound **0.078**, final-season availability bound **0.056** (§10). **These are NOT drift thresholds** and are not re-derived by the monitoring module; it imports the existing constants. |
| **Baseline / reference** | **CONSTRAINED — LOCKED to non-test historical partitions only; 2025/26 EXCLUDED.** Built from the documented per-family partition evidence already in the repository (`docs/PHASE5_V2_CANDIDATE_REVIEW.md` Part 4): `fold_1 train`, `fold_2 train`, `fold_3 train`, `final train (2020/21–2024/25)`. **Residual open item:** which of those non-test partitions (or which aggregation across them) is the operational baseline is **not selected by this specification**; the implementation therefore requires the caller to state it explicitly and provides **no default** — see the decision matrix §13.1. |
| **Timing** | **Prediction-time.** Requires no labels. |
| **Implementation** | `src/monitoring/signals.py` — `evaluate_null_rate_ceiling()` and `evaluate_null_rate_drift()`, emitting signal ids `S1_CEILING` and `S1_DRIFT`. Consumes observed data separately from the availability check; does not modify or replace `candidate_contract.check_feature_availability`. |

**Ceiling and drift must never be merged.** They answer different questions. Worked case from recorded evidence: `form` was 0.0074 on the 2025/26 partition; a rise to 0.070 is a ~9.5× increase in missingness yet stays under the 0.078 ceiling — the ceiling cannot see it. Conversely an absolute level never trained on is a ceiling matter, not a drift matter.

### 12.2 Signal 2 — unseen `competition_id` occurrences

| Field | Status |
|---|---|
| **Signal definition** | **LOCKED.** Occurrence and count of `competition_id` values outside the known set **{200, 419, 423, 477, 499}** (imported from `candidate_contract.KNOWN_COMPETITION_IDS`; not restated). |
| **Threshold** | **LOCKED — categorical.** Any unseen ID: **MUST** emit a monitoring event and be flagged; **MUST NOT** automatically fail inference. |
| **Status emitted** | **`DEGRADED`** (flagged), **never `HARD_FAIL`**. |
| **Baseline / reference** | **LOCKED.** The enumerated known set. No statistical baseline, no minimum sample size. |
| **Timing** | **Prediction-time.** |
| **Implementation** | `evaluate_unseen_competition_ids()`, signal id `S2_UNSEEN_COMPETITION`. |

Predictive quality for unseen leagues remains **not established by current evidence**. **No new league behaviour is defined.**

### 12.3 Signal 3 — predicted class distribution vs. historical base rates

| Field | Status |
|---|---|
| **Signal definition** | **LOCKED: mean predicted probability mass per class** is the primary signal. **Argmax class share is a secondary diagnostic only** and **must not** replace it. |
| **Threshold** | **LOCKED: ±5 percentage points per class**, with **H, D and A evaluated independently**. |
| **Baseline / reference** | **LOCKED: stored historical base-rate reference from already-recorded non-test evidence** — `data/audit/phase3_probability_diagnostics.json` → `diagnostics.actual_class_frequency` (folds 1–3 validation, n=5331): H 0.43632 / D 0.25192 / A 0.31176. Read from the artifact rather than hardcoded. **2025/26 is not used.** |
| **Timing** | **PREDICTION-TIME ONLY (INC-1 resolved).** The rolling live realized-frequency baseline (former option **C-4**) is **EXCLUDED**. Signal 3 may use **only a stored historical reference**, so it never requires labels. |
| **Implementation** | `evaluate_class_distribution()`, signal id `S3_CLASS_DISTRIBUTION`. Argmax shares are recorded in the event's diagnostic block and **never** participate in status determination. |

**Recorded evidence for the locked reading** (folds 1–3, n=5331):

| Class | Historical base rate | Mean predicted probability mass (**primary**) | Argmax class share (**secondary diagnostic**) |
|---|---|---|---|
| H | 0.43632 | 0.42330 | 0.63797 |
| D | 0.25192 | **0.25382** | **0.00825** |
| A | 0.31176 | 0.32288 | 0.35378 |

Probability mass deviates from base rate by +0.00190 for Draw; argmax share deviates by −0.24367. Phase 5B accepted the Draw limitation **under the probability-output contract**, so probability mass is the coherent primary quantity; an argmax-share alert would fire permanently against known, accepted behaviour.

**No claim is made that the candidate is a good Draw-class classifier. It is not** — Draw recall **0.018**, F1 **0.034**, argmax share **0.00825** despite Draw ranking second of three in ~59% of fixtures. The candidate **must never be presented as a reliable Draw-class predictor**.

### 12.4 Signal 4 — mean predicted probability per class vs. realized outcome frequency

| Field | Status |
|---|---|
| **Signal definition** | **LOCKED. POST-OUTCOME ONLY.** Mean predicted probability per class compared against realized outcome frequency over the eligible window. |
| **Threshold** | **LOCKED: ±5 percentage points per class.** |
| **Minimum sample size** | **LOCKED: 200 settled predictions.** |
| **Observation window** | **LOCKED: rolling 30 days.** |
| **Settlement maturity** | **LOCKED: only outcomes settled at least 7 days after the prediction are eligible.** (Implemented exactly as written: `settled_at − predicted_at ≥ 7 days`. See the decision matrix §13.2 for a recorded operational observation about this rule — the rule is implemented as specified and was not reinterpreted.) |
| **Insufficient data** | **LOCKED.** Below 200 eligible observations the emitted status is **`INSUFFICIENT_SAMPLE`** — never `PASS`, never `ALERT`. |
| **Baseline / reference** | Realized outcome frequency **within the eligible window**. **2025/26 is not used as a pre-production baseline.** |
| **Timing** | **POST-OUTCOME ONLY.** |
| **Implementation** | `evaluate_calibration_post_outcome()`, signal id `S4_CALIBRATION`. |

**This signal cannot be operationally exercised until real predictions and real settled outcomes exist.** It is implemented and unit-tested against synthetic records; it has produced no production measurement, and cannot until exposure exists. Its Gate 10 acceptance therefore rests on specified-and-implemented emission, never on observed operation.

### 12.5 Signal 5 — probability-validity failures

| Field | Status |
|---|---|
| **Signal definition** | **LOCKED.** Count of invalid **rows** and invalid **batches** failing the §8 validity rule. |
| **Threshold** | **LOCKED: zero tolerance.** Any invalid probability vector is a **hard inference failure**; status **`HARD_FAIL`**. |
| **Authority** | **`evaluate.validate_probabilities` remains authoritative.** **`src/models/evaluate.py` is NOT modified.** |
| **Mechanism** | **LOCKED.** Monitoring records the failure **externally** and then **re-raises the original exception unchanged** (same object, same type, same message, same traceback chain). |
| **Prohibited** | Monitoring must **not** swallow, normalize, clip, repair, or downgrade invalid probabilities. |
| **Timing** | **Prediction-time.** |
| **Implementation** | `src/monitoring/validity.py` — `monitored_predict_candidate()`, an **external wrapper around `predict_candidate`**. Existing prediction semantics are unchanged; `candidate_contract.py` is not modified. Row-level counts are **diagnostic detail only**; pass/fail authority remains solely with `validate_probabilities`. |

### 12.6 Signal 6 — draw-class behaviour

| Field | Status |
|---|---|
| **Signal definition** | **LOCKED: mean predicted P(draw) versus realized draw frequency.** **The previously rejected ±0.03 heuristic is NOT adopted** and is not referenced by the implementation. |
| **Threshold and policy** | **LOCKED — identical to Signal 4:** ±5 percentage points, minimum n = 200, rolling 30-day window, 7-day outcome-settlement maturity, `INSUFFICIENT_SAMPLE` below the minimum. |
| **Secondary diagnostic** | Draw **argmax share** is recorded as a diagnostic only. **No separate, conflicting argmax-based Draw alert exists.** |
| **Timing** | **POST-OUTCOME ONLY** (same policy as Signal 4). |
| **Implementation** | `evaluate_draw_behaviour()`, signal id `S6_DRAW_BEHAVIOUR`. |

**Poor historical Draw classification remains an accepted limitation under Condition 2** (`DRAW_LIMITATION_ACCEPTED`, scoped to the probability-output contract, reopenable if a hard predicted class enters the contract). Recorded figures: Draw recall **0.018**, F1 **0.034**, argmax share **0.00825**. **No claim is made that the model is a reliable Draw classifier.**

### 12.7 Emission contract (LOCKED)

**Store.** Append-only **JSONL** at a caller-supplied path (`src/monitoring/store.py`). JSONL was chosen over SQLite because it is durable, ordered, trivially inspectable, and requires **no new dependency and no schema migration**; the project already uses SQLite for feature/match data, and adding monitoring tables there would touch existing databases, which is forbidden. No database table is added to any existing database.

**Every event contains** (`src/monitoring/events.py`, `MonitoringEvent`): `schema_version`, `event_id`, `timestamp`, `model_version`, `signal_id`, `signal_name`, `status`, `severity`, `metrics`, `baseline`, `threshold`, `sample_count`, `context` (fixture/batch context where applicable), `message`.

**Statuses (exactly these five):** `PASS`, `ALERT`, `HARD_FAIL`, `INSUFFICIENT_SAMPLE`, `DEGRADED`.

| Case | Status |
|---|---|
| Signal 2 unseen league | **`DEGRADED`** (flagged, not a failure) |
| Signal 5 invalid probability | **`HARD_FAIL`** |
| Signals 4 / 6 below minimum n | **`INSUFFICIENT_SAMPLE`** |

`model_version` is read from `config.MODEL_VERSION` and is therefore `v1.0`; the monitoring module never writes it. **Severity** is a mechanical function of status (not an independent policy): `PASS`/`INSUFFICIENT_SAMPLE` → `INFO`, `DEGRADED`/`ALERT` → `WARNING`, `HARD_FAIL` → `CRITICAL`. Severity *policy* remained a deferrable item, so this mapping is deliberately derived rather than invented as policy.

**Explicitly not built:** dashboards, web APIs, cloud infrastructure, schedulers, alert integrations, UI. No deployment.

### 12.8 Consolidated status

| # | Signal | Definition | Baseline | Threshold | Timing | Emitted by implementation |
|---|---|---|---|---|---|---|
| 1 | Per-family null-rate drift | **LOCKED** | Non-test partitions; 2025/26 excluded; **partition selection caller-supplied** | **LOCKED +2 pp** (ceiling 0.078 / 0.056 kept separate) | Prediction-time | **Yes** — `S1_DRIFT` + `S1_CEILING` |
| 2 | Unseen `competition_id` | **LOCKED** | **LOCKED** known set | **LOCKED** categorical → `DEGRADED` | Prediction-time | **Yes** — `S2_UNSEEN_COMPETITION` |
| 3 | Class distribution | **LOCKED** probability mass | **LOCKED** stored non-test reference | **LOCKED ±5 pp**, per class | **Prediction-time only** (C-4 excluded) | **Yes** — `S3_CLASS_DISTRIBUTION` |
| 4 | Probability vs realized frequency | **LOCKED** | Window-realized | **LOCKED ±5 pp**, n≥200, 30 d, 7 d maturity | **Post-outcome only** | **Yes** — `S4_CALIBRATION` |
| 5 | Probability-validity failures | **LOCKED** | None required | **LOCKED** zero tolerance → `HARD_FAIL` | Prediction-time | **Yes** — `S5_PROBABILITY_VALIDITY` |
| 6 | Draw-class behaviour | **LOCKED** probability mass | Window-realized | **LOCKED** = Signal 4 policy; ±0.03 rejected | **Post-outcome only** | **Yes** — `S6_DRAW_BEHAVIOUR` |

**All six required signals are emitted by the additive implementation. All required thresholds are defined. Monitoring has produced no production measurement, and Gate 10 is not passed merely because the code exists** — see the Gate 10 status note under the acceptance-gate table.

No signal accesses 2025/26. No signal modifies frozen V1 code.

## Acceptance gates

Implementation may not begin until every gate has a defined owner and passes. Each gate below states its requirement, verification method, pass/fail condition and stop condition.

| Gate | Requirement | Verification method | Pass condition | Stop condition |
|---|---|---|---|---|
| **1 — Feature contract** | Exactly the 80 contract columns, in contract order; Conditions 1 & 2 resolved | Automated test comparing the implemented column list to `MODEL_B_COLUMNS`; written resolution of both conditions | Exact set and order match; both conditions closed in writing | Any column added/removed/reordered, or either condition unresolved |
| **2 — Temporal safety** | No feature uses information from or after the predicted fixture | Re-run Phase 2 leakage suite + static audit of all 80 columns | Leakage suite passes; zero features classified UNKNOWN | Any leakage path found, or any feature unprovable |
| **3 — Feature availability** | All 80 columns present with null rates in documented bounds | Pre-inference coverage check against §10 | All present; no all-null column; null rates within bounds | Missing column, all-null column, or out-of-bounds null rate |
| **4 — Preprocessing reproducibility** | Imputation/scaling/encoding fit on training partition only and reproducible | Code audit + fit-isolation tests (fold-swap style) | Statistics depend only on training rows; identical across repeated fits | Any statistic influenced by validation/test data |
| **5 — Deterministic training** | Identical inputs produce identical model and predictions | Train twice, compare predictions byte-for-byte | Byte-identical outputs | Any nondeterminism |
| **6 — Probability validity** | All outputs valid 3-class probability vectors | `validate_probabilities` over a full prediction batch | 100% pass | Any NaN/inf/negative/non-summing row |
| **7 — Regression against Phase 4C** | Implementation reproduces the audited Phase 4C Model B result | Compare implementation output on the 2025/26 rows to `phase4c_final_comparison.json` (**verification only — not a new evaluation, not a new selection input**) | log loss reproduces 0.9960487649063601 to <1e-9 | Any deviation ≥1e-9 |
| **8 — Inference integration** | Candidate callable through the intended prediction path | Integration test on a held-out-shaped batch | Correct shape, ordering, `fixture_id` alignment, valid probabilities | Interface mismatch or silent column substitution |
| **9 — V1 rollback** | V1 restorable without retraining or data regeneration | Rehearse rollback; re-verify V1 checksums and reproduce V1's locked metrics | V1 restored; metrics reproduce to <1e-9 | Rollback fails or V1 metrics do not reproduce |
| **10 — Monitoring** | §12 monitoring in place before any production exposure | Review monitoring implementation and alert thresholds | All §12 signals emitted with defined thresholds | Any required signal absent |

**Gate 10 status note (the gate row above is unchanged and remains binding):**

> **GATE 10 SPECIFICATION: DECISIONS LOCKED** — including all numeric thresholds; recorded in §12 and in `docs/GATE10_MONITORING_SPECIFICATION_DECISION_MATRIX.md`.
> **GATE 10 IMPLEMENTATION: ADDITIVE MODULE IMPLEMENTED** — `src/monitoring/` emits all six required signals. No frozen V1 file modified. No dashboard, API, scheduler, alert integration or deployment.
> **GATE 10 ACCEPTANCE: see the Gate 10 report** — the pass condition is *"All §12 signals emitted with defined thresholds"*; acceptance additionally requires the Gate 10 test suite to execute and pass, and is **not** satisfied merely by the existence of code.
> Monitoring has produced **no production measurement**; Signals 4 and 6 cannot be operationally exercised until real predictions and settled outcomes exist. The absence of production exposure does not satisfy Gate 10. Any future production exposure requires Gate 10 PASS first.

---

## Explicit non-authorizations

This draft does **not** authorize: creating V2, bumping `MODEL_VERSION`, modifying `src/models/config.py` or any V1 source, retraining any model, re-evaluating 2025/26, tuning any hyperparameter, calibrating the candidate, creating a production artifact, deployment, or API/UI work.

**Next authorized action:** human review of `docs/PHASE5_V2_CANDIDATE_REVIEW.md` and this draft, followed by an explicit decision on Conditions 1 and 2.
