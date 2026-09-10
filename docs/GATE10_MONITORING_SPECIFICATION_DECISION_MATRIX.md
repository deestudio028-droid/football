# Gate 10 — Monitoring Specification / Decision Matrix

**Status: GATE 10 — PASS (specification locked; additive monitoring implemented and tested)**

> **Superseding status note (revision 3).** This document was issued at **BLOCKED / SPECIFICATION GAP**, revised to **SPECIFICATION LOCKED / IMPLEMENTATION NOT STARTED**, and is now at **PASS**. Read it as a layered record, not a single snapshot:
> - **§§1–7** — original analysis, retained **unaltered**. Where they say a decision is unresolved, that is the *pre-decision* position.
> - **§8** — governance decisions locked (first authorization round).
> - **§9** — numeric decisions that were still open at that point; **superseded by §13.1**, which records the authorized values.
> - **§§11–12** — consistency audit and the readiness position while thresholds were still open; **§12 is superseded by §13**.
> - **§13** — authorized thresholds, the implementation, acceptance evidence, and the two residual governance notes. **§13 is the current record.**
>
> Gate 10 PASS certifies that specified monitoring exists, emits, and is tested. It is **not** promotion authorization, does not create V2, does not change `MODEL_VERSION` (still `v1.0`), and does not authorize deployment or production exposure.
>
> **GATE 10 SPECIFICATION: DECISIONS LOCKED**
> **GATE 10 IMPLEMENTATION: NOT AUTHORIZED / NOT STARTED**
> **GATE 10 ACCEPTANCE: OPEN**
>
> **Gate 10 is NOT PASS.** Monitoring is not implemented; 0 of 6 signals are emitted.

**Document type:** documentation and decision analysis only. For human approval.

**Date:** 2026-08-17 (original analysis); same-day revision adding §8 and §9.

---

## 0. Scope, authority and non-authorizations

**What this document is.** A formal decision matrix converting the Gate 10 read-only audit findings into a set of explicit, approvable governance decisions. It exists because Gate 10's pass condition — *"All §12 signals emitted with defined thresholds"* — cannot be met by writing code: four of the six required signals have **no threshold defined anywhere in the specification**, and inventing them would make any subsequent "Gate 10 PASS" meaningless.

**Authoritative basis.** Only two sources are treated as authoritative:

1. `docs/PHASE5_V2_MODEL_SPEC_DRAFT.md` — §8, §10, §11, §12, Condition 2, and the acceptance-gate table (Gate 10 row).
2. The Gate 10 read-only audit (this session), whose findings are restated here with file/line references so each claim is independently checkable.

Recorded empirical values are quoted from existing frozen artifacts (`data/audit/phase3_probability_diagnostics.json`, `docs/PHASE5_V2_CANDIDATE_REVIEW.md` Part 4). **No new computation was performed to produce this document.**

**What this document does NOT do.**

- Does **not** invent, adopt, select, default to, or recommend-by-omission any threshold. Every numeric candidate below is explicitly labelled **PROPOSAL — NOT ADOPTED**.
- Does **not** modify production code. No file under `src/` was changed.
- Does **not** modify any frozen V1 source or pinned artifact.
- Does **not** train, evaluate, tune, calibrate, or create any V2 artifact.
- Does **not** access 2025/26 for any new computation.
- Does **not** create monitoring infrastructure.
- Does **not** declare Gate 10 PASS. *(At original issue the status was BLOCKED / SPECIFICATION GAP; as of the later same-day revision it is SPECIFICATION LOCKED / IMPLEMENTATION NOT STARTED — see §8 and §10. Gate 10 is still **not** PASS.)*
- Contains **no tests** and creates **no data artifact**.

`MODEL_VERSION` remains `v1.0`. No V2 exists. No production exposure exists or is authorized.

**Reading convention used throughout.**

| Label | Meaning |
|---|---|
| **SPECIFIED** | Stated in `PHASE5_V2_MODEL_SPEC_DRAFT.md`. Binding. |
| **IMPLEMENTED** | Code exists that computes it. Says nothing about whether it is wired to anything. |
| **PROPOSAL — NOT ADOPTED** | An option written down for the approver to accept, reject, or replace. Carries no authority. Silence is not acceptance. |
| **GAP** | A decision that only the approver can make. |
| **VERBATIM** | Quoted exactly from the source. |

---

## 1. The requirement, quoted verbatim

**Gate 10 row** (`docs/PHASE5_V2_MODEL_SPEC_DRAFT.md:127`):

> | **10 — Monitoring** | §12 monitoring in place before any production exposure | Review monitoring implementation and alert thresholds | All §12 signals emitted with defined thresholds | Any required signal absent |

**§12 in full** (`docs/PHASE5_V2_MODEL_SPEC_DRAFT.md:106–108`) — this is the *entire* section, reproduced with nothing omitted:

> ## 12. Monitoring requirements
>
> Post-promotion (if ever authorized), at minimum: per-family null-rate drift, unseen-`competition_id` occurrences, predicted class distribution vs. historical base rates, mean predicted probability per class vs. realized outcome frequency, probability-validity failures, and draw-class behaviour specifically (given §Condition 2).

**Observation of record.** §12 is two sentences. It enumerates *what to observe* and is silent on *every* operational parameter: threshold, baseline, metric, cadence, window, minimum sample size, emission format, sink, severity, response action, and owner. The word "threshold" does not appear in §12 at all — it appears only in the Gate 10 row's verification and pass columns, which *presuppose* thresholds that §12 never supplies.

This is the structural cause of the block: **Gate 10 tests for the presence of thresholds that no section of the specification defines.**

**Supporting sections relied on below.**

§10 (`:97–100`, VERBATIM, relevant bullets):

> - Null rate per family within documented historical bounds (≤ ~7.7% observed historically; ~5.5% on 2025/26).
> - `competition_id` within the known set {200, 419, 423, 477, 499}; an unseen league must be flagged (behaviour is degraded-not-failed, and predictive quality for it is **not established by current evidence**).

§8 (`:88`, VERBATIM):

> Every emitted probability set must pass `evaluate.validate_probabilities`: exactly 3 classes, all finite, all non-negative, each row summing to 1 within `atol=1e-6`. A failure is a hard stop, not a warning.

Condition 2 (`:10`, VERBATIM, abbreviated to the operative clauses):

> **Condition 2 — draw-class limitation: CLOSED — `DRAW_LIMITATION_ACCEPTED` (scoped).** … **The acceptance is scoped to the current probability-output contract and must be reopened if a predicted class enters the contract, or if the deferred Poisson/scoreline layer or any UI/API consumer introduces a hard-class requirement.** Draw performance is not solved (recall 0.018, F1 0.034) and the candidate must never be presented as a reliable Draw-class predictor.

---

## 2. Cross-cutting findings that apply to all six signals

These five facts shape every row of the matrix and are stated once here rather than repeated six times.

**F1 — No monitoring implementation exists.** There is no monitoring module, alerting module, metrics emitter, prediction log, drift job, scheduler, or dashboard anywhere in the repository. `src/` contains exactly three packages (`ingestion`, `features`, `models`); none contains a monitoring component. `logs/` holds ingestion run logs only.

**F2 — No production exposure surface exists.** The only inference path is `candidate_contract.predict_candidate` (`src/models/candidate_contract.py:340`), a library function currently called only from tests. There is no serving entrypoint, no scheduled prediction run, and no store that persists predictions. Gate 10 is written as a precondition — *"before any production exposure"* — and that trigger has not occurred.

**F3 — The distinction between "computable" and "emitted" is the crux.** Five of six signals are *computable today* from existing code. **Zero are emitted as operational monitoring signals.** Gate 10's pass condition uses the word "emitted". Offline functions that write into a Phase 3 JSON audit file, or that raise an exception, satisfy "computable" but not "emitted".

**F4 — `check_feature_availability` has no operational call site.** `src/models/candidate_contract.py:189` implements the §10 pre-inference check and correctly sets `passed=False` on breach — but a repository-wide search finds its only caller is `tests/test_gate3_feature_availability.py`. In particular `predict_candidate` does **not** call it; it calls `verify_feature_contract()` and `select_candidate_features()` only. Signals 1 and 2 therefore have working detection logic attached to nothing.

**F5 — Three signals are structurally post-hoc.** Signals 3 (partly), 4 and 6 require *realized outcomes*. They cannot be evaluated at prediction time under any threshold choice, because the label does not exist until the fixture is played and settled. Any monitoring design must therefore include a label-arrival delay and an accumulation window. §12 mentions neither.

---

## 3. Decision matrix

Each signal below follows the identical 16-field structure requested.

---

### SIGNAL 1 — Per-family null-rate drift

**1.1 Exact signal definition (as the approver must ultimately fix it).**
For each of the six documented feature families — `goals_core` (18 columns), `form` (20), `strength` (5), `competition_id` (1), `shots_core` (18), `shots_on_core` (18) — the per-family null rate of an inference batch, compared against a reference baseline, to detect **change over time**.

The family sizes above are quoted from `CANDIDATE_FAMILY_COUNTS` (`src/models/candidate_contract.py:79`), whose own docstring states it is *"Used only for verification reporting -- the authoritative membership is the tuple above."* **A family→columns mapping does not exist in code.** Only counts exist. This is a prerequisite gap, not a threshold gap.

**1.2 What is currently implemented.**
`check_feature_availability()` (`src/models/candidate_contract.py:189`) computes `null_rate_by_column` for all 80 columns and reports `max_null_rate_observed`, `max_null_rate_column`, `out_of_bounds_columns`, `all_null_columns`, `constant_columns`, `passed`, `failures`. Bounds enforced in-function: `MAX_HISTORICAL_NULL_RATE = 0.078`, `MAX_FINAL_SEASON_NULL_RATE = 0.056` (`:114–115`).

Two defects relative to §12: the granularity is **per column, not per family**; and the measure is an **absolute ceiling, not drift**. Per F4, the function is never called operationally.

**1.3 What §12 explicitly requires.**
VERBATIM: *"per-family null-rate drift"*. Two words matter: **per-family** (not per-column) and **drift** (not ceiling).

**1.4 Does the specification define a threshold?**

> **PARTIALLY — and the two constructs must not be conflated (SPECIAL RULE A).**

This document keeps them strictly separate:

| Construct | Status | Value | What it answers |
|---|---|---|---|
| **Absolute availability ceiling** | **SPECIFIED** (§10) | ≤ ~7.7% historical; ~5.5% on 2025/26; encoded 0.078 / 0.056 | *"Is this batch's missingness within the band ever observed historically?"* |
| **Drift from baseline** | **NOT SPECIFIED — GAP** | none exists | *"Has missingness moved away from its reference level, regardless of whether it is still under the ceiling?"* |

These are **not** substitutes. A concrete illustration using recorded values: `form` was 0.0074 on the 2025/26 partition. If it rose to 0.070, that is a **9.5× increase** in missingness for that family — an unambiguous feed-behaviour change — yet it remains comfortably under the 0.078 ceiling and the ceiling check would report `passed=True` and stay silent. The ceiling cannot detect that event. Conversely a drift check with no ceiling would not catch an absolute level that has never been trained on. **Both are needed; neither implies the other.**

**1.5 Missing governance decisions (GAPs).**

- **GAP-1a** — Reference baseline: which recorded partition is the drift baseline (see 1.6)?
- **GAP-1b** — Drift metric: absolute percentage-point delta, relative/ratio change, or a distributional distance? These rank breaches differently, especially for `strength` (baseline ~0.0028), where a +0.01 absolute move is trivial in points but a ~4.6× relative move.
- **GAP-1c** — Drift tolerance: unspecified.
- **GAP-1d** — Whether the §10 ceiling is retained *alongside* drift, or whether the ceiling alone is accepted as Gate 10's null-rate signal with drift-proper explicitly deferred.
- **GAP-1e** — Family→columns mapping authority (derive from the 80-column contract, or state it in the spec).
- **GAP-1f** — Granularity: does family-level monitoring *replace* or *supplement* the existing per-column check? A single pathological column inside an 18-column family is diluted ~18× at family level and could pass while the column is entirely null.

**1.6 Required baseline / reference population.**
A per-family baseline **already exists as recorded evidence** — `docs/PHASE5_V2_CANDIDATE_REVIEW.md:129–137`, Part 4, "Max null rate per family, per partition":

| Partition | goals_core | form | strength | competition_id | shots_core | shots_on_core |
|---|---|---|---|---|---|---|
| fold_1 train | 0.0764 | 0.0764 | 0.0027 | 0.0000 | 0.0772 | 0.0772 |
| fold_2 train | 0.0555 | 0.0555 | 0.0027 | 0.0000 | 0.0560 | 0.0560 |
| fold_3 train | 0.0542 | 0.0444 | 0.0028 | 0.0000 | 0.0550 | 0.0550 |
| final train (2020/21–2024/25) | 0.0543 | 0.0374 | 0.0028 | 0.0000 | 0.0551 | 0.0550 |
| **final test (2025/26)** | 0.0548 | 0.0074 | 0.0029 | 0.0000 | **0.0548** | **0.0548** |

**This is materially favourable for Gate 10:** the baseline is a *read* of an existing frozen document, so establishing it requires **no new computation and no 2025/26 access**.

Baseline options — **PROPOSAL — NOT ADOPTED**, approver must choose:

- **(B-1)** `final train (2020/21–2024/25)` row. Rationale: it is the partition a promoted model would be fitted on, and it contains no test-season information.
- **(B-2)** `final test (2025/26)` row. Rationale: most recent, closest to live conditions. **Caution:** although reading the recorded row involves no new computation, adopting the test season as an operational reference makes 2025/26 a tuning-adjacent input, which the standing constraints have consistently excluded. Flagged as a governance concern, not recommended by this document.
- **(B-3)** Per-family max across the three fold-train rows. Rationale: most conservative; widest tolerance.
- **(B-4)** A rolling live baseline computed after exposure begins. Rationale: measures true drift. **Consequence:** unavailable at Gate 10 time, so Gate 10 could not close on this signal until after exposure — which contradicts Gate 10 being a pre-exposure gate.

**1.7 Required measurement window / cadence.** **GAP.** Not specified. Options — **PROPOSAL — NOT ADOPTED**: per inference batch; per matchday; rolling 7 days; rolling N batches.

**1.8 Minimum sample size.** **GAP.** Applicable and material. A null rate over a 6-fixture batch is extremely noisy; a single missing value is 16.7%, which would breach any plausible tolerance for reasons of batch size alone. **PROPOSAL — NOT ADOPTED:** a minimum batch size below which the signal is recorded but suppressed from alerting.

**1.9 Alert severity options.** **GAP.** **PROPOSAL — NOT ADOPTED:** ceiling breach → blocking (§10 already frames availability as a pre-inference gate); drift-only breach (still under ceiling) → warning/investigate; all-null column → blocking.

**1.10 Expected response / action.** **GAP.** Note §10's own language for a related case is *"degraded-not-failed"* for unseen leagues but is silent on whether an out-of-bounds null rate blocks inference. Per `docs/PHASE5_V2_CANDIDATE_REVIEW.md:153` (VERBATIM): *"Behaviour if an entire feed (e.g. shots) were to go missing at prediction time is **not established by current evidence** and is a monitoring requirement, not a demonstrated capability."* **The response action is therefore genuinely undetermined and must be decided, not inferred.**

**1.11 Evaluable at prediction time, or only after outcomes arrive?** **At prediction time.** Requires no labels. This is a pre-inference signal.

**1.12 Would the historical baseline require 2025/26 access?** **No** under options B-1/B-3/B-4 — the baseline is a read of an existing recorded table. Under B-2 it is still only a *read* of a recorded row (no new computation), but adopting the test season as an operational reference is a governance decision in its own right.

**1.13 Would implementation require modifying frozen V1 code?** **No.** Additive only. Would touch `src/models/candidate_contract.py` (not pinned; already extended by Gates 1/3/8) **only if** the family mapping is placed there — itself an authorization item.

**1.14 Implement now or defer?** **PROPOSAL — NOT ADOPTED:** partially now. The family mapping and the family-level computation are pre-exposure-testable and could be specified and built before exposure; a *live* rolling drift baseline is inherently post-exposure.

---

### SIGNAL 2 — Unseen `competition_id` occurrences

**2.1 Exact signal definition.** Occurrence and count of `competition_id` values in an inference batch outside the known set {200, 419, 423, 477, 499}, plus null `competition_id` values.

**2.2 What is currently implemented.** `check_feature_availability()` returns a `competition_id` block: `observed_ids`, `known_ids`, `unseen_ids`, `unseen_flagged`, `null_count`, and a documented degraded-not-failed note. `KNOWN_COMPETITION_IDS = frozenset({200, 419, 423, 477, 499})` (`src/models/candidate_contract.py:122`). Correctly, `unseen_flagged` does **not** set `passed=False` (consistent with §10's degraded-not-failed language), whereas a null `competition_id` **does** append a failure. Per F4, never called operationally.

**2.3 What §12 explicitly requires.** VERBATIM: *"unseen-`competition_id` occurrences"*.

**2.4 Does the specification define a threshold?**

> **YES — and per SPECIAL RULE B, no numeric threshold is needed or proposed.**

§10 enumerates the known set and states the required behaviour: *"an unseen league must be flagged (behaviour is degraded-not-failed…)"*. This is a **complete categorical rule**: any unseen ID → flag; do not fail. There is nothing to tune. **This is the only one of the six signals whose detection rule is fully specified.**

**2.5 Missing governance decisions.** Exactly four, and only four — all operational, none numeric:

- **GAP-2a — Emission.** Where does the flag go? Today it is a dictionary field returned to a caller that does not exist (F4).
- **GAP-2b — Cadence.** Per batch, or aggregated?
- **GAP-2c — Severity.** §10 fixes the *model* behaviour (degraded-not-failed) but not the *alert* severity. Degraded-not-failed for the prediction is compatible with a high-severity operational alert, because §10 also says predictive quality for an unseen league is *"not established by current evidence"*.
- **GAP-2d — Response.** Serve the degraded prediction; serve it with an explicit caveat attached; suppress it; or escalate for a human decision. **Undetermined.** This interacts with Condition 2's reopening clause: if a consumer ever presents these outputs, an unflagged degraded prediction becomes a presentation risk.

**2.6 Required baseline / reference population.** The enumerated known set, already **SPECIFIED** in §10 and encoded. No statistical baseline required.

**2.7 Required measurement window / cadence.** GAP-2b. **PROPOSAL — NOT ADOPTED:** evaluate every batch (detection is exact, not statistical, so no accumulation is needed for validity); aggregate counts for reporting.

**2.8 Minimum sample size.** **Not applicable.** Detection is exact set membership; a single occurrence is fully informative.

**2.9 Alert severity options.** GAP-2c. **PROPOSAL — NOT ADOPTED:** (i) informational — degraded-not-failed implies tolerated; (ii) warning — quality not established; (iii) high — a new league is a scope change the evidence base does not cover. Separately, **null `competition_id`** is arguably distinct and more severe than an *unseen* one, since the existing code already treats it as a failure while treating unseen as a flag. The approver should confirm this asymmetry is intended.

**2.10 Expected response / action.** GAP-2d.

**2.11 Evaluable at prediction time, or only after outcomes arrive?** **At prediction time.** No labels required.

**2.12 Would the historical baseline require 2025/26 access?** **No.** The known set is a specified constant.

**2.13 Would implementation require modifying frozen V1 code?** **No.** Wiring and emission only; detection logic already exists. **This is the lowest-effort signal to close.**

**2.14 Implement now or defer?** **PROPOSAL — NOT ADOPTED:** now. Detection exists, the rule is specified, and only emission/cadence/severity/response remain — all decidable pre-exposure.

---

### SIGNAL 3 — Predicted class distribution vs. historical base rates

**3.1 Exact signal definition.** **This is precisely what is ambiguous — see 3.4.**

**3.2 What is currently implemented.** `compute_probability_diagnostics()` (`src/models/calibration.py:229`) emits `argmax_counts`, `argmax_percentage`, `actual_class_frequency`, `mean_predicted_probability`, `mean_predicted_probability_by_actual_class`, plus log loss / Brier / ECE. Recorded in `data/audit/phase3_probability_diagnostics.json`. Offline, descriptive, written to a JSON audit file; nothing consumes it and nothing fails on it. `calibration.py:62` pins `SELECTED_MODEL = MODEL_LOGREG` — *"frozen per Step 3; never changed by this module"* — so these are **V1-frozen** diagnostics.

**3.3 What §12 explicitly requires.** VERBATIM: *"predicted class distribution vs. historical base rates"*.

**3.4 Does the specification define a threshold?**

> **NO — and there is a prior ambiguity that must be resolved before any threshold is even meaningful (SPECIAL RULE C).**

The phrase *"predicted class distribution"* admits two materially different readings:

- **Reading (i) — argmax class share.** The proportion of fixtures whose highest-probability class is H / D / A.
- **Reading (ii) — mean predicted probability mass.** The average probability assigned to each class.

**The existing historical diagnostics show these differ materially** — recorded folds-1–3 validation values, n=5331, from `data/audit/phase3_probability_diagnostics.json`:

| Class | Historical base rate | Reading (i): argmax share | Reading (ii): mean probability mass |
|---|---|---|---|
| H | 0.43632 | 0.63797 | 0.42330 |
| D | 0.25192 | **0.00825** | **0.25382** |
| A | 0.31176 | 0.35378 | 0.32288 |

For the Draw class the two readings differ by roughly **31×** (0.00825 vs 0.25382). Against the same base rate of 0.25192:

- Under **Reading (i)**, Draw deviates from base rate by **−0.24367** — a catastrophic-looking breach under any plausible tolerance, and it would fire permanently from day one as the model's *normal, known, accepted* behaviour (Condition 2, `DRAW_LIMITATION_ACCEPTED`). A monitor that always fires conveys no information and trains operators to ignore it.
- Under **Reading (ii)**, Draw deviates by **+0.00190** — well within any plausible tolerance, and the signal would be quiet and informative.

**The choice of reading therefore determines whether this monitor is usable at all.** It is not a stylistic preference and **must not be made without authorization.** A further consequence: under Reading (i), Draw's deviation is *structural and accepted*, so a Reading (i) monitor would additionally require a Draw-specific exemption or a separate Draw tolerance — which entangles Signal 3 with Signal 6.

**3.5 Missing governance decisions.**

- **GAP-3a** — Choose Reading (i), Reading (ii), or both-as-separate-signals.
- **GAP-3b** — Fix the historical base-rate reference (see 3.6).
- **GAP-3c** — Tolerance: absolute deviation per class, relative deviation, or a distributional distance across all three classes.
- **GAP-3d** — Whether a per-class tolerance or a single aggregate is used; and if Reading (i) is chosen, whether Draw is exempted.
- **GAP-3e** — Whether the base rate is static (frozen) or re-estimated over time as leagues evolve.

**3.6 Required baseline / reference population.** **PROPOSAL — NOT ADOPTED:**

- **(C-1)** Recorded folds-1–3 validation base rates (H 0.43632 / D 0.25192 / A 0.31176, n=5331). Available today as a read of a frozen artifact; no new computation; no 2025/26 access.
- **(C-2)** Final-train (2020/21–2024/25) label frequencies. Rationale: matches the fitted partition. **Requires new computation** over historical data (not 2025/26) — needs authorization since it is a new computation even though it touches no test season.
- **(C-3)** A published/external league base rate. Introduces an external dependency; not recommended by this document.
- **(C-4)** Rolling live realized frequencies. Post-exposure only.

**3.7 Required measurement window / cadence.** **GAP.** Under Reading (i) the signal is a proportion over a batch and needs enough fixtures to be stable; under Reading (ii) it is a mean of probabilities and is far more stable at small n. **PROPOSAL — NOT ADOPTED:** rolling window of N fixtures, or per-matchday.

**3.8 Minimum sample size.** **GAP.** Material, especially under Reading (i). With Draw's argmax share at ~0.8% historically, a batch would need to be very large before a *zero* Draw-argmax count carried any information at all. **PROPOSAL — NOT ADOPTED:** a stated minimum-fixture count below which the signal is recorded but not alerted.

**3.9 Alert severity options.** **GAP.** **PROPOSAL — NOT ADOPTED:** warning (distribution shift is diagnostic, not a correctness failure) rather than blocking, since a shifted distribution does not make an individual probability invalid.

**3.10 Expected response / action.** **GAP.** **PROPOSAL — NOT ADOPTED:** investigate upstream feature/feed changes; consider recalibration review; escalate to a rollback decision under §11 if sustained. Note recalibration is **not currently authorized** under any standing instruction.

**3.11 Evaluable at prediction time, or only after outcomes arrive?** **Mixed, and this is a useful property.** The *predicted* side (both readings) is available **at prediction time**. The *base-rate* side is a stored reference under C-1/C-2/C-3, so the comparison is available at prediction time. Only under C-4 (rolling live realized frequencies) does the signal become post-hoc.

**3.12 Would the historical baseline require 2025/26 access?** **No** under C-1 (read of an existing artifact). **No new 2025/26 access** under C-2 either, but C-2 does require a new historical computation and therefore authorization.

**3.13 Would implementation require modifying frozen V1 code?** **No** if implemented additively. **`calibration.py` must not be modified** — it is frozen-by-behaviour for Phase 3 reproducibility (`SELECTED_MODEL` comment at `:62`), and Phase 3/4C artifacts are pinned. The required computations are simple enough to implement additively without importing its mutable state.

**3.14 Implement now or defer?** **PROPOSAL — NOT ADOPTED:** specification now (GAP-3a is required regardless of timing and is the single highest-value decision in this document); implementation deferrable until exposure, since there are no predictions to monitor.

---

### SIGNAL 4 — Mean predicted probability per class vs. realized outcome frequency

**4.1 Exact signal definition.** For each class in H/D/A: the mean predicted probability assigned to that class over a window, compared against the realized frequency of that outcome over the same window — i.e. an operational calibration check.

**4.2 What is currently implemented.** `compute_probability_diagnostics()` emits `mean_predicted_probability`, `actual_class_frequency`, `mean_predicted_probability_by_actual_class`. `compute_ece()` (`src/models/calibration.py:179`) emits top-label ECE plus a 10-bin table. `compute_reliability_bins()` (`:345`) emits per-class reliability bins. All offline, all descriptive, all V1-frozen inputs, none enforced, none emitted operationally.

Recorded folds-1–3 values (n=5331): mean predicted H 0.42330 / D 0.25382 / A 0.32288 versus realized H 0.43632 / D 0.25192 / A 0.31176. Per-class absolute gaps: H **−0.01302**, D **+0.00190**, A **+0.01112**.

**4.3 What §12 explicitly requires.** VERBATIM: *"mean predicted probability per class vs. realized outcome frequency"*. Note this phrasing is unambiguous about the quantity — unlike Signal 3, it explicitly says *mean predicted probability*. It is silent on everything operational.

**4.4 Does the specification define a threshold?**

> **NO.** No calibration tolerance, no ECE limit, no per-class gap bound exists anywhere in the specification.

**4.5 Missing governance decisions (SPECIAL RULE D — all five explicitly identified as unresolved).**

- **GAP-4a — Tolerance / threshold.** **UNRESOLVED.** No value is specified. For scale only, and explicitly **not** as a proposal: the largest recorded per-class gap on folds 1–3 is 0.01302 (H). Quoting an observation is not proposing a threshold, and a bound written to describe an observation cannot be violated by that observation — the same trap documented for the §10 ceiling at `src/models/candidate_contract.py:102–113`.
- **GAP-4b — Minimum sample size.** **UNRESOLVED.** Critical here. A realized frequency over a small batch is dominated by sampling noise: over 10 fixtures the realized Draw frequency can only take values in multiples of 0.1, so it cannot be within 0.02 of 0.25192 *even if the model is perfectly calibrated*. Without a minimum-n rule this monitor produces false alarms by construction.
- **GAP-4c — Outcome/label availability delay.** **UNRESOLVED.** Labels exist only after the fixture is played and settled. The specification does not state the assumed delay, how late-arriving or voided/abandoned fixtures are handled, or whether a partially-settled window may be evaluated.
- **GAP-4d — Comparison window.** **UNRESOLVED.** Rolling fixtures, rolling days, matchday, or season-to-date. Interacts directly with GAP-4b.
- **GAP-4e — Baseline definition.** **UNRESOLVED.** Two distinct things could serve as the reference: the *realized frequency within the same window* (self-referential calibration check — the natural reading of §12's wording), or a *fixed historical base rate* (which would make this a duplicate of Signal 3 Reading (ii)). §12's phrase *"realized outcome frequency"* points to the former, but this should be confirmed rather than assumed.

Additional undecided item: whether the signal is the simple mean-vs-frequency gap, or a binned calibration measure (ECE / reliability bins) — the latter is already implemented offline and is strictly more informative, but §12's literal wording asks only for the former.

**4.6 Required baseline / reference population.** Per GAP-4e. **PROPOSAL — NOT ADOPTED:** realized outcomes within the same evaluation window (self-referential), with the recorded folds-1–3 values retained only as a *historical context* figure, not as the comparison target.

**4.7 Required measurement window / cadence.** GAP-4d. **PROPOSAL — NOT ADOPTED:** a rolling window sized from the GAP-4b minimum-n decision, evaluated on a fixed cadence after the label-delay allowance.

**4.8 Minimum sample size.** GAP-4b. **Applicable and decisive.** Any tolerance chosen without a matching minimum-n is not implementable.

**4.9 Alert severity options.** **GAP.** **PROPOSAL — NOT ADOPTED:** warning on a single window breach; escalate on consecutive breaches. Not blocking — this signal is retrospective and cannot gate a prediction that has already been served.

**4.10 Expected response / action.** **GAP.** **PROPOSAL — NOT ADOPTED:** investigate; review calibration; consider §11 rollback if sustained. Recalibration is **not currently authorized**.

**4.11 Evaluable at prediction time, or only after outcomes arrive?** **Only after outcomes arrive.** Structurally post-hoc (F5). This has a direct Gate 10 consequence: **this signal cannot be empirically demonstrated before production exposure exists**, because there are no served predictions and no subsequent realized outcomes to pair with them. Gate 10's pass condition ("emitted with defined thresholds") can be met by *specification plus implemented emission*, but not by *observed operation*, until exposure exists.

**4.12 Would the historical baseline require 2025/26 access?** **No** under the self-referential reading (4.6). A tempting alternative — computing a calibration baseline from the 2025/26 test season — **would** constitute new 2025/26 computation and is therefore **excluded** under standing constraints. Recorded Phase 4C values may be *read*, but must not be recomputed or used as a tuning input.

**4.13 Would implementation require modifying frozen V1 code?** **No** if additive. `calibration.py` must not be modified (see 3.13).

**4.14 Implement now or defer?** **PROPOSAL — NOT ADOPTED:** specify now, defer implementation until exposure. This is the signal with the weakest case for building now, since it cannot produce a single valid measurement until predictions have been served and settled.

---

### SIGNAL 5 — Probability-validity failures

**5.1 Exact signal definition.** Count and rate of prediction rows or batches that fail the §8 validity rule: not exactly 3 classes, non-finite, negative, or row sum not equal to 1 within tolerance.

**5.2 What is currently implemented.** `validate_probabilities()` (`src/models/evaluate.py:28`) enforces exactly this rule and **raises `MetricInputError`**. It **is** wired into the inference path: `predict_candidate` calls it after a shape check (`src/models/candidate_contract.py`). This makes Signal 5 the **only** §12 signal whose check is actually enforced operationally.

**The gap is emission, not detection (SPECIAL RULE E).** An exception is a *control*: it stops the batch. It is not a *signal*: no counter, no rate, no persisted record, no aggregate. Nothing can answer "how many validity failures occurred last week?" — because a raised exception leaves no trace. §12 asks for failures to be *monitored*; today they are only *raised*.

**5.3 What §12 explicitly requires.** VERBATIM: *"probability-validity failures"*. Read with §8 (VERBATIM): *"A failure is a hard stop, not a warning."*

**5.4 Does the specification define a threshold?**

> **YES — effectively zero-tolerance, and this is the least ambiguous threshold of the six.**

§8 makes any failure a hard stop. The Gate 6 row states pass = *"100% pass"* and stop = *"Any NaN/inf/negative/non-summing row"*. The threshold is therefore **any occurrence**. No numeric tuning is required or proposed.

**5.5 Missing governance decisions.**

- **GAP-5a — Emission mechanism.** How is a failure counted and persisted, given that it currently manifests as a raised exception that must continue to propagate (§8's hard-stop requirement must not be softened)?
- **GAP-5b — Whether "hard stop" and "counted" can coexist.** They can — a counting wrapper can record and then re-raise — but this must be authorized explicitly, since anything touching this path is safety-relevant.
- **GAP-5c — Severity.** Presumably maximum, but not stated.
- **GAP-5d — Response.** §8 fixes the *immediate* behaviour (hard stop). It does not state the operational escalation, or whether a validity failure triggers a §11 rollback consideration.
- **GAP-5e — Whether near-misses are also recorded.** Rows summing to within tolerance but at the edge would be a leading indicator. Not required by §12; noted as an option only.

**5.6 Required baseline / reference population.** **None.** Absolute zero-tolerance rule; no statistical baseline required.

**5.7 Required measurement window / cadence.** Per prediction call for detection (already the case). Aggregation window for reporting is a **GAP**. **PROPOSAL — NOT ADOPTED:** cumulative count plus per-window count.

**5.8 Minimum sample size.** **Not applicable.** A single failure is fully informative.

**5.9 Alert severity options.** GAP-5c. **PROPOSAL — NOT ADOPTED:** highest severity, consistent with §8's hard-stop framing.

**5.10 Expected response / action.** GAP-5d. **PROPOSAL — NOT ADOPTED:** the batch already fails closed; escalate immediately; treat as a candidate §11 rollback trigger.

**5.11 Evaluable at prediction time, or only after outcomes arrive?** **At prediction time.** No labels required. **This is the only §12 signal that is both pre-outcome and already enforced** — and therefore the one closest to closable.

**5.12 Would the historical baseline require 2025/26 access?** **No.**

**5.13 Would implementation require modifying frozen V1 code?**

> **It must NOT — and this is a hard constraint (SPECIAL RULE E).**

`src/models/evaluate.py` is one of the **seven frozen V1 production files pinned in `data/audit/phase4c_prerun_manifest.json`** (`config.py`, `train.py`, `splits.py`, `evaluate.py`, `data.py`, `baselines.py`, `run_experiments.py`), verified byte-identical under Gate 9. Modifying it would break the Gate 9 checksum verification and the §11 rollback guarantee.

**PROPOSAL — NOT ADOPTED (additive external approach only):** a monitoring counter placed *outside* `evaluate.py` that invokes the unmodified `validate_probabilities`, records the outcome, and re-raises unchanged on failure — preserving §8's hard stop exactly. Concretely this means a wrapper at the monitoring layer, not a change to the validator and not a change to `predict_candidate`'s existing call. Whether such a wrapper may be introduced, and whether it wraps or sits alongside the existing call in `predict_candidate`, both require authorization (the latter modifies inference behaviour — see AUTH-I).

**5.14 Implement now or defer?** **PROPOSAL — NOT ADOPTED:** now. The detection rule is fully specified, the threshold is unambiguous, no baseline is needed, no labels are needed, and only additive emission is missing. **This is the single most closable signal.**

---

### SIGNAL 6 — Draw-class behaviour

**6.1 Exact signal definition.** **Not determinable from the specification.** §12 says *"draw-class behaviour specifically (given §Condition 2)"* without naming which behaviour. Candidate quantities that "draw-class behaviour" could denote, all already computed offline by `investigate_draw_probability()`: mean predicted P(draw); the gap between mean predicted P(draw) and empirical draw frequency; Draw argmax share; Draw's rank distribution (1st/2nd/3rd of 3); Draw recall/F1; Draw reliability bins.

**6.2 What is currently implemented.** `investigate_draw_probability()` (`src/models/calibration.py:274`), recorded in `data/audit/phase3_probability_diagnostics.json` → `draw_probability_investigation`, plus the governance review `docs/PHASE5B_DRAW_LIMITATION_REVIEW.md`. Recorded folds-1–3 values (n=5331):

| Quantity | Recorded value |
|---|---|
| `empirical_draw_frequency` | 0.25192 |
| `mean_predicted_p_draw_overall` | 0.25382 |
| `mean_predicted_minus_empirical_gap` | **+0.00190** |
| `argmax_draw_percentage` | **0.00825** |
| `pct_draw_ranked_2nd_of_3` | 0.59032 |

Also recorded in Condition 2 (VERBATIM): Draw *"recall 0.018, F1 0.034"*.

These figures capture the accepted limitation precisely: Draw probability mass is **well calibrated in aggregate** (gap +0.0019), while Draw **almost never wins the argmax** (0.8%) despite being ranked second in ~59% of fixtures. Any Draw monitor must be built on a quantity chosen with this structure in mind.

**6.3 What §12 explicitly requires.** VERBATIM: *"draw-class behaviour specifically (given §Condition 2)"*. The cross-reference to Condition 2 is the interpretive key: Condition 2 accepted the limitation **scoped to the current probability-output contract** and requires reopening *"if a predicted class enters the contract, or if the deferred Poisson/scoreline layer or any UI/API consumer introduces a hard-class requirement."* A defensible reading is that Signal 6 exists to detect (a) degradation of Draw *probability* quality, and (b) the arrival of a condition that reopens Condition 2. **This is a reading, not a finding — it requires confirmation.**

**6.4 Does the specification define a threshold?**

> **NO — and the existing ±0.03 heuristic must NOT be treated as one (SPECIAL RULE F).**

`investigate_draw_probability()` contains a `±0.03` band on `mean_predicted_minus_empirical_gap`. **This document does not promote it into an approved monitoring threshold.** Its own source describes it, VERBATIM (docstring, `src/models/calibration.py:274–281`):

> *"The `top_line_heuristic_verdict` is an explicit, stated rule (not a black-box judgment) meant as a quick summary -- the reliability-bin table for Draw (Step 3) is the rigorous, per-probability-level answer and should be read alongside this."*

and its returned `caveat` field, VERBATIM:

> *"This is a descriptive top-line heuristic, not a statistical test, and does not alter the model regardless of its verdict. Read alongside the Draw reliability-bin table."*

Three reasons it cannot be silently adopted:

1. **It was authored as descriptive, explicitly non-statistical, and explicitly non-consequential.** Operational monitoring is by definition consequential — it fires alerts and can trigger rollback consideration. Repurposing it inverts its stated design intent.
2. **It was calibrated to a Phase 3 diagnostic question** ("is Draw's low argmax share an argmax artifact or systematic underestimation?"), not to an operational drift question. Fitness for one does not transfer to the other.
3. **Phase 5B is explicit on the governing principle** (`docs/PHASE5B_DRAW_LIMITATION_REVIEW.md:89`, VERBATIM): *"**No epsilon threshold exists** and none was invented. No delta in this review is described as meaningful or significant."*

**Therefore: the ±0.03 band requires explicit ratification before any operational use.** It is listed below as a ratification *option*, clearly labelled, with no presumption in its favour.

**6.5 Missing governance decisions.**

- **GAP-6a — Which quantity constitutes "draw-class behaviour"?** Must be fixed before anything else. Choosing the argmax share would create a monitor that fires permanently against accepted behaviour (see Signal 3, 3.4). Choosing the probability gap yields a quiet, informative monitor. These are opposite outcomes from the same §12 sentence.
- **GAP-6b — Ratify, reject, or replace the ±0.03 heuristic.** Options: (i) ratify as-is for operational use, with the reinterpretation recorded in writing; (ii) reject and set an independent threshold; (iii) reject and defer. **No option is preselected.**
- **GAP-6c — Whether Signal 6 monitors probability quality, Condition-2 reopening triggers, or both.**
- **GAP-6d — Whether Draw is monitored *in addition to* or *instead of* its treatment inside Signals 3 and 4.** Without a decision, Draw could be triple-counted (Signal 3 Reading (i), Signal 4 per-class, Signal 6) and generate three correlated alerts from one underlying event.
- **GAP-6e — Whether the "must never be presented as a reliable Draw-class predictor" clause (Condition 2, VERBATIM) implies a *presentation* control that monitoring must verify.** This is arguably a compliance check rather than a statistical monitor, and is currently unowned by any gate.

**6.6 Required baseline / reference population.** **PROPOSAL — NOT ADOPTED:** the recorded folds-1–3 Draw figures in §6.2 (a read of a frozen artifact; no new computation; no 2025/26 access), or a rolling live realized draw frequency (post-exposure only).

**6.7 Required measurement window / cadence.** **GAP.** **PROPOSAL — NOT ADOPTED:** aligned with whichever window is chosen for Signal 4, to avoid inconsistent Draw verdicts between two monitors reading the same data.

**6.8 Minimum sample size.** **GAP.** Applicable and severe if an argmax-based quantity is chosen: at a ~0.8% historical Draw-argmax rate, a window of 100 fixtures has an expected Draw-argmax count below 1, so a count of zero is entirely unremarkable and carries almost no information. Any argmax-based Draw monitor needs a very large window to be meaningful at all — a strong practical argument the approver should weigh under GAP-6a.

**6.9 Alert severity options.** **GAP.** **PROPOSAL — NOT ADOPTED:** informational/warning for probability-gap movement; **high** for a Condition-2 reopening trigger (a hard-class requirement entering the contract), since that invalidates a closed condition and would require reopening Phase 5B.

**6.10 Expected response / action.** **GAP.** **PROPOSAL — NOT ADOPTED:** for a probability-gap breach, investigate and review calibration (recalibration not currently authorized). For a reopening trigger, **stop and reopen Condition 2** — this is the response Condition 2's own text implies, and it is a governance action, not a technical one.

**6.11 Evaluable at prediction time, or only after outcomes arrive?** **Mixed.** Mean predicted P(draw), Draw argmax share and Draw rank distribution are available **at prediction time**. Any comparison to *empirical* draw frequency — including the gap the ±0.03 heuristic operates on — is **post-outcome**. So the answer depends entirely on GAP-6a.

**6.12 Would the historical baseline require 2025/26 access?** **No** using the recorded folds-1–3 values. Recomputing Draw behaviour on 2025/26 **would** be new test-season computation and is **excluded**; recorded Phase 4C values may be read but not recomputed or used as a tuning input.

**6.13 Would implementation require modifying frozen V1 code?** **No** if additive. `calibration.py` must not be modified (see 3.13). Note `investigate_draw_probability` is importable and could be reused read-only, but it consumes a `combined` frame of V1 fold predictions — its input contract, not just its logic, would need consideration. **Not a proposal; noted for the approver.**

**6.14 Implement now or defer?** **PROPOSAL — NOT ADOPTED:** specification now (GAP-6a and GAP-6b are required regardless), implementation deferred until exposure for any post-outcome component.

---

## 4. Consolidated matrix (summary view)

| # | Signal | Detection exists | Emitted operationally | Threshold in spec | Enforced | Pre-outcome? | Needs 2025/26? | Touches frozen V1? | Now / defer (proposal) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Per-family null-rate drift | Per **column** only | No | **Ceiling only — no drift threshold** | Inside an uncalled function | Yes | No | No | Partial now |
| 2 | Unseen `competition_id` | Yes | No | **Yes** (set + flag rule) | Inside an uncalled function | Yes | No | No | Now |
| 3 | Class distribution vs base rates | Yes (offline) | No | **No** + reading ambiguity | No | Yes | No | No | Spec now |
| 4 | Mean prob vs realized frequency | Yes (offline) | No | **No** | No | **No** | No | No | Spec now, build later |
| 5 | Probability-validity failures | Yes | **Raises, never counted** | **Yes** (zero tolerance) | **Yes** | Yes | No | **Must not** | Now |
| 6 | Draw-class behaviour | Yes (offline) | No | **No** (heuristic unratified) | No | Mixed | No | No | Spec now |

**Signals emitted as operational monitoring with a defined threshold: 0 of 6.**
**Signals whose threshold is fully specified today: 2 of 6 (Signals 2 and 5).**
**Signals requiring a governance threshold decision: 4 of 6 (Signals 1-drift, 3, 4, 6).**
**Signals requiring no 2025/26 access under the recommended baselines: 6 of 6.**
**Signals requiring frozen-V1 modification: 0 of 6.**

The last two rows are the encouraging part of this analysis: **nothing in Gate 10 forces a constraint violation.** Every baseline needed can be sourced from already-recorded evidence, and every implementation can be additive. The block is purely a specification/governance block, not a technical one.

---

## 5. Previously open governance discrepancies

Both are restated here because Gate 10 inherits them. **Neither is resolved by this document.**

### 5.1 Gate 3 / §10 wording — documented "≤ ~7.7%" versus observed 0.077218

**The discrepancy.** §10 states (VERBATIM): *"Null rate per family within documented historical bounds (≤ ~7.7% observed historically; ~5.5% on 2025/26)."* The figure traces to the Phase 5 review coverage table, where fold_1's training partition shows 0.0772 for `shots_core` / `shots_on_core`. The exact observed value is **0.077218**. Read literally, a bound of "≤ 0.077" **excludes the very observation it was written to describe**, by 0.000218.

The implementation documents this rather than hiding it (`src/models/candidate_contract.py:102–113`, VERBATIM):

> *"A bound written to describe an observation cannot be violated by that same observation, but '<= 0.077' read literally would exclude it by 0.000218. The bound is therefore encoded at a precision that admits the documented observation, and the discrepancy is reported rather than papered over -- see the Gate 3 report."*

Encoded values: `MAX_HISTORICAL_NULL_RATE = 0.078`, `MAX_FINAL_SEASON_NULL_RATE = 0.056`.

**Why Gate 10 inherits it.** §12 requires **per-family** null-rate monitoring, and 0.077218 is a **family-level** observation. Any family-level monitor is measured against this same bound, so the wording ambiguity propagates directly into the monitoring threshold.

**Wording options — PROPOSAL — NOT ADOPTED. Approver selects one:**

- **(W-1) Amend §10 to state the exact observed value.** Replace *"≤ ~7.7%"* with *"≤ 0.077218 (exact maximum observed; fold_1 train, shots_core/shots_on_core)"*. **Pro:** exact, self-consistent, no rounding artifact. **Con:** a bound set exactly at the observed maximum has zero headroom, so any future value above the historical max breaches immediately — arguably correct for an "availability ceiling", but it must be a deliberate choice.
- **(W-2) Amend §10 to ratify the encoded value.** State the bound as **0.078**, describing it as a documented ceiling admitting the 0.077218 observation with explicit minimal headroom. **Pro:** matches the code exactly, so spec and implementation agree with no interpretation step. **Con:** 0.078 is a rounding-derived number, not an observation; the headroom (0.000782) has no empirical justification.
- **(W-3) Amend §10 to separate the two constructs explicitly.** State the *observation* as 0.077218 and the *enforced ceiling* as a separately-justified number, making clear the ceiling is a policy choice informed by — not equal to — the observation. **Pro:** eliminates the category error at the root, and aligns with SPECIAL RULE A's insistence on separating ceiling from drift. **Con:** requires the approver to justify the ceiling independently.
- **(W-4) Retain "~7.7%" and add a precision note to §10** pointing at the code comment. **Pro:** minimal edit. **Con:** leaves an approximate figure as the binding bound; "~" is not enforceable, and Gate 10 would inherit an unenforceable threshold.

**Constraint reminder:** whichever option is chosen, changing the encoded constants would alter `candidate_contract.py` and must be authorized separately. **No change is made here.**

### 5.2 Gate 6 / §8 tolerance — documented `atol=1e-6` versus effective ≈1.1e-5

**The discrepancy.** §8 states (VERBATIM): *"each row summing to 1 within `atol=1e-6`"*. The implementation (`src/models/evaluate.py:40–42`) is:

```
if not np.allclose(row_sums, 1.0, atol=1e-6):
```

`np.allclose` applies `|a − b| ≤ atol + rtol·|b|` with a **default `rtol=1e-5`**. With `b = 1.0`, the effective tolerance is `1e-6 + 1e-5·1.0 = 1.1e-5` — approximately **11× looser than §8 documents**. `np.isclose` in the adjacent diagnostic line carries the same default, so the two are at least mutually consistent.

**Why Gate 10 inherits it.** Signal 5 is *"probability-validity failures"*, and this is precisely the rule defining a validity failure. A monitoring counter counts breaches of whatever tolerance is actually in force. If §8 and the code disagree, the monitor's reported failure count is ambiguous — it measures 1.1e-5, while any report citing §8 would imply 1e-6.

**Options — PROPOSAL — NOT ADOPTED. Approver selects one:**

- **(T-1) Amend §8 to document the effective tolerance.** State the rule as `np.allclose(row_sums, 1.0, atol=1e-6)` with NumPy's default `rtol=1e-5`, effective ≈1.1e-5 at row sums near 1. **Pro:** documentation-only; **no code change**; `evaluate.py` stays byte-identical, preserving Gate 9 checksums and the §11 rollback guarantee; nothing downstream is revalidated. **Con:** the accepted tolerance becomes 11× looser than originally written — the approver must confirm 1.1e-5 is acceptable rather than merely convenient.
- **(T-2) Tighten the implementation to match §8** by passing `rtol=0`. **Pro:** the code would then mean exactly what §8 says. **Con — decisive and must be stated plainly:** `evaluate.py` is **one of the seven frozen V1 files pinned in `data/audit/phase4c_prerun_manifest.json`** and verified byte-identical under Gate 9. Modifying it would break the Gate 9 checksum verification, invalidate the §11 rollback guarantee's evidence base, and require re-verification of every gate that depends on it. **This document does not modify `evaluate.py` and does not recommend T-2.**
- **(T-3) Record the discrepancy as a known, accepted deviation** in the spec and in the Gate 6 record, without amending §8's stated intent or the code. **Pro:** no code change; full traceability; preserves the original documented intent as the aspiration. **Con:** leaves spec and implementation formally divergent, which a future reader may re-flag.
- **(T-4) Defer** until any tightening is independently required. **Pro:** no action, no risk. **Con:** Signal 5's threshold semantics remain ambiguous, so Gate 10 would close on this signal with a known ambiguity in its definition — acceptable only if recorded explicitly.

**Practical note for the approver:** the magnitudes involved are far from the boundary. Probability rows produced by a softmax/`predict_proba` path deviate from 1.0 at the level of floating-point residue (~1e-16), roughly ten orders of magnitude below either candidate tolerance. **The choice between 1e-6 and 1.1e-5 is therefore near-certainly immaterial in practice** — which is an argument for resolving it by documentation (T-1 or T-3) rather than by touching a frozen, pinned V1 file. That is an observation about risk, not a recommendation.

---

## 6. Why Gate 10 cannot be closed by writing code

Stated once, plainly, because it is the central conclusion.

Gate 10's pass condition is *"All §12 signals emitted with defined thresholds"*, and its stop condition is *"Any required signal absent."* Today **all six signals are absent as emitted monitoring signals** (F1, F3), and **four of six have no threshold defined anywhere in the specification** (Signals 1-drift, 3, 4, 6).

Four of the six thresholds are decisions only the approver can make. If this document chose them, a subsequent "Gate 10 PASS" would mean *"PASS against thresholds Claude invented"* — which is worthless as governance evidence and would silently convert an unresolved specification gap into an apparently-closed gate. That failure mode is exactly what the standing instructions exist to prevent.

**Status: BLOCKED — SPECIFICATION GAP.** Not PASS. Not FAIL — no implementation was ever authorized to exist, so there is nothing to have failed. Not INCONCLUSIVE — that label has been used in this project for *correct code that could not be executed in the sandbox*; here the **requirement itself is underdetermined**, which is a different and more fundamental condition.

**One structural question remains genuinely open and is not decided here.** Gate 10 requires monitoring *"before any production exposure"*, and no production exposure exists or is authorized (F2). Two readings follow, and they lead to different work:

- **Reading (i) — build now.** Treat Gate 10 as due, close the specification, and implement monitoring so it is ready in advance.
- **Reading (ii) — not yet due.** Treat Gate 10 as a documented open precondition attached to any future promotion decision, deferring implementation until exposure is actually proposed. Under this reading Gate 10 legitimately remains open, and that is not a defect.

Reading (ii) has a specific supporting argument worth weighing: Signal 4 **cannot produce a single valid measurement** before exposure (4.11), since it requires served predictions paired with subsequent realized outcomes. Under Reading (i), Gate 10 would close on Signal 4 based on implemented-but-never-exercised code — weaker evidence than every other gate in this project has been held to. **This document does not choose between the readings.** It is decision **AUTH-A** below.

---

## Gate 10 Decision Checklist

Decisions the human must explicitly approve before any implementation can begin. **Nothing below is preselected. Silence is not approval.**

### REQUIRED TO CLOSE THE SPECIFICATION

These block Gate 10 regardless of timing. Without them the gate has no testable pass criterion.

| ID | Decision | Signal | Options |
|---|---|---|---|
| **D-01** | Choose the Gate 10 reading: build monitoring now, or record it as a documented open precondition on promotion | All | Reading (i) / Reading (ii) — see §6; = AUTH-A |
| **D-02** | **Confirm the ceiling/drift separation.** Either define a drift baseline + metric + tolerance, **or** explicitly record that the §10 absolute ceiling alone satisfies Gate 10's null-rate signal with drift-proper deferred | 1 | GAP-1a–1d; **must be an explicit choice, not an equation of the two** (SPECIAL RULE A) |
| **D-03** | **Resolve the class-distribution ambiguity: argmax class share vs mean predicted probability mass** | 3 | GAP-3a — highest-value single decision; the two readings differ ~31× for Draw and determine whether the monitor is usable at all |
| **D-04** | Set the class-distribution base-rate reference and tolerance | 3 | GAP-3b, 3c; baseline options C-1…C-4 |
| **D-05** | Set the calibration tolerance for mean-probability vs realized-frequency | 4 | GAP-4a |
| **D-06** | Set the minimum sample size for the calibration signal | 4 | GAP-4b — without this, D-05 is not implementable |
| **D-07** | Define the label-availability delay and handling of late/voided/abandoned fixtures | 4 | GAP-4c |
| **D-08** | Define the comparison window for the calibration signal | 4 | GAP-4d |
| **D-09** | Define the calibration baseline: same-window realized outcomes, or a fixed historical base rate | 4 | GAP-4e |
| **D-10** | **Fix which quantity constitutes "draw-class behaviour"** | 6 | GAP-6a — determines whether the monitor is quiet and informative or fires permanently against accepted behaviour |
| **D-11** | **Ratify, reject, or replace the ±0.03 Phase 3 draw heuristic.** It was authored as descriptive and explicitly non-statistical and therefore **requires explicit ratification before operational use** | 6 | GAP-6b (SPECIAL RULE F) |
| **D-12** | Decide whether Draw is monitored in addition to, or instead of, its treatment inside Signals 3 and 4 (avoids triple-counting one event) | 3, 4, 6 | GAP-6d |
| **D-13** | Confirm the family→columns mapping and its authority | 1 | GAP-1e; only counts exist today |
| **D-14** | Decide whether family-level monitoring replaces or supplements the per-column check (a single all-null column is diluted ~18× at family level) | 1 | GAP-1f |
| **D-15** | **Resolve the §10 wording discrepancy** ("≤ ~7.7%" vs observed 0.077218) | 1, Gate 3 | W-1 / W-2 / W-3 / W-4 — §5.1 |
| **D-16** | **Resolve the §8 tolerance discrepancy** (`atol=1e-6` vs effective ≈1.1e-5) **without modifying `evaluate.py`** | 5, Gate 6 | T-1 / T-2 / T-3 / T-4 — §5.2; T-2 would break Gate 9 checksums and is not recommended |
| **D-17** | Define "production exposure" — the gate's own trigger condition, currently undefined anywhere | All | Prerequisite for D-01 to be meaningful |

### OPTIONAL / CAN BE DEFERRED UNTIL PRODUCTION EXPOSURE EXISTS

Needed for an operating monitoring system, but not required to make Gate 10's pass criterion testable.

| ID | Decision | Signal |
|---|---|---|
| **D-18** | Cadence and measurement window for the null-rate signal | 1 |
| **D-19** | Minimum batch size below which null-rate alerting is suppressed | 1 |
| **D-20** | Alert severity and response action for null-rate ceiling vs drift breaches (including whether a ceiling breach blocks inference) | 1 |
| **D-21** | Emission target, cadence, severity and response for unseen `competition_id` — **the only four items outstanding for Signal 2** | 2 |
| **D-22** | Confirm the intended asymmetry whereby a **null** `competition_id` fails but an **unseen** one only flags | 2 |
| **D-23** | Whether unseen-league predictions are served, served-with-caveat, suppressed, or escalated | 2 |
| **D-24** | Cadence, severity and response for the class-distribution signal | 3 |
| **D-25** | Whether the class base rate is static or periodically re-estimated | 3 |
| **D-26** | Whether Signal 4 uses the simple mean-vs-frequency gap or binned calibration (ECE / reliability bins) | 4 |
| **D-27** | Severity, escalation rule and response for calibration breaches | 4 |
| **D-28** | Aggregation/reporting window for validity-failure counts | 5 |
| **D-29** | Whether validity **near-misses** are recorded as a leading indicator | 5 |
| **D-30** | Severity and escalation for validity failures beyond §8's existing hard stop | 5 |
| **D-31** | Whether Signal 6 monitors probability quality, Condition-2 reopening triggers, or both | 6 |
| **D-32** | Whether Condition 2's *"must never be presented as a reliable Draw-class predictor"* clause implies a presentation-compliance control, and which gate owns it | 6 |
| **D-33** | Monitoring cadence, ownership, on-call routing and alert channel — none defined anywhere | All |
| **D-34** | Which monitoring signals, if any, are §11 rollback triggers | All |

### IMPLEMENTATION AUTHORIZATIONS REQUIRED AFTER SPEC IS LOCKED

Requested **only** after the REQUIRED section is approved. Listed for completeness; **none is requested now.**

| ID | Authorization | Constraint carried |
|---|---|---|
| **AUTH-A** | Confirm the Gate 10 reading (= D-01) | Governs whether any of the below is needed |
| **AUTH-H** | Authorize a new, additive monitoring module | Must not modify the 7 pinned V1 files (`config.py`, `train.py`, `splits.py`, `evaluate.py`, `data.py`, `baselines.py`, `run_experiments.py`) or `calibration.py`'s frozen behaviour |
| **AUTH-I** | Decide whether `check_feature_availability` may be wired into `predict_candidate`, and whether a validity-counting wrapper may sit on that path | Changes existing inference behaviour in `candidate_contract.py` — will not be assumed. §8's hard stop must be preserved exactly: record, then re-raise unchanged |
| **AUTH-J** | Decide whether a prediction/outcome store may be created | Signals 3/4/6 need served predictions paired with realized labels; **must not use 2025/26 for new computation**, so no historical monitoring baseline may be computed from the test season |
| **AUTH-K** | Authorize the family→columns mapping's location (derive in `candidate_contract.py`, or state in the spec) | `candidate_contract.py` is not pinned but is contract-bearing |
| **AUTH-L** | Authorize the §10 and §8 wording amendments selected in D-15 / D-16 | Documentation-only if W-2/W-4 and T-1/T-3 are chosen; **W-1/W-3 may require changing encoded constants, and T-2 would modify a pinned V1 file** |
| **AUTH-M** | Authorize any new historical computation required by the chosen baseline (e.g. C-2 final-train label frequencies) | Historical partitions only; **2025/26 excluded** |
| **AUTH-N** | Authorize Gate 10 tests once the specification is locked | No tests are written by this document |

---

## 7. Status at original issue (superseded — retained for the record)

> ## GATE 10 — BLOCKED / SPECIFICATION GAP  *(superseded by §8; see the status note at the top of this document)*

**Not PASS. Not FAIL. Not INCONCLUSIVE.** The specification does not define thresholds for four of the six required §12 signals, and no signal is emitted as an operational monitoring signal. The block is a governance/specification block, not a technical one: every required baseline is available from already-recorded evidence, no signal requires 2025/26 access, and no signal requires modifying frozen V1 code.

**Compliance statement for this task.** Documentation and decision analysis only. No threshold invented, adopted, or defaulted to. No production code modified. No frozen V1 source or pinned artifact modified. No model trained, evaluated, tuned, or calibrated. No V2 artifact created. No 2025/26 access for any new computation. No monitoring infrastructure created. No tests written. No data artifact created. Gate 10 not declared PASS. `MODEL_VERSION` remains `v1.0`.

**Next authorized action (at original issue):** human review of this document and explicit decisions on the **REQUIRED TO CLOSE THE SPECIFICATION** checklist. No implementation until then. **This has now occurred — see §8.**

---

# 8. Locked Decision Record

**Authority:** explicit human authorization, 2026-08-17. Each entry states the decision, the evidence relied on, and what it deliberately does **not** decide. Every numeric threshold left open is listed in §9 and was **not** invented here.

## 8.1 D-01 — Gate 10 timing / production exposure — **LOCKED**

**Decision.** Gate 10 monitoring implementation is a **mandatory pre-promotion requirement**, but operational monitoring **does not need to be exercised** before a production exposure exists.

**Recorded explicitly:**

1. **No production exposure currently exists.**
2. **Gate 10 remains OPEN** as a promotion precondition.
3. Monitoring **specification may be finalized now** — and is, as of this revision.
4. Monitoring **implementation may be authorized now only if explicitly requested later.** It is not requested and not authorized.
5. **Gate 10 cannot be marked PASS** until the required monitoring implementation exists and its acceptance checks pass.
6. **Any future production exposure requires Gate 10 PASS first.**

**Explicitly NOT claimed:** that "no exposure" means Gate 10 is automatically satisfied. It does not. Gate 10 is **open, not vacuous**, and the absence of exposure must never be recorded as satisfying it.

**Evidence.** Gate 10 row (`PHASE5_V2_MODEL_SPEC_DRAFT.md:127`) — *"§12 monitoring in place before any production exposure"*; §2 finding F2 (only inference path is `predict_candidate`, called from tests; no serving entrypoint, no scheduler, no prediction store); §3 Signal 4 field 4.11 (post-outcome signals cannot be exercised pre-exposure).

**Resolves:** checklist D-01 and AUTH-A. **Partially resolves D-17** — the *governance relationship* to exposure is now fixed; the *operational definition* of "production exposure" remains open (§9.6).

## 8.2 D-03 — Predicted class distribution — **LOCKED: mean predicted probability mass per class**

**Decision.** The primary distribution-monitoring quantity is **mean predicted probability mass per class**. **Argmax class share is NOT the primary signal** (it may be retained as a secondary diagnostic).

**Evidence** (recorded, folds 1–3 validation, n=5331, `data/audit/phase3_probability_diagnostics.json`):

| Class | Empirical base rate | Mean predicted probability mass | Argmax share |
|---|---|---|---|
| H | 0.43632 | 0.42330 | 0.63797 |
| D | **0.25192** | **0.25382** | **0.00825** |
| A | 0.31176 | 0.32288 | 0.35378 |

Draw deviation from base rate: **+0.00190** under probability mass; **−0.24367** under argmax share — the two interpretations produce materially different conclusions (~31× apart for Draw). Phase 5B accepted the Draw limitation **under the probability-output contract** (Condition 2, `DRAW_LIMITATION_ACCEPTED`, scoped). Mean probability mass is therefore the coherent quantity for the current output contract; an argmax monitor would fire permanently against known, documented, accepted behaviour.

**Explicitly NOT claimed.** This decision does **not** claim the candidate is a good Draw-class classifier. **It is not.** Draw recall **0.018**, Draw F1 **0.034**, argmax Draw share **0.00825**, Draw ranked 2nd of 3 in **0.59032** of fixtures. The severe argmax Draw limitation **remains explicitly documented** in §12.3, Condition 2, and `docs/PHASE5B_DRAW_LIMITATION_REVIEW.md`, and the candidate **must never be presented as a reliable Draw-class predictor**.

**Threshold.** The numeric tolerance is **NOT decided here and was not invented.** Recorded as a separate implementation authorization — §9.2.

**Resolves:** D-03. **Leaves open:** D-04 (tolerance and formal baseline-row adoption).

## 8.3 Signal 1 — null-rate drift — **CONCEPT LOCKED, NUMERIC OPEN**

**Decision.** Two constructs are kept **strictly separate** and neither is redefined as the other:

1. **Absolute availability ceiling (§10).** Retained **unchanged**: 0.078 historical / 0.056 final-season. This is a ceiling, not a drift threshold.
2. **Drift from historical baseline (§12 Signal 1).** The Gate 10 signal is defined as **deviation from the documented historical baseline**, per family.

**Explicitly NOT done:** the 0.078 / 0.056 ceilings are **NOT** redefined as drift thresholds.

**Baseline.** Source **LOCKED** as the recorded per-family, per-partition table in `docs/PHASE5_V2_CANDIDATE_REVIEW.md` Part 4 (lines 129–137) — a read of existing recorded evidence requiring **no new computation**. **Which non-test row is the baseline remains OPEN** (§9.1). The **2025/26 row is excluded** as an operational reference under the standing no-test-season-as-input constraint — a narrowing that follows from existing constraints, not a new choice.

**Threshold.** Drift metric and tolerance are **explicitly OPEN and were not invented** — §9.1.

**Evidence.** §10 (as revised, see §8.7); Part 4 coverage table; the worked `form` case (0.0074 → 0.070 = ~9.5× increase, still under ceiling, ceiling reports `passed=True`) demonstrating the ceiling structurally cannot detect drift.

**Resolves:** D-02 (the separation), the baseline *source*. **Leaves open:** baseline row, metric, tolerance (§9.1); D-13/D-14 (family mapping and granularity — §9.5, §9.7).

## 8.4 Signal 2 — unseen `competition_id` — **LOCKED**

**Decision.** Known IDs: **{200, 419, 423, 477, 499}**. Any unseen ID:

- **MUST** be emitted as a monitoring event.
- **MUST** be flagged.
- **MUST NOT** automatically fail inference under the current **degraded-not-failed** contract.

Predictive quality for unseen leagues **remains unestablished by current evidence**. **Severity, response and cadence are DEFERRED** (not authorized — §9.6). **No additional league behaviour is defined.**

**Evidence.** §10 (verbatim: *"an unseen league must be flagged (behaviour is degraded-not-failed, and predictive quality for it is **not established by current evidence**)"*); `KNOWN_COMPETITION_IDS` (`src/models/candidate_contract.py:122`); existing detection returning `unseen_ids` / `unseen_flagged` without setting `passed=False`.

**No numeric threshold required, requested, or invented** — detection is exact set membership.

**Resolves:** the Signal 2 rule (matrix GAP-2 detection semantics). **Leaves open:** D-21/D-22/D-23 (emission target, null-vs-unseen asymmetry, serve/suppress response).

## 8.5 Signal 4 — probability vs realized outcome — **TIMING LOCKED, ALL NUMERICS OPEN**

**Decision.** Locked as a **POST-OUTCOME** monitoring signal. The specification states explicitly that it **cannot be fully evaluated at prediction time because realized outcomes do not yet exist** at prediction time.

**Unresolved governance fields, recorded separately and NOT invented:** minimum sample size; observation window; tolerance; baseline/reference period; alert severity; response — §9.3, §9.4, §9.6.

**Evidence.** §12 wording (*"vs. realized outcome frequency"*); §2 finding F5; §3 field 4.11.

**Resolves:** the timing classification. **Leaves open:** D-05 through D-09, D-26, D-27.

## 8.6 Signal 5 — probability validity — **LOCKED**

**Decision.**

- `evaluate.validate_probabilities()` **remains authoritative**; semantics unchanged.
- **Zero tolerance** remains the validity rule; any invalid probability vector is a **hard failure**.
- **`src/models/evaluate.py` is NOT modified** and must not be.
- Future monitoring implementation **must count/emit failures externally and additively**.
- The monitoring counter **must not alter validation semantics** — it records, then **re-raises unchanged**, preserving §8's hard stop.

**Evidence.** §8 (verbatim: *"A failure is a hard stop, not a warning."*); Gate 6 row (*"100% pass"*); `evaluate.py` is one of the seven frozen V1 files pinned in `data/audit/phase4c_prerun_manifest.json`, verified byte-identical under Gate 9; `predict_candidate` already calls the validator.

**Gap that remains a gap:** emission. Failures currently raise and leave **no trace**, so "how many validity failures occurred last week?" is unanswerable. This is an implementation item (AUTH-H / AUTH-I), not a threshold item — **no numeric decision is outstanding for Signal 5.**

**Resolves:** the Signal 5 rule and its implementation constraints. **Leaves open:** D-28/D-29/D-30 (reporting window, near-misses, escalation).

## 8.7 Signal 6 — draw behaviour — **±0.03 EXPLICITLY REJECTED; CRITERION OPEN**

**Decision.** The existing ±0.03 heuristic is **NOT adopted** as a monitoring threshold.

**Recorded reasons:**

- Phase 3's ±0.03 band is **descriptive only**.
- It was **explicitly documented as not being a statistical test** — its returned `caveat`, verbatim: *"This is a descriptive top-line heuristic, not a statistical test, and does not alter the model regardless of its verdict."*
- Phase 5B states, verbatim: *"**No epsilon threshold exists** and none was invented."*
- **Therefore it is NOT an approved production threshold.**

**Draw monitoring remains REQUIRED by §12**, and its exact criterion is an **OPEN governance item** — the matrix contains **no authorized alternative**, so nothing is adopted in its place (§9.5). Any criterion later adopted must be consistent with the §8.2 probability-mass lock. Also open: whether Draw is monitored in addition to, or instead of, its treatment in Signals 3 and 4 (D-12).

**Evidence.** `src/models/calibration.py:274–281` docstring and returned `caveat`; `docs/PHASE5B_DRAW_LIMITATION_REVIEW.md:89`; Condition 2.

## 8.8 Gate 3 precision wording — **RESOLVED (documentation only)**

**Decision.** §10's null-rate bullet is reworded so it **no longer implies an exact 0.0770 ceiling**, using the authorized wording:

> **"Historical maximum observed null rate: 0.077218. Operational availability bound: 0.078."**

This corresponds to matrix option **W-3** (separating the *observation* from the *enforced ceiling*). The revised §10 additionally records the final-season figures (0.0548 observed on 2025/26; 0.056 bound) and states that the bound is an **absolute availability ceiling, not a drift threshold**.

**Explicitly NOT done:** no empirical data changed; **no implementation modified**. `MAX_HISTORICAL_NULL_RATE = 0.078` and `MAX_FINAL_SEASON_NULL_RATE = 0.056` in `src/models/candidate_contract.py` are **unchanged** — the revised wording now matches those encoded constants, so no code change is required or authorized by this task.

**Evidence.** Observed 0.077218 (fold_1 train, `shots_core`/`shots_on_core`), traced in `src/models/candidate_contract.py:102–113`; Part 4 coverage table (fold_1 0.0772; final test 0.0548).

**Resolves:** D-15.

## 8.9 Gate 6 tolerance wording — **RESOLVED AS DISCLOSURE; TIGHTENING REMAINS OPEN**

**Decision.** Applied as a **wording-only resolution** (matrix options **T-1 + T-3** combined): §8 now discloses both the **intended/documented** tolerance (`atol=1e-6`) and the **actual behaviour** of the current implementation (`np.allclose(..., atol=1e-6)` with NumPy's default `rtol=1e-5`, effective ≈**1.1e-5** at row sums near 1, ~11× looser), and records it as a known, disclosed deviation.

**Explicitly NOT done:** **`evaluate.py` is NOT modified.** The current V1 implementation and its checksum (`4e9d9313867d47a19001a383a301c2fe`) are **preserved**, so Gate 9's checksum verification and the §11 rollback evidence base are intact. Matrix option **T-2** (passing `rtol=0`) is **not taken** — it would modify a pinned frozen V1 file.

**Remains OPEN.** Whether to tighten the implementation to match the documented figure is recorded as an **OPEN technical-governance item requiring explicit authorization** (§9.6). Practical note carried into §8: real row sums deviate at ~1e-16, roughly ten orders of magnitude below either candidate tolerance, so the distinction is near-certainly immaterial in practice — an observation, not a reason to leave it formally unresolved.

**Resolves:** D-16 as a disclosure. **Leaves open:** the tightening decision.

---

# 9. Remaining Numeric Decisions

**Only genuinely unresolved items appear below. No values are manufactured, proposed-as-default, or implied.** Each entry states why it is needed, what it controls, what evidence exists, and what authorization is required.

## 9.1 Null-rate drift tolerance (and drift metric, and baseline row)

- **Why needed.** Signal 1 is now defined as *deviation from baseline*, but "deviation" is not measurable without a metric and a bound. Without them the signal cannot fire, so Gate 10's pass condition cannot be met for Signal 1.
- **Controls.** Signal 1 (per-family null-rate drift). Does **not** affect the §10 absolute ceiling, which is already defined and separate.
- **Existing evidence.** The recorded per-family, per-partition table (`PHASE5_V2_CANDIDATE_REVIEW.md` Part 4): non-test baseline candidates are fold_1/2/3 train and final train (2020/21–2024/25); e.g. `goals_core` 0.0764 / 0.0555 / 0.0542 / 0.0543, `form` 0.0764 / 0.0555 / 0.0444 / 0.0374, `strength` 0.0027 / 0.0027 / 0.0028 / 0.0028, `competition_id` 0.0000 throughout, `shots_core` 0.0772 / 0.0560 / 0.0550 / 0.0551. **No drift tolerance has ever been computed, proposed or recorded anywhere in the project.**
- **Authorization required.** Three linked decisions: (a) which non-test row is the baseline (final train, or per-family max across fold-train rows); (b) the metric (absolute percentage-point delta, relative/ratio change, or a distributional distance) — this matters most for `strength`, whose ~0.0028 baseline makes a +0.01 absolute move trivial in points but ~4.6× in relative terms; (c) the numeric tolerance. A minimum batch size (§9.3) must be set alongside, since a single missing value in a 6-fixture batch is 16.7%.

## 9.2 Class-distribution tolerance

- **Why needed.** The *quantity* is locked (mean predicted probability mass, §8.2) but the allowed deviation from base rate is not. Without it Signal 3 cannot fire.
- **Controls.** Signal 3.
- **Existing evidence.** Recorded folds-1–3 per-class deviations under the locked quantity: H **−0.01302**, D **+0.00190**, A **+0.01112** (largest magnitude 0.01302). **These are observations, not a proposed threshold** — a bound written to describe an observation cannot be violated by that observation, the same trap documented at `src/models/candidate_contract.py:102–113`.
- **Authorization required.** The numeric tolerance; whether it is per-class or aggregate; whether the base rate is static or periodically re-estimated (D-25); and formal adoption of the folds-1–3 row as the reference versus a final-train recomputation (the latter would additionally need AUTH-M, since it is a new historical computation — 2025/26 excluded either way).

## 9.3 Minimum sample size

- **Why needed.** Both Signal 1 and Signal 4 are rates over a window; below some window size they are dominated by sampling noise and produce false alarms **by construction**. Any tolerance set without a matching minimum-n is not implementable.
- **Controls.** Signals 1, 3 and 4 (and Signal 6 if an argmax-based quantity were ever chosen).
- **Existing evidence.** Structural, not empirical: over 10 fixtures a realized Draw frequency can only take multiples of 0.1, so it cannot land within 0.02 of 0.25192 **even under perfect calibration**. At the recorded ~0.8% argmax Draw rate, a 100-fixture window has an expected Draw-argmax count below 1, so a count of zero is unremarkable. **No minimum-n has ever been specified.**
- **Authorization required.** A minimum fixture count per signal, plus the suppression rule (record-but-do-not-alert below the minimum).

## 9.4 Probability-vs-realized-frequency tolerance (and window, baseline period)

- **Why needed.** Signal 4's timing is locked (post-outcome) but every numeric field is open, so the signal cannot be evaluated at all.
- **Controls.** Signal 4.
- **Existing evidence.** Recorded folds-1–3 mean predicted vs realized: H 0.42330 vs 0.43632; D 0.25382 vs 0.25192; A 0.32288 vs 0.31176. Offline machinery exists (`compute_ece`, `compute_reliability_bins`) but no tolerance or ECE limit has ever been set.
- **Authorization required.** Tolerance; observation window; baseline/reference period (same-window realized outcomes versus a fixed historical base rate — the latter would duplicate Signal 3); whether the measure is the simple mean-vs-frequency gap or a binned calibration measure (D-26); and the label-availability delay with handling of late, voided and abandoned fixtures (D-07).

## 9.5 Draw monitoring threshold (and criterion)

- **Why needed.** §12 requires Draw monitoring specifically. The ±0.03 heuristic is **rejected** (§8.7) and **no authorized alternative exists**, so the signal currently has neither a quantity nor a bound.
- **Controls.** Signal 6.
- **Existing evidence.** Recorded folds-1–3 Draw figures: empirical frequency 0.25192; mean predicted P(draw) 0.25382; gap **+0.00190**; argmax Draw share **0.00825**; Draw ranked 2nd of 3 in **0.59032**; Draw recall **0.018**, F1 **0.034**. The ±0.03 band exists in code but is **explicitly descriptive and non-statistical** and is **not** approved.
- **Authorization required.** First the criterion/quantity (D-10) — which must be consistent with the §8.2 probability-mass lock — then either explicit **ratification** of ±0.03 for operational use with the reinterpretation recorded in writing, or an **independent** threshold, or explicit **deferral**. Also required: whether Draw is monitored in addition to or instead of its treatment in Signals 3 and 4 (D-12), to avoid three correlated alerts from one event.

## 9.6 Non-numeric items still open (recorded here for completeness — no values involved)

These are **not** thresholds, so they are listed separately and do not belong to the numeric set above:

| Item | Signal | Note |
|---|---|---|
| Operational definition of "production exposure" | All | D-17 partially resolved by §8.1; the operational definition must be fixed **before promotion**, not before spec lock |
| Family→columns mapping and its authority | 1 | D-13 / AUTH-K. Only family **counts** exist today (`CANDIDATE_FAMILY_COUNTS`, whose docstring states it is *"used only for verification reporting"*). Required before implementation, not before spec lock |
| Family-level vs per-column granularity | 1 | D-14. A single all-null column is diluted ~18× at family level |
| Emission target, cadence, severity, response | 1, 2, 3, 4, 5, 6 | D-18/D-20/D-21/D-23/D-24/D-27/D-30/D-33. Explicitly **deferred** for Signal 2 per §8.4 |
| Null-vs-unseen `competition_id` asymmetry | 2 | D-22. Existing code fails on null but flags on unseen; confirm intended |
| Validity-failure reporting window; near-miss recording | 5 | D-28 / D-29 |
| Whether any monitoring signal is a §11 rollback trigger | All | D-34 |
| Whether to tighten `np.allclose` to match documented `atol=1e-6` | 5 | §8.9. **`evaluate.py` must not be modified**; T-2 would break Gate 9 checksums |
| Condition 2 presentation-compliance control ownership | 6 | D-32. *"must never be presented as a reliable Draw-class predictor"* — currently unowned by any gate |

## 9.7 Summary of numeric status

| Signal | Numeric threshold status |
|---|---|
| 1 — Per-family null-rate drift | **OPEN** (§9.1) — ceiling separately defined and unchanged |
| 2 — Unseen `competition_id` | **NONE REQUIRED** — categorical rule, fully locked |
| 3 — Class distribution | **OPEN** (§9.2) — quantity locked, tolerance open |
| 4 — Mean probability vs realized frequency | **OPEN** (§9.4) — timing locked, all numerics open |
| 5 — Probability-validity failures | **NONE REQUIRED** — zero tolerance, fully locked |
| 6 — Draw-class behaviour | **OPEN** (§9.5) — ±0.03 rejected, no alternative authorized |

**Numeric decisions genuinely outstanding: 4** (Signals 1, 3, 4, 6), plus the cross-cutting minimum sample size (§9.3). **Numeric decisions manufactured in this document: 0.**

---

# 10. Final status (current)

> ## GATE 10 — SPECIFICATION LOCKED / IMPLEMENTATION NOT STARTED

**GATE 10 SPECIFICATION: DECISIONS LOCKED** — every governance decision enumerated for this task is resolved and recorded in §8 and mirrored in `PHASE5_V2_MODEL_SPEC_DRAFT.md` §12.

**GATE 10 IMPLEMENTATION: NOT AUTHORIZED / NOT STARTED** — no monitoring module, emitter, sink, counter, or alert exists.

**GATE 10 ACCEPTANCE: OPEN** — the pass condition (*"All §12 signals emitted with defined thresholds"*) is **not met**: 0 of 6 signals are emitted, and four numeric thresholds remain open (§9).

**Gate 10 is NOT PASS**, and the absence of production exposure does not satisfy it (§8.1). **Any future production exposure requires Gate 10 PASS first.**

**Compliance statement for the decision-locking task.** Documentation and specification changes only. No production code changed. No tests written. No monitoring implemented. No model trained, evaluated, tuned, or calibrated. No V2 model or artifact created. No deployment, no production exposure. 2025/26 not accessed or recomputed. No frozen V1 source modified; `evaluate.py` untouched. No Phase 3/4C/5A artifact modified. No threshold, baseline, or production behaviour invented. No existing heuristic promoted to a production threshold. `MODEL_VERSION` remains `v1.0`.

**Next authorized action:** explicit human decisions on §9.1–§9.5 (the four open numeric thresholds plus minimum sample size), after which implementation authorization (AUTH-H, AUTH-I, AUTH-J, AUTH-K, AUTH-N) may be requested.

---

# 11. Consistency Audit Record

**Scope.** Cross-comparison of: this matrix (§§1–7 retained analysis, §8 Locked Decision Record, §9 Remaining Numeric Decisions, §10 status), `PHASE5_V2_MODEL_SPEC_DRAFT.md` §12 (all subsections), the Acceptance Gate 10 row, and the revised §8 / §10 of the spec. Read-only; documented, not resolved.

## 11.1 Checks performed and results

| Check | Result |
|---|---|
| A decision LOCKED in one place but OPEN elsewhere | **PASS** — §12.7 and §9.7 agree signal-by-signal. Thresholds open for 1, 3, 4, 6; locked for 2 and 5. |
| A threshold described as defined in one place but unresolved elsewhere | **PASS** — no location asserts a numeric tolerance for Signals 1, 3, 4 or 6. |
| Signal described as prediction-time in one document, post-outcome in another | **FAIL — see INC-1 (Signal 3).** Signals 1, 2, 4, 5, 6 consistent. |
| Baseline treated as authoritative without explicit authorization | **PASS** — Signal 1 and Signal 3 baselines both carry **PARTIALLY LOCKED** with the row/adoption explicitly OPEN in both documents. |
| "Implementation not authorized" contradicted by approval-implying language | **PASS** — no instance of implementation being described as authorized, approved, started or complete. |
| D-01 exposure timing inconsistent | **PASS** — §12.0, §8.1 and the gate-table status note state the same six points, including that absence of exposure does not satisfy Gate 10. |
| Signal 6 accidentally adopting ±0.03 | **PASS** — every mention is a rejection, a description of it as descriptive/non-statistical, or an option requiring ratification. No adoption anywhere. |
| Signal 3 reverting to argmax share | **PASS** — argmax appears only as the rejected primary, a permitted secondary diagnostic, or recorded evidence. |
| Signal 5 hard-stop semantics altered | **PASS** — record-then-re-raise-unchanged stated in both documents; zero tolerance intact; `evaluate.py` untouched. |
| 2025/26 becoming a baseline or tuning input | **PASS** — explicitly excluded as a Signal 1 operational reference; see OBS-2 for an apparent-only tension. |
| Gate 10 row altered | **PASS** — byte-identical; status recorded in a separate note beneath the table. |
| Gate 10 claimed PASS anywhere | **PASS** — no such claim; all occurrences are "NOT PASS" or "requires Gate 10 PASS first". |

## 11.2 INC-1 — Signal 3 timing: unconditional in §12, conditional in retained analysis (DOCUMENTED, NOT RESOLVED)

**The inconsistency.** The locked specification states Signal 3's timing unconditionally:

- `PHASE5_V2_MODEL_SPEC_DRAFT.md` §12.3 timing: *"**Prediction-time** for the predicted side; the base-rate side is a stored reference, so the comparison is available at prediction time."* §12.7 records **Prediction-time**.

The retained pre-decision analysis states it conditionally, and in one place as partly post-hoc:

- §2 finding **F5**: *"Signals 3 (partly), 4 and 6 require realized outcomes. They cannot be evaluated at prediction time under any threshold choice…"*
- §3 field **3.11**: *"**Mixed**… available at prediction time under C-1/C-2/C-3. Only under C-4 (rolling live realized frequencies) does the signal become post-hoc."*

**Why it arises.** Field 3.11 is correct as written: Signal 3's timing was *conditional on the baseline choice*. The §8.2 lock chose the probability-mass quantity and §12.3 names only stored-reference baselines (recorded folds-1–3 row, or a final-train recomputation), which makes prediction-time timing correct — **but option C-4 (rolling live realized frequencies) was never explicitly struck out.** F5's parenthetical "(partly)" is inherited from a world where C-4 was live.

**Status: DOCUMENTED, NOT RESOLVED.** The blanket superseding note at the top of this document covers §§1–7 generically, but a reader consulting F5 or 3.11 directly would reach a different timing conclusion than §12.7. Resolving it requires one human confirmation — **that baseline option C-4 is excluded for Signal 3** — after which F5 and 3.11 can be annotated. **No text is changed here, and no exclusion is assumed.** Practical impact: low (both remaining baseline options are stored references, so prediction-time holds either way), but it is a live documentation contradiction and is recorded as such rather than tidied away.

## 11.3 Precision observations (not contradictions)

- **OBS-1 — the 0.078 ceiling still lacks an independent justification.** §8.8 records the Gate 3 wording as matrix option **W-3**, which separates the *observation* (0.077218) from the *enforced ceiling* (0.078). W-3's stated cost was that the ceiling then *"requires the approver to justify the ceiling independently"*. The applied wording supplies the separation but **not** that independent justification — 0.078 remains a rounding-derived number with 0.000782 of unexplained headroom. Not a contradiction (the value matches the encoded constant and predates this task), but the justification is outstanding. Listed in §12.2 below as a deferrable item.
- **OBS-2 — apparent-only tension between §10's 0.056 and Signal 1's exclusion of 2025/26.** §10's final-season availability bound (0.056) derives from a 2025/26 observation (0.0548), while §12.1 excludes the 2025/26 row as a **drift baseline**. This reads as a contradiction only if ceiling and drift are conflated — which both documents explicitly forbid. The §10 ceiling is a pre-existing absolute availability bound; the Signal 1 exclusion governs the *drift reference*. **Already resolved by existing text; no edit needed.** Flagged only because a future reader may re-raise it.
- **OBS-3 — "emission" means two different things.** §12.2 locks that an unseen ID **MUST be emitted** (a requirement), while §9.6 lists "emission target, cadence, severity, response" as open (a destination). Both are correct and non-contradictory, but the shared word invites misreading. No edit made.
- **OBS-4 — §12.6's consistency constraint is directional, not determinative.** §12.6 requires any Draw criterion to be *"consistent with the §12.3 lock on probability-mass framing"*. This constrains the choice without making it; Signal 6's quantity remains genuinely OPEN. Confirmed not to function as a covert adoption.

**Audit conclusion.** The locked decisions are internally consistent across both documents. One documentation-level timing inconsistency (INC-1) is recorded for human resolution; no substantive contradiction affects any locked decision, threshold, baseline or semantic guarantee.

---

# 12. Implementation Authorization Readiness

**Nothing in this section authorizes implementation.** It states only what *could* be built without guessing, if implementation were later authorized.

## 12.1 Remaining numeric decisions (compact)

| ID | Signal | Decision required | Current evidence | Why it matters | Status |
|---|---|---|---|---|---|
| **N-1a** | 1 | Drift **metric** (absolute pp delta / relative ratio / distributional distance) | Per-family baselines recorded (`PHASE5_V2_CANDIDATE_REVIEW.md` Part 4); `strength` baseline ≈0.0028 vs `shots_core` ≈0.055 | Ranks breaches differently by orders of magnitude; a +0.01 move is trivial in points for `strength` but ≈4.6× relative | **OPEN** |
| **N-1b** | 1 | Baseline **row** (final train, or per-family max across fold-train rows) | fold_1/2/3 train + final train rows recorded; 2025/26 excluded | Different rows differ materially (`form` 0.0764 vs 0.0374) so the same batch drifts differently | **OPEN** |
| **N-1c** | 1 | Drift **tolerance** | **None ever computed or recorded in the project** | Without it Signal 1 cannot fire | **OPEN** |
| **N-3a** | 3 | Baseline **reference** (recorded folds-1–3 row vs final-train recomputation) | H 0.43632 / D 0.25192 / A 0.31176, n=5331 | Recomputation needs AUTH-M; also bears on INC-1 | **OPEN** |
| **N-3b** | 3 | **Tolerance** | Recorded deviations H −0.01302 / D +0.00190 / A +0.01112 (observations, **not** proposals) | Without it Signal 3 cannot fire | **OPEN** |
| **N-3c** | 3 | **Per-class vs aggregate** comparison | Per-class deviations differ ~7× in magnitude (0.00190 vs 0.01302) | An aggregate can mask a single-class breach | **OPEN** |
| **N-4a** | 4 | **Tolerance** | Recorded mean-predicted vs realized pairs (folds 1–3) | Without it Signal 4 cannot fire | **OPEN** |
| **N-4b** | 4 | **Minimum sample size** | Structural: over 10 fixtures realized Draw frequency can only take multiples of 0.1, so cannot land within 0.02 of 0.25192 even under perfect calibration | Without it the monitor false-alarms **by construction** | **OPEN** |
| **N-4c** | 4 | **Observation window** | None specified | Determines noise level; couples to N-4b | **OPEN** |
| **N-4d** | 4 | **Label-arrival handling** (delay; late / voided / abandoned fixtures) | None specified | Determines when a window may be evaluated at all | **OPEN** |
| **N-4e** | 4 | **Baseline interpretation** (same-window realized outcomes vs fixed historical base rate) | §12 wording *"realized outcome frequency"* points to same-window | Fixed-base-rate reading would duplicate Signal 3 | **OPEN** |
| **N-6a** | 6 | **Exact quantity** monitored | Recorded Draw figures: gap +0.00190; argmax share 0.00825; ranked-2nd 0.59032; recall 0.018; F1 0.034 | An argmax-based quantity fires permanently against accepted behaviour | **OPEN** |
| **N-6b** | 6 | **Threshold** | No authorized value exists | Without it Signal 6 cannot fire | **OPEN** |
| **N-6c** | 6 | ±0.03: **rejected / ratified / replaced** | Currently **REJECTED** (§8.7); described in source as descriptive, non-statistical | Determines whether a value already exists to reuse | **REJECTED; replacement OPEN** |
| **N-6d** | 6 | Whether Signal 6 **duplicates** Signals 3/4 | Draw appears in all three | Risks three correlated alerts from one event | **OPEN** |
| **N-x** | 1, 3, 4 | Cross-cutting **minimum sample size / suppression rule** | See N-4b | Any tolerance without a minimum-n is not implementable | **OPEN** |

**No numbers are chosen in this table.**

## 12.2 Pre-implementation classification

**Principle applied:** a decision must be resolved before implementation if the implementation **cannot be deterministic or testable** without it; it may be deferred if the implementation can be built generically **without changing monitoring semantics later**. Classified on that basis, not on convenience.

### A — MUST RESOLVE BEFORE IMPLEMENTATION

| Item | Why it blocks |
|---|---|
| N-1a, N-1b, N-1c (Signal 1 metric, baseline row, tolerance) | A drift computation is undefined without a metric and reference; no deterministic output, nothing to assert in a test |
| N-3a, N-3b, N-3c (Signal 3 baseline, tolerance, per-class vs aggregate) | Quantity is locked but the comparison and its shape are not; per-class vs aggregate changes the emitted record's structure, not just a constant |
| N-4a, N-4b, N-4c, N-4e (Signal 4 tolerance, minimum-n, window, baseline interpretation) | Window and baseline interpretation determine *what is computed*; N-4e specifically decides whether Signal 4 is distinct from Signal 3 |
| N-4d (label-arrival handling) | Determines when a measurement is admissible; a later change would silently alter every historical measurement |
| N-6a (Signal 6 quantity) | The signal has no definition without it; nothing can be built |
| N-6b / N-6c (Signal 6 threshold; ±0.03 disposition) | No firing rule exists |
| N-6d (Signal 6 vs Signals 3/4 overlap) | Changes how many signals exist and their alert semantics |
| N-x (minimum sample size / suppression rule) | Determines whether an emitted value is alertable; retrofitting changes past semantics |
| **Family→columns mapping** (D-13 / AUTH-K) | Signal 1 is per-**family** and only family *counts* exist; membership is not derivable from the spec alone |
| **Family vs per-column granularity** (D-14) | A single all-null column is diluted ~18× at family level; changes what the signal detects |
| **Emission contract** (record schema, identifiers, sink interface) | Every signal writes through it; changing it later rewrites all emitters and invalidates stored history |

### B — CAN REMAIN DEFERRED UNTIL EXPOSURE

| Item | Why deferral is safe |
|---|---|
| **Severity levels** (all signals) | Severity is consumer-side metadata; can be attached to an already-correct record without altering the measurement |
| **Response / escalation actions** (all signals) | Operational procedure, not computation. No monitoring semantics change |
| **Cadence / scheduling** | A generic emitter can be invoked at any cadence; cadence does not change the value computed for a given batch |
| **Alert routing, ownership, on-call** (D-33) | Purely organisational |
| **§11 rollback-trigger designation** (D-34) | A governance mapping over existing signals; adding it later changes nothing already emitted |
| **Validity-failure reporting window** (D-28) | Aggregation over an exact per-event count; derivable retrospectively from stored events |
| **Near-miss recording** (D-29) | Additive extra field, not required by §12 |
| **Null-vs-unseen `competition_id` asymmetry confirmation** (D-22) | Current behaviour is already specified and implemented; confirmation records intent |
| **Operational definition of "production exposure"** (D-17) | Required **before promotion**, not before building monitoring. §8.1 already fixes the governance relationship |
| **OBS-1 — independent justification of the 0.078 ceiling** | Documentation of an existing, unchanged constant; affects no monitoring computation |
| **Condition 2 presentation-compliance ownership** (D-32) | A separate control surface; no §12 signal depends on it |
| **`np.allclose` tightening decision** (§8.9) | Disclosed; real row sums deviate ~1e-16, so no measurement changes either way. **Must not** be resolved by editing `evaluate.py` |

**Deliberately NOT classified as deferrable, despite being convenient to defer:** the emission contract, family mapping, and minimum sample size. Each would silently change the meaning of already-emitted data if altered later, which fails the stated principle.

## 12.3 Ready now

Items already specified precisely enough to implement **without guessing**, if authorized:

1. **Signal 2 — categorical unseen-`competition_id` detection.** Known set enumerated in §10; rule fully locked (MUST emit, MUST flag, MUST NOT auto-fail); detection logic already exists in `check_feature_availability`; no threshold, baseline or minimum-n required. **Caveat:** its *emission target* depends on the emission contract (Class A), so the detection is ready but its sink is not.
2. **Signal 5 — validity-failure counting**, subject to one placement decision. Semantics are fully locked (zero tolerance, hard stop, record-then-re-raise-unchanged, `evaluate.py` untouched). **Placement is not yet defined:** whether the counter wraps the existing call inside `predict_candidate` (which modifies inference behaviour → **AUTH-I**) or sits alongside it at a monitoring layer. Ready **only if** placement can be fixed without modifying frozen code and without altering §8 semantics.
3. **Nothing else.** In particular, an "additive monitoring infrastructure skeleton" is **NOT** listed as ready: any skeleton necessarily fixes the emission contract, which is a Class A blocker. Building one now would encode unresolved semantics — precisely the failure mode this gate exists to prevent.

## 12.4 Blocked

Implementation would require inventing at least one of a threshold, baseline, metric, window, minimum sample size, response action, or ownership rule:

| Blocked item | Would require inventing |
|---|---|
| **Signal 1** (per-family null-rate drift) | metric, baseline row, tolerance, family mapping |
| **Signal 3** (class distribution) | baseline reference, tolerance, per-class-vs-aggregate shape |
| **Signal 4** (probability vs realized frequency) | tolerance, minimum-n, window, label-arrival rule, baseline interpretation |
| **Signal 6** (draw behaviour) | the quantity itself, plus a threshold |
| **Emission contract / sink** | record schema and interface — unblocks nothing until fixed, blocks everything once wrong |
| **Any alerting layer** | severity and response rules |
| **Prediction/outcome store** | required by Signals 3/4/6 post-hoc paths; needs **AUTH-J**; must not use 2025/26 for new computation |

## 12.5 Explicitly prohibited

Constraints that materially bind the next implementation stage:

- **No frozen V1 modification** — the seven files pinned in `data/audit/phase4c_prerun_manifest.json` (`config.py`, `train.py`, `splits.py`, `evaluate.py`, `data.py`, `baselines.py`, `run_experiments.py`).
- **No `evaluate.py` modification** — under any justification, including tightening `rtol`.
- **No 2025/26 recomputation or access** for any new computation; recorded values may be read only.
- **No tuning.** **No calibration.** (Recalibration is not authorized even as a response action.)
- **No V2 model or artifact; no `MODEL_VERSION` change** — remains `v1.0`.
- **No deployment; no production exposure** — and exposure requires Gate 10 PASS first.
- **No Phase 3 / 4C / 5A artifact modification.**
- **No threshold, baseline, metric, window, sample size, or response invented** in place of an authorization.

## 12.6 Readiness verdict

**Gate 10 is NOT ready for implementation authorization.** Two of six signals are implementation-ready in their detection semantics (Signals 2 and 5); both still depend on the emission contract, which is itself a Class A blocker. Fifteen numeric decisions and three structural decisions (family mapping, granularity, emission contract) must be resolved first.

**Gate 10 status unchanged: SPECIFICATION LOCKED / IMPLEMENTATION NOT STARTED. Acceptance OPEN. Not PASS.**

*(§12 above records the readiness position at the time thresholds were still open. It is superseded by §13, which records the authorized thresholds, the additive implementation, and the acceptance evidence.)*

---

# 13. Gate 10 Implementation and Acceptance Record

**Authority:** explicit human authorization of all remaining monitoring-policy decisions (thresholds, baselines, emission contract, implementation boundary, testing scope). No threshold in this section was invented, tuned, or inferred.

## 13.1 Locked thresholds as implemented

| Signal | Quantity | Threshold | Baseline | Timing |
|---|---|---|---|---|
| 1 — ceiling (`S1_CEILING`) | per-family null rate | §10 ceilings **0.078** / **0.056**, imported unchanged | absolute | prediction-time |
| 1 — drift (`S1_DRIFT`) | absolute percentage-point deviation from baseline | **+2 pp**, above-baseline only | non-test historical partitions; 2025/26 excluded | prediction-time |
| 2 (`S2_UNSEEN_COMPETITION`) | unseen-id occurrence | categorical → **`DEGRADED`**, never `HARD_FAIL` | known set {200, 419, 423, 477, 499} | prediction-time |
| 3 (`S3_CLASS_DISTRIBUTION`) | **mean predicted probability mass** per class | **±5 pp**, H/D/A independently | stored non-test base rates (Phase 3 artifact) | **prediction-time only** |
| 4 (`S4_CALIBRATION`) | mean predicted probability vs realized frequency | **±5 pp**, n ≥ **200**, **30**-day window, **7**-day maturity | window-realized frequency | **post-outcome only** |
| 5 (`S5_PROBABILITY_VALIDITY`) | invalid rows and batches | **zero tolerance** → **`HARD_FAIL`** | absolute rule | prediction-time |
| 6 (`S6_DRAW_BEHAVIOUR`) | mean predicted P(draw) vs realized draw frequency | **identical to Signal 4** | window-realized draw frequency | **post-outcome only** |

**Ceiling and drift are emitted as two separate events and are never merged.** The ±0.03 Phase 3 heuristic is **not adopted** and is asserted absent from the entire package by an AST constant scan.

## 13.2 Implementation completion status

**Additive module `src/monitoring/`** — `__init__.py`, `config.py`, `events.py`, `store.py`, `signals.py`, `validity.py`. Nothing outside this new package was created or modified by the implementation.

- **Family mapping** is *derived*, not restated: the six frozen `models.ablation` tuples are asserted at import time to be pairwise disjoint, to equal `MODEL_B_COLUMNS` as a set, and to match its order. A contract change breaks loudly rather than silently skewing every drift measurement.
- **Emission contract:** append-only **JSONL** (`JsonlEventStore`), caller-supplied path, no default path inside the repository, no update/delete/truncate methods. No table was added to `features.db` or `matches.db`. Every event carries `schema_version`, `event_id`, `timestamp`, `model_version`, `signal_id`, `signal_name`, `status`, `severity`, `metrics`, `baseline`, `threshold`, `sample_count`, `context`, `message`, and is schema-validated at construction *and* on write.
- **Statuses:** exactly `PASS`, `ALERT`, `HARD_FAIL`, `INSUFFICIENT_SAMPLE`, `DEGRADED`. **Severity** is a mechanical function of status, not an independent policy (severity policy remained deferrable).
- **Signal 5 boundary:** `monitored_predict_candidate` is an **external wrapper** around `predict_candidate`. `evaluate.py` and `candidate_contract.py` are unmodified. On failure the wrapper records the event and re-raises via a **bare `raise`**, preserving type, message, instance identity and traceback. AST tests assert every `raise` in the wrapper is bare, and that no `clip`/`normalize`/`fillna`/`nan_to_num`/`repair` call exists. Row-level counts are explicitly diagnostic; `validate_probabilities` remains the sole authority.
- **Not built:** dashboards, web APIs, cloud infrastructure, schedulers, alert integrations, UI, deployment. Asserted by an import-allowlist test.

## 13.3 Test evidence

**Gate 10 suite: 134 tests — 134 passed, 0 failed, 0 errors, 0 skipped.**
**Full repository suite: 975 tests — 975 passed, 0 failed, 0 errors, 87 skipped.**

Baseline before this work was 841 passed / 0 failed / 87 skipped. The change is **+134 tests, all new Gate 10 tests**; no pre-existing test was modified, removed, or skipped, and the skip count is unchanged at 87 (the same sklearn/data-gated tests as before — Gate 10's own tests have no such dependency and all execute).

All 24 required areas are covered, plus §12-coverage and negative-control tests. Boundary cases are asserted from **both** sides (e.g. +1.9 pp PASS / +2.0 pp PASS / +2.1 pp ALERT; 199 `INSUFFICIENT_SAMPLE` / 200 verdict; 6.99-day maturity excluded / 7.00-day included; 29-, 30-, 31-day window).

**Three defective tests were repaired — test defects, not implementation faults. No threshold or invariant was weakened.**

1. **`test_no_reliable_draw_classifier_claim`** — the original was a source-text grep that flagged the implementation's own *disclaimer* ("nothing here claims the model is a reliable Draw classifier") as a prohibited *claim*. Replaced with a **negation-aware semantic check** applied to (a) every string actually emitted in a monitoring event — what a consumer sees — and (b) every **non-docstring** string constant in the package (docstrings identified via AST, not heuristics). A match is a violation only if it is *not* preceded by a negation. Two controls were added: the detector must still flag `"this model is a reliable draw classifier"` and must not flag `"the candidate is not a reliable draw classifier"`. **The invariant is preserved and is now stronger**, because it now also covers runtime-emitted strings, which the grep never inspected.
2. **`test_no_training_function_defined`** — the original matched the token `calibrat` inside `evaluate_calibration_post_outcome`, a monitoring evaluator that fits nothing. Replaced with **two behavioural AST checks**: no estimator/transformer/calibrator is *constructed* (explicit constructor allowlist-violation set), and no fitting/training/tuning/calibration-fitting *call* occurs (`fit`, `fit_transform`, `partial_fit`, `train*`, `tune`, `_fit_platt_1d`, `select_calibration_method`, …). A third test guards against the check becoming over-broad again by asserting `evaluate_calibration_post_outcome` remains present, callable, and free of any fitting call.
3. **`test_row_order_does_not_change_aggregate_metrics`** — the original asserted bit-for-bit order invariance, which floating-point summation does not provide (observed 0.4000000000000001 vs 0.39999999999999997). **Row-order invariance is not a Gate 10 requirement and is no longer claimed.** Replaced with a numerically appropriate invariant: reordering perturbs aggregates only at machine-epsilon scale (≤ 1e-12), and the *verdict* is unchanged — plus a 500-row permutation test asserting a reorder can never flip a threshold verdict (the perturbation is ~14 orders of magnitude below the tightest threshold). **Bit-for-bit determinism where it is genuinely required — identical input producing identical output across repeated calls — remains asserted for all six signals.**

## 13.4 Residual governance notes

Both are recorded openly rather than resolved by invention.

### Residual 1 — N-1b: baseline partition/aggregation remains caller-supplied

The authorization fixed the **constraint** (baseline built only from non-test historical partitions, 2025/26 excluded) but did **not** select which partition, or which aggregation across partitions, is operational. **No default was invented.** `build_null_rate_baseline(aggregation, partitions=None)` therefore requires `aggregation` explicitly — calling it with no argument raises `TypeError`, asserted by test — and offers `"max"`, `"mean"`, or `"single"` over the four recorded non-test partitions (`fold_1_train`, `fold_2_train`, `fold_3_train`, `final_train`). Any partition name outside that set, including every spelling of the test season, raises `MonitoringInputError`; 2025/26 is excluded **structurally**, not by convention.

**Effect:** Signal 1 emits correctly for any stated baseline. What remains undecided is which baseline an operator should state.

### Residual 2 — settlement window interpreted on `settled_at`

The authorized policy fixes a **30-day rolling observation window** and a **7-day settlement maturity**, but does not state which timestamp the window is measured against. The implementation applies:

- **maturity** exactly as written — `settled_at − predicted_at ≥ 7 days` (the literal rule, deliberately **not** reinterpreted);
- **window** to `settled_at` — every eligible observation is by definition settled, and a post-outcome signal measures recently-settled results.

**This is recorded, not silently adopted:** the reading is documented in `select_eligible_observations`, in spec §12.4, and here; **both** timestamps are preserved in the emitted metrics so the choice is auditable rather than hidden; and the exclusion breakdown reports `unsettled` / `immature` / `outside_window` counts separately. **The specification was not changed.** A separate operational observation is also recorded without acting on it: settlement in practice follows kickoff by hours, so a literal "settled ≥ 7 days after prediction" rule excludes any fixture predicted less than a week before kickoff. That is the authorized rule as written and is implemented faithfully; if a different intent was meant, it requires an explicit decision, not an inference.

### Do these residuals prevent Gate 10 acceptance?

**No — neither blocks acceptance, and this is stated with the reasoning rather than asserted.**

Gate 10's pass condition is *"All §12 signals emitted with defined thresholds."* All six §12 requirements are emitted with defined thresholds, and the full acceptance suite passes. Residual 1 concerns which *input* an operator supplies to a correctly-implemented and fully-tested signal, not whether the signal or its threshold exists — the threshold (+2 pp) is defined and enforced. Residual 2 concerns a documented, auditable reading of an under-specified field, not a missing or invented threshold.

**Both must nevertheless be resolved before Signal 1 or Signals 4/6 are *operated* in production**, and both are carried forward as required pre-promotion decisions. They are acceptance-neutral, not consequence-free.

## 13.5 Verification record

| Check | Result |
|---|---|
| 13 pinned baselines | **all byte-identical**, zero mismatches |
| 7 frozen V1 sources (`config`, `train`, `splits`, `evaluate`, `data`, `baselines`, `run_experiments`) | **unchanged** |
| `evaluate.py` specifically | **unchanged** (`4e9d9313867d47a19001a383a301c2fe`) |
| `calibration.py` | unchanged (`8a43c6b5361f7b369be85b4944880506`) |
| `candidate_contract.py` | unchanged (`20ef06aba787ab4be8b3372c6684b9bc`) |
| `ablation.py` | unchanged |
| `features.db`, `matches.db` | unchanged |
| Phase 3 / 4C / 5A artifacts | unchanged |
| `MODEL_VERSION` | **`v1.0`** |
| Model training / tuning / calibration fitting | **none** — asserted by AST tests |
| 2025/26 access | **none** — asserted structurally and by AST tests |
| V2 model or artifact | **none created** |
| Deployment / production exposure | **none** |

## 13.6 Gate 10 status

> ## GATE 10 — PASS

Recorded against the authorized pass criteria: all six required signals are emitted by the implementation; all required thresholds are defined; the full acceptance suite executes and passes (134/134, plus 975/975 repository-wide, 0 failures, 0 errors); no frozen V1 file changed; no forbidden data or model operation occurred.

**What this PASS does and does not mean.** It certifies that specified monitoring **exists, emits, and is tested**. It does **not** certify observed operation: monitoring has produced **no production measurement**, and Signals 4 and 6 cannot produce one until real predictions and settled outcomes exist. **Gate 10 PASS is not promotion authorization**, does not create V2, does not change `MODEL_VERSION`, and does not authorize deployment or production exposure. Promotion remains subject to its own explicit authorization, and the two residual decisions in §13.4 must be closed before Signal 1 or Signals 4/6 are operated.
