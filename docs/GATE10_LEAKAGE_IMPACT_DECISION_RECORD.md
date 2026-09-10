# Leakage Impact Decision Record — Simultaneous-Kickoff Contamination

**Document type:** governance decision record. Documentation only.

**Date:** 2026-08-18

> **LEAKAGE IMPACT DIAGNOSTIC — COMPLETE**
> **GOVERNANCE DECISION — OPEN**
> **V1 — FROZEN**
> **V2 / STRICT CANDIDATE — NOT AUTHORIZED**

---

## 1. Executive Summary

A discrepancy was found between the project's **written** Phase 2 temporal rule and its **implemented** feature-generation ordering. The written rule admits history strictly earlier than the target fixture's kickoff (`unix < T.unix`). The implementation walks fixtures in `(unix ASC, fixture_id ASC)` order and records each fixture's result immediately after featurising it, so for two fixtures sharing an **identical** kickoff timestamp the later-ordered one sees the earlier-ordered one's result — information that was not available strictly before its own kickoff.

The contamination is **narrow and precisely bounded**. It reaches the model through exactly one channel: the league-season accumulator, which feeds `league_mean_goals_per_team_match()` into the empirical-Bayes shrinkage behind the strength family. Team histories are provably unaffected (a team cannot play two fixtures simultaneously). No strictly-later fixture ever entered any history.

An authorized, read-only diagnostic compared two conditions that differ **only** in that historical cutoff, holding the frozen Model B configuration, the 80-column contract, V1 preprocessing, and the walk-forward protocol constant. The impact is **real and measurable, but small and directionally mixed**:

- CURRENT has slightly **lower** mean validation log loss (Δ +2.600305e-04 for STRICT).
- STRICT has slightly **lower** 2025/26 log loss (Δ −1.828235e-04).
- Accuracy, macro-F1 and balanced accuracy move in the **opposite** direction to log loss on the validation folds.
- Largest single-fixture probability movement observed: **1.775986e-01**; typical movement is ~1e-03.

**The evidence does not establish that STRICT is better.** It establishes that the discrepancy exists, that it changes model outputs measurably, and that it cannot be dismissed as numerically irrelevant. **No decision is made in this document.** Two options are placed before the owner in §10; neither is selected, and no threshold of materiality is invented.

**V1 remains frozen. No V1 artifact, database, checksum, or governance decision was changed.**

---

## 2. Scope and Non-Scope

### In scope

- Recording the specification/implementation discrepancy precisely.
- Recording the completed diagnostic's evidence.
- Defining the boundary that any change to the feature-generation semantics would cross.
- Enumerating the open governance options.

### Explicitly NOT in scope, and NOT performed

This record does **not**: modify frozen V1 code; rebuild or regenerate `features.db`; train, retrain, tune, or calibrate any model; run a new evaluation; generate predictions; create a V2 module or artifact; change `MODEL_VERSION`; access 2025/26 for any new model-development purpose; alter any locked governance decision; choose between CURRENT and STRICT; declare V1 invalid; declare Gate 10 invalid; declare STRICT superior; or invent an acceptance or materiality threshold.

The only file created by this task is this document.

---

## 3. Governing Temporal Rule

### 3.1 The written rule

`docs/PHASE2_FEATURE_SPEC.md` states the universal leakage rule verbatim:

> **Leakage rule (universal):** for a target fixture T with kickoff time `T.unix`, every feature is computed using only fixtures with `unix < T.unix` and `status = 'FT'` (or `'AWARDED'`, which the audit found has real scores — see §0.1), for the relevant team(s)/league. `T` itself is always excluded from its own history, unconditionally, even if some other fixture shares its exact `unix` (defensive tie-break: exclude by `fixture_id != T.fixture_id` in addition to the time filter).

Two clauses matter and must be read separately:

1. The **cutoff** is `unix < T.unix` — strictly less than.
2. The **defensive tie-break** guarantees that `T` is excluded from its own history when another fixture shares its `unix`. It guarantees nothing about *other* fixtures sharing that `unix`.

### 3.2 The implemented behaviour

`src/features/history.py` loads fixtures with:

```sql
SELECT * FROM fixtures ORDER BY unix ASC, fixture_id ASC
```

`src/features/feature_builder.py::build_feature_dataset` then applies the compute-then-record discipline: `build_feature_row(fixture, ctx)` is called, and only afterwards `ctx.record(fixture)`. That ordering is the project's central leakage guarantee and it functions correctly for strictly-ordered fixtures.

For two fixtures **A** and **B** with identical `unix` where `A.fixture_id < B.fixture_id`, the walk featurises A, records A, then featurises B. B's history therefore contains A — a fixture whose `unix` equals, not precedes, B's.

### 3.3 The precise discrepancy

The implementation's effective cutoff is:

```
unix < T.unix   OR   (unix == T.unix AND fixture_id < T.fixture_id)
```

The written rule's cutoff is:

```
unix < T.unix
```

The second disjunct is the discrepancy. `fixture_id` is a record identifier, not a temporal quantity, so it cannot establish precedence between simultaneous fixtures.

### 3.4 Why only one channel is affected

- **Team history — unaffected.** `FeatureContext.history_before(team_id)` is per team. A team cannot appear in two fixtures at the same instant; this was verified empirically with **zero** occurrences across all 10,735 fixtures.
- **League-season accumulator — affected.** `LeagueSeasonAccumulator` is keyed by `(competition_id, season_id)` and aggregates every fixture in that league-season, including simultaneous ones. Its `league_mean_goals_per_team_match()` is consumed by `features/strength.py::strength_features` and passed to `shrunk_rate()`, which produces the strength scores.

Of the league-derived features, `league_home_advantage_season` and `league_mean_goals_per_team_match_season` are **not** in the Model B contract. The contamination therefore reaches Model B **only** through the strength family.

---

## 4. Leakage Reconstruction Evidence

Read-only reconstruction; `features.db` was never regenerated and no reconstructed value was written to disk.

| Measure | Value |
|---|---|
| Fixtures walked | **10,735** |
| Affected by same-kickoff contamination | **2,877** |
| Unaffected | **7,858** |
| Unaffected fixtures bit-identical across all five strength columns | **7,858 / 7,858** |
| Strictly-later contamination (`unix > T.unix`) | **0** |
| Team-history temporal contamination | **0** |
| Channel | League-season accumulator only |

### 4.1 Columns differing

| Column | Classification |
|---|---|
| `home_attack_strength_score` | **material difference** |
| `home_defence_strength_score` | **material difference** |
| `away_attack_strength_score` | **material difference** |
| `away_defence_strength_score` | **material difference** |
| `strength_diff` | **floating-point residue only** — maximum ≈ **1.33e-15** |

`strength_diff` is **algebraically invariant** to the league mean, which is why only residue appears. With `shrunk_rate(sum, n, L) = (sum + k·L)/(n + k)`:

```
attack − defence = (gf + k·L)/(n + k) − (ga + k·L)/(n + k) = (gf − ga)/(n + k)
```

`L` cancels on both the home and away side independently, so `strength_diff = (home_att − home_def) − (away_att − away_def)` cannot depend on it. The residue is the expected numerical signature of that cancellation, not a small real difference.

**Net exposure to Model B: four of the eighty contract columns.**

### 4.2 2025/26 La Liga

| Measure | Value |
|---|---|
| Fixtures affected | **31 of 380** |
| Same-kickoff clusters | **17** |
| Clusters of size 2 | 15 |
| Clusters of size 9 | 2 |

---

## 5. Model Impact

**Everything in this section is a measurement.** No value here is an acceptance criterion, a materiality threshold, or a pass/fail bound. None was compared against any threshold, because no such threshold exists or was invented.

### 5.1 Diagnostic controls

Both conditions used the identical frozen configuration: `MODEL_B_COLUMNS` (80, contract order), `LogisticRegression(C=1.0, max_iter=2000, random_state=0)`, V1 preprocessing, `CLASS_ORDER = ["H", "D", "A"]`, no calibration, the existing walk-forward protocol and splits. **The league-season historical cutoff was the only intentional difference.**

### 5.2 Walk-forward validation (existing protocol)

| Metric | CURRENT | STRICT | Δ (STRICT − CURRENT) |
|---|---:|---:|---:|
| Mean log loss | 0.9993791056968738 | 0.9996391362118540 | **+2.600305e-04** |
| Mean Brier | 0.5965016957578898 | 0.5966577484351560 | +1.561e-04 |
| Mean accuracy | 0.5184135481726413 | 0.5195160492160965 | +1.103e-03 |
| Mean macro-F1 | 0.4034836046490057 | 0.4042627797238625 | +7.792e-04 |
| Mean balanced accuracy | 0.4532925378053693 | 0.4543823247610101 | +1.090e-03 |

**Fold ordering: unchanged.** On the primary metric (log loss), CURRENT is lower. On accuracy, macro-F1 and balanced accuracy, STRICT is higher.

### 5.3 2025/26 final partition — Gate 7 / Phase 4C verification role only

This partition was used **solely** in its existing verification role. It was **not** used to fit, tune, calibrate, select a model, select features, select thresholds, or choose between CURRENT and STRICT.

| Metric | CURRENT | STRICT | Δ (STRICT − CURRENT) |
|---|---:|---:|---:|
| Log loss | 0.9960487649063601 | 0.9958659414519916 | **−1.828235e-04** |
| Accuracy | 0.5151342090234152 | 0.5139920045688178 | −1.142e-03 |
| Macro-F1 | 0.3938491505819635 | 0.3931308693346867 | −7.183e-04 |
| Balanced accuracy | 0.4489548797121166 | 0.4480902018046372 | −8.647e-04 |

**Harness fidelity:** CURRENT reproduces the locked Phase 4C Model B reference log loss `0.9960487649063601` **exactly**. This independently corroborates that the diagnostic reproduced the audited path faithfully rather than an approximation of it. STRICT differs from that locked reference by **1.828235e-04** — a difference attributable to the cutoff change, not to any model change.

**The direction reverses between §5.2 and §5.3.** On log loss, CURRENT is better in validation and STRICT is better on the final partition; on the class-assignment metrics the reversal runs the other way. This is recorded as observed, without inference.

### 5.4 Probability movement

**Validation folds** — every evaluated fixture had non-identical probabilities:

| Fold | Mean absolute movement |
|---|---:|
| fold 1 | 2.094266e-03 |
| fold 2 | 1.264576e-03 |
| fold 3 | 1.148278e-03 |

Largest individual probability movement observed: **1.775986e-01**.

**Final 2025/26 partition** — 1,751 fixtures evaluated, **1,751** non-identical:

| Statistic | Value |
|---|---:|
| Mean absolute movement | 9.397055e-04 |
| Median per-fixture max movement | 9.532521e-04 |
| p95 | 3.156939e-03 |
| Maximum | 4.365619e-02 |

**111 fixtures** across the measured validation and final partitions moved by more than 0.01.

**Affected 2025/26 La Liga subset (31 fixtures)** — probability movement only; **no accuracy or performance claim is made for this subset**:

| Statistic | Value |
|---|---:|
| Mean absolute movement | 6.942176e-04 |
| Median | 8.942674e-04 |
| p95 | 2.290939e-03 |
| Maximum | 3.484420e-03 |
| Fixtures exceeding 0.01 | **0** |

A point worth recording plainly: the movement is **not confined to the 31 directly contaminated fixtures**. Every fixture's probabilities moved, because the cutoff change alters four features across the *training* partitions and therefore the fitted coefficients. The largest movements occur outside the directly affected subset.

---

## 6. Interpretation

Stated carefully, because the evidence does not point one way.

1. **CURRENT wins the walk-forward primary metric.** Mean validation log loss is lower for CURRENT by 2.600305e-04.
2. **STRICT wins the 2025/26 log loss** by 1.828235e-04 — but that partition is a verification surface, not a selection surface, and using it to choose would violate the standing governance boundary.
3. **The class-assignment metrics move in mixed directions.** STRICT is higher on validation accuracy, macro-F1 and balanced accuracy; CURRENT is higher on the same metrics on the final partition. Log loss and the class metrics disagree with each other within the same partition.
4. **Therefore the evidence does NOT establish that STRICT is universally better**, and it does not establish that CURRENT is better either. The magnitudes are small, the directions conflict across partitions and metrics, and no significance testing was performed or authorized.
5. **Nevertheless, the discrepancy is real and must not be silently ignored.** The implementation does not satisfy the written rule. That is a specification-conformance fact, and it is independent of whether correcting it would improve any metric. A conformance defect is not validated by a favourable metric, nor excused by an unfavourable one.
6. **What is genuinely established:** the contamination exists; it is confined to one channel; it affects four contract columns materially and one only at floating-point residue; it changes every downstream prediction by a small amount; and it is not numerically negligible at the individual-fixture level, where the largest observed movement is 0.178.

No materiality threshold is applied, because none exists. Whether a maximum movement of 0.178 on a single fixture, or ~1e-03 typically, is tolerable is a judgement for the owner, not a computation.

---

## 7. Frozen V1 Status

- **V1 remains FROZEN.**
- **No V1 artifact was changed.** All 13 pinned baselines verified byte-identical before and after the diagnostic.
- **No feature database was regenerated.** `features.db` and `matches.db` are unchanged; all reconstruction was in memory and discarded.
- **No V2 was created.** No V2 module, artifact, or version exists.
- **`MODEL_VERSION` remains `v1.0`.**
- **No serialized estimator exists.** Both diagnostic fits were in memory and were not persisted.
- **Gate 1 → Gate 10 remain as previously verified.** Nothing in this record invalidates any gate, and this document makes no such claim.

---

## 8. Decision Boundary

Changing feature generation from CURRENT to STRICT is **not a code edit**. It would cross a governance boundary because it changes, in order:

1. **Feature-generation semantics** — the effective temporal cutoff of the league-season accumulator, in `src/features/`.
2. **`features.db`** — regeneration is required for the change to take effect. `features.db` is a **pinned baseline** (`e7ebe7fc07040a5927683c35b6371e63`).
3. **Pinned checksums** — regenerating `features.db` breaks the pin recorded in `data/audit/phase4c_prerun_manifest.json`, and with it the Gate 9 checksum verification and the §11 rollback evidence base.
4. **Downstream artifacts** — every Phase 3, Phase 4A, Phase 4B, Phase 4C and Phase 5A result was computed on the current feature values. Under regenerated features they would no longer be reproducible as recorded, which affects Gate 7's `<1e-9` regression evidence and the Phase 4C V1-replacement decision record.

**Consequence:** a CURRENT → STRICT change requires **separate, explicit authorization**, and is not implied by the discovery, by this record, or by any measurement within it.

---

## 9. Proposed Next Stage — **NOT AUTHORIZED**

Described only so the scope is visible before any decision. **None of this has been performed, and none of it is authorized by this document.**

1. **Separately versioned strict-temporal candidate reconstruction.** Correct the league-season accumulator so that same-`unix` fixtures are recorded only after every fixture in that timestamp cluster has been featurised, leaving team-history handling untouched. Version it distinctly; do not overwrite the existing feature version.
2. **Regenerate the necessary feature data into a separate, additively named store** — never overwriting `features.db` — so both feature generations remain simultaneously inspectable and the current pins stay intact.
3. **Re-run the relevant regression and validation gates** against the reconstructed data: Gate 2 (temporal safety, extended to cover simultaneous kickoffs), Gate 3, Gate 4, Gate 5, and Gate 7's reproduction boundary, plus the Phase 2 leakage suite with an added same-kickoff case.
4. **Compare clean CURRENT vs STRICT baselines** under the pre-registered protocol, with the selection rule fixed in advance and 2025/26 excluded from selection.
5. **Only after all of the above**, consider whether any model improvement work is warranted.

Each step needs its own authorization. Step 2 in particular touches pinned state and cannot proceed on the strength of this record.

---

## 10. Governance Decision — **OPEN**

**The decision is the owner's. This document does not make it, does not recommend either option, and does not treat silence as assent.**

### Option A — Preserve CURRENT frozen V1 as the historical baseline, and document the discrepancy

V1 and every downstream artifact stay exactly as they are. The discrepancy is recorded permanently (this document) and carried as a known, disclosed deviation. Pins, gates, and reproducibility evidence remain intact. Nothing is regenerated.

### Option B — Authorize a separately versioned strict-temporal candidate experiment

Proceed with §9 under separate authorization, keeping CURRENT intact and additive throughout, so both generations can be compared on clean evidence before any further decision.

**Neither option is selected. Choosing between them requires an explicit instruction.** Note that Option A is not "do nothing" — it is an affirmative decision to accept and document a known specification/implementation divergence, and should be recorded as such if chosen.

---

## 11. Residual Governance Notes

Carried forward unchanged from the Gate 10 record. **No default is invented for either.**

- **N-1b — Signal 1 baseline partition/aggregation remains caller-supplied.** The authorization fixed the constraint (non-test partitions only; 2025/26 excluded) but did not select which partition or aggregation is operational. `build_null_rate_baseline` accordingly requires `aggregation` explicitly and provides no default.
- **Settlement-window interpretation remains `settled_at`.** The 7-day maturity rule is implemented literally as `settled_at − predicted_at ≥ 7 days`, and the 30-day rolling window is applied to `settled_at`. This reading is documented, not silently changed.

Neither residual blocks Gate 10 acceptance. Both must be resolved before Signal 1 or Signals 4/6 are **operated** in production.

---

## 12. Integrity / Audit Trail

Verified before and after the diagnostic:

| Item | Result |
|---|---|
| 13 pinned V1 baselines | **13/13 identical** |
| `features.db` | unchanged |
| `matches.db` | unchanged |
| `calibration.py` | unchanged |
| `candidate_contract.py` | unchanged |
| `ablation.py` | unchanged |
| Locked Phase 3 / 4C / 5A artifacts | unchanged |
| `MODEL_VERSION` | **`v1.0`** |
| Serialized estimator persisted | **NONE** |
| V2 artifact created | **NONE** |
| Monitoring artifact created | **NONE** |
| Prediction artifact created | **NONE** |
| Training / tuning / calibration | **NONE** beyond the authorized diagnostic's in-memory frozen-path fits |
| Production exposure | **NONE** |
| Deployment | **NONE** |
| Feature database regeneration | **NONE** |

The diagnostic was **read-only with respect to repository state**. The 2025/26 partition was used solely in its existing Gate 7 / Phase 4C verification role.

---

## 13. Final Status

> **LEAKAGE IMPACT DIAGNOSTIC — COMPLETE**
> **GOVERNANCE DECISION — OPEN**
> **V1 — FROZEN**
> **V2 / STRICT CANDIDATE — NOT AUTHORIZED**

No model improvement work, feature regeneration, or match prediction is authorized by this record.
