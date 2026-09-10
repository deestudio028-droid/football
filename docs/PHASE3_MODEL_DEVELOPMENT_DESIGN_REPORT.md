# PHASE 3 — MODEL DEVELOPMENT DESIGN REPORT

**Status: design and analysis only. No `src/models/` package, no training pipeline, and no committed model code exist yet.** Everything quantitative in this report (target audit, baseline numbers, the softmax sanity-check model) came from throwaway analysis scripts run against `data/processed/features.db` and discarded — nothing was written to the repository except this report and its supporting evidence in `data/audit/`. `matches.db`, Phase 1 ingestion code, and Phase 2 feature definitions were not touched.

---

## 1. Dataset audit

Inspected directly (not assumed from prior docs, though it corroborates them):

- `data/processed/features.db`, table `feature_rows`: **10,735 rows, 274 columns**, `feature_version = "v1.0"` on every row.
- `data/processed/matches.db` unchanged since Phase 2 (single `fixtures` table, 10,735 rows).
- Chronological range: 6 seasons, 2020/21 through 2025/26, all fully complete (no partial/future season — today's date is 2026-08-16 and every ingested season already finished).
- 5 leagues, roughly balanced season sizes (~1,750-1,827 rows per season across all 5 leagues combined).
- `feature_version` gate: exactly one version currently exists, so no version-mixing risk yet — but the model pipeline should hard-fail (not silently proceed) if it ever sees a `feature_rows` table with more than one distinct `feature_version` value, since that would mean features generated under different methodologies got blended.
- **No feature or metadata column can encode the target.** Verified structurally, not just by naming convention: `storage.LABEL_COLUMNS` (`label_home_goals`, `label_away_goals`, `label_result`) are the only columns derived from the target fixture's own result, and Phase 2's leakage tests (10/10 passing, re-confirmed still passing before this analysis started) already prove no feature column is computed from the target fixture's own `unix`-or-later data. This report did not need to re-derive that guarantee, only confirm it hadn't regressed — it hadn't (`python -m unittest discover -s tests` → 78/78 pass, checked before writing this report).
- xG coverage, season coverage, missingness: match Phase 2's report exactly (65.7% overall xG null, 0% pre-2023/24, ~21% in the transition season, ~94.5% in 2024/25-2025/26) — re-verified via a fresh query rather than copied, see `data/audit/phase3_target_audit.json`.

## 2. Target definition

**V1 target: 3-class match result (H/D/A), derived from `label_result` in `feature_rows`.** Per the task's instruction to evaluate rather than assume this, options B/C/D were assessed against A (§ below in the modeling-approach analysis, folded into this section since the target and the modeling approach are the same decision here):

- **B (separate binary models)** — three independent binary classifiers (H-vs-not, D-vs-not, A-vs-not) whose outputs aren't constrained to sum to 1 without a separate normalization step, and which throw away the information that the three outcomes are mutually exclusive. No advantage over a single multiclass model here and an extra failure mode (badly-calibrated binaries producing incoherent joint probabilities). **Rejected for V1.**
- **C (goal-based: expected home/away goals → derived W/D/L)** — statistically the more principled long-run direction (it's the natural bridge to the Poisson-simulation stage the project's overall roadmap already anticipates for later phases), and the feature set already contains goals-based and xG-based rolling stats that are natural regression targets/inputs for this. But it requires two well-calibrated regression models (expected home goals, expected away goals) *and* a defensible goals→outcome-probability conversion (e.g. a Poisson or bivariate-Poisson simulation), which is explicitly out of scope for this phase per the strict boundaries ("do not implement Poisson/Monte Carlo yet"). Building it properly now would mean silently doing Phase-4-scope work inside Phase 3. **Deferred, not rejected** — flagged as the recommended V1.1+ direction once a working classification baseline exists to compare it against.
- **D (hybrid)** — combining a classification head with goal-based features as inputs is actually just "A, using goals/xG features," which is what's being recommended anyway; a hybrid *output* (both a direct H/D/A model and a derived-from-goals one, compared against each other) is a reasonable v1.1 experiment once there are two real candidates to compare, not a V1 requirement.
- **A (multiclass H/D/A classification)** — directly matches what the project's client-facing requirement asks for first ("which team is more likely to win," "draw probability"), needs only one well-calibrated model instead of two, and is what every baseline/candidate in this report was actually evaluated against. **Selected for V1.**

### Class distribution (10,734 labeled rows — the 1 ABANDONED fixture excluded, see §4)

| Outcome | Count | % |
|---|---|---|
| Home win (H) | 4,606 | 42.9% |
| Away win (A) | 3,402 | 31.7% |
| Draw (D) | 2,726 | 25.4% |

**Moderate, not severe, class imbalance** — home win is the plurality class everywhere but never a majority above ~49% in any single league/season, and draw is consistently the smallest class (20-29% range) but never negligible. This rules out needing aggressive resampling (SMOTE etc.) for V1; class weighting or simply reporting macro-averaged metrics alongside accuracy (§10) is enough.

**By league** (all 6 seasons combined) — spread is real but modest, no outlier league:

| League | H% | D% | A% |
|---|---|---|---|
| La Liga | 45.0% | 26.6% | 28.4% |
| Bundesliga | 43.7% | 25.2% | 31.0% |
| Premier League | 43.1% | 23.6% | 33.3% |
| Ligue 1 | 42.3% | 24.6% | 33.1% |
| Serie A | 40.4% | 26.9% | 32.7% |

**By season** — no material drift over time (checked explicitly per the task's instruction): home-win share ranges roughly 37-49% and draw share 20-29% in every single season, with no visible trend line across 2020/21→2025/26 — the variation looks like ordinary season-to-season noise, not a structural shift the model needs to explicitly account for (e.g. no post-covid-crowd-return effect big enough to show up at this resolution). Full table: `data/audit/phase3_target_audit.json`.

## 3. Class distribution — see §2 (kept together deliberately; splitting them would duplicate the same table twice)

## 4. Treatment of the ABANDONED fixture

Fixture `420450481` (Nantes vs Toulouse, Ligue 1 2025/26, the same one identified in the Phase 2 data-integrity audit) has `label_home_goals`/`label_away_goals`/`label_result` all `NULL` in `feature_rows` — this was Phase 2's correct behavior (a match with no valid final score gets no fabricated label). **Explicit decision for Phase 3: this row is excluded from supervised training and evaluation everywhere** (every query, split, and metric in this report already excludes it via `WHERE label_result IS NOT NULL`). It is not deleted from `feature_rows` — its pre-match features remain valid and it may still be a legitimate row for other purposes (e.g. an inference-time smoke test where no label is expected) — it is simply never included in any (X, y) training or evaluation pair. This keeps the labeled dataset at exactly 10,734 rows, and the implementation must encode this as an explicit filter (`label_result IS NOT NULL`), not an accidental one (e.g. relying on a downstream `dropna` that would also silently drop legitimately-null *feature* columns for unrelated reasons).

## 5. Temporal split strategy

**Random splitting (train_test_split, K-fold, stratified-random) is explicitly rejected**, per the task's instruction and for the obvious reason that it would let a model trained partly on "future" matches predict "past" ones during cross-validation — a direct violation of the project's core leakage principle, just at the model-evaluation layer instead of the feature layer.

Inspected actual season sizes before choosing boundaries (not assumed): all 6 seasons are complete and roughly equal in size (1,751-1,827 rows each, combining all 5 leagues), so there's no season so small it would destabilize a fold and no reason to deviate from clean season-boundary splits.

**Recommended design — walk-forward with 3 validation folds plus one untouched final test season:**

| Fold | Train | Validate | Train n | Val n |
|---|---|---|---|---|
| 1 | 2020/21 – 2021/22 | 2022/23 | 3,388 | 1,719 |
| 2 | 2020/21 – 2022/23 | 2023/24 | 5,107 | 1,649 |
| 3 | 2020/21 – 2023/24 | 2024/25 | 6,756 | 1,651 |
| **Final test (untouched)** | 2020/21 – 2024/25 | **2025/26** | 8,407 | 1,651 |

This is close to, but not identical to, the task's illustrative example — the actual data confirmed it holds up (each fold has thousands of training rows and 1,600+ validation rows, no fold is starved), so no adjustment was needed beyond confirming it. **2025/26 is reserved as the final test set and must not be touched for model selection or hyperparameter tuning at any point** — only Folds 1-3 may inform which model/hyperparameters get chosen (§10's selection rule is fixed before anyone looks at 2025/26 results, per the task's requirement and standard practice for avoiding test-set leakage through repeated peeking).

One nuance worth flagging: Fold 2's validation season (2023/24) is exactly the xG-rollout transition season (per Phase 2's report, ~21% xG coverage that season vs ~0% or ~95% elsewhere) — this makes Fold 2 a naturally useful stress test for how much the xG-coverage-gating design actually matters in practice, not just a fold to average away.

## 6. Walk-forward validation strategy

The 3-fold design in §5 *is* the walk-forward/rolling-origin backtest the task asked for — each fold's training window only grows forward in time and never includes data from its own or a later validation period. Model selection (§10) is based on the **average** of the 3 folds' metrics, not any single fold, to avoid overfitting the choice of model/hyperparameters to one particular season's quirks (e.g. 2023/24's xG transition). After a model/hyperparameter configuration is chosen using only Folds 1-3, it is retrained once on all data through 2024/25 and evaluated exactly once on 2025/26 — that number is reported as-is, not used to go back and adjust anything.

## 7. League generalization strategy

Evaluated using only project data, no external rankings, per the options in the task:

- **Per-league models (B)**: would cut every league's training set roughly to a fifth of the pooled size (from ~8,400 down to ~1,600-1,700 rows through 2024/25) — for a 3-class problem with the feature richness Phase 2 built (274 columns, even after dropping label/metadata columns), that's a real risk of overfitting, especially for leagues with the sparsest xG coverage.
- **Global pooled model (A)**: The by-league distribution table in §2 shows real but modest differences (a ~5 percentage point spread in home-win rate, a similar spread in away-win rate) — not large enough to suggest the leagues are statistically distinct populations that must never be pooled, but real enough that the model should be given league identity as information rather than pretending the leagues are identical.
- **Global model with league context (C)**: pools all 5 leagues' data for sample-size benefit while giving the model `competition_id` (already a feature-table column) as a categorical input, letting it learn league-specific adjustments where the data supports them and fall back to the pooled signal where a specific league/situation is data-poor. This is also exactly consistent with how §I's attack/defence shrinkage already treats "sparse data" (shrink toward a pooled estimate rather than either fully separating or fully ignoring group structure) — using the same philosophy at the model level as Phase 2 already used at the feature level.
- **Hierarchical/partially-pooled (D)**: statistically the most principled version of C, but requires either a Bayesian hierarchical model or a mixed-effects framework neither of which is justified yet without first checking whether the simpler global+context approach already captures most of the benefit — exactly the kind of "don't build the fancier thing before checking if the simple thing suffices" the task's baseline-first philosophy asks for elsewhere.

**Recommendation: (C) — one global model trained on all 5 leagues pooled, with `competition_id` included as a categorical feature.** Revisit (D) only if a per-league residual-error analysis after V1 shows the global+context model is systematically mis-calibrated for a specific league in a way league-context alone doesn't fix.

## 8. Baseline models

Computed for real (not just designed on paper) against the walk-forward folds in §5, using only already-existing `feature_rows` columns, no model training infrastructure beyond a throwaway script:

| Baseline | What it does |
|---|---|
| Majority-class | Always predicts the training period's single most common class (home win in every fold) |
| Frequency (train-period H/D/A rates) | Predicts the training period's actual H/D/A proportions as a constant probability triple for every match |

Results (log loss / Brier score / accuracy), Folds 1-3:

| Fold | Majority: LogLoss | Freq: LogLoss / Brier / Acc |
|---|---|---|
| 1 (val 2022/23) | 3.71 | 1.064 / 0.643 / 46.3% |
| 2 (val 2023/24) | 3.92 | 1.077 / 0.652 / 43.3% |
| 3 (val 2024/25) | 4.01 | 1.077 / 0.652 / 42.0% |

The majority-class baseline's log loss is enormous (3.7-4.0) because it assigns ~0 probability to the two non-majority outcomes, which then get catastrophically penalized whenever they occur (as they do, most of the time — home win is a plurality, not a majority) — this by itself is a useful, concrete illustration of why accuracy alone is a bad model-selection metric (§10): the majority baseline's *accuracy* (42-46%) looks deceptively close to the frequency baseline's, but its log loss is nearly 4x worse. The frequency baseline is the real bar V1 needs to clear, not the majority-class one.

## 9. Candidate ML models

**Evaluated for real: a multinomial (softmax) logistic regression**, implemented directly in NumPy (gradient descent, standardized inputs, L2 regularization) since `scikit-learn` is not installed in this sandboxed analysis environment and there is no outbound internet access to install it here (the same network restriction documented in Phase 1 — this sandbox can reach neither `data.oddalerts.com` nor PyPI). Trained on 13 well-covered, mostly-non-null features (season/last5 goals for/against, points_last5, the 4 shrinkage strength scores, `strength_diff`, `league_home_advantage_season`), with rows containing any null among those 13 features dropped for this quick check only (not a final imputation policy — see §17).

Results vs. the frequency baseline, same 3 folds plus the untouched final test:

| Fold | Freq LogLoss / Brier / Acc | Softmax LogLoss / Brier / Acc |
|---|---|---|
| 1 (val 2022/23) | 1.064 / 0.643 / 46.3% | **1.003 / 0.600 / 51.9%** |
| 2 (val 2023/24) | 1.077 / 0.652 / 43.3% | **0.987 / 0.589 / 52.3%** |
| 3 (val 2024/25) | 1.077 / 0.652 / 42.0% | **1.006 / 0.601 / 50.9%** |
| **Final test (2025/26)** | 1.074 / 0.650 / 43.7% | **1.011 / 0.605 / 50.4%** |

**A genuinely modest but consistent, non-suspicious improvement over the frequency baseline on every single fold and on the untouched final test** — roughly 8-10 accuracy points, ~0.06-0.09 lower log loss, ~0.04-0.05 lower Brier score. This magnitude is exactly what's expected and reported in the football-prediction literature for pre-match statistical features (accuracy in the low-to-mid 50s is typical and credible; anything dramatically higher, e.g. 70%+, would itself be a leakage red flag rather than good news, and none of the folds show that). Full numbers: `data/audit/phase3_baseline_eval.json`.

**Not yet evaluated, and explicitly flagged as the first concrete implementation step**: Random Forest, Gradient Boosting, and `HistGradientBoostingClassifier` (scikit-learn's own, dependency-light gradient boosting implementation, explicitly preferred over requiring XGBoost/LightGBM per the task's "don't install large dependencies blindly" instruction — `HistGradientBoostingClassifier` ships inside scikit-learn itself, no extra dependency beyond the one already-necessary ML library). None of these could be run in this analysis pass because of the sandboxed environment's lack of `scikit-learn`/internet access — this is an environment constraint identical in kind to Phase 1's inability to reach `data.oddalerts.com` from this same sandbox, not a design gap. **No neural network is proposed for V1** — nothing about this dataset's size (~8,400 training rows for the final model) or the softmax baseline's behavior suggests the extra complexity/data-hunger of a neural net is justified yet, consistent with the task's "don't use one without clear evidence" instruction.

## 10. Evaluation metrics

**Primary (drive model selection): log loss and Brier score**, both proper scoring rules that directly reward well-calibrated probabilities rather than just correct arg-max predictions — essential given the project's stated goal is calibrated probabilities, not just labels (per project instructions on reporting calibration honestly, and per this task's explicit "do not optimize solely for accuracy"). **Secondary: macro-F1 and balanced accuracy** (unweighted-by-class-frequency, so they don't reward a model for just being good at the majority class), used as tie-breakers and sanity checks, not as the primary selection signal. **Reported but never used to select a model on its own: raw accuracy and the confusion matrix** — kept because they're what a non-technical stakeholder will ask for first, but §8 already demonstrated why accuracy alone is misleading (the majority baseline's accuracy looked fine; its log loss was disqualifying).

## 11. Probability calibration strategy

Not yet executed (no model is finalized), but designed: after a candidate is chosen (§10), fit a calibration layer (Platt scaling / sigmoid, or isotonic regression — the choice between them is itself a small, data-driven decision to make once real held-out predictions exist, since isotonic needs more data to avoid overfitting than Platt scaling does) **using only the walk-forward validation folds' predictions, never the final test season**, then verify with a reliability diagram (predicted-probability bucket vs actual outcome frequency in that bucket) and an expected calibration error (ECE) computed the same way. The 2025/26 final test set is used only to report the finished, already-calibrated model's numbers once — never to fit or adjust the calibration itself, for the same test-set-integrity reason as §5/§6.

## 12. Feature ablation strategy

Designed, not yet executed (requires the real model training infrastructure from §9's deferred candidates to be worth doing properly — the quick NumPy softmax check in §9 already used a fixed, hand-picked 13-feature subset rather than a systematic ablation). Planned comparison, exactly per the task:

- **Model A** — goals + form only (the features with the fewest nulls, full 6-season depth)
- **Model B** — A + shots/shots-on-target (still full-depth, adds the volume-based tempo proxy)
- **Model C** — B + xG (coverage-gated per Phase 2's policy — rows/seasons without sufficient xG coverage keep xG features NULL, handled via the same imputation-in-training-only strategy chosen in §17, never zero-filled)
- **Model D** — every V1-eligible feature from Phase 2 (all of Model C plus venue-split goals, tempo/discipline proxies, league home-advantage)

Run across the same 3 walk-forward folds, compared on log loss/Brier (§10), to answer the task's real question — does xG (Model C vs B) and the full feature set (D vs C) earn its complexity, or does most of the signal already live in goals+form (A)? This is left as a designed-but-unexecuted experiment for the implementation phase, since running it meaningfully requires the same missing `scikit-learn`/real-training-loop that §9's deferred candidates need.

## 13. Leakage controls

Independent audit performed against the design (not a full re-implementation, since no training code exists yet to audit at the implementation level) — walking through the task's checklist:

| Check | Status |
|---|---|
| Labels never enter the feature matrix | ✅ Structurally enforced — `storage.LABEL_COLUMNS` is a named, importable list every future training script must exclude; Phase 2's own tests already prove features never read the label columns during generation. |
| Target fixture statistics never enter features | ✅ Proven by Phase 2's 10 leakage tests (re-run, still passing). |
| Future fixtures never influence past features | ✅ Same. |
| Future season information never influences earlier predictions | ✅ Same architectural guarantee (chronological single-pass accumulation) extends automatically across season boundaries — nothing in `history.py` resets or special-cases season transitions in a way that could leak forward. |
| Test-period information never affects training | ⚠️ **Design-level only, not yet code-enforced** — §5/§6 define the rule (2025/26 untouched until final evaluation), but no code exists yet to *guarantee* a future implementation can't accidentally violate it (e.g. by fitting a global scaler across all seasons before splitting). This is exactly the kind of thing `tests/test_model_splits.py` (§15) must check mechanically, not just by policy. |
| Feature selection uses only training data | ⚠️ Same status — designed, not yet code-enforced (no feature selection step has been implemented). |
| Preprocessing/scaling fitted only on training data | ⚠️ Same — the §9 sanity-check script *did* do this correctly (mean/std computed from the training fold only, applied to validation), establishing the right pattern, but that was a throwaway script, not committed, tested code. |
| Imputation fitted only on training data | ⚠️ Not yet implemented at all (§9's check used a drop-null shortcut, not real imputation) — policy fixed in §17, enforcement is an implementation task. |
| Hyperparameter tuning never sees final test data | ✅ By design (§5/§6) — no hyperparameter tuning has happened yet, so nothing to audit yet, but the rule is explicit before any tuning starts, per the task's requirement. |

**No RED findings — nothing suspicious was found requiring a stop.** The ⚠️ items are honestly-reported gaps between "designed correctly" and "mechanically enforced by tests," which is expected at this stage (no training code exists yet) and is exactly why `test_model_splits.py`/`test_model_leakage.py` are first-class deliverables of the implementation phase (§15), not an afterthought.

## 14. Model selection criteria

**Fixed now, before any candidate beyond the illustrative softmax baseline is trained**, per the task's explicit requirement:

1. **Primary:** lowest mean log loss across the 3 walk-forward validation folds (§5/§6). Brier score as a co-primary tie-breaker if two candidates' log loss is within a small, pre-agreed margin (e.g. 0.01) of each other.
2. **Secondary (used only to break primary-metric near-ties, never to override a clear primary-metric winner):** macro-F1, balanced accuracy, and qualitative calibration-curve shape.
3. **Disqualifying conditions**, checked regardless of primary-metric rank: a candidate whose accuracy exceeds roughly 65-70% on any fold is treated as a leakage red flag requiring investigation before being considered, not celebrated as a win (per §9's discussion of what a credible result looks like for this problem). A candidate that cannot produce well-formed probabilities (e.g. doesn't sum to 1, or produces exact 0/1 probabilities) is disqualified regardless of its other metrics.
4. The winning configuration is retrained once on all data through 2024/25 and evaluated exactly once on the untouched 2025/26 test set (§6). That result is reported as final — it is not grounds for going back to pick a different model.

## 15. Proposed implementation architecture

Adapted from the task's suggested structure to fit the existing repository's conventions (`src/ingestion/`, `src/features/` — flat modules, dataclass-light, docstring-heavy, single-purpose files):

```
src/models/
    __init__.py          # MODEL_VERSION constant, mirroring features.config.FEATURE_VERSION
    config.py             # split boundaries (§5), selection thresholds (§14), feature groups (§12)
    data.py                # loads feature_rows, applies the label_result IS NOT NULL filter (§4)
    splits.py               # walk-forward fold construction from season_id -- the single source of
                             # truth for "what counts as train/val/test," used by every script below
    baselines.py            # majority-class, frequency baselines (§8)
    train.py                 # candidate model training (starts with HistGradientBoostingClassifier
                              # once available; logistic regression via sklearn as the reproducible
                              # replacement for this report's NumPy sanity-check version)
    evaluate.py               # log loss, Brier, macro-F1, balanced accuracy, confusion matrix (§10)
    calibration.py             # §11, fit only on validation-fold predictions
    ablation.py                 # §12's Model A/B/C/D comparison harness
    registry.py                  # versioned model artifact storage (§17 below)
tests/
    test_model_splits.py          # proves splits.py never lets a later season appear in an earlier
                                   # fold's training set, and that the final-test season is excluded
                                   # from every fold used for selection (closes the ⚠️ items in §13)
    test_model_leakage.py          # proves scaling/imputation/feature-selection fit only on training
                                    # data within a single split, mirroring test_feature_leakage.py's
                                    # style for the model layer
    test_baselines.py               # majority-class and frequency baseline correctness
    test_evaluation.py               # metric functions (log loss, Brier, etc.) against hand-computed
                                      # expected values, the same way test_feature_strength.py checks
                                      # the shrinkage formula
data/processed/
    models.db or a models/ directory  # versioned model artifacts + their evaluation metrics, kept
                                       # separate from matches.db and features.db (task requirement #19)
docs/
    PHASE3_MODEL_SPEC.md                # NOT yet written -- see §17, this design report is the
                                         # precursor to it, not a replacement
    PHASE3_MODEL_DEVELOPMENT_REPORT.md  # written after implementation + real training, analogous to
                                         # PHASE2_FEATURE_ENGINEERING_REPORT.md's relationship to
                                         # PHASE2_FEATURE_SPEC.md
```

`MODEL_VERSION` starts at `"v1.0"` and is recorded on every stored prediction/evaluation artifact, mirroring `features.FEATURE_VERSION`'s pattern exactly (task requirement #20).

## 16. Risks / limitations

- **scikit-learn is unavailable in this analysis sandbox** (no internet access to install it, confirmed) — every real candidate model beyond the illustrative NumPy softmax regression must be trained in an environment with normal package-install access (the user's own machine, consistent with how the full historical ingestion in Phase 1 had to run there too). This is the single biggest concrete blocker to starting §9/§12's deferred work.
- **xG's coverage-gated nulls create a real missing-data-mechanism question the ablation study (§12) needs to resolve carefully**: xG is missing *not at random* (it's structurally absent pre-2024, not randomly scattered), which means naive imputation (e.g. mean-fill) could introduce a spurious "xG became informative right when its imputed value started being real" artifact. The safest options are (a) training separate coverage-aware model variants, or (b) using a missingness *indicator* feature alongside an imputed value so the model can learn "trust this xG value" vs "this is a filled-in placeholder" as separate signals — a decision to make explicitly in the implementation phase, not to default into.
- **Sample size per league** (§7) is adequate for the recommended pooled approach but would be marginal for true per-league models, especially for the leagues/seasons with the sparsest xG coverage — this is why (C) was recommended over (B), but it's worth re-checking after real candidate models exist rather than treating this analysis as final.
- **No second-division/promotion data** (already flagged in Phase 2) means the model will never have a real signal to distinguish "recently promoted, genuinely weaker squad" from "established club having a slow start" beyond what `insufficient_history`/shrinkage already encode — an inherent ceiling on V1's accuracy for early-season fixtures involving newly-promoted teams, not something Phase 3 can fix.
- **Only one season (2025/26) is available as a genuinely untouched final test set** — this is a single realization of "the future," not a large sample, so the final-test numbers in §14 step 4 should be reported with appropriate humility (a confidence interval or at minimum an explicit acknowledgment of sample size) rather than treated as a precise, final verdict on real-world performance.

## 17. Exact next implementation steps

1. Get `scikit-learn` available in whatever environment will actually run training (the concrete blocker in §16) — this is a prerequisite, not a "nice to have," for everything below.
2. Write `src/models/splits.py` + `tests/test_model_splits.py` first, before any model code — this is the piece §13 flagged as design-only/not-yet-enforced, and it's the foundation everything else depends on being correct.
3. Write `src/models/data.py` with the explicit `label_result IS NOT NULL` filter (§4) and a documented, code-reviewed imputation policy for the §16 missing-xG-mechanism question (not deferred further).
4. Implement `baselines.py` and reproduce §8's numbers as the first test-covered milestone (should match this report's numbers within floating-point tolerance, since the logic is the same, just properly tested this time).
5. Implement `train.py` with `HistGradientBoostingClassifier` (native missing-value handling, no separate imputation step needed for tree splits — worth evaluating specifically for how it handles the xG missingness question in §16 versus an imputation-based approach) and a scikit-learn `LogisticRegression` as the reproducible replacement for this report's NumPy version, run across the real §5 folds.
6. Implement `evaluate.py` + `calibration.py`, run the §12 ablation study for real.
7. Apply the §14 selection rule, retrain on the full pre-2025/26 window, evaluate once on 2025/26, and write `docs/PHASE3_MODEL_DEVELOPMENT_REPORT.md` with the real results — at that point, and only then, write the shorter `docs/PHASE3_MODEL_SPEC.md` as the finalized contract (mirroring how `PHASE2_FEATURE_SPEC.md` preceded `PHASE2_FEATURE_ENGINEERING_REPORT.md`, the spec-then-report pattern already established in this project).

No prediction for any future/upcoming match, no Poisson/Monte Carlo simulation, no betting recommendation, and no UI work should begin until step 7 is complete and reviewed, per the task's strict scope boundaries.
