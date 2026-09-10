"""POISSON V2 -- STEP 4: VENUE-FEATURE DESIGN AUDIT (read-only, NOT the experiment).

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_step_poisson_4_venue_design_audit.py

WHAT THIS IS
The audit and PRE-REGISTRATION for the next controlled experiment. It traces
the venue features' construction and chronology, measures their coverage and
redundancy, and prints the experiment specification and decision rule so both
are fixed BEFORE any fitting.

WHAT THIS IS NOT
It does not fit anything. No PoissonRegressor, no LogisticRegression, no
preprocessing fitted on any partition, no lambda, no probability, no metric
comparison. The experiment itself is a separate STEP 5 script.

Writes nothing. No production source, database or artifact is touched.
2025/26 is never read -- not even for coverage.

------------------------------------------------------------------------------
THE BASELINE HAS MOVED
------------------------------------------------------------------------------
STEP 3 returned a pre-registered BETTER verdict:

    V1 (walk-forward)          pooled log loss 0.9993791057   Brier 0.5965017
    Poisson + existing 80      pooled log loss 0.9928234559   Brier 0.5919683
    log loss improved 3/3 folds

So the incumbent to beat is now POISSON + THE EXISTING 80 COLUMNS, not V1. V1
remains a secondary reference only. An experiment that beats V1 but not the
Poisson baseline is NOT an improvement.

------------------------------------------------------------------------------
A CAVEAT ABOUT THIS BEING THE SECOND EXPERIMENT -- STATED UP FRONT
------------------------------------------------------------------------------
STEP 3 was the first thing tried and it won. This is the second. Running
candidate changes sequentially against the SAME three folds and stopping at the
first that improves inflates the chance of a spurious win, and three folds
support ranges only -- never a standard deviation, never a significance claim.

That is a reason to state the risk plainly, not a reason to skip the
experiment. Two mitigations are built into the design:
  * the decision rule is fixed here, before fitting, and is not re-openable;
  * the fold-level deltas are reported individually, so a pooled improvement
    driven entirely by one fold is visible rather than hidden in the mean.
A second BETTER verdict on the same three folds is weaker evidence than the
first one was, and any write-up must say so.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---- pinned reference values (asserted at experiment time, not here) --------
V1_LOG_LOSS = 0.9993791056968738
V1_BRIER = 0.5965016957578898
POISSON_BASE_LOG_LOSS = 0.9928234559
POISSON_BASE_BRIER = 0.5919683

EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
ARTIFACT_MD5 = "5e504427712b35778bb8a62a8496c7cd"
TRAIN_PY_MD5 = "21425459195311492f49e73f5ae38fe0"
REDUNDANCY_THRESHOLD = 0.90        # the D-39/D-40 project convention, reused

#: The four VALUE columns. The matching `_n` sample-size columns are
#: deliberately EXCLUDED -- see the pre-registration section for why.
VENUE_VALUE_COLUMNS = (
    "home_goals_for_home_venue_season",
    "home_goals_against_home_venue_season",
    "away_goals_for_away_venue_season",
    "away_goals_against_away_venue_season",
)
VENUE_N_COLUMNS = tuple(c + "_n" for c in VENUE_VALUE_COLUMNS)

PRODUCTION_ASSUMPTIONS = (
    'SimpleImputer(strategy="median")',
    "StandardScaler()",
    "np.hstack([scaled, onehot])",
    "self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]",
    "LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
)

#: The venue construction chain, each link a literal from production source.
VENUE_CHAIN = (
    ("rolling.py", 'season_matches = [m for m in history if m["season_id"] == target_season_id]',
     "restricts to the CURRENT season, from history only"),
    ("rolling.py", 'venue_matches = [m for m in season_matches if m["is_home"] == home_only]',
     "restricts to the team's own home (or away) matches"),
    ("rolling.py", "mean, n, coverage_n = windowed_mean(venue_matches, extractor)",
     "aggregates with the shared windowed_mean"),
    ("rolling.py", "value = mean if coverage_n >= VENUE_MIN_COVERAGE else None",
     "emits NULL below the minimum-coverage gate rather than a thin estimate"),
    ("feature_builder.py", 'EXTRACTORS["goals_for"]', "goals-for channel"),
    ("feature_builder.py", 'EXTRACTORS["goals_against"]', "goals-against channel"),
    ("feature_builder.py", "_venue_features(home_history, season_id, \"home\", home_only=True)",
     "home team measured at HOME"),
    ("feature_builder.py", "_venue_features(away_history, season_id, \"away\", home_only=False)",
     "away team measured AWAY"),
)


def rule(t): print("\n" + "=" * 126); print(t); print("=" * 126)
def sub(t): print("\n" + "-" * 126); print(t); print("-" * 126)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def abort(msg):
    print("\n" + "!" * 126)
    print("STEP 4 ABORTED -- no experiment is designed")
    print(msg)
    print("!" * 126)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        abort(f"stop condition: {label}")


def snapshot():
    snap = {}
    for rel in EXPECTED_MD5:
        snap[rel] = md5(REPO / rel)
    for rel in ("src/models/train.py", "src/models/config.py",
                "src/features/feature_builder.py", "src/features/rolling.py",
                "src/features/history.py"):
        snap[rel] = md5(REPO / rel)
    art = REPO / "data/models/v1_logreg.pkl"
    if art.exists():
        snap["data/models/v1_logreg.pkl"] = md5(art)
    for rel in json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                          .read_text())["locked_input_checksums"]:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


def main():
    # ================================================================= RULE 1
    rule("RULE 1 -- PRODUCTION SOURCE TRACE (source text, never memory)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:76]}", tok in tsrc)

    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.config import (
        FINAL_TEST_SEASONS, MODEL_VERSION, RECOMMENDED_CONTEXT_FEATURE,
        REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS, WALK_FORWARD_FOLDS,
        X_EXCLUDED_COLUMNS,
    )
    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    check("production contract == exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("numeric production features == exactly 79", len(numeric) == 79)
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")

    rule("STEP 0 -- INTEGRITY GATE")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  pandas {pd.__version__}")
    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(f"{rel} unchanged", before[rel] == exp, before[rel])
    check("src/models/train.py unchanged", before["src/models/train.py"] == TRAIN_PY_MD5)
    check("v1_logreg.pkl unchanged", before.get("data/models/v1_logreg.pkl") == ARTIFACT_MD5)
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    bad = [r for r, v in pins.items() if before[f"pin:{r}"] != v["expected"]]
    check(f"{len(pins)}/{len(pins)} LOCKED_INPUTS unchanged", not bad, str(bad))

    # =================================================================== 4A
    rule("STEP 4A -- SOURCE CONSTRUCTION AND CHRONOLOGY TRACE")
    fb = (REPO / "src/features/feature_builder.py").read_text(encoding="utf-8")
    rl = (REPO / "src/features/rolling.py").read_text(encoding="utf-8")
    hs = (REPO / "src/features/history.py").read_text(encoding="utf-8")
    srcs = {"feature_builder.py": fb, "rolling.py": rl, "history.py": hs}
    for fname, token, meaning in VENUE_CHAIN:
        check(f"{fname}: {meaning}", token in srcs[fname], f"[{token[:58]}]")

    print("\n  FULL CONSTRUCTION PATH, current fixture -> feature:")
    print("    1. fixtures loaded ORDER BY unix ASC, fixture_id ASC")
    print("    2. for fixture F: build_feature_row(F, ctx) runs BEFORE ctx.record(F)")
    print("    3. history = ctx.history_before(team_id)          <- past only")
    print("    4. filter: m['season_id'] == target_season_id      <- current season")
    print("    5. filter: m['is_home'] == home_only               <- venue-restricted")
    print("    6. windowed_mean(venue_matches, goals_for/goals_against)")
    print("    7. value = mean if coverage_n >= VENUE_MIN_COVERAGE else None")
    print("\n  SEMANTICS (they are not symmetric, and that is deliberate):")
    print("    home_*_home_venue_season : the HOME team's record IN ITS OWN HOME matches")
    print("    away_*_away_venue_season : the AWAY team's record IN ITS AWAY matches")
    print("  i.e. each side is measured in the venue role it will actually occupy.")

    # =================================================================== 4C
    rule("STEP 4C -- NO CURRENT-FIXTURE / POST-MATCH INFORMATION")
    tree = ast.parse(fb)
    walker = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_walk_and_build"), None)
    check("the single chronological walker exists", walker is not None)
    loop = next((n for n in ast.walk(walker) if isinstance(n, ast.For)), None)
    build_line = record_line = None
    for node in ast.walk(loop):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "build_feature_row":
                build_line = node.lineno
            if isinstance(fn, ast.Attribute) and fn.attr == "record":
                record_line = node.lineno
    check("build_feature_row precedes ctx.record in the same iteration",
          build_line is not None and record_line is not None and build_line < record_line,
          f"build@{build_line} < record@{record_line}")
    n_record = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == "record")
    check("exactly one ctx.record call site (no second write path)", n_record == 1)

    # The venue helper's only data input is `history`; it cannot see the fixture.
    vfun = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "_venue_features"), None)
    check("_venue_features exists", vfun is not None)
    vargs = [a.arg for a in vfun.args.args]
    print(f"  _venue_features parameters: {vargs}")
    check("_venue_features receives NO fixture object and no goals argument",
          "fixture" not in vargs and not any("goal" in a for a in vargs), str(vargs))
    vsrc = ast.get_source_segment(fb, vfun) or ""
    for banned in ("fixture", "label_", "home_goals", "away_goals"):
        check(f"_venue_features body never references `{banned}`", banned not in vsrc)
    print("\n  => the venue features are a function of ctx.history_before() ONLY.")
    print("     The current fixture's goals are appended by ctx.record() AFTER the")
    print("     row is built, so they cannot enter. Same guarantee as the 79")
    print("     contract features, from the same single walker.")

    # =================================================================== 4B
    rule("STEP 4B -- EXACT COLUMNS AND PER-FOLD COVERAGE")
    fdb = REPO / "data/processed/features.db"
    con = sqlite3.connect(f"file:{fdb.resolve()}?mode=ro", uri=True)
    try:
        all_cols = [r[1] for r in con.execute("PRAGMA table_info(feature_rows)")]
        test_ids = set()
        for s in FINAL_TEST_SEASONS:
            test_ids.update(SEASON_NAME_TO_IDS[s])
        train_all = sorted({i for f in WALK_FORWARD_FOLDS
                            for s in (f.train_seasons + f.validation_seasons)
                            for i in SEASON_NAME_TO_IDS[s]})
        check("the fold season universe excludes 2025/26",
              not (set(train_all) & test_ids))
        q = ",".join(str(i) for i in train_all)
        df = pd.read_sql_query(
            f"SELECT * FROM feature_rows WHERE season_id IN ({q}) "
            f"AND label_result IS NOT NULL", con)
    finally:
        con.close()
    check("no 2025/26 row entered the audit frame",
          not set(df["season_id"]) & test_ids)
    print(f"  rows in the fold universe (2025/26 excluded): {len(df)}")

    for c in VENUE_VALUE_COLUMNS + VENUE_N_COLUMNS:
        check(f"column exists in features.db: {c}", c in all_cols)
    check("no venue column is in the frozen 80-column contract",
          not (set(VENUE_VALUE_COLUMNS) & set(MODEL_B_COLUMNS)))
    check("no venue column is an excluded/label column",
          not (set(VENUE_VALUE_COLUMNS) & set(X_EXCLUDED_COLUMNS)))

    print("\n| column | non-null | coverage | mean | sd | min | max |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for c in VENUE_VALUE_COLUMNS:
        s = df[c].astype(float)
        print(f"| {c} | {int(s.notna().sum())} | {s.notna().mean():.4f} | {s.mean():.4f} "
              f"| {s.std():.4f} | {s.min():.4f} | {s.max():.4f} |")
    all4 = df[list(VENUE_VALUE_COLUMNS)].notna().all(axis=1)
    print(f"\n  rows with ALL FOUR venue values present: {int(all4.sum())} "
          f"({all4.mean():.4f})")

    sub("PER-FOLD COVERAGE (train and validation partitions separately)")
    print("| fold | partition | n | all-4 coverage | " +
          " | ".join(c.replace("_season", "").replace("_venue", "") for c in VENUE_VALUE_COLUMNS)
          + " |")
    print("|---|---|---:|---:|" + "---:|" * 4)
    for f in WALK_FORWARD_FOLDS:
        for part, seasons in (("train", f.train_seasons), ("validation", f.validation_seasons)):
            ids = {i for s in seasons for i in SEASON_NAME_TO_IDS[s]}
            m = df["season_id"].isin(ids)
            d = df[m]
            cov4 = d[list(VENUE_VALUE_COLUMNS)].notna().all(axis=1).mean()
            per = " | ".join(f"{d[c].notna().mean():.4f}" for c in VENUE_VALUE_COLUMNS)
            print(f"| {f.name} | {part} | {int(m.sum())} | {cov4:.4f} | {per} |")

    sub("WHY THE MISSING ~12% IS MISSING (this shapes the design)")
    print("  `value = mean if coverage_n >= VENUE_MIN_COVERAGE else None` -- a team")
    print("  with fewer than the minimum prior SAME-VENUE matches in the CURRENT")
    print("  season gets NULL. Missingness is therefore concentrated in early")
    print("  matchdays, by construction, not at random.")
    nvals = df[VENUE_N_COLUMNS[0]].astype(float)
    print("\n| prior same-venue matches (home team) | rows | value present |")
    print("|---:|---:|---:|")
    for k in range(0, 6):
        m = nvals == k
        if m.sum():
            print(f"| {k} | {int(m.sum())} | {df.loc[m, VENUE_VALUE_COLUMNS[0]].notna().mean():.4f} |")
    m = nvals >= 6
    if m.sum():
        print(f"| >=6 | {int(m.sum())} | {df.loc[m, VENUE_VALUE_COLUMNS[0]].notna().mean():.4f} |")
    print("\n  CONSEQUENCE, stated rather than hidden: the production preprocessing")
    print("  imputes NULLs with the TRAINING-partition median. About 12% of rows")
    print("  will therefore carry a league-typical constant instead of a team value,")
    print("  which ATTENUATES the feature exactly where teams differ most (early")
    print("  season). The experiment keeps that behaviour unchanged -- it is V1's")
    print("  existing imputation, and changing it would confound this test.")

    # ============================================== redundancy (pre-fit, descriptive)
    rule("STEP 4B2 -- REDUNDANCY AGAINST THE EXISTING 80 COLUMNS (pre-fit, descriptive)")
    print(f"  Threshold |r| >= {REDUNDANCY_THRESHOLD} is the D-39/D-40 project convention,")
    print("  reused unchanged. Correlation is ASSOCIATION only -- never importance.")
    print("  A venue column that merely restates an existing column cannot add")
    print("  information, so this measurement sets the prior expectation BEFORE")
    print("  any result is seen.")
    print("\n| venue column | closest existing contract column | r | |r| >= threshold? |")
    print("|---|---|---:|---|")
    worst = 0.0
    for c in VENUE_VALUE_COLUMNS:
        x = df[c].astype(float).to_numpy()
        best, bestr = None, 0.0
        for o in numeric:
            y = df[o].astype(float).to_numpy()
            m = ~(np.isnan(x) | np.isnan(y))
            if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
                continue
            r = float(np.corrcoef(x[m], y[m])[0, 1])
            if abs(r) > abs(bestr):
                best, bestr = o, r
        worst = max(worst, abs(bestr))
        print(f"| {c} | {best} | {bestr:+.4f} | "
              f"{'YES -- redundant' if abs(bestr) >= REDUNDANCY_THRESHOLD else 'no'} |")
    print(f"\n  highest |r| against any existing contract column: {worst:.4f}")
    print("  This is reported, not acted on. No column is dropped, merged or")
    print("  reweighted, and a high value does NOT cancel the experiment -- it")
    print("  simply lowers the prior odds of a real gain.")

    sub("ASSOCIATION WITH THE POISSON TARGETS (descriptive, no fitting)")
    hg = df["label_home_goals"].astype(float).to_numpy()
    ag = df["label_away_goals"].astype(float).to_numpy()
    print("| venue column | r with home goals | r with away goals |")
    print("|---|---:|---:|")
    for c in VENUE_VALUE_COLUMNS:
        x = df[c].astype(float).to_numpy()
        out = []
        for y in (hg, ag):
            m = ~np.isnan(x)
            out.append(float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 2 else float("nan"))
        print(f"| {c} | {out[0]:+.4f} | {out[1]:+.4f} |")
    print("\n  These are targets of the two Poisson arms, so association here is the")
    print("  relevant descriptive quantity. It is NOT evidence of improvement:")
    print("  the existing 80 columns already correlate with the same targets and")
    print("  the model already consumes them.")

    # =================================================================== 4D/4E
    rule("STEP 4D/4E -- PRE-REGISTERED EXPERIMENT SPECIFICATION (fixed before fitting)")
    print("  ARMS (exactly two; both fitted identically):")
    print("    BASELINE : two-Poisson on the frozen 80 columns          [the incumbent]")
    print("    CANDIDATE: two-Poisson on the 80 columns + the 4 venue value columns")
    print()
    print("  CANDIDATE INPUT SET -- 84 columns, enumerated and closed:")
    for i, c in enumerate(VENUE_VALUE_COLUMNS, 1):
        print(f"    + {i}. {c}")
    print("\n  EXCLUDED ON PURPOSE, and why:")
    print("    - the four *_n sample-size columns: they encode HOW MUCH history")
    print("      exists, not WHAT it says. Including them would test two different")
    print("      ideas at once and make a result unattributable. Candidate for a")
    print("      later, separate experiment; not this one.")
    print("    - xG, rest days, congestion, head-to-head, Dixon-Coles, bivariate")
    print("      Poisson, alpha tuning, calibration, any other column: OUT OF SCOPE.")
    print()
    print("  HELD IDENTICAL BETWEEN ARMS (the whole point):")
    print("    estimator      : PoissonRegressor(alpha=1.0, max_iter=2000)")
    print("    preprocessing  : production LogisticRegressionPreprocessor, fitted")
    print("                     inside each fold's TRAIN partition only")
    print("    targets        : label_home_goals, label_away_goals")
    print("    conversion     : tail-safe independent-Poisson scoreline -> H/D/A")
    print("                     (adaptive grid, P(A) by complement, as fixed in STEP 3)")
    print("    folds          : the same three from config.WALK_FORWARD_FOLDS")
    print("    metrics        : identical")
    print("    seed/ordering  : identical; no randomness anywhere")
    print("  The ONLY difference between the arms is the four extra columns.")
    print()
    print("  REFERENCE VALUES TO BE ASSERTED AT EXPERIMENT TIME (not assumed here):")
    print(f"    V1 walk-forward   log loss {V1_LOG_LOSS:.10f}   Brier {V1_BRIER:.10f}")
    print(f"    Poisson baseline  log loss {POISSON_BASE_LOG_LOSS:.10f}   "
          f"Brier {POISSON_BASE_BRIER:.10f}")
    print("    The STEP 5 script MUST recompute the baseline arm and hard-fail if it")
    print("    does not reproduce these, so the comparison cannot drift silently.")

    sub("PRE-REGISTERED DECISION RULE -- FIXED NOW, NOT RE-OPENABLE")
    print("  Comparison is CANDIDATE vs POISSON BASELINE. (V1 is reported as a")
    print("  secondary reference and cannot promote anything.)")
    print()
    print("    BETTER  requires ALL THREE:")
    print("      1. pooled log loss (candidate) < pooled log loss (baseline)   [PRIMARY]")
    print("      2. log loss improves on >= 2 of 3 folds")
    print("      3. pooled Brier does not materially contradict, i.e.")
    print("         Brier(candidate) <= Brier(baseline)                        [SECONDARY]")
    print()
    print("    WORSE   if pooled log loss AND pooled Brier are both worse.")
    print("    INCONCLUSIVE otherwise.")
    print()
    print("  Draw-vs-rest AUC, Draw argmax share, mean/sd P(D) and P(D)>1/3 are")
    print("  REPORTED for every arm but CANNOT promote the candidate.")
    print()
    print("  STEP 8 DRAW FAILURE SIGNATURE -- explicit test, must be run and named:")
    print("    if Draw argmax share INCREASES while Draw-vs-rest AUC is FLAT OR")
    print("    FALLS, the run must print 'STEP 8 FAILURE SIGNATURE RECURRED' and")
    print("    the candidate must NOT be described as more balanced or improved.")
    print("    (STEP 8 precedent: argmax 0.0240 -> 0.1313 while AUC 0.5361 ->")
    print("     0.5349 and log loss worsened by +0.0897.)")
    print()
    print("  No threshold is defined for 'materially'. Condition 3 is a plain")
    print("  inequality precisely so it cannot be re-interpreted after the fact.")

    sub("SAFETY CONDITIONS THE STEP 5 SCRIPT MUST ENFORCE")
    for i, s in enumerate((
        "2025/26 appears in no partition of any fold",
        "train and validation fixture sets are disjoint per fold",
        "validation is strictly later than train by unix",
        "preprocessing fitted on TRAIN partitions only, never on validation",
        "no production source, database or artifact is written",
        "the frozen v1_logreg.pkl is not replaced or modified",
        "no feature column is written to features.db (the 4 columns already exist)",
        "lambdas finite and strictly positive",
        "scoreline residual mass < 1e-12 and P(A) complement control < 1e-10",
        "every probability row sums to 1",
        "the baseline arm reproduces the pinned Poisson reference values",
    ), 1):
        print(f"    {i:2d}. {s}")

    # ============================================================== SUMMARY
    rule("WHAT THIS EXPERIMENT WILL AND WILL NOT ANSWER")
    print("  ANSWERS: with the estimator, preprocessing, targets, conversion and")
    print("  folds all held identical, do these four venue columns improve pooled")
    print("  log loss over the current Poisson baseline?")
    print()
    print("  DOES NOT ANSWER:")
    print("    - whether venue information helps in general (one construction, one")
    print("      season window, one minimum-coverage gate)")
    print("    - whether the *_n columns would help")
    print("    - whether a different imputation of the missing 12% would help")
    print("    - anything about 2025/26 or about generalisation to future seasons")
    print("    - anything with statistical significance: three folds give ranges,")
    print("      never a standard deviation, never a p-value")

    # ============================================================= INTEGRITY
    rule("POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified anywhere", not changed, str(changed))
    print("    models fitted      : NONE      preprocessing fitted : NONE")
    print("    probabilities      : NONE      metrics compared     : NONE")
    print("    production changes : NONE      artifact replaced    : NONE")
    print("    database writes    : NONE      files created        : NONE")
    print("    2025/26 access     : NONE")
    for k in ("data/processed/features.db", "data/models/v1_logreg.pkl",
              "src/models/train.py", "src/features/rolling.py"):
        print(f"    {k:38s} {after[k]}")
    print(f"    LOCKED_INPUTS                          {len(pins)}/{len(pins)} unchanged")

    print("\n" + "=" * 126)
    print("STEP 4 COMPLETE -- DESIGN AND PRE-REGISTRATION ONLY")
    print("=" * 126)
    print("No experiment has been run. The decision rule above is now fixed and")
    print("must not be changed after seeing any result. STEP 5 is a separate script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
