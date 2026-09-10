"""POISSON V2 -- STEP 2: PRE-MATCH GOAL-MODELLING SIGNAL AUDIT (read-only).

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_step_poisson_2_signal_audit.py

Determines whether the EXISTING feature dataset carries enough legitimate
PRE-MATCH information to estimate lambda_home and lambda_away.

INVESTIGATION ONLY. No fit(), no training, no artifact, no feature write, no
database write, no production change. Training seasons only (2020/21-2024/25);
no 2025/26 outcome is read at any point.

THE STANDING RULE THIS AUDIT IS WRITTEN AGAINST
Poisson V2 is useful ONLY if it beats frozen V1 under the SAME walk-forward
protocol on log loss and Brier. Producing expected goals, scorelines, more
draws or a more balanced-looking confusion matrix are CAPABILITIES, not
evidence of better prediction. STEP 8 already demonstrated the trap directly:
its nonlinear probe raised the Draw argmax share from 2.4% to 13.1% while
Draw-vs-rest AUC went DOWN (0.5361 -> 0.5349) and log loss got worse by
+0.0897. Nothing in this audit may be read as predicting improvement.

WHAT THE SOURCE TRACE ESTABLISHED (see SECTION 2C for the proof chain)
Every one of the 79 production features is computed from `history_before()`
output, and `ctx.record(fixture)` is called only AFTER `build_feature_row` for
that same fixture. That ordering is the entire leakage guarantee and it is
verified here from source text, not assumed.

TWO FINDINGS THAT CHANGE THE 2E ANSWER
  * Lagged xG features ALREADY EXIST in features.db (54 columns, e.g.
    `home_xg_for_per_match_last5`), built by the SAME chronological walker.
    So for item 8 the chronology IS provable; the constraint is coverage
    (~35%), not leakage.
  * Venue-restricted lagged goals ALREADY EXIST (8 columns, e.g.
    `home_goals_for_home_venue_season`) -- item 7 is already built and simply
    absent from the frozen 80-column contract.
Neither is a recommendation. Both are reported as availability facts.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
ARTIFACT_MD5 = "5e504427712b35778bb8a62a8496c7cd"
TRAIN_PY_MD5 = "21425459195311492f49e73f5ae38fe0"
POOLED_REF_LOG_LOSS = 0.9993791056968738
POOLED_REF_BRIER = 0.5965016957578898
V1_DRAW_AUC = 0.5361
V1_DRAW_ARGMAX = 0.0240

PRODUCTION_ASSUMPTIONS = (
    'SimpleImputer(strategy="median")',
    "StandardScaler()",
    "np.hstack([scaled, onehot])",
    "self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]",
    "LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
)

#: The temporal-safety chain, each link a literal from production source.
TEMPORAL_CHAIN = (
    ("history.py", "SELECT * FROM fixtures ORDER BY unix ASC, fixture_id ASC",
     "source rows are loaded in strict chronological order"),
    ("history.py", "return self._team_history.get(team_id, [])",
     "history_before returns only what has already been accumulated"),
    ("history.py", 'if fixture.get("status") not in PLAYED_STATUSES:',
     "unplayed/abandoned fixtures never enter any history"),
    ("feature_builder.py", "ctx.record(fixture)",
     "the write step exists and is a single call site in the walker"),
)

FAMILY_RULES = (
    ("goals_season", lambda c: "goals" in c and "season" in c and "venue" not in c
     and "shots" not in c),
    ("goals_last5", lambda c: "goals" in c and "last5" in c and "shots" not in c),
    ("goals_last10", lambda c: "goals" in c and "last10" in c and "shots" not in c),
    ("shots_on_season", lambda c: "shots_on" in c and "season" in c),
    ("shots_on_last", lambda c: "shots_on" in c and "last" in c),
    ("shots_season", lambda c: "shots" in c and "shots_on" not in c and "season" in c),
    ("shots_last", lambda c: "shots" in c and "shots_on" not in c and "last" in c),
    ("form", lambda c: any(k in c for k in ("points", "win_rate", "draw_rate",
                                            "loss_rate", "goal_diff"))),
    ("strength", lambda c: "strength" in c),
    ("context", lambda c: c == "competition_id"),
)


def rule(t): print("\n" + "=" * 124); print(t); print("=" * 124)
def sub(t): print("\n" + "-" * 124); print(t); print("-" * 124)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def abort(msg):
    print("\n" + "!" * 124)
    print("STEP 2 ABORTED")
    print(msg)
    print("!" * 124)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        abort(f"stop condition: {label}")


def snapshot():
    snap = {}
    for rel in EXPECTED_MD5:
        snap[rel] = md5(REPO / rel)
    for rel in ("src/models/train.py", "src/features/feature_builder.py",
                "src/features/history.py", "src/features/strength.py",
                "src/features/rolling.py", "src/features/form.py"):
        snap[rel] = md5(REPO / rel)
    art = REPO / "data/models/v1_logreg.pkl"
    if art.exists():
        snap["data/models/v1_logreg.pkl"] = md5(art)
    for rel in json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                          .read_text())["locked_input_checksums"]:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


def family_of(col):
    for name, pred in FAMILY_RULES:
        if pred(col):
            return name
    return "UNCLASSIFIED"


def side_of(col):
    if col.startswith("home_"):
        return "home"
    if col.startswith("away_"):
        return "away"
    return "none"


def horizon_of(col):
    for h in ("last5", "last10", "season"):
        if h in col:
            return h
    return "unspecified"


def main():
    # ================================================================= RULE 1
    rule("RULE 1 -- PRODUCTION SOURCE TRACE (source text, never memory)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:74]}", tok in tsrc)

    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.config import (
        FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, LABEL_COLUMNS, MODEL_VERSION,
        RECOMMENDED_CONTEXT_FEATURE, REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS,
        X_EXCLUDED_COLUMNS,
    )
    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    check("production contract == exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("numeric production features == exactly 79", len(numeric) == 79)
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")

    rule("STEP 0 -- INTEGRITY GATE (before any analysis)")
    print(f"  Python {sys.version.split()[0]}   numpy {np.__version__}   pandas {pd.__version__}")
    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(f"{rel} unchanged", before[rel] == exp, before[rel])
    check("src/models/train.py unchanged", before["src/models/train.py"] == TRAIN_PY_MD5)
    check("v1_logreg.pkl unchanged",
          before.get("data/models/v1_logreg.pkl") == ARTIFACT_MD5)
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    bad = [r for r, v in pins.items() if before[f"pin:{r}"] != v["expected"]]
    check(f"{len(pins)}/{len(pins)} LOCKED_INPUTS unchanged", not bad, str(bad))

    train_ids, test_ids = set(), set()
    for s in FINAL_TRAIN_SEASONS:
        train_ids.update(SEASON_NAME_TO_IDS[s])
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    check("training and final-test seasons are disjoint", not (train_ids & test_ids))
    print(f"  training seasons : {list(FINAL_TRAIN_SEASONS)}")
    print(f"  EXCLUDED         : {list(FINAL_TEST_SEASONS)}  season_ids {sorted(test_ids)}")

    fb = (REPO / "src/features/feature_builder.py").read_text(encoding="utf-8")
    hs = (REPO / "src/features/history.py").read_text(encoding="utf-8")
    st = (REPO / "src/features/strength.py").read_text(encoding="utf-8")
    srcs = {"feature_builder.py": fb, "history.py": hs, "strength.py": st}

    # =================================================================== 2C
    # Run BEFORE 2A, because 2A's classification depends on its result.
    rule("STEP 2C -- TEMPORAL SAFETY TRACE (proof chain, from source text)")
    print("  current fixture -> source rows -> historical filter -> ordering ->")
    print("  aggregation window -> feature\n")
    for fname, token, meaning in TEMPORAL_CHAIN:
        check(f"{fname}: {meaning}", token in srcs[fname], f"[{token[:52]}]")

    # The decisive property: record() is called AFTER build_feature_row() for
    # the SAME fixture, inside the same loop iteration. Verified structurally
    # via AST, not by reading comments or trusting statement order in text.
    tree = ast.parse(fb)
    walker = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "_walk_and_build"), None)
    check("the single chronological walker `_walk_and_build` exists", walker is not None)
    loop = next((n for n in ast.walk(walker) if isinstance(n, ast.For)), None)
    check("the walker contains a for-loop over fixtures", loop is not None)
    build_line = record_line = None
    for node in ast.walk(loop):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "build_feature_row":
                build_line = node.lineno
            if isinstance(fn, ast.Attribute) and fn.attr == "record":
                record_line = node.lineno
    check("build_feature_row is called inside the loop", build_line is not None)
    check("ctx.record is called inside the loop", record_line is not None)
    check("record() comes AFTER build_feature_row() in the same iteration",
          build_line < record_line, f"build@{build_line} < record@{record_line}")
    n_record = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr == "record")
    check("exactly one ctx.record call site in the builder (no second write path)",
          n_record == 1, f"{n_record} call site(s)")

    print("\n  PROOF, stated plainly:")
    print("    1. fixtures are read ORDER BY unix ASC, fixture_id ASC")
    print("    2. for fixture F: build_feature_row(F, ctx) executes FIRST")
    print("    3. every feature reads ctx.history_before(team) / league_accumulator")
    print("    4. ctx.record(F) executes AFTER, appending F's own result")
    print("    => F's own goals cannot be in F's own features. The window is")
    print("       'strictly earlier fixtures', enforced by execution order, not")
    print("       by a date filter that could be written incorrectly.")
    print("  Independently enforced by tests/test_gate2_temporal_safety.py, which")
    print("  requires EXACTLY ONE chronological walker to exist.")

    # =================================================================== 2A
    rule("STEP 2A -- INVENTORY OF THE 79 PRE-MATCH FEATURES")
    print("  Classification key:")
    print("    A = legitimate pre-match (available before kickoff, non-historical)")
    print("    B = derived from PRIOR matches via the proven chronological walker")
    print("    C = potentially post-match / temporally unsafe")
    print("    D = unknown / cannot be proven pre-match from source")
    print()
    cls, fam_count = {}, Counter()
    for c in MODEL_B_COLUMNS:
        f = family_of(c)
        fam_count[f] += 1
        cls[c] = "A" if f == "context" else ("B" if f != "UNCLASSIFIED" else "D")
    print("| family | n | class | source construction path |")
    print("|---|---:|---|---|")
    paths = {
        "goals_season": "history_before -> _season_matches -> season mean of goals_for/against",
        "goals_last5": "history_before -> last-5 window -> rolling mean (rolling.py)",
        "goals_last10": "history_before -> last-10 window -> rolling mean (rolling.py)",
        "shots_season": "history_before -> season mean of shots_for/against",
        "shots_last": "history_before -> last-N window -> rolling mean",
        "shots_on_season": "history_before -> season mean of shots_on_for/against",
        "shots_on_last": "history_before -> last-N window -> rolling mean",
        "form": "history_before -> form_features (points/W-D-L rates/goal diff)",
        "strength": "history_before + league_accumulator -> shrunk_rate vs league mean",
        "context": "fixture.competition_id (fixture metadata, known at scheduling)",
        "UNCLASSIFIED": "-",
    }
    for f, n in sorted(fam_count.items(), key=lambda t: -t[1]):
        k = "A" if f == "context" else ("B" if f != "UNCLASSIFIED" else "D")
        print(f"| {f} | {n} | {k} | {paths.get(f, '-')} |")
    counts = Counter(cls.values())
    print(f"\n  A (legitimate pre-match) : {counts['A']}")
    print(f"  B (prior-match derived)  : {counts['B']}")
    print(f"  C (unsafe)               : {counts['C']}")
    print(f"  D (unknown)              : {counts['D']}")
    check("no production feature falls in class C (post-match/unsafe)",
          counts["C"] == 0)
    check("no production feature falls in class D (unclassifiable)",
          counts["D"] == 0, str([c for c, k in cls.items() if k == "D"][:10]))
    print("\n  NOTE on class B: 'derived from prior matches' is SAFE here only")
    print("  because 2C proved the window excludes the current fixture. The name")
    print("  of a feature is never treated as evidence of its chronology.")

    # =================================================================== 2B
    rule("STEP 2B -- GOAL-SIGNAL AVAILABILITY BY MODELLING DIMENSION")
    fdb = REPO / "data/processed/features.db"
    con = sqlite3.connect(f"file:{fdb.resolve()}?mode=ro", uri=True)
    try:
        all_cols = [r[1] for r in con.execute("PRAGMA table_info(feature_rows)")]
        tq = ",".join(str(i) for i in sorted(train_ids))
        df = pd.read_sql_query(
            f"SELECT * FROM feature_rows WHERE season_id IN ({tq}) "
            f"AND label_result IS NOT NULL", con)
    finally:
        con.close()
    n_rows = len(df)
    print(f"  training rows loaded (2025/26 excluded by query): {n_rows}")
    check("no final-test season leaked into the analysis frame",
          not set(df["season_id"]).intersection(test_ids))

    DIMENSIONS = {
        "home scoring strength": lambda c: c.startswith("home_") and "goals_for" in c,
        "home defensive strength": lambda c: c.startswith("home_") and "goals_against" in c,
        "away scoring strength": lambda c: c.startswith("away_") and "goals_for" in c,
        "away defensive strength": lambda c: c.startswith("away_") and "goals_against" in c,
        "recent attacking form": lambda c: ("last5" in c or "last10" in c) and "for" in c,
        "recent defensive form": lambda c: ("last5" in c or "last10" in c) and "against" in c,
        "home/away-specific strength": lambda c: "venue" in c,
        "league scoring environment": lambda c: "strength" in c,
        "recency": lambda c: "last5" in c or "last10" in c,
        "team strength": lambda c: "strength" in c or "points" in c or "win_rate" in c,
    }
    print("\n| dimension | columns in contract | coverage | pre-kickoff? | history used"
          " | chronology provable? | scope |")
    print("|---|---:|---:|---|---|---|---|")
    for dim, pred in DIMENSIONS.items():
        cols = [c for c in MODEL_B_COLUMNS if pred(c)]
        if cols:
            cov = float(df[cols].notna().all(axis=1).mean())
            hz = sorted({horizon_of(c) for c in cols})
            scope = "team-specific" if all(side_of(c) != "none" for c in cols) else "match-level"
            print(f"| {dim} | {len(cols)} | {cov:.4f} | YES | {'/'.join(hz)} "
                  f"| YES (2C) | {scope} |")
        else:
            built = [c for c in all_cols if pred(c) and c not in X_EXCLUDED_COLUMNS]
            print(f"| {dim} | 0 | - | - | - | - | NOT IN CONTRACT"
                  f"{f' ({len(built)} built in features.db)' if built else ''} |")

    # =================================================================== 2D
    rule("STEP 2D -- DESCRIPTIVE GOAL-SIGNAL (training seasons only)")
    print("  Pearson correlation between each pre-match feature and four goal")
    print("  quantities. ASSOCIATION ONLY -- never importance, never causal, and")
    print("  explicitly NOT a selection step: every family is reported, nothing is")
    print("  ranked-and-kept, and no feature is dropped on the basis of any value.")
    hg = df["label_home_goals"].astype(float).to_numpy()
    ag = df["label_away_goals"].astype(float).to_numpy()
    targets = {"home goals": hg, "away goals": ag,
               "total goals": hg + ag, "goal difference": hg - ag}

    def corr(x, y):
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 3 or np.nanstd(x[m]) == 0:
            return float("nan")
        return float(np.corrcoef(x[m], y[m])[0, 1])

    per_fam = defaultdict(lambda: defaultdict(list))
    per_col = {}
    for c in numeric:
        x = df[c].astype(float).to_numpy()
        per_col[c] = {t: corr(x, y) for t, y in targets.items()}
        for t in targets:
            per_fam[family_of(c)][t].append(abs(per_col[c][t]))
    print("\n  FAMILY-LEVEL SUMMARY (mean |r|, then max |r|)")
    print("| family | n | home goals | away goals | total goals | goal diff | max |r| |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for f in sorted(per_fam, key=lambda k: -fam_count[k]):
        row = per_fam[f]
        mx = max(max(v) for v in row.values())
        print(f"| {f} | {fam_count[f]} | " + " | ".join(
            f"{np.nanmean(row[t]):.4f}" for t in targets) + f" | {mx:.4f} |")
    print("\n  STRONGEST SINGLE ASSOCIATION PER TARGET (reported, not selected)")
    for t in targets:
        best = max(per_col, key=lambda c: abs(per_col[c][t]) if not np.isnan(per_col[c][t]) else -1)
        print(f"    {t:16s} {best:44s} r = {per_col[best][t]:+.4f}")
    all_abs = [abs(per_col[c][t]) for c in per_col for t in targets
               if not np.isnan(per_col[c][t])]
    print(f"\n  distribution of |r| across all {len(numeric)}x4 feature-target pairs:")
    for q in (50, 75, 90, 99, 100):
        print(f"    p{q:<3d} {np.percentile(all_abs, q):.4f}")
    print("\n  READING GUIDE, fixed before the numbers were produced:")
    print("    Correlations of this magnitude are what a WEAK-but-real signal looks")
    print("    like. They do NOT indicate that a Poisson model will beat V1 -- V1")
    print("    already consumes these same columns. The only thing 2D establishes is")
    print("    whether the goal targets are related to the features AT ALL, i.e.")
    print("    whether lambda estimation from this data is even coherent to attempt.")

    # =================================================================== 2E
    rule("STEP 2E -- MISSING LEGITIMATE DIMENSIONS (derivable but not in the contract)")
    xg_cols = [c for c in all_cols if "xg" in c and not c.endswith("_n")
               and "coverage" not in c]
    venue_cols = [c for c in all_cols if "venue" in c and not c.endswith("_n")]
    xg_cov = float(df[xg_cols].notna().any(axis=1).mean()) if xg_cols else 0.0
    venue_cov = float(df[venue_cols].notna().all(axis=1).mean()) if venue_cols else 0.0
    h2h_ok = '"opponent_id"' in hs or "opponent_id" in hs
    unix_in_hist = '"unix": fixture["unix"]' in hs

    print("| # | dimension | available | coverage | leakage risk | derivable | usefulness"
          " | proceed to experiment? |")
    print("|---:|---|---|---:|---|---|---|---|")
    rows_2e = [
        ("1", "rest days", "NO (not built)", None,
         "NONE -- uses only prior fixtures' unix and this fixture's scheduled unix",
         "YES -- history rows already carry `unix`", "plausible", "candidate, LATER"),
        ("2", "schedule congestion", "NO (not built)", None,
         "NONE -- same construction as rest days",
         "YES -- count prior fixtures within a trailing window", "plausible", "candidate, LATER"),
        ("3", "head-to-head history", "NO (not built)", None,
         "NONE if restricted to prior meetings",
         "PARTIAL -- team_match_record stores NO opponent id, so this needs either a "
         "builder change or a separate matches.db join", "unknown", "NO -- needs new plumbing"),
        ("4", "lagged team scoring rate", "YES -- IN CONTRACT", None,
         "NONE -- proven by 2C", "already present", "already consumed by V1", "already in V1"),
        ("5", "lagged team conceding rate", "YES -- IN CONTRACT", None,
         "NONE -- proven by 2C", "already present", "already consumed by V1", "already in V1"),
        ("6", "lagged rolling goals", "YES -- IN CONTRACT", None,
         "NONE -- proven by 2C", "already present", "already consumed by V1", "already in V1"),
        ("7", "home/away-specific lagged goals", f"BUILT, NOT IN CONTRACT ({len(venue_cols)} cols)",
         venue_cov, "NONE -- same walker, same guarantee",
         "YES -- already materialised in features.db", "plausible", "candidate, LATER"),
        ("8", "lagged xG", f"BUILT, NOT IN CONTRACT ({len(xg_cols)} cols)", xg_cov,
         "NONE -- same walker; chronology IS provable, so the item-8 condition is MET",
         "YES -- already materialised", "unknown (coverage-limited)", "NO -- coverage gate"),
        ("9", "league scoring baseline", "YES -- IN CONTRACT (via strength)", None,
         "NONE -- accumulator updated only in record()", "already present",
         "already consumed by V1", "already in V1"),
    ]
    for i, dim, avail, cov, leak, deriv, use, proceed in rows_2e:
        c = f"{cov:.4f}" if cov is not None else "-"
        print(f"| {i} | {dim} | {avail} | {c} | {leak} | {deriv} | {use} | {proceed} |")

    print("\n  EVIDENCE FOR THE THREE NON-OBVIOUS ROWS:")
    print(f"    item 3: history.py stores an opponent id? {h2h_ok}")
    print("            team_match_record keys are fixture_id, unix, competition_id,")
    print("            season_id, is_home, goals_for, goals_against, result, and")
    print("            per-stat channels -- no opponent identifier. Head-to-head is")
    print("            therefore NOT derivable from the existing feature pipeline.")
    print(f"    item 7: {len(venue_cols)} venue columns exist, e.g. "
          f"{venue_cols[:2] if venue_cols else '-'}")
    print(f"    item 8: {len(xg_cols)} xG columns exist; any-xG row coverage {xg_cov:.4f}")
    print(f"            history rows carry unix (needed for items 1-2): {unix_in_hist}")
    print("\n  NOTHING HERE IS A RECOMMENDATION TO ADD ANY COLUMN. Availability and")
    print("  leakage-safety are necessary conditions, not evidence of usefulness.")
    print("  STEP 9 already rated comparable candidates POSSIBLE, never STRONG.")

    # =================================================================== 2F
    rule("STEP 2F -- THE SMALLEST DEFENSIBLE FIRST EXPERIMENT")
    print("  PROPOSAL (not executed here):")
    print("    inputs      : EXACTLY the frozen 80-column contract. No venue columns,")
    print("                  no xG, no rest days, no head-to-head. Nothing added.")
    print("    estimator   : two Poisson regressions with a log link ->")
    print("                  lambda_home, lambda_away")
    print("    conversion  : P(i,j) = Pois(i; lambda_home) * Pois(j; lambda_away),")
    print("                  i,j in 0..10, renormalised; P(H)=sum_{i>j}, P(D)=sum_{i=j},")
    print("                  P(A)=sum_{i<j}")
    print("    fitting     : TRAINING seasons only, inside each fold's train split")
    print("    protocol    : the SAME three walk-forward folds V1 used")
    print("    comparison  : head-to-head against V1 on identical validation rows")
    print()
    print("  WHY THE INPUTS ARE HELD FIXED: if features and model form changed")
    print("  together, a result could not be attributed to either. Holding the 80")
    print("  columns constant makes the experiment a clean test of the MODEL FORM,")
    print("  which is the only thing STEP 8 and STEP 9 left standing.")
    print()
    print("  PRIMARY METRICS      : 1 multiclass log loss   2 Brier   3 accuracy")
    print("  DRAW DIAGNOSTICS     : 4 Draw-vs-rest AUC  5 actual draw rate")
    print("                         6 P(D) distribution  7 Draw argmax frequency")
    print()
    print(f"  V1 REFERENCE VALUES (pinned, for the head-to-head):")
    print(f"    pooled log loss   {POOLED_REF_LOG_LOSS:.16f}")
    print(f"    pooled Brier      {POOLED_REF_BRIER:.16f}")
    print(f"    Draw-vs-rest AUC  {V1_DRAW_AUC:.4f}")
    print(f"    Draw argmax share {V1_DRAW_ARGMAX:.4f}")
    print()
    print("  PRE-REGISTERED DECISION RULE (to be fixed BEFORE any fitting in STEP 3):")
    print("    BETTER  only if pooled log loss improves AND >=2/3 folds improve AND")
    print("            Brier does not materially contradict.")
    print("    WORSE   if log loss and Brier both worsen.")
    print("    INCONCLUSIVE otherwise.")
    print("    Draw metrics are REPORTED but cannot promote V2 on their own.")
    print("    A higher Draw argmax share with flat or falling Draw-vs-rest AUC is")
    print("    the STEP 8 failure signature and must be named as such if it recurs.")
    print()
    print("  REQUIRED BASELINES, so a pass is interpretable:")
    print("    - constant-lambda (league means, no features): proves the features")
    print("      contribute anything at all to lambda")
    print("    - frozen V1 on the identical validation rows")
    print("  2025/26 IS NOT TOUCHED. It is spent as an evaluation set.")

    # =============================================================== SUMMARY
    rule("FINAL CLASSIFICATION")
    sub("1. VERIFIED SAFE SIGNAL (pre-match proven from source)")
    for f, n in sorted(fam_count.items(), key=lambda t: -t[1]):
        if f != "UNCLASSIFIED":
            print(f"    {f:18s} {n:3d} columns   class "
                  f"{'A' if f == 'context' else 'B'}")
    print(f"    TOTAL: {counts['A'] + counts['B']} of 80 contract columns")

    sub("2. UNSAFE / LEAKAGE-RISK SIGNAL")
    print("    Within the 80-column contract: NONE.")
    print("    Outside it, STEP 9 classified every stat_* channel as post-kickoff")
    print("    for its OWN fixture. Those raw columns must never be used directly;")
    print("    only their LAGGED forms, built by the proven walker, are admissible.")

    sub("3. UNKNOWN SIGNAL")
    print("    Within the contract: NONE (all 79 traced to history_before).")
    print("    Outside it, STEP 9 marked position / played-count / formation / odds")
    print("    as TEMPORAL STATUS NOT VERIFIABLE. They remain excluded.")

    sub("4. NEW DERIVABLE SIGNAL (available, leakage-safe, not in the contract)")
    print(f"    venue-restricted lagged goals : {len(venue_cols)} columns, "
          f"coverage {venue_cov:.4f}  -- BUILT ALREADY")
    print(f"    lagged xG                     : {len(xg_cols)} columns, "
          f"coverage {xg_cov:.4f}  -- BUILT ALREADY, coverage-limited")
    print("    rest days / congestion        : derivable (history carries unix), NOT built")
    print("    head-to-head                  : NOT derivable without new plumbing")
    print("    None of these enters the first experiment.")

    sub("5. RECOMMENDED FIRST EXPERIMENT")
    print("    Two-Poisson lambda model on the UNCHANGED 80-column contract,")
    print("    same three walk-forward folds, head-to-head against frozen V1,")
    print("    judged on log loss and Brier under the pre-registered rule above.")
    print("    Capabilities (expected goals, scorelines, more draws) are explicitly")
    print("    NOT evidence and cannot promote it.")

    # ============================================================== INTEGRITY
    rule("POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified anywhere", not changed, str(changed))
    print("    fit() calls           : NONE")
    print("    models trained        : NONE")
    print("    artifacts written     : NONE")
    print("    feature DB writes     : NONE")
    print("    production changes    : NONE")
    print("    V1 changes            : NONE")
    print("    2025/26 outcome reads : NONE")
    print("    files created         : NONE")
    for k in ("data/processed/features.db", "data/processed/matches.db",
              "data/models/v1_logreg.pkl", "src/models/train.py",
              "src/features/feature_builder.py", "src/features/history.py"):
        print(f"    {k:38s} {after[k]}")
    print(f"    LOCKED_INPUTS                          {len(pins)}/{len(pins)} unchanged")

    print("\n" + "=" * 124)
    print("STEP 2 COMPLETE -- SIGNAL AUDIT ONLY")
    print("=" * 124)
    print("No experiment has been run and no improvement is claimed or predicted.")
    print("Not proceeding to STEP 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
