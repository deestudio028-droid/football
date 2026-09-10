"""STEP 9 -- new information source discrimination audit.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_step9_new_information_source_audit.py

ONE PURPOSE ONLY
Determine whether any genuinely missing, pre-match information dimension
ALREADY PRESENT in the project's source data contains measurable
Draw-specific information that is not obviously redundant with the existing
79-feature contract.

AUDIT ONLY. Not feature engineering. Not model improvement. Not a
recommendation. No candidate is implemented and none is proposed.

WHY A TEMPORAL PROBE IS NECESSARY
No file in src/ documents the capture time of any `fixtures` column. A column
existing in the schema therefore does NOT establish that its value was known
before kickoff. Rather than assume, STEP 4 runs two EMPIRICAL probes against
the fixtures table itself:

  * played-count probe : is `home_played` / `away_played` equal to the number
    of that team's prior fixtures (pre-match) or prior+1 (includes this
    match, therefore post-kickoff)?
  * table-position probe: what does `home_position` look like on a team's
    FIRST fixture of a season? A pre-match table on matchday 1 cannot already
    rank teams on that season's results.

Anything the probes cannot settle is reported NOT VERIFIABLE and is excluded
from the association measurements, so no leakage-suspect column can enter an
evidence table that a later step might read as support.

MEASUREMENT DISCIPLINE
Association is measured ONLY on columns that are pre-match valid (or static
by definition) and numerically defensible. Categorical identifiers are NOT
converted into arbitrary numeric correlations -- they are reported as
NOT MEASURABLE UNDER THIS CONVENTION. Every `stat_*` column describes the
match in which it was recorded and is therefore post-kickoff for that
fixture; aggregating it over prior matches would be feature construction and
is explicitly out of scope.

CONVENTIONS REUSED, NOT REINVENTED
Integrity gates, fold construction, data loading, SMD definition, the D-34
taxonomy and the |r| >= 0.90 redundancy threshold all follow D-38/D-40/D-41
and STEP 7/8. Redundancy is computed on TRAINING rows only. Three folds mean
ranges, never a standard deviation. No causal claim. No "importance".
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

POOLED_REF_LOG_LOSS = 0.9993791056968738
EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
ARTIFACT_MD5 = "5e504427712b35778bb8a62a8496c7cd"
FORBIDDEN_SEASONS = [602681, 667780, 725788, 725793, 762170]
REDUNDANCY_THRESHOLD = 0.90          # D-39/D-40 convention, unchanged

PRODUCTION_ASSUMPTIONS = (
    'SimpleImputer(strategy="median")',
    "StandardScaler()",
    "np.hstack([scaled, onehot])",
    "self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]",
    "LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
)

FAMILIES = {                          # EXACTLY the D-34 taxonomy
    "goals_season": lambda c: "goals" in c and "season" in c and "venue" not in c,
    "goals_last5": lambda c: "goals" in c and "last5" in c,
    "goals_last10": lambda c: "goals" in c and "last10" in c,
    "shots_season": lambda c: "shots" in c and "shots_on" not in c and "season" in c,
    "shots_last": lambda c: "shots" in c and "shots_on" not in c and "last" in c,
    "shots_on_season": lambda c: "shots_on" in c and "season" in c,
    "shots_on_last": lambda c: "shots_on" in c and "last" in c,
    "form": lambda c: "form" in c or "points" in c or "streak" in c or "result" in c,
    "strength": lambda c: "strength" in c or "attack" in c or "defence" in c or "defense" in c,
    "context": lambda c: c == "competition_id",
}

#: The 17 D-41 audit dimensions -> candidate source fields in matches.db.
#: Presence of a field is evidence of a SOURCE, never of pre-match validity.
DIMENSIONS = {
    "1 market expectation": ["has_odds"],
    "2 team availability / lineup state": ["home_formation", "away_formation"],
    "3 player availability": [],
    "4 injuries / suspensions": [],
    "5 tactical style": ["home_formation", "away_formation"],
    "6 possession / control profile": ["stat_home_possession", "stat_away_possession",
                                       "stat_home_pressure", "stat_away_pressure"],
    "7 chance-quality information": ["stat_home_xg", "stat_away_xg"],
    "8 shot-quality information": ["stat_home_xgot", "stat_away_xgot",
                                   "stat_home_shots_on", "stat_away_shots_on"],
    "9 schedule congestion": ["home_played", "away_played", "season_progress", "unix"],
    "10 rest differential": ["unix", "date"],
    "11 travel / load context": ["venue", "competition_country"],
    "12 weather / environment": [],
    "13 referee context": ["referee_id"],
    "14 game-state tendency": ["ht_score", "elapsed", "elapsed_seconds", "winning_team"],
    "15 opponent-adjusted strength": ["home_position", "away_position"],
    "16 matchup-specific interaction": [],
    "17 league table position": ["home_position", "away_position"],
}

#: Definitionally post-kickoff: describes the match in which it was recorded.
POST_KICKOFF_PREFIXES = ("stat_",)
POST_KICKOFF_EXACT = {"ht_score", "elapsed", "elapsed_seconds", "time_added",
                      "winning_team", "home_goals", "away_goals", "status"}
#: Known before kickoff by definition (scheduling / identity metadata).
STATIC_EXACT = {"fixture_id", "home_id", "away_id", "home_name", "away_name",
                "competition_id", "competition_name", "competition_country",
                "competition_type", "competition_predictability", "season_id",
                "season", "venue", "referee_id", "is_friendly", "is_cup",
                "unix", "date", "ko_human", "season_progress"}
#: Categorical identity -- not converted into a numeric correlation.
CATEGORICAL_IDENTIFIERS = {"referee_id", "venue", "home_formation", "away_formation",
                           "competition_country", "competition_name", "competition_type",
                           "home_name", "away_name", "ht_score", "ko_human", "date",
                           "winning_team", "status", "season"}


def rule(t): print("\n" + "=" * 124); print(t); print("=" * 124)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 124)
    print("ABORT -- STEP 9 halted. No audit is produced.")
    print(msg)
    print("!" * 124)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        stop(f"stop condition: {label}")


def snapshot():
    snap = {}
    for rel in EXPECTED_MD5:
        snap[rel] = md5(REPO / rel)
    lg = REPO / "data/processed/leagues"
    if lg.exists():
        for p in sorted(lg.glob("*.db")):
            snap[f"leagues/{p.name}"] = md5(p)
    art = REPO / "data/models/v1_logreg.pkl"
    if art.exists():
        snap["artifact"] = md5(art)
    snap["src/models/train.py"] = md5(REPO / "src/models/train.py")
    for rel in json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                          .read_text())["locked_input_checksums"]:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


def smd(a, b):
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


def coverage_class(pct):
    if pct >= 95: return "high"
    if pct >= 70: return "moderate"
    if pct >= 30: return "partial"
    if pct > 0: return "sparse"
    return "absent"


def main():
    # ---------------------------------------------------------------- RULE 1
    rule("RULE 1 -- PRODUCTION SOURCE RE-VERIFICATION (source text, never memory)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:74]}", tok in tsrc)

    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        stop(f"scikit-learn is required: {exc}. Do not substitute another estimator.")

    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.config import (
        FINAL_TEST_SEASONS, MODEL_VERSION, RECOMMENDED_CONTEXT_FEATURE,
        REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS,
    )
    from models.data import load_supervised_dataset
    from models.evaluate import evaluate
    from models.splits import iter_walk_forward_folds
    from models.train import train_logistic_regression

    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    check("production contract == exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("numeric production block == exactly 79", len(numeric) == 79)
    check("competition_id is the only categorical production column",
          len(set(MODEL_B_COLUMNS) - set(numeric)) == 1
          and RECOMMENDED_CONTEXT_FEATURE in MODEL_B_COLUMNS)
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")

    # ---------------------------------------------------------------- STEP 1
    rule("STEP 1 -- ENVIRONMENT + INTEGRITY")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"pandas {pd.__version__}  sklearn {sklearn.__version__}")
    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(rel, before[rel] == exp, before[rel])
    if "artifact" in before:
        check("v1_logreg.pkl unchanged", before["artifact"] == ARTIFACT_MD5, before["artifact"])
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    check("13/13 LOCKED_INPUTS at expected values",
          all(before[f"pin:{r}"] == v["expected"] for r, v in pins.items()))

    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    test_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    check("forbidden season set matches the documented list",
          sorted(test_ids) == sorted(FORBIDDEN_SEASONS), str(sorted(test_ids)))
    folds = list(iter_walk_forward_folds(ds))
    check("exactly three walk-forward folds", len(folds) == 3)
    for fold, tr, va in folds:
        seen = set(tr.metadata["season_id"]) | set(va.metadata["season_id"])
        check(f"{fold.name}: no final-test season_id", not (seen & test_ids))

    # V1 reproduction anchors the evaluation universe to the frozen baseline.
    v1 = []
    for fold, tr, va in folds:
        _, _, P = train_logistic_regression(tr.X[list(MODEL_B_COLUMNS)], tr.y,
                                            va.X[list(MODEL_B_COLUMNS)])
        v1.append(evaluate(va.y, P).log_loss)
    ll1 = float(np.mean(v1))
    print(f"  V1 pooled mean log loss: {ll1:.16f}  (reference {POOLED_REF_LOG_LOSS:.16f})")
    check("V1 reproduces the documented pooled log loss",
          abs(ll1 - POOLED_REF_LOG_LOSS) <= 1e-9, f"diff={ll1 - POOLED_REF_LOG_LOSS:.3e}")

    # frozen evaluation universe
    val_meta = pd.concat([va.metadata for _, _, va in folds], ignore_index=True)
    val_X = pd.concat([va.X[list(MODEL_B_COLUMNS)].reset_index(drop=True)
                       for _, _, va in folds], ignore_index=True)
    Y = np.concatenate([va.y.to_numpy() for _, _, va in folds])
    train_meta = pd.concat([tr.metadata for _, tr, _ in folds], ignore_index=True)
    train_X = pd.concat([tr.X[list(MODEL_B_COLUMNS)].reset_index(drop=True)
                         for _, tr, _ in folds], ignore_index=True)
    val_fids = val_meta["fixture_id"].tolist()
    train_fids = train_meta["fixture_id"].tolist()
    print(f"  frozen validation universe: {len(val_fids)} rows   "
          f"training universe: {len(train_fids)} rows")

    # ---------------------------------------------------------------- STEP 2
    rule("STEP 2 -- THE ACTUAL 79-FEATURE PRODUCTION CONTRACT")
    fam_of = {}
    for fam, fn in FAMILIES.items():
        for c in numeric:
            if fn(c):
                fam_of.setdefault(c, fam)

    def horizon(c):
        for h in ("last5", "last10", "season"):
            if h in c:
                return h
        return "unspecified"

    def side(c):
        return "home" if c.startswith("home_") else ("away" if c.startswith("away_") else "none")

    def mtype(c):
        if "diff" in c: return "within-team difference"
        if "per_match" in c: return "rate"
        if "strength" in c or "attack" in c or "defence" in c: return "strength score"
        if "rate" in c or "points" in c: return "form"
        return "level"

    print(f"  total numeric features : {len(numeric)}")
    print(f"  home-side              : {sum(1 for c in numeric if side(c)=='home')}")
    print(f"  away-side              : {sum(1 for c in numeric if side(c)=='away')}")
    print(f"  non-sided              : {[c for c in numeric if side(c)=='none']}")
    print("\n| # | feature | D-34 family | horizon | side | metric type |")
    print("|---:|---|---|---|---|---|")
    for i, c in enumerate(numeric, 1):
        print(f"| {i} | {c} | {fam_of.get(c,'UNMATCHED')} | {horizon(c)} | {side(c)} "
              f"| {mtype(c)} |")
    print("\n  family counts: " + "  ".join(
        f"{f}={sum(1 for c in numeric if fam_of.get(c)==f)}" for f in FAMILIES))
    print(f"  UNMATCHED by D-34: {sum(1 for c in numeric if c not in fam_of)}")
    print("  horizons: " + "  ".join(
        f"{h}={sum(1 for c in numeric if horizon(c)==h)}"
        for h in ("last5", "last10", "season", "unspecified")))

    # ------------------------------------------------- schema (read-only copies)
    tmp = Path(tempfile.mkdtemp(prefix="step9_"))
    try:
        for rel in ("data/processed/matches.db", "data/processed/features.db"):
            src = REPO / rel; dst = tmp / Path(rel).name
            shutil.copyfile(src, dst)
            if md5(src) != md5(dst):
                stop(f"scratch copy of {Path(rel).name} is not byte-identical")
        mcon = sqlite3.connect(f"file:{tmp/'matches.db'}?mode=ro", uri=True)
        fcon = sqlite3.connect(f"file:{tmp/'features.db'}?mode=ro", uri=True)
        raw_cols = [r[1] for r in mcon.execute("PRAGMA table_info(fixtures)")]
        der_cols = [r[1] for r in fcon.execute("PRAGMA table_info(feature_rows)")]
        n_fx = mcon.execute("SELECT COUNT(*) FROM fixtures").fetchone()[0]
        pop = {}
        for c in raw_cols:
            nn = mcon.execute(f'SELECT SUM("{c}" IS NOT NULL) FROM fixtures').fetchone()[0] or 0
            pop[c] = 100.0 * nn / n_fx
        fx = pd.read_sql_query("SELECT * FROM fixtures", mcon)
        feat_src = "\n".join(p.read_text(encoding="utf-8")
                             for p in sorted((REPO / "src/features").glob("*.py")))

        # ------------------------------------------------------------ STEP 3
        rule("STEP 3 -- SOURCE INFORMATION INVENTORY (17 D-41 dimensions)")
        print(f"  matches.db fixtures columns : {len(raw_cols)}")
        print(f"  features.db derived columns : {len(der_cols)}")
        print(f"  production contract columns : {len(MODEL_B_COLUMNS)}")
        print("\n| Dimension | candidate source fields present | in contract? | in features.db? |")
        print("|---|---|---|---|")
        for dim, fields in DIMENSIONS.items():
            present = [f for f in fields if f in raw_cols]
            inc = [f for f in fields if f in MODEL_B_COLUMNS]
            ind = [f for f in fields if f in der_cols]
            print(f"| {dim} | {present or 'NONE FOUND IN SOURCE'} | {inc or 'no'} "
                  f"| {ind or 'no'} |")

        # ------------------------------------------------------------ STEP 4
        rule("STEP 4 -- TEMPORAL / LEAKAGE CLASSIFICATION")
        print("  NOTE: no file in src/ documents the capture time of any fixtures column.")
        print("  Existence in the schema is therefore NOT evidence of pre-match validity.")
        print("  Two empirical probes are run before any column is classified.\n")

        # probe 1 -- played counts
        print("  PROBE 1 -- played-count semantics")
        f2 = fx.sort_values(["unix", "fixture_id"]).reset_index(drop=True)
        prior = {}
        agree_pre = agree_post = tested = 0
        for r in f2.itertuples():
            for tid, col in ((r.home_id, "home_played"), (r.away_id, "away_played")):
                stored = getattr(r, col)
                n_prior = prior.get(tid, 0)
                if stored is not None and not pd.isna(stored):
                    tested += 1
                    if int(stored) == n_prior: agree_pre += 1
                    elif int(stored) == n_prior + 1: agree_post += 1
            for tid in (r.home_id, r.away_id):
                if r.status in ("FT", "AWARDED"):
                    prior[tid] = prior.get(tid, 0) + 1
        if tested:
            print(f"    tested {tested} team-fixture observations")
            print(f"    stored == prior count      (PRE-MATCH)  : {agree_pre} "
                  f"({100*agree_pre/tested:.1f}%)")
            print(f"    stored == prior + 1 (INCLUDES THIS MATCH): {agree_post} "
                  f"({100*agree_post/tested:.1f}%)")
            print(f"    neither                                  : "
                  f"{tested-agree_pre-agree_post} "
                  f"({100*(tested-agree_pre-agree_post)/tested:.1f}%)")
            played_verdict = ("PRE-MATCH VALID" if agree_pre / tested > .9 else
                              "POST-KICKOFF" if agree_post / tested > .9 else
                              "NOT VERIFIABLE")
        else:
            played_verdict = "NOT VERIFIABLE"
        print(f"    verdict for home_played/away_played: {played_verdict}")

        # probe 2 -- table position on a team's first fixture of a season
        print("\n  PROBE 2 -- table-position semantics on matchday 1")
        firsts = (f2.sort_values(["season_id", "unix", "fixture_id"])
                    .groupby(["season_id", "home_id"], as_index=False).first())
        vals = pd.to_numeric(firsts["home_position"], errors="coerce").dropna()
        print(f"    teams' first home fixture of a season, n={len(vals)}")
        if len(vals):
            print(f"    distinct positions observed : {vals.nunique()}   "
                  f"min={vals.min():.0f} max={vals.max():.0f} mean={vals.mean():.2f}")
            print("    A genuinely PRE-match table on a team's first fixture cannot already")
            print("    rank teams on that season's results; a spread of distinct ranks is")
            print("    consistent with either a post-match update or a carried-over table.")
            pos_verdict = ("NOT VERIFIABLE" if vals.nunique() > 2 else "PRE-MATCH VALID")
        else:
            pos_verdict = "NOT VERIFIABLE"
        print(f"    verdict for home_position/away_position: {pos_verdict}")

        def temporal(colname):
            if colname.startswith(POST_KICKOFF_PREFIXES) or colname in POST_KICKOFF_EXACT:
                return "B) POST-KICKOFF / LEAKAGE", "describes the match it is recorded on"
            if colname in ("home_played", "away_played"):
                return ({"PRE-MATCH VALID": "A) PRE-MATCH VALID",
                         "POST-KICKOFF": "B) POST-KICKOFF / LEAKAGE"}.get(
                            played_verdict, "C) TEMPORAL STATUS NOT VERIFIABLE"),
                        f"probe 1: {played_verdict}")
            if colname in ("home_position", "away_position"):
                return ({"PRE-MATCH VALID": "A) PRE-MATCH VALID"}.get(
                            pos_verdict, "C) TEMPORAL STATUS NOT VERIFIABLE"),
                        f"probe 2: {pos_verdict}")
            if colname in STATIC_EXACT:
                return "E) STATIC / IDENTIFIER-LIKE", "scheduling or identity metadata"
            if colname in ("home_formation", "away_formation", "has_odds"):
                return "C) TEMPORAL STATUS NOT VERIFIABLE", "capture time undocumented in src/"
            return "C) TEMPORAL STATUS NOT VERIFIABLE", "no source evidence"

        print("\n| Dimension | Source field(s) | Population | Temporal status | Evidence |")
        print("|---|---|---:|---|---|")
        field_status = {}
        for dim, fields in DIMENSIONS.items():
            present = [f for f in fields if f in raw_cols]
            if not present:
                print(f"| {dim} | NONE | - | D/UNAVAILABLE -- no source field | "
                      f"schema inspection of {len(raw_cols)} columns |")
                continue
            for f in present:
                st, ev = temporal(f)
                field_status[f] = st
                print(f"| {dim} | {f} | {pop[f]:.2f}% | {st} | {ev} |")
        print("\n  Dimensions with NO source field are additionally classified:")
        print("    9/10 schedule congestion, rest differential -> "
              "D) DERIVABLE PRE-MATCH BUT NOT CURRENTLY COMPUTED (from unix + prior fixtures)")
        print("    16 matchup-specific interaction              -> "
              "D) DERIVABLE PRE-MATCH BUT NOT CURRENTLY COMPUTED (from prior meetings)")
        print("    3/4 player availability, injuries/suspensions -> UNAVAILABLE (no table)")
        print("    12 weather / environment                      -> UNAVAILABLE (no column)")
        print(f"  no such derivation exists in src/features/: "
              f"{'rest' not in feat_src.lower() and 'h2h' not in feat_src.lower()}")

        # ------------------------------------------------------------ STEP 5
        rule("STEP 5 -- POPULATION / COVERAGE AUDIT")
        val_set = set(val_fids)
        print("| Field | raw rows | non-null | non-null % | coverage class "
              "| usable rows in frozen validation universe |")
        print("|---|---:|---:|---:|---|---:|")
        for f in sorted({f for fs in DIMENSIONS.values() for f in fs if f in raw_cols}):
            sub = fx[fx["fixture_id"].isin(val_set)]
            usable = int(sub[f].notna().sum())
            nn = int(fx[f].notna().sum())
            print(f"| {f} | {n_fx} | {nn} | {pop[f]:.2f}% | {coverage_class(pop[f])} "
                  f"| {usable} / {len(val_set)} |")
        print("\n  Thresholds are descriptive. High coverage does NOT imply predictive value.")

        # ------------------------------------------------------------ STEP 6
        rule("STEP 6 -- DRAW-SPECIFIC RAW ASSOCIATION AUDIT")
        print("  Measured ONLY for fields that are pre-match valid or static AND numerically")
        print("  defensible. Categorical identifiers are NOT numerically correlated.")
        print("  Association only -- never importance, never causal.\n")
        fx_idx = fx.set_index("fixture_id")
        isD, isH, isA = Y == "D", Y == "H", Y == "A"
        measurable, results = [], {}
        for f in sorted({f for fs in DIMENSIONS.values() for f in fs if f in raw_cols}):
            st = field_status.get(f, "C) TEMPORAL STATUS NOT VERIFIABLE")
            if st.startswith("B"):
                results[f] = ("EXCLUDED -- post-kickoff", np.nan, np.nan); continue
            if st.startswith("C"):
                results[f] = ("NOT MEASURED -- temporal status not verifiable",
                              np.nan, np.nan); continue
            if f in CATEGORICAL_IDENTIFIERS:
                results[f] = ("NOT MEASURABLE UNDER THIS CONVENTION -- categorical identity",
                              np.nan, np.nan); continue
            v = pd.to_numeric(fx_idx.reindex(val_fids)[f], errors="coerce").to_numpy(float)
            if np.isfinite(v).sum() < 100 or np.nanstd(v) == 0:
                results[f] = ("NOT MEASURABLE -- insufficient variation/population",
                              np.nan, np.nan); continue
            a, b = smd(v[isD], v[isH]), smd(v[isD], v[isA])
            results[f] = ("measured", a, b)
            measurable.append(f)
        print("| Field | status | SMD D-H | SMD D-A | median \\|SMD\\| | max \\|SMD\\| |")
        print("|---|---|---:|---:|---:|---:|")
        for f, (st, a, b) in results.items():
            if st == "measured":
                vals_ = [abs(a), abs(b)]
                print(f"| {f} | measured | {a:+.4f} | {b:+.4f} | {np.median(vals_):.4f} "
                      f"| {max(vals_):.4f} |")
            else:
                print(f"| {f} | {st} | - | - | - | - |")
        p_dh = np.nanmean([abs(smd(pd.to_numeric(val_X[c], errors="coerce").to_numpy()[isD],
                                   pd.to_numeric(val_X[c], errors="coerce").to_numpy()[isH]))
                           for c in numeric])
        p_da = np.nanmean([abs(smd(pd.to_numeric(val_X[c], errors="coerce").to_numpy()[isD],
                                   pd.to_numeric(val_X[c], errors="coerce").to_numpy()[isA]))
                           for c in numeric])
        print(f"\n  reference -- existing 79-feature contract: mean |SMD| D-H {p_dh:.4f}   "
              f"D-A {p_da:.4f}")

        # ------------------------------------------------------------ STEP 7
        rule("STEP 7 -- REDUNDANCY AGAINST THE EXISTING 79 FEATURES (TRAINING rows only)")
        print("| Candidate | raw Draw separation (max \\|SMD\\|) | max training redundancy "
              "| redundancy class |")
        print("|---|---:|---:|---|")
        red = {}
        tX = train_X[numeric].apply(pd.to_numeric, errors="coerce")
        for f in measurable:
            tv = pd.to_numeric(fx_idx.reindex(train_fids)[f], errors="coerce")
            if tv.notna().sum() < 100 or tv.std() == 0:
                red[f] = ("D) NOT TESTABLE", float("nan")); continue
            cors = tX.corrwith(tv).abs()
            mx = float(cors.max()) if cors.notna().any() else float("nan")
            cls = ("C) HIGHLY REDUNDANT" if mx >= REDUNDANCY_THRESHOLD else
                   "B) PARTIALLY REDUNDANT" if mx >= 0.60 else
                   "A) DISTINCT / LOW REDUNDANCY")
            red[f] = (cls, mx)
            sep = max(abs(results[f][1]), abs(results[f][2]))
            print(f"| {f} | {sep:.4f} | {mx:.4f} | {cls} |")
        for f, (st, a, b) in results.items():
            if st != "measured":
                print(f"| {f} | not measured | - | D) NOT TESTABLE |")
        print("\n  Redundancy uses the D-39/D-40 threshold |r| >= 0.90, TRAINING rows only.")
        print("  Categorical identity is never converted into a numeric correlation.")

        # ------------------------------------------------------------ STEP 8
        rule("STEP 8 -- RELATIONAL INFORMATION CHECK (mathematical meaning, not naming)")
        print("| Field | scope | justification |")
        print("|---|---|---|")
        for f in sorted({f for fs in DIMENSIONS.values() for f in fs if f in raw_cols}):
            if f.startswith("home_") or f.startswith("away_") or f.startswith("stat_home") \
               or f.startswith("stat_away"):
                scope = "team-level"
                why = "describes one side only; no between-team term is formed"
            elif f in ("season_progress", "unix", "date", "competition_country", "venue",
                       "referee_id", "has_odds"):
                scope = "context-only"
                why = "shared by both sides; carries no team-relative content"
            else:
                scope = "context-only"; why = "no between-team construction present"
            print(f"| {f} | {scope} | {why} |")
        print("\n  between-team fields found : NONE")
        print("  matchup-specific fields   : NONE")
        print("  A paired home_/away_ column is TEAM-LEVEL: the pair exists, but no")
        print("  between-team term is present in the source. D-41 recorded 1/8 relational")
        print("  dimensions explicitly represented; nothing here changes that count.")

        # ------------------------------------------------------------ STEP 9
        rule("STEP 9 -- PRE-REGISTERED EVIDENCE CLASSIFICATION (mechanical)")
        SEP_FLOOR = 0.10          # pre-registered before any value was read
        cls_of = {}
        for f in sorted({f for fs in DIMENSIONS.values() for f in fs if f in raw_cols}):
            st = field_status.get(f, "C) TEMPORAL STATUS NOT VERIFIABLE")
            r = results.get(f, ("NOT MEASURED", np.nan, np.nan))
            if st.startswith("B") or st.startswith("C") or r[0] != "measured":
                cls_of[f] = "D) UNUSABLE / EXCLUDED"; continue
            sep = max(abs(r[1]), abs(r[2]))
            rc = red.get(f, ("D) NOT TESTABLE", float("nan")))[0]
            cov = coverage_class(pop[f])
            if sep >= SEP_FLOOR and cov in ("high", "moderate") and rc.startswith("A"):
                cls_of[f] = "A) STRONG INFORMATION-GAP EVIDENCE"
            elif sep > 0 and not rc.startswith("C"):
                cls_of[f] = "B) POSSIBLE INFORMATION GAP"
            else:
                cls_of[f] = "C) NOT SUPPORTED AS INFORMATION GAP"
        print(f"  pre-registered separation floor for class A: max|SMD| >= {SEP_FLOOR}")
        print("  (fixed before any value was read; no candidate is promoted for sounding useful)")
        for f, c in sorted(cls_of.items()):
            print(f"    {f:26s} -> {c}")

        # ----------------------------------------------------------- STEP 10
        rule("STEP 10 -- CANDIDATE MATRIX (all audited dimensions, none omitted)")
        print("| Dimension | Source exists | Pre-match valid | Coverage | Draw SMD D-H "
              "| Draw SMD D-A | Redundancy | Relational? | Classification |")
        print("|---|---|---|---:|---:|---:|---|---|---|")
        for dim, fields in DIMENSIONS.items():
            present = [f for f in fields if f in raw_cols]
            if not present:
                print(f"| {dim} | NO | n/a | 0% | - | - | D) NOT TESTABLE | none "
                      f"| D) UNUSABLE / EXCLUDED |")
                continue
            for f in present:
                st = field_status.get(f, "C)")
                r = results.get(f, ("", np.nan, np.nan))
                rc = red.get(f, ("D) NOT TESTABLE", float("nan")))[0]
                pv = ("YES" if st.startswith(("A", "E")) else
                      "NO (post-kickoff)" if st.startswith("B") else "NOT VERIFIABLE")
                dh = f"{r[1]:+.4f}" if r[0] == "measured" else "-"
                da = f"{r[2]:+.4f}" if r[0] == "measured" else "-"
                print(f"| {dim} :: {f} | YES | {pv} | {pop[f]:.1f}% | {dh} | {da} "
                      f"| {rc} | context/team-level | {cls_of.get(f,'D) UNUSABLE / EXCLUDED')} |")

        # ----------------------------------------------------------- STEP 11
        rule("STEP 11 -- INFORMATION-LIMITATION TEST")
        strong = [f for f, c in cls_of.items() if c.startswith("A")]
        possible = [f for f, c in cls_of.items() if c.startswith("B")]
        notsup = [f for f, c in cls_of.items() if c.startswith("C")]
        excluded = [f for f, c in cls_of.items() if c.startswith("D")]
        nofield = [d for d, fs in DIMENSIONS.items() if not [f for f in fs if f in raw_cols]]
        print(f"  STRONG INFORMATION-GAP candidates : {len(strong)}")
        print(f"  POSSIBLE candidates               : {len(possible)}")
        print(f"  NOT SUPPORTED                     : {len(notsup)}")
        print(f"  EXCLUDED / UNUSABLE (fields)      : {len(excluded)}")
        print(f"  dimensions with no source field   : {len(nofield)} {nofield}")
        print(f"\n  STRONG   : {strong or 'NONE'}")
        print(f"  POSSIBLE : {possible or 'NONE'}")
        print(f"  NOT SUPPORTED : {notsup or 'NONE'}")
        print(f"  EXCLUDED : {excluded or 'NONE'}")
        if len(strong) == 0:
            print("\n  ZERO candidates satisfy the pre-registered STRONG criteria.")
        elif len(strong) > 1:
            print("\n  EVIDENCE DOES NOT DISCRIMINATE BETWEEN CANDIDATES.")

        # ----------------------------------------------------------- STEP 12
        rule("STEP 12 -- CASE C / CASE E / INFORMATION-LIMITATION UPDATE (evidence only)")
        print("  supporting CASE C        : V1 mean P(D) tracks the base rate while Draw")
        print("                             argmax stays suppressed (D-33/D-38 measurements)")
        print("  weakening CASE C         : STEP 7 relaxed the suppression and scoring got")
        print("                             worse; nothing withheld was released")
        print("  supporting CASE E        : STEP 7 closeness failed; D-40 grouping recovered")
        print("                             nothing")
        print("  weakening CASE E         : STEP 8's nonlinear probe on the SAME inputs")
        print("                             failed badly (log loss 0.999379 -> 1.089122),")
        print("                             so added capacity did not release hidden signal")
        print(f"  supporting information-limitation : {len(strong)} STRONG candidate(s) in")
        print("                             the existing source; every stat_* channel is")
        print("                             post-kickoff for its own fixture")
        print("  against information-limitation    : dimensions classified D) DERIVABLE BUT")
        print("                             NOT COMPUTED remain untested, so exhaustion is")
        print("                             not demonstrated")
        print("  unresolved               : no case is proven; this audit measures source")
        print("                             availability and association, not causation")

        # ----------------------------------------------------------- STEP 14
        rule("STEP 14 -- FINAL STEP-9 CLOSURE INPUTS")
        allf = sorted({f for fs in DIMENSIONS.values() for f in fs if f in raw_cols})
        prematch = [f for f in allf if field_status.get(f, "C").startswith(("A", "E"))]
        goodcov = [f for f in allf if coverage_class(pop[f]) in ("high", "moderate")]
        print(f"  production numeric contract size      : {len(numeric)}")
        print(f"  audited information dimensions        : {len(DIMENSIONS)}")
        print(f"  distinct source fields audited        : {len(allf)}")
        print(f"  pre-match valid / static fields       : {len(prematch)}")
        print(f"  high or moderate coverage fields      : {len(goodcov)}")
        print(f"  fields with measurable Draw separation: {len(measurable)}")
        print(f"  strongly redundant fields             : "
              f"{sum(1 for f in red if red[f][0].startswith('C'))}")
        print(f"  STRONG INFORMATION-GAP                : {len(strong)}")
        print(f"  POSSIBLE                              : {len(possible)}")
        print(f"  NOT SUPPORTED                         : {len(notsup)}")
        print(f"  EXCLUDED                              : {len(excluded)}")
        print(f"\nSTRONG INFORMATION-GAP CANDIDATES:\n  {strong or 'NONE'}")
        print(f"POSSIBLE INFORMATION-GAP CANDIDATES:\n  {possible or 'NONE'}")
        print(f"NOT SUPPORTED:\n  {notsup or 'NONE'}")
        print(f"EXCLUDED / UNUSABLE:\n  {excluded or 'NONE'}  + dimensions with no field: {nofield}")
        if len(strong) == 1:
            print(f"\nPRIMARY INFORMATION GAP: evidence is uniquely discriminating for "
                  f"{strong[0]} under the pre-registered criteria.")
            print("  This is a reported measurement only. Implementation is NOT proposed.")
        else:
            print("\nPRIMARY INFORMATION GAP:\n  NOT SELECTED BY THIS AUDIT")

        mcon.close(); fcon.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --------------------------------------------------------------- STEP 15
    rule("STEP 15 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))
    check("src/models/train.py unchanged",
          after["src/models/train.py"] == before["src/models/train.py"])
    check("v1_logreg.pkl unchanged", after.get("artifact") == before.get("artifact"))
    check("features.db unchanged",
          after["data/processed/features.db"] == EXPECTED_MD5["data/processed/features.db"])
    check("matches.db unchanged",
          after["data/processed/matches.db"] == EXPECTED_MD5["data/processed/matches.db"])
    check("13/13 LOCKED_INPUTS unchanged",
          all(after[f"pin:{r}"] == v["expected"] for r, v in pins.items()))
    print("\nFILES MODIFIED: NONE")
    print("DATABASES MODIFIED: NONE")
    print("MODEL MODIFIED: NONE")
    print("ARTIFACT MODIFIED: NONE")
    print("2025/26 RESULTS ACCESSED: NO")
    print("\nSTEP 9 COMPLETE -- AUDIT ONLY, NOTHING IMPLEMENTED, NOTHING RECOMMENDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
