"""D-41 -- missing information / information gap diagnosis (STEP 6).

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_d41_missing_information_diagnosis.py

READ-ONLY AUDIT. NOTHING IS ADDED, ENGINEERED, TUNED, RETRAINED OR
PERSISTED. NO RECOMMENDATION IS MADE.

THE QUESTION (STEP 6)
What information is missing from the current feature contract that
prevents the existing linear representation from expressing
Draw-relevant match structure?

This is an information-gap AUDIT. It is not feature engineering, not an
improvement experiment, and not permission to add anything. Every
classification below must be backed by source or schema evidence that
the script itself prints; anything unverifiable is marked
NOT VERIFIABLE rather than guessed.

EVIDENCE HIERARCHY USED FOR EVERY CLASSIFICATION
    1. is the dimension in MODEL_B_COLUMNS?          -> in the contract
    2. else is it a derived column in features.db?   -> derived, excluded
    3. else is it a raw column in matches.db?        -> raw, unused
    4. else is it computed anywhere in src/features/ ?
    5. else                                          -> ABSENT / NOT VERIFIABLE
Column population (non-null %) is reported alongside, because a column
that exists but is 35% populated is not the same as one that is present.

STANDING CAVEATS RESTATED IN THE OUTPUT
  - No "feature importance" terminology. SMD is an association measure.
  - No causal claims from SMD, correlation or coverage.
  - Three folds => ranges only, never a standard deviation.
  - A dimension is called missing ONLY where the contract demonstrably
    does not contain it -- never because it sounds useful.
  - No dimension is recommended, proposed or tested.
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

LEAGUES = {423: "Premier League", 477: "Bundesliga", 419: "La Liga",
           200: "Ligue 1", 499: "Serie A"}
POOLED_REF_LOG_LOSS = 0.9993791056968738
EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
ARTIFACT_MD5 = "5e504427712b35778bb8a62a8496c7cd"
FORBIDDEN_SEASONS = [602681, 667780, 725788, 725793, 762170]

PRODUCTION_ASSUMPTIONS = (
    'SimpleImputer(strategy="median")',
    "StandardScaler()",
    "np.hstack([scaled, onehot])",
    "self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]",
    "LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
)

#: EXACTLY the D-34 taxonomy. Not redefined, not extended.
FAMILIES = {
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

#: STEP 3 diagnostic categories -> name tokens, matched against the CONTRACT only.
CATEGORIES = {
    "A RESULT / OUTCOME HISTORY": ("result", "win", "loss", "draw", "points", "streak", "form"),
    "B GOAL PRODUCTION": ("goals_for",),
    "C GOAL CONCESSION": ("goals_against",),
    "D SHOT VOLUME": ("shots_for", "shots_against", "shots_diff"),
    "E SHOT-ON-TARGET VOLUME": ("shots_on_for", "shots_on_against", "shots_on_diff"),
    "F RECENT FORM": ("last5", "last10"),
    "G TEAM STRENGTH": ("strength", "attack", "defence", "defense"),
    "H HOME/AWAY SPLITS": ("home_venue", "away_venue"),
    "I SEASON-LONG PERFORMANCE": ("season",),
    "J LEAGUE CONTEXT": ("competition_id", "league"),
}

#: STEP 6/9 candidate dimensions -> (raw schema tokens, derived tokens, contract tokens).
#: Classification is decided by evidence, never asserted.
CANDIDATES = {
    "market expectation": (("odds", "price", "implied"), ("odds",), ("odds",)),
    "team availability / lineup state": (("formation", "lineup", "squad"), ("formation",), ("formation",)),
    "player availability": (("player", "injur", "suspend", "absent"), ("player",), ("player",)),
    "injuries / suspensions": (("injur", "suspend"), ("injur", "suspend"), ("injur", "suspend")),
    "tactical style": (("formation", "style", "tactic"), ("formation", "style"), ("style",)),
    "possession / control profile": (("possession",), ("possession",), ("possession",)),
    "chance-quality information": (("xg", "xgot"), ("xg",), ("xg",)),
    "shot-quality information": (("xgot", "shots_on"), ("xgot", "shots_on"), ("shots_on",)),
    "schedule congestion": (("played", "season_progress", "unix", "date"), ("congest", "rest"), ("congest", "rest")),
    "rest differential": (("unix", "date"), ("rest", "days_since"), ("rest", "days_since")),
    "travel / load context": (("venue", "country"), ("travel", "distance"), ("travel", "distance")),
    "weather / environment": (("weather", "temp", "pitch", "wind"), ("weather",), ("weather",)),
    "referee context": (("referee",), ("referee",), ("referee",)),
    "game-state tendency": (("ht_score", "elapsed", "winning_team"), ("game_state", "ht_"), ("game_state",)),
    "opponent-adjusted strength": (("position",), ("strength", "shrunk", "adjusted"), ("strength",)),
    "matchup-specific interaction": (("h2h", "head_to_head"), ("h2h", "matchup", "opponent"), ("h2h", "matchup")),
    "league table position": (("position",), ("position", "rank"), ("position", "rank")),
}

#: STEP 9 relational dimensions -> contract tokens that would express them explicitly.
RELATIONAL = {
    "relative strength (home vs away)": ("strength_diff", "relative_strength", "strength_gap"),
    "relative attack": ("attack_diff", "attack_gap"),
    "relative defence": ("defence_diff", "defense_diff", "defence_gap"),
    "home-vs-away form gap": ("form_diff", "form_gap"),
    "rest differential": ("rest_diff", "days_since_diff"),
    "rating differential": ("rating_diff", "elo", "rating_gap"),
    "opponent-adjusted difference": ("opponent_adjusted", "opp_adj"),
    "explicit interaction state": ("interaction", "x_", "product"),
}


def rule(t): print("\n" + "=" * 122); print(t); print("=" * 122)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def stop(msg):
    print("\n" + "!" * 122)
    print("HARD FAIL -- D-41 halted. No audit is produced.")
    print(msg)
    print("!" * 122)
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


def hit(tokens, names):
    return sorted({n for n in names if any(t in n.lower() for t in tokens)})


def main():
    rule("RULE 1 -- PRODUCTION CODE TRACE (source text, never memory)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:68]}", tok in tsrc)

    rule("STEP 1 -- ENVIRONMENT + INTEGRITY")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  pandas {pd.__version__}")
    try:
        import sklearn
        print(f"  sklearn {sklearn.__version__}")
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
    check("contract is exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("numeric block == 79", len(numeric) == 79)
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])

    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(rel, before[rel] == exp, before[rel])
    if "artifact" in before:
        check("v1_logreg.pkl unchanged", before["artifact"] == ARTIFACT_MD5, before["artifact"])
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    check("13/13 LOCKED_INPUTS at expected values",
          all(before[f"pin:{r}"] == v["expected"] for r, v in pins.items()))
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")

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

    # in-memory reproduction, no artifact substitution
    P_l, y_l, c_l, f_l, X_l, per_fold = [], [], [], [], [], []
    for fold, tr, va in folds:
        Xtr, Xva = tr.X[list(MODEL_B_COLUMNS)], va.X[list(MODEL_B_COLUMNS)]
        _, _, P = train_logistic_regression(Xtr, tr.y, Xva)
        r = evaluate(va.y, P)
        per_fold.append((fold.name, len(va), r))
        P_l.append(P); y_l.append(va.y.to_numpy())
        c_l.append(va.X[RECOMMENDED_CONTEXT_FEATURE].to_numpy())
        f_l.append(np.repeat(fold.name, len(va))); X_l.append(Xva.reset_index(drop=True))
    Y = np.concatenate(y_l); CID = np.concatenate(c_l); FOLD = np.concatenate(f_l)
    X = pd.concat(X_l, ignore_index=True)
    pooled_ll = float(np.mean([r.log_loss for _, _, r in per_fold]))
    print(f"\n  pooled mean log loss: {pooled_ll:.16f}  (reference {POOLED_REF_LOG_LOSS:.16f})")
    check("pooled log loss reproduces documented V1",
          abs(pooled_ll - POOLED_REF_LOG_LOSS) < 1e-9,
          f"diff={pooled_ll - POOLED_REF_LOG_LOSS:.3e}")
    isD, isH, isA = Y == "D", Y == "H", Y == "A"

    # ---- schema evidence, read-only, via bit-identical scratch copies ----
    tmp = Path(tempfile.mkdtemp(prefix="d41_"))
    try:
        for rel in ("data/processed/matches.db", "data/processed/features.db"):
            src = REPO / rel
            dst = tmp / Path(rel).name
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
        feat_src = "\n".join((p.read_text(encoding="utf-8"))
                             for p in sorted((REPO / "src/features").glob("*.py")))

        rule("STEP 2 -- ACTUAL FEATURE CONTRACT INVENTORY (79 numeric)")
        fam_of = {}
        for fam, fn in FAMILIES.items():
            for c in numeric:
                if fn(c):
                    fam_of.setdefault(c, fam)
        unmatched = [c for c in numeric if c not in fam_of]

        def horizon(c):
            for h in ("last5", "last10", "season"):
                if h in c:
                    return h
            return "unspecified"

        def side(c):
            return "home" if c.startswith("home") else ("away" if c.startswith("away") else "n/a")

        def kind(c):
            if "diff" in c:
                return "difference"
            if "per_match" in c:
                return "rate"
            if "strength" in c or "attack" in c or "defence" in c:
                return "strength"
            if "form" in c or "points" in c or "streak" in c:
                return "form"
            if "venue" in c:
                return "season statistic (venue-restricted)"
            return "level"

        print("| # | feature | D-34 family | horizon | side | metric type |")
        print("|---:|---|---|---|---|---|")
        for i, c in enumerate(numeric, 1):
            print(f"| {i} | {c} | {fam_of.get(c, 'UNMATCHED')} | {horizon(c)} | {side(c)} "
                  f"| {kind(c)} |")
        print(f"\n  D-34-UNMATCHED features ({len(unmatched)}):")
        for c in unmatched:
            print(f"    {c}")

        rule("STEP 3 -- INFORMATION COVERAGE AUDIT (contract only)")
        print("| Category | present features (n) | horizons covered | home | away "
              "| direct or derived |")
        print("|---|---:|---|---:|---:|---|")
        cat_members = {}
        for cat, toks in CATEGORIES.items():
            ms = hit(toks, numeric)
            cat_members[cat] = ms
            hz = sorted({horizon(c) for c in ms}) or ["-"]
            print(f"| {cat} | {len(ms)} | {','.join(hz)} | "
                  f"{sum(1 for c in ms if side(c)=='home')} | "
                  f"{sum(1 for c in ms if side(c)=='away')} | "
                  f"{'derived (rolling/season aggregates)' if ms else 'none in contract'} |")
        print("\n  absent dimensions are reported in STEP 6 with schema evidence, not here.")

        rule("STEP 4 -- DRAW-SPECIFIC INFORMATION PROFILE (validation-only associations)")
        raw_dh = {c: smd(X[c].to_numpy(float)[isD], X[c].to_numpy(float)[isH]) for c in numeric}
        raw_da = {c: smd(X[c].to_numpy(float)[isD], X[c].to_numpy(float)[isA]) for c in numeric}
        print("| Group | n | mean \\|SMD\\| D-H | median | max | sign coherent D-H "
              "| home | away |")
        print("|---|---:|---:|---:|---:|---|---:|---:|")
        for label, members in [(f"FAMILY {k}", [c for c in numeric if fam_of.get(c) == k])
                               for k in FAMILIES] + \
                              [("UNMATCHED", unmatched)] + \
                              [(f"CATEGORY {k}", v) for k, v in cat_members.items()]:
            if not members:
                print(f"| {label} | 0 | - | - | - | - | 0 | 0 |")
                continue
            v = np.array([raw_dh[c] for c in members], dtype=float)
            ok = ~np.isnan(v)
            print(f"| {label} | {len(members)} | {np.nanmean(np.abs(v)):.4f} | "
                  f"{np.nanmedian(np.abs(v)):.4f} | {np.nanmax(np.abs(v)):.4f} | "
                  f"{len({np.sign(x) for x in v[ok]}) == 1 if ok.any() else 'n/a'} | "
                  f"{sum(1 for c in members if side(c)=='home')} | "
                  f"{sum(1 for c in members if side(c)=='away')} |")
        print("\n  Association only. Not importance. Not causal.")

        rule("STEP 5 -- REDUNDANCY vs ABSENCE (per information category)")
        Xn = X[numeric].apply(pd.to_numeric, errors="coerce")
        R = np.nan_to_num(Xn.corr().to_numpy(), nan=0.0)
        ix = {c: i for i, c in enumerate(numeric)}
        print("| Category | n | raw sep (mean\\|SMD\\|) | max within-category \\|r\\| "
              "| max cross-category \\|r\\| | reading |")
        print("|---|---:|---:|---:|---:|---|")
        for cat, ms in cat_members.items():
            if not ms:
                print(f"| {cat} | 0 | - | - | - | category ABSENT from contract |")
                continue
            jj = [ix[c] for c in ms]
            sub = np.abs(R[np.ix_(jj, jj)])
            m = ~np.eye(len(jj), dtype=bool) if len(jj) > 1 else np.zeros_like(sub, bool)
            other = [i for i in range(len(numeric)) if i not in jj]
            cross = np.abs(R[np.ix_(jj, other)]).max() if other else float("nan")
            sep = np.nanmean([abs(raw_dh[c]) for c in ms])
            within = sub[m].max() if m.any() else float("nan")
            if sep < 0.10:
                rd = "exists but TOO WEAK"
            elif within >= 0.90 or cross >= 0.90:
                rd = "exists and REDUNDANT"
            else:
                rd = "exists, separation present"
            print(f"| {cat} | {len(ms)} | {sep:.4f} | {within:.4f} | {cross:.4f} | {rd} |")
        print("\n  These four readings are kept distinct: redundant / poorly represented /")
        print("  too weak / genuinely absent. STEP 6 supplies the absence evidence.")

        rule("STEP 6 -- CANDIDATE DIMENSIONS: SCHEMA + SOURCE EVIDENCE")
        print("| Dimension | in contract | derived in features.db | raw in matches.db "
              "| computed in src/features | raw population | CLASSIFICATION |")
        print("|---|---|---|---|---|---:|---|")
        cls_count = {"PRESENT": 0, "PARTIALLY REPRESENTED": 0,
                     "ABSENT FROM CURRENT CONTRACT": 0, "NOT VERIFIABLE FROM AVAILABLE SOURCE": 0}
        for dim, (rawtok, dertok, contok) in CANDIDATES.items():
            inc = hit(contok, numeric)
            der = hit(dertok, der_cols)
            raw = hit(rawtok, raw_cols)
            insrc = any(t in feat_src.lower() for t in dertok)
            popv = max([pop.get(c, 0.0) for c in raw], default=float("nan"))
            if inc:
                klass = "PRESENT"
            elif der or raw:
                klass = ("ABSENT FROM CURRENT CONTRACT" if (der or raw) and not inc
                         else "PARTIALLY REPRESENTED")
                if der and raw:
                    klass = "ABSENT FROM CURRENT CONTRACT"
            elif insrc:
                klass = "PARTIALLY REPRESENTED"
            else:
                klass = "NOT VERIFIABLE FROM AVAILABLE SOURCE"
            cls_count[klass] = cls_count.get(klass, 0) + 1
            print(f"| {dim} | {len(inc)} col(s) | {len(der)} col(s) | {len(raw)} col(s) "
                  f"| {insrc} | {popv:.1f}% | {klass} |")
        print("\n  evidence detail (first 6 matching column names per source):")
        for dim, (rawtok, dertok, contok) in CANDIDATES.items():
            print(f"    {dim}")
            print(f"      contract : {hit(contok, numeric)[:6] or 'none'}")
            print(f"      derived  : {hit(dertok, der_cols)[:6] or 'none'}")
            print(f"      raw      : {hit(rawtok, raw_cols)[:6] or 'none'}")

        rule("STEP 7 -- TEMPORAL INFORMATION AUDIT (contract definitions only)")
        print("| Horizon | features in contract | families covered |")
        print("|---|---:|---|")
        for h in ("last5", "last10", "season", "unspecified"):
            ms = [c for c in numeric if horizon(c) == h]
            fams = sorted({fam_of.get(c, "UNMATCHED") for c in ms})
            print(f"| {h} | {len(ms)} | {','.join(fams) or '-'} |")
        for lab, toks in (("opponent-adjusted recent performance", ("opp", "adjusted")),
                          ("changing team state / trend", ("trend", "delta", "change", "momentum")),
                          ("current match-state", ("ht_", "elapsed", "game_state"))):
            print(f"  {lab:42s} contract={hit(toks, numeric) or 'none'}   "
                  f"derived={hit(toks, der_cols)[:4] or 'none'}   raw={hit(toks, raw_cols)[:4] or 'none'}")

        rule("STEP 8 -- HOME/AWAY INFORMATION AUDIT")
        print("| Aspect | contract evidence | CLASSIFICATION |")
        print("|---|---|---|")
        aspects = {
            "home team's home-specific state": ("home_venue",),
            "away team's away-specific state": ("away_venue",),
            "relative home-vs-away state": ("relative", "vs_", "_gap"),
            "matchup-specific balance": ("h2h", "matchup", "opponent"),
        }
        for asp, toks in aspects.items():
            ms = hit(toks, numeric)
            der = hit(toks, der_cols)
            klass = ("PRESENT" if ms else
                     "ABSENT FROM CURRENT CONTRACT" if der else
                     "ABSENT" if not der else "NOT VERIFIABLE")
            print(f"| {asp} | {ms[:4] or 'none'} | {klass} |")
        n_home = sum(1 for c in numeric if side(c) == "home")
        n_away = sum(1 for c in numeric if side(c) == "away")
        n_rel = len(hit(("diff", "gap", "relative"), numeric))
        n_side_diff = len([c for c in numeric if "diff" in c and side(c) in ("home", "away")])
        print(f"\n  home-side features {n_home}   away-side features {n_away}   "
              f"features containing 'diff'/'gap'/'relative': {n_rel}")
        print(f"  of those, WITHIN-TEAM differences (e.g. a team's own for-minus-against): "
              f"{n_side_diff}")
        print("  A within-team difference describes ONE team; it is not a home-vs-away")
        print("  relational term. STEP 9 tests for BETWEEN-team terms explicitly.")

        rule("STEP 9 -- RELATIONAL INFORMATION GAP (between-team terms)")
        print("| Relational dimension | explicit in contract | derived in features.db "
              "| indirectly derivable | CLASSIFICATION |")
        print("|---|---|---|---|---|")
        for dim, toks in RELATIONAL.items():
            inc = hit(toks, numeric)
            der = hit(toks, der_cols)
            derivable = bool(n_home and n_away)
            klass = ("PRESENT" if inc else
                     "ABSENT FROM CURRENT CONTRACT" if der else
                     "ABSENT")
            print(f"| {dim} | {inc or 'none'} | {der[:3] or 'none'} | "
                  f"{'yes (home and away blocks both present)' if derivable else 'no'} | {klass} |")
        print("\n  'Indirectly derivable' means a linear model COULD form the difference")
        print("  itself given both sides -- it does not mean the term is represented.")

        rule("STEP 10 -- INFORMATION GAP MATRIX")
        print("| Dimension | current representation | Draw separation evidence | redundancy "
              "| linear alignment | stability | contract status | evidence source | conclusion |")
        print("|---|---|---|---|---|---|---|---|---|")
        for cat, ms in cat_members.items():
            if ms:
                jj = [ix[c] for c in ms]
                sub = np.abs(R[np.ix_(jj, jj)])
                m = ~np.eye(len(jj), dtype=bool) if len(jj) > 1 else np.zeros_like(sub, bool)
                within = sub[m].max() if m.any() else float("nan")
                sep = np.nanmean([abs(raw_dh[c]) for c in ms])
                concl = ("REDUNDANTLY REPRESENTED" if within >= 0.90 else
                         "WEAKLY REPRESENTED" if sep < 0.10 else
                         "POORLY REPRESENTED")
                print(f"| {cat} | {len(ms)} contract cols | mean\\|SMD\\|={sep:.4f} | "
                      f"max\\|r\\|={within:.4f} | D-38 r=+0.069/+0.054 | D-39/40 unstable "
                      f"| in contract | features.db + contract | {concl} |")
            else:
                print(f"| {cat} | none | n/a | n/a | n/a | n/a | not in contract "
                      f"| contract inspection | ABSENT |")
        for dim, toks in RELATIONAL.items():
            inc = hit(toks, numeric)
            print(f"| RELATIONAL: {dim} | {'contract' if inc else 'none'} | n/a | n/a | n/a "
                  f"| n/a | {'in contract' if inc else 'not in contract'} "
                  f"| contract inspection | {'SUFFICIENTLY REPRESENTED' if inc else 'ABSENT'} |")

        rule("STEP 11 -- SINGLE INFORMATION GAP IDENTIFICATION (measurement only)")
        absent_rel = [d for d, t in RELATIONAL.items() if not hit(t, numeric)]
        absent_dim = [d for d, (rt, dt, ct) in CANDIDATES.items()
                      if not hit(ct, numeric) and (hit(dt, der_cols) or hit(rt, raw_cols))]
        print(f"  relational dimensions with NO explicit contract term: {len(absent_rel)}")
        for d in absent_rel:
            print(f"    {d}")
        print(f"\n  candidate dimensions present in data but NOT in contract: {len(absent_dim)}")
        for d in absent_dim:
            print(f"    {d}")
        print("\n  PRIMARY CANDIDATE / ALTERNATIVES / NON-DISCRIMINATION:")
        print("  This script prints the measurements. It does NOT nominate a primary")
        print("  candidate, because selection requires weighing evidence across D-38..D-41")
        print("  and that is the written diagnosis's job. Where several dimensions are")
        print("  simultaneously absent and consistent with the observed Draw failure, the")
        print("  evidence above does not discriminate between them, and the lists are")
        print("  printed in full rather than reduced to one.")

        rule("STEP 12 -- PRE-REGISTERED STEP 6 CLOSURE INPUTS")
        print(f"  current numeric contract size            : {len(numeric)}")
        print(f"  information categories with >=1 contract feature : "
              f"{sum(1 for v in cat_members.values() if v)}/{len(CATEGORIES)}")
        print(f"  relational dimensions explicitly represented     : "
              f"{len(RELATIONAL) - len(absent_rel)}/{len(RELATIONAL)}")
        print(f"  candidate dimensions by classification           : {cls_count}")
        print(f"  D-34-unmatched contract features                 : {len(unmatched)}")
        print("\n  PRIMARY INFORMATION GAP : not selected by this script")
        print("  EVIDENCE / COUNTER-EVIDENCE / AMBIGUITIES / CONFIDENCE : written diagnosis")
        print("\n  No improvement is proposed. No feature is selected. No model is modified.")

        mcon.close(); fcon.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rule("STEP 13 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))
    print("\nFILES MODIFIED: NONE")
    print("DATABASES MODIFIED: NONE")
    print("MODEL MODIFIED: NONE")
    print("ARTIFACT MODIFIED: NONE")
    print("2025/26 RESULTS ACCESSED: NO")
    print("\nD-41 COMPLETE -- AUDIT ONLY, NOTHING CHANGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
