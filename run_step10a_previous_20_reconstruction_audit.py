"""STEP 10A -- PREVIOUS 20-MATCH RECONSTRUCTION AUDIT.

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_step10a_previous_20_reconstruction_audit.py

OBJECTIVE
Recover the exact 20 fixture ids that produced the recorded 9/20, using
project evidence only. This script does NOT run a 40-match evaluation, does
not retrain, does not fit, does not tune, and writes nothing.

WHAT THIS SCRIPT WILL NOT DO
It will not search for a selection that happens to yield 9/20. Candidates are
admitted ONLY if a project artifact documents the mechanism. Any count value
other than the one documented in predict_blind_2025_26.py's own usage block is
reported as UNRESOLVED, never accepted -- even if it reproduces 9/20 -- because
accepting it would be outcome-optimised selection, which the governing rules
forbid.

WHY THE 9/20 ANCHOR IS WEAKER THAN IT LOOKS -- ESTABLISHED BY SOURCE READING
  * predict_blind_2025_26.py: "No prediction is saved -- stdout only.
    No aggregate metric is computed here"
  * reveal_results_2025_26.py: "It computes NO aggregate metric" /
    "compare manually against the prediction output captured earlier"
Neither tool ever computed a correctness count, and neither persisted anything.
The 9/20 figure was therefore produced by a MANUAL comparison of two separately
printed 20-row tables, and no machine-readable record of it exists anywhere in
the project. That does not make 9/20 wrong. It does mean the anchor and the
reconstruction are not the same kind of evidence, and a one-match discrepancy
between them cannot be resolved by this script alone.

GROUND-TRUTH SOURCE IS NOT THE DISCREPANCY
reveal_results_2025_26.py derives the outcome from matches.db home_goals /
away_goals; the earlier 40-match script derived it from features.db
label_result. STEP 4 below re-checks both sources on all 20 fixtures. They were
measured to agree on every one, so the 8-vs-9 gap is NOT a ground-truth
definition artefact.
"""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RECORDED_CORRECT = 9
RECORDED_N = 20
EXPECTED_MD5 = {
    "data/processed/features.db": "e7ebe7fc07040a5927683c35b6371e63",
    "data/processed/matches.db": "fdeed042096fa1c851aaee6c84995247",
}
ARTIFACT_MD5 = "5e504427712b35778bb8a62a8496c7cd"
FORBIDDEN_SEASONS = [602681, 667780, 725788, 725793, 762170]
BLIND_SCRIPT = "predict_blind_2025_26.py"
REVEAL_SCRIPT = "reveal_results_2025_26.py"

PRODUCTION_ASSUMPTIONS = (
    'SimpleImputer(strategy="median")',
    "StandardScaler()",
    "np.hstack([scaled, onehot])",
    "self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]",
    "LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
)
EVIDENCE_TOKENS = ("9/20", "9 / 20", "20 matches", "20-match", "blind", "2025/26",
                   "predict_blind_2025_26", "fixture_id", "correct")
LABEL = {"H": "HOME WIN", "D": "DRAW", "A": "AWAY WIN"}


def rule(t): print("\n" + "=" * 126); print(t); print("=" * 126)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def abort(code, msg):
    print("\n" + "!" * 126)
    print(msg)
    print("!" * 126)
    print(f"\n{code}")
    raise SystemExit(1)


def stop(msg):
    abort("STEP 10A ABORTED\nHISTORICAL 20-MATCH UNIVERSE NOT RECOVERED", msg)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        stop(f"stop condition: {label}")


def snapshot():
    snap = {}
    for rel in EXPECTED_MD5:
        snap[rel] = md5(REPO / rel)
    snap["src/models/train.py"] = md5(REPO / "src/models/train.py")
    art = REPO / "data/models/v1_logreg.pkl"
    if art.exists():
        snap["artifact"] = md5(art)
    lg = REPO / "data/processed/leagues"
    if lg.exists():
        for p in sorted(lg.glob("*.db")):
            snap[f"leagues/{p.name}"] = md5(p)
    for rel in json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                          .read_text())["locked_input_checksums"]:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


def main():
    # ================================================================= RULE 1
    rule("RULE 1 -- PRODUCTION SOURCE TRACE (printed before any selection or prediction)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:76]}", tok in tsrc)
    try:
        import sklearn
    except ImportError as exc:
        stop(f"scikit-learn is required: {exc}. Do not substitute another estimator.")

    from models import artifact as artifact_mod
    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.candidate_contract import predict_candidate
    from models.config import (
        FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS, MODEL_VERSION,
        RECOMMENDED_CONTEXT_FEATURE, REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS,
    )
    from models.data import load_supervised_dataset

    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    check("production contract == exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("numeric production features == exactly 79", len(numeric) == 79)
    check("competition_id is the only categorical production column",
          set(MODEL_B_COLUMNS) - set(numeric) == {RECOMMENDED_CONTEXT_FEATURE})
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")
    check("C == 1.0 in source", "C=1.0" in tsrc)
    check("random_state == 0 in source", "random_state=0" in tsrc)
    check("max_iter == 2000 in source", "max_iter=2000" in tsrc)

    # ================================================================= STEP 1
    rule("STEP 1 -- INTEGRITY GATE")
    print(f"  Python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"pandas {pd.__version__}  sklearn {sklearn.__version__}")
    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(rel, before[rel] == exp, before[rel])
    check("v1_logreg.pkl unchanged", before.get("artifact") == ARTIFACT_MD5,
          str(before.get("artifact")))
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    bad = [r for r, v in pins.items() if before[f"pin:{r}"] != v["expected"]]
    check("13/13 LOCKED_INPUTS unchanged", not bad, str(bad))
    print(f"  src/models/train.py : {before['src/models/train.py']}")
    test_ids = set()
    for s in FINAL_TEST_SEASONS:
        test_ids.update(SEASON_NAME_TO_IDS[s])
    print(f"  forbidden / final-test season ids : {sorted(test_ids)}")
    check("forbidden seasons match the documented list",
          sorted(test_ids) == sorted(FORBIDDEN_SEASONS))

    # ================================================================= STEP 2
    rule("STEP 2 -- HISTORICAL EVIDENCE INVENTORY (filenames AND contents)")
    skip_dirs = {".git", "__pycache__", ".venv", "venv", "node_modules"}
    exts = {".py", ".md", ".txt", ".log", ".json", ".csv", ".ipynb", ".ps1",
            ".bat", ".cmd", ".sh", ".rst", ".yaml", ".yml"}
    hits = []
    for p in sorted(REPO.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        if any(d in p.parts for d in skip_dirs):
            continue
        if p.name == Path(__file__).name:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        found = [t for t in EVIDENCE_TOKENS if t in txt]
        if not found:
            continue
        rel = p.relative_to(REPO).as_posix()
        hits.append({
            "path": rel,
            "type": p.suffix.lstrip("."),
            "fixture_ids": "fixture_id" in txt,
            "predictions": any(k in txt for k in ("prob_home", "P(H)", "predicted", "prob_draw")),
            "selection_logic": "step = len(" in txt or "evenly spaced" in txt,
            "result_9_20": ("9/20" in txt) or ("9 / 20" in txt),
            "tokens": found,
        })
    print(f"  files scanned for evidence tokens; {len(hits)} contain at least one\n")
    print("| path | type | fixture ids | predictions | selection logic | records 9/20 |")
    print("|---|---|---|---|---|---|")
    for h in hits:
        print(f"| {h['path']} | {h['type']} | {'YES' if h['fixture_ids'] else 'no'} "
              f"| {'YES' if h['predictions'] else 'no'} "
              f"| {'YES' if h['selection_logic'] else 'no'} "
              f"| {'YES' if h['result_9_20'] else 'no'} |")

    holds_the_20 = [h for h in hits if h["result_9_20"] and h["fixture_ids"]
                    and h["path"] != BLIND_SCRIPT]
    print(f"\n  artifacts holding BOTH a 9/20 record AND fixture ids: {len(holds_the_20)}")
    print("  version control present in repo : "
          f"{'YES' if (REPO / '.git').exists() else 'NO -- no prior script versions recoverable'}")
    print("  shell/PowerShell history inside the project : "
          f"{'YES' if list(REPO.rglob('*history*.ps1')) else 'NO'}")
    print("\n  KEY SOURCE FACTS (quoted from the tools themselves):")
    bsrc = (REPO / BLIND_SCRIPT).read_text(encoding="utf-8")
    rsrc = (REPO / REVEAL_SCRIPT).read_text(encoding="utf-8")
    check(f"{BLIND_SCRIPT} states it saves no prediction",
          "No prediction is saved" in bsrc)
    check(f"{BLIND_SCRIPT} states it computes no aggregate metric",
          "No aggregate metric is computed" in bsrc)
    check(f"{REVEAL_SCRIPT} states it computes NO aggregate metric",
          "computes NO aggregate metric" in rsrc)
    check(f"{REVEAL_SCRIPT} instructs manual comparison",
          "compare manually" in rsrc)
    print("    -> no tool in this project ever COMPUTED or STORED a correctness count.")
    print("    -> the recorded 9/20 is a manual tally with no machine-readable record.")

    # ================================================================= STEP 3
    rule("STEP 3 -- RECOVER THE ORIGINAL SELECTION MECHANISM (full path, via AST)")
    tree = ast.parse(bsrc)
    args_found = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            name = node.args[0].value if node.args else "?"
            kw = {k.arg: ast.unparse(k.value) for k in node.keywords}
            args_found[name] = kw
    print("  command-line arguments and defaults:")
    for k, v in args_found.items():
        print(f"    {k:16s} default={v.get('default', '<none>'):8s} "
              f"type={v.get('type', '-'):6s} nargs={v.get('nargs', '-')}")
    check("--count has NO default (a value must have been supplied explicitly)",
          "default" not in args_found.get("--count", {}))
    print("\n  documented usage in the script's own docstring:")
    for line in bsrc.splitlines()[:6]:
        if "predict_blind_2025_26.py" in line:
            print(f"    {line.strip()}")

    forbidden_calls = {"sample", "shuffle", "choice", "seed", "randint"}
    called = {n.func.attr for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    check("no sampling / shuffling / random seed anywhere in the selection path",
          not (forbidden_calls & called), str(sorted(forbidden_calls & called)))
    check("no hard-coded manual fixture list in the script",
          "selected = [" not in bsrc.replace("selected = [int(x) for x in args.fixture_ids]", ""))
    for tok, desc in (
        ("WHERE season_id IN ({q}) AND status = 'FT'", "season + status filter"),
        ("ORDER BY unix ASC, fixture_id ASC", "ordering"),
        ("step = len(ordered) / args.count", "step"),
        ("idx = sorted({int(i * step) for i in range(args.count)})", "index generation + dedup"),
    ):
        check(f"selection path component recovered: {desc}", tok in bsrc)
    check("no date filter beyond season_id", "unix >" not in bsrc and "unix <" not in bsrc)
    check("no competition filter in the selection query",
          "competition_id" not in bsrc.split("STEP 3")[-1].split("STEP 4")[0])
    print("\n  RECONSTRUCTED HISTORICAL SELECTION PATH (complete):")
    print("    1. matches.db, mode=ro")
    print("    2. WHERE season_id IN FINAL_TEST_SEASONS AND status = 'FT'")
    print("    3. ORDER BY unix ASC, fixture_id ASC")
    print("    4. step = len(ordered) / count")
    print("    5. idx  = sorted({int(i * step) for i in range(count)})   [set = dedup]")
    print("    6. selected = [ordered[i] for i in idx]")
    print("    7. hard-fail if any selected fixture lacks a v1.0 feature row")
    print("  No randomness, no seed, no sampling, no manual list, no environment")
    print("  dependence, no hidden exclusion. Fully determined by ONE integer: count.")

    # ================================================================= STEP 4
    rule("STEP 4 -- DATABASE STATE / HISTORICAL DATA AUDIT (read-only)")
    mdb = REPO / "data/processed/matches.db"
    con = sqlite3.connect(f"file:{mdb.resolve()}?mode=ro", uri=True)
    try:
        q = ",".join(str(i) for i in sorted(test_ids))
        rows = con.execute(
            f"SELECT fixture_id, home_name, away_name, unix, season_id, competition_id, "
            f"home_goals, away_goals, status FROM fixtures "
            f"WHERE season_id IN ({q}) AND status = 'FT' "
            f"ORDER BY unix ASC, fixture_id ASC").fetchall()
        n_all = con.execute(
            f"SELECT COUNT(*) FROM fixtures WHERE season_id IN ({q})").fetchone()[0]
        statuses = con.execute(
            f"SELECT status, COUNT(*) FROM fixtures WHERE season_id IN ({q}) "
            f"GROUP BY status").fetchall()
    finally:
        con.close()
    ordered = [r[0] for r in rows]
    meta = {r[0]: r for r in rows}
    L = len(ordered)
    print(f"  2025/26 fixtures (all statuses) : {n_all}")
    print(f"  status breakdown               : {dict(statuses)}")
    print(f"  FT universe size L             : {L}")
    print(f"  unique fixture_ids             : {len(set(ordered))}")
    check("no duplicate fixture_id in the universe", len(set(ordered)) == L)
    nulls = [r[0] for r in rows if r[3] is None]
    check("no NULL unix in the universe (ordering is total)", not nulls, str(nulls[:5]))
    ties = L - len({r[3] for r in rows})
    print(f"  fixtures sharing a unix with another : {ties} "
          f"(broken deterministically by fixture_id ASC)")
    print("  NOTE: any change to matches.db would change L and therefore every index.")
    print(f"  matches.db checksum is at its pinned value, so L={L} is the historical L.")

    ds = load_supervised_dataset(REPO / "data/processed/features.db")
    have_features = set(ds.metadata["fixture_id"].tolist())

    # ================================================================= STEP 5
    rule("STEP 5 -- CANDIDATE RECONSTRUCTION (evidence-supported mechanisms only)")

    def select(count):
        step = L / count
        return [ordered[i] for i in sorted({int(i * step) for i in range(count)})]

    # The mechanism's entire parameter space is one integer. Report which counts
    # yield exactly 20 unique fixtures -- as a PROPERTY OF THE MECHANISM, not as
    # a search. Only the count documented in the script's usage block is admitted
    # as an evidence-supported candidate.
    yields_20 = [c for c in range(1, L + 1) if len(select(c)) == RECORDED_N]
    print(f"  counts yielding exactly {RECORDED_N} unique fixtures: {yields_20}")
    documented = [c for c in yields_20 if f"--count {c}" in bsrc]
    print(f"  of those, documented in {BLIND_SCRIPT}'s usage block: {documented}")
    check("exactly one count value is documented by a project artifact",
          len(documented) == 1, str(documented))
    cand_count = documented[0]
    candidates = {
        f"--count {cand_count} (documented in {BLIND_SCRIPT} usage block)": select(cand_count),
    }
    print(f"\n  ADMITTED CANDIDATE MECHANISMS: {len(candidates)}")
    for name in candidates:
        print(f"    - {name}")
    others = [c for c in yields_20 if c not in documented]
    if others:
        print(f"  NOT admitted (no artifact documents them): --count {others}")
        print("    These are NOT evaluated against 9/20. Selecting among them by their")
        print("    correctness count would be outcome-optimised selection, which the")
        print("    governing rules forbid.")

    # ================================================================= STEP 6
    rule("STEP 6 -- HISTORICAL 9/20 VERIFICATION (frozen artifact; fit() never called)")
    art = artifact_mod.load(REPO / "data/models/v1_logreg.pkl")
    check("artifact contract == frozen 80 columns, in order",
          tuple(art.feature_columns) == tuple(MODEL_B_COLUMNS))
    check("artifact class_order == ['H','D','A']", art.class_order == ["H", "D", "A"])
    check("artifact estimator is the frozen configuration",
          art.model.C == 1.0 and art.model.max_iter == 2000 and art.model.random_state == 0)
    train_ids = set()
    for s in FINAL_TRAIN_SEASONS:
        train_ids.update(SEASON_NAME_TO_IDS[s])
    stats = np.asarray(art.preprocessor._imputer.statistics_, dtype=float)
    dists = {
        "blind (excludes 2025/26)": float(np.nanmax(np.abs(
            stats - ds.X[ds.metadata["season_id"].isin(train_ids).values][numeric]
            .median(skipna=True).to_numpy(dtype=float)))),
        "all rows (includes 2025/26)": float(np.nanmax(np.abs(
            stats - ds.X[numeric].median(skipna=True).to_numpy(dtype=float)))),
    }
    for k, v in dists.items():
        print(f"    imputer max|diff| vs {k:32s} = {v:.6e}")
    check("artifact was NOT fitted on 2025/26",
          min(dists, key=dists.get).startswith("blind") and min(dists.values()) < 1e-9)

    verdicts = {}
    for name, sel in candidates.items():
        print(f"\n  --- CANDIDATE: {name} ---")
        print(f"      source artifact: {BLIND_SCRIPT} (usage block + selection code)")
        print(f"      exact rule     : step = L/{cand_count}; "
              f"idx = sorted({{int(i*step) for i in range({cand_count})}}); L = {L}")
        if len(set(sel)) != RECORDED_N:
            print(f"      REJECTED: {len(set(sel))} unique fixtures, expected {RECORDED_N}")
            continue
        miss = [f for f in sel if f not in have_features]
        check("all candidate fixtures have a v1.0 feature row", not miss, str(miss[:5]))
        check("all candidate fixtures are in the 2025/26 holdout",
              all(meta[f][4] in test_ids for f in sel))

        mask = ds.metadata["fixture_id"].isin(sel).values
        X = ds.X[mask][list(art.feature_columns)]
        m = ds.metadata[mask].reset_index(drop=True)
        check("X carries exactly the 80 contract columns",
              list(X.columns) == list(MODEL_B_COLUMNS))
        out = predict_candidate(art.model, art.preprocessor, X, m["fixture_id"].tolist())
        check("prediction class order == ['H','D','A']", list(out.class_order) == ["H", "D", "A"])
        pos = {f: i for i, f in enumerate(out.fixture_ids)}
        P = np.array([np.asarray(out.probabilities)[pos[f]] for f in sel], dtype=float)
        check("every probability row sums to 1",
              bool(np.all(np.abs(P.sum(axis=1) - 1.0) < 1e-9)),
              f"max|sum-1|={float(np.max(np.abs(P.sum(axis=1)-1.0))):.3e}")
        pred = np.array(list(out.class_order))[P.argmax(axis=1)]

        # Ground truth from BOTH sources, cross-checked.
        lab_feat = {int(r.fixture_id): r.label_result for r in
                    pd.DataFrame({"fixture_id": ds.metadata["fixture_id"][mask].values,
                                  "label_result": ds.y[mask].values}).itertuples()}
        actual, disagree = [], []
        for f in sel:
            hg, ag = meta[f][6], meta[f][7]
            g = "H" if hg > ag else ("A" if ag > hg else "D")
            if g != lab_feat[f]:
                disagree.append(f)
            actual.append(g)
        actual = np.array(actual)
        check("matches.db goals and features.db label_result agree on every fixture",
              not disagree, str(disagree))

        correct = pred == actual
        n_correct = int(correct.sum())
        print(f"\n| # | fixture_id | date | home | away | P(H) | P(D) | P(A) | predicted "
              f"| confidence | actual | correct |")
        print("|---:|---:|---|---|---|---:|---:|---:|---|---:|---|---|")
        for i, f in enumerate(sel):
            r = meta[f]
            d = datetime.datetime.utcfromtimestamp(r[3]).strftime("%Y-%m-%d")
            print(f"| {i+1} | {f} | {d} | {r[1][:20]} | {r[2][:20]} | {P[i,0]:.4f} "
                  f"| {P[i,1]:.4f} | {P[i,2]:.4f} | {LABEL[pred[i]]} | {P[i].max():.4f} "
                  f"| {LABEL[actual[i]]} | {'YES' if correct[i] else 'no'} |")
        print(f"\n      correct: {n_correct}/{RECORDED_N}      recorded: "
              f"{RECORDED_CORRECT}/{RECORDED_N}      "
              f"{'MATCH' if n_correct == RECORDED_CORRECT else 'MISMATCH'}")
        verdicts[name] = (sel, n_correct)

    # ================================================================= STEP 7
    rule("STEP 7 -- RECONCILIATION")
    reproducing = {k: v for k, v in verdicts.items() if v[1] == RECORDED_CORRECT}
    print(f"  evidence-supported candidates evaluated : {len(verdicts)}")
    print(f"  candidates reproducing exactly {RECORDED_CORRECT}/{RECORDED_N} : {len(reproducing)}")

    if len(reproducing) > 1:
        for k, v in reproducing.items():
            print(f"    - {k}: {v[0]}")
        abort("STEP 10A ABORTED\nHISTORICAL 20-MATCH UNIVERSE AMBIGUOUS",
              "More than one evidence-supported candidate reproduces the recorded result. "
              "Per rule 7, no candidate is chosen.")

    if len(reproducing) == 0:
        only = next(iter(verdicts.items()), None)
        print("\n  WHAT IS MISSING, PRECISELY:")
        print("   1. No project artifact stores the historical 20 fixture ids.")
        print(f"      {BLIND_SCRIPT} saves nothing; {REVEAL_SCRIPT} saves nothing.")
        print("   2. No project artifact stores the historical predictions.")
        print("   3. No tool ever COMPUTED the correctness count -- 9/20 was tallied")
        print("      by hand from two separately printed tables, and that tally is the")
        print("      only record of it.")
        print("   4. There is no version control in the repository, so an earlier")
        print("      variant of the selection code cannot be recovered or ruled out.")
        print("   5. The mechanism is fully determined by one integer (count), and the")
        print("      only value any artifact documents is the one evaluated above.")
        if only:
            print(f"\n   The documented mechanism yields {only[1][1]}/{RECORDED_N}, "
                  f"not {RECORDED_CORRECT}/{RECORDED_N}.")
            print("   Exactly one of the following must be true, and this script cannot")
            print("   decide between them from project evidence:")
            print("     (a) the historical run used a different argument, no longer recorded")
            print("     (b) the manual 9/20 tally was off by one")
            print("\n   The ONE artifact that would settle it is the historical terminal")
            print(f"   output of {BLIND_SCRIPT} and {REVEAL_SCRIPT}. Compare the table")
            print("   above against that output fixture-by-fixture. If the fixture ids")
            print("   match, (b) is established and the anchor should be corrected to")
            print(f"   {only[1][1]}/{RECORDED_N}. If they differ, the historical ids are")
            print("   recovered directly from that output and no inference is needed.")
        abort("STEP 10A ABORTED\nHISTORICAL 20-MATCH UNIVERSE NOT RECOVERED",
              "No evidence-supported candidate reproduces the recorded 9/20. "
              "Nothing is guessed and no substitute selection is adopted.")

    name, (sel, n_correct) = next(iter(reproducing.items()))
    print("\n  PASS -- HISTORICAL 20-MATCH UNIVERSE RECOVERED")
    print(f"    mechanism        : {name}")
    print(f"    source artifact  : {BLIND_SCRIPT}")
    print(f"    fixture ids      : {sel}")
    print(f"    verification     : {n_correct}/{RECORDED_N} == recorded "
          f"{RECORDED_CORRECT}/{RECORDED_N}")
    print(f"    uniquely determined : YES (single admitted candidate, single documented count)")

    # ================================================================= STEP 8
    rule("STEP 8 -- 40-MATCH EXTENSION ASSESSMENT (assessment only -- NOT RUN)")
    s40 = select(2 * cand_count)
    nests = set(sel) <= set(s40)
    print(f"  the same mechanism at count={2 * cand_count} yields {len(set(s40))} unique fixtures")
    print(f"  it contains the recovered 20 : {nests}")
    print(f"  identity int(2k*L/{2*cand_count}) == int(k*L/{cand_count}) for all k : "
          f"{all(int(2*k*L/(2*cand_count)) == int(k*L/cand_count) for k in range(cand_count))}")
    if nests:
        print("  DEFENSIBLE: the extension is the SAME mechanism at a doubled count, and")
        print("  the nesting is an arithmetic property, not a chosen construction.")
        print(f"  overlap {len(set(sel) & set(s40))}, additional {len(set(s40) - set(sel))}")
    else:
        print("  NOT DEFENSIBLE: the mechanism does not nest at a doubled count.")
    print("  The 40-match evaluation is NOT run here. It remains a separate step.")

    # ================================================================= STEP 9
    rule("STEP 9 -- POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified", not changed, str(changed))
    print("    FILES MODIFIED     : NONE")
    print("    DATABASES MODIFIED : NONE")
    print("    MODEL MODIFIED     : NONE")
    print("    ARTIFACTS MODIFIED : NONE")
    print(f"    features.db        : {after['data/processed/features.db']}")
    print(f"    matches.db         : {after['data/processed/matches.db']}")
    print(f"    v1_logreg.pkl      : {after.get('artifact')}")
    print(f"    train.py           : {after['src/models/train.py']}")
    print(f"    LOCKED_INPUTS      : 13/13 unchanged")

    print("\nSTEP 10A COMPLETE")
    print("HISTORICAL 20-MATCH UNIVERSE RECOVERED")
    print("UNIQUE: YES")
    print(f"RESULT: {n_correct}/{RECORDED_N}")
    print("40-MATCH EXTENSION STATUS: READY FOR SEPARATE STEP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
