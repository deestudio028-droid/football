"""POISSON V2 -- STEP 1: GOAL TARGET AUDIT (read-only).

    cd "E:\\Football Prediction Project"
    $env:PYTHONPATH="$PWD\\src"
    python run_step_poisson_1_goal_target_audit.py

Audits whether `label_home_goals` and `label_away_goals` are complete, valid
and internally consistent with `label_result` on the TRAINING seasons only
(2020/21-2024/25). Creates no feature, trains no model, writes no file, and
touches no production source, database or artifact.

2025/26 IS NOT INSPECTED. Its row count is reported once, from metadata only,
purely to prove it was excluded. No 2025/26 goal value or outcome is read.

------------------------------------------------------------------------------
A CAVEAT THAT CHANGES WHAT SECTION 3 CAN PROVE -- READ THIS FIRST
------------------------------------------------------------------------------
`src/features/feature_builder.py` builds all three label columns in ONE dict
literal, from the SAME two source fields:

    "label_home_goals": fixture.get("home_goals"),
    "label_away_goals": fixture.get("away_goals"),
    "label_result": ("H" if (fixture.get("home_goals") or 0)
                          > (fixture.get("away_goals") or 0) ... )

So re-deriving H/D/A from the two goal columns and comparing it to
`label_result` compares a value against a value computed from the same inputs
by the same comparison. It CANNOT fail unless the database was corrupted after
writing. Reporting only that check, and calling the target "verified", would
overstate the evidence.

This audit therefore runs BOTH:

  3a. internal consistency  -- features.db goals vs features.db label_result
        (weak: agreement is guaranteed by construction; a disagreement would
         indicate post-write corruption, which is still worth excluding)
  3b. INDEPENDENT cross-check -- features.db goal labels vs matches.db
        fixtures.home_goals / away_goals, the upstream source
        (strong: a genuinely separate table, joined on fixture_id)

Only 3b can establish that the targets are the real recorded outcomes. The
decision gate requires BOTH.

A further consequence of the shared construction: the `or 0` coalescing in the
`label_result` expression means a NULL goal would silently become 0 in the
comparison while the goal column itself stayed NULL. The None-guard at the end
of the expression prevents this in the normal path, but section 1 checks for
the resulting signature (label_result present while a goal column is NULL)
explicitly rather than assuming the guard held.
------------------------------------------------------------------------------
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from collections import Counter
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
EXPECTED_TRAIN_ROWS = 8983
GOAL_CEILING = 20  # sanity bound only; any breach is REPORTED, never corrected

PRODUCTION_ASSUMPTIONS = (
    'SimpleImputer(strategy="median")',
    "StandardScaler()",
    "np.hstack([scaled, onehot])",
    "self.numeric_columns = [c for c in X_train.columns if c != RECOMMENDED_CONTEXT_FEATURE]",
    "LogisticRegression(max_iter=2000, C=1.0, random_state=0)",
)
LABEL_SOURCE_TOKENS = (
    '"label_home_goals": fixture.get("home_goals")',
    '"label_away_goals": fixture.get("away_goals")',
)


def rule(t): print("\n" + "=" * 122); print(t); print("=" * 122)
def md5(p): return hashlib.md5(Path(p).read_bytes()).hexdigest()


def abort(msg):
    print("\n" + "!" * 122)
    print("STEP 1 ABORTED -- GOAL TARGETS NOT VERIFIED")
    print(msg)
    print("!" * 122)
    raise SystemExit(1)


def check(label, ok, extra=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    if not ok:
        abort(f"stop condition: {label}")


def report(label, ok, extra=""):
    """Records a gate result WITHOUT halting, so the full audit still prints."""
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{('  ' + extra) if extra else ''}")
    return bool(ok)


def snapshot():
    snap = {}
    for rel in EXPECTED_MD5:
        snap[rel] = md5(REPO / rel)
    snap["src/models/train.py"] = md5(REPO / "src/models/train.py")
    snap["src/features/feature_builder.py"] = md5(REPO / "src/features/feature_builder.py")
    art = REPO / "data/models/v1_logreg.pkl"
    if art.exists():
        snap["data/models/v1_logreg.pkl"] = md5(art)
    for rel in json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                          .read_text())["locked_input_checksums"]:
        snap[f"pin:{rel}"] = md5(REPO / rel)
    return snap


def is_integral(v):
    """True only for a genuine whole number. Rejects bool, str and non-integral
    floats. SQLite stores these columns with NO declared type affinity, so the
    Python type is checked directly rather than trusted."""
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, np.integer)):
        return True
    if isinstance(v, (float, np.floating)):
        return float(v).is_integer()
    return False


def main():
    # ================================================================= RULE 1
    rule("RULE 1 -- PRODUCTION SOURCE TRACE (read from source; nothing from memory)")
    tsrc = (REPO / "src/models/train.py").read_text(encoding="utf-8")
    for tok in PRODUCTION_ASSUMPTIONS:
        check(f"train.py contains: {tok[:72]}", tok in tsrc)

    from models.ablation import MODEL_B_COLUMNS
    from models.baselines import CLASS_ORDER
    from models.config import (
        ALLOWED_LABELS, FINAL_TEST_SEASONS, FINAL_TRAIN_SEASONS,
        LABEL_COLUMNS, MODEL_VERSION, RECOMMENDED_CONTEXT_FEATURE,
        REQUIRED_FEATURE_VERSION, SEASON_NAME_TO_IDS, X_EXCLUDED_COLUMNS,
    )
    numeric = [c for c in MODEL_B_COLUMNS if c != RECOMMENDED_CONTEXT_FEATURE]
    check("production contract == exactly 80 columns", len(MODEL_B_COLUMNS) == 80)
    check("numeric production features == exactly 79", len(numeric) == 79)
    check("CLASS_ORDER == ['H','D','A']", list(CLASS_ORDER) == ["H", "D", "A"])
    check("MODEL_VERSION == v1.0", MODEL_VERSION == "v1.0")
    check("REQUIRED_FEATURE_VERSION == v1.0", REQUIRED_FEATURE_VERSION == "v1.0")

    # ================================================================= STEP 0
    rule("STEP 0 -- INTEGRITY GATE + SCHEMA / PARTITION VERIFICATION")
    print(f"  Python {sys.version.split()[0]}   numpy {np.__version__}   pandas {pd.__version__}")
    before = snapshot()
    for rel, exp in EXPECTED_MD5.items():
        check(f"{rel} unchanged", before[rel] == exp, before[rel])
    check("src/models/train.py unchanged",
          before["src/models/train.py"] == TRAIN_PY_MD5, before["src/models/train.py"])
    check("v1_logreg.pkl unchanged",
          before.get("data/models/v1_logreg.pkl") == ARTIFACT_MD5,
          str(before.get("data/models/v1_logreg.pkl")))
    pins = json.loads((REPO / "data/audit/phase4c_prerun_manifest.json")
                      .read_text())["locked_input_checksums"]
    bad = [r for r, v in pins.items() if before[f"pin:{r}"] != v["expected"]]
    check(f"{len(pins)}/{len(pins)} LOCKED_INPUTS unchanged", not bad, str(bad))

    # ---- exact column names, from source constants, not typed by hand -------
    print(f"\n  LABEL_COLUMNS (from src/features/storage.py) : {list(LABEL_COLUMNS)}")
    check("LABEL_COLUMNS is exactly the three expected label columns",
          list(LABEL_COLUMNS) == ["label_home_goals", "label_away_goals", "label_result"],
          str(list(LABEL_COLUMNS)))
    check("every label column is excluded from X by production config",
          set(LABEL_COLUMNS) <= set(X_EXCLUDED_COLUMNS))
    check("no label column appears in the 80-column feature contract",
          not (set(LABEL_COLUMNS) & set(MODEL_B_COLUMNS)))
    print(f"  ALLOWED_LABELS : {tuple(ALLOWED_LABELS)}")

    fdb = REPO / "data/processed/features.db"
    mdb = REPO / "data/processed/matches.db"
    fcon = sqlite3.connect(f"file:{fdb.resolve()}?mode=ro", uri=True)
    mcon = sqlite3.connect(f"file:{mdb.resolve()}?mode=ro", uri=True)
    try:
        info = {r[1]: r[2] for r in fcon.execute("PRAGMA table_info(feature_rows)")}
        for col in LABEL_COLUMNS:
            check(f"feature_rows has column {col}", col in info)
        print("\n  declared SQLite type affinity for the label columns:")
        for col in LABEL_COLUMNS:
            print(f"    {col:20s} {repr(info[col])}")
        print("  NOTE: no declared affinity means SQLite will store whatever Python")
        print("  supplied. Value types are therefore checked directly in section 2.")

        # ---- season partition, from config constants ----------------------
        train_ids, test_ids = set(), set()
        for s in FINAL_TRAIN_SEASONS:
            train_ids.update(SEASON_NAME_TO_IDS[s])
        for s in FINAL_TEST_SEASONS:
            test_ids.update(SEASON_NAME_TO_IDS[s])
        print(f"\n  FINAL_TRAIN_SEASONS : {list(FINAL_TRAIN_SEASONS)}")
        print(f"  FINAL_TEST_SEASONS  : {list(FINAL_TEST_SEASONS)}")
        print(f"  training season_ids : {sorted(train_ids)}")
        print(f"  excluded season_ids : {sorted(test_ids)}")
        check("training and final-test season sets are disjoint",
              not (train_ids & test_ids))
        tq = ",".join(str(i) for i in sorted(train_ids))
        eq = ",".join(str(i) for i in sorted(test_ids))

        # =============================================================== (5)
        rule("SECTION 5 -- ARE THE TARGETS PRE-EXISTING OUTCOMES, OR CONSTRUCTED?")
        fbsrc = (REPO / "src/features/feature_builder.py").read_text(encoding="utf-8")
        for tok in LABEL_SOURCE_TOKENS:
            check(f"feature_builder assigns verbatim: {tok}", tok in fbsrc)
        print("\n  Both goal targets are a direct pass-through of the upstream fixture")
        print("  fields `home_goals` / `away_goals`. They are NOT computed from any")
        print("  engineered feature column.")

        # Prove no feature column contributes to the labels: the labels are
        # assigned in the dict literal BEFORE any row.update(...) call adds
        # features, so no feature value can reach them.
        lit_at = fbsrc.index('"label_home_goals"')
        first_update = fbsrc.index("row.update(")
        check("labels are assigned before any feature block is merged into the row",
              lit_at < first_update, f"label@{lit_at} < first row.update@{first_update}")

        feature_names_in_label_expr = [
            c for c in MODEL_B_COLUMNS
            if c in fbsrc[lit_at:fbsrc.index("row.update(")]
        ]
        check("no production feature name appears in the label construction block",
              not feature_names_in_label_expr, str(feature_names_in_label_expr))

        print("\n  CONSEQUENCE, stated plainly: `label_result` is built in the SAME dict")
        print("  literal from the SAME two fields by the SAME comparison. Section 3a")
        print("  therefore cannot independently validate the targets -- only section 3b,")
        print("  against matches.db, can. Both are run; the gate requires both.")

        # =============================================================== (1)
        rule("SECTION 1 -- COVERAGE (training seasons 2020/21-2024/25 only)")
        total = fcon.execute(
            f"SELECT COUNT(*) FROM feature_rows WHERE season_id IN ({tq})").fetchone()[0]
        n_home = fcon.execute(
            f"SELECT COUNT(*) FROM feature_rows WHERE season_id IN ({tq}) "
            f"AND label_home_goals IS NOT NULL").fetchone()[0]
        n_away = fcon.execute(
            f"SELECT COUNT(*) FROM feature_rows WHERE season_id IN ({tq}) "
            f"AND label_away_goals IS NOT NULL").fetchone()[0]
        n_both = fcon.execute(
            f"SELECT COUNT(*) FROM feature_rows WHERE season_id IN ({tq}) "
            f"AND label_home_goals IS NOT NULL AND label_away_goals IS NOT NULL").fetchone()[0]
        n_res = fcon.execute(
            f"SELECT COUNT(*) FROM feature_rows WHERE season_id IN ({tq}) "
            f"AND label_result IS NOT NULL").fetchone()[0]
        n_orphan = fcon.execute(
            f"SELECT COUNT(*) FROM feature_rows WHERE season_id IN ({tq}) "
            f"AND label_result IS NOT NULL AND (label_home_goals IS NULL "
            f"OR label_away_goals IS NULL)").fetchone()[0]
        print(f"  total training fixtures                 : {total}")
        print(f"  rows with non-null label_home_goals     : {n_home}")
        print(f"  rows with non-null label_away_goals     : {n_away}")
        print(f"  rows with BOTH goals present            : {n_both}")
        print(f"  rows with non-null label_result         : {n_res}")
        print(f"  missing home goals                      : {total - n_home}  "
              f"({100 * (total - n_home) / total:.4f}%)")
        print(f"  missing away goals                      : {total - n_away}  "
              f"({100 * (total - n_away) / total:.4f}%)")
        print(f"  missing either goal                     : {total - n_both}  "
              f"({100 * (total - n_both) / total:.4f}%)")
        print(f"  expected (design doc)                   : ~{EXPECTED_TRAIN_ROWS}")
        gate_cov_a = report("both goal labels complete for every training row",
                            n_both == total, f"{n_both}/{total}")
        gate_cov_b = report("label_result present for exactly the same rows",
                            n_res == n_both, f"{n_res} vs {n_both}")
        gate_cov_c = report("no row has label_result while a goal column is NULL "
                            "(the `or 0` coalescing signature)", n_orphan == 0, str(n_orphan))
        report("training row count matches the design-document figure",
               total == EXPECTED_TRAIN_ROWS, f"{total} vs {EXPECTED_TRAIN_ROWS}")

        excl = fcon.execute(
            f"SELECT COUNT(*) FROM feature_rows WHERE season_id IN ({eq})").fetchone()[0]
        print(f"\n  2025/26 rows EXCLUDED from this audit    : {excl}  "
              f"(count only -- no goal or outcome value read)")

        # =============================================================== (2)
        rule("SECTION 2 -- VALIDITY")
        rows = fcon.execute(
            f"SELECT fixture_id, label_home_goals, label_away_goals, label_result, "
            f"season_id, competition_id FROM feature_rows WHERE season_id IN ({tq}) "
            f"ORDER BY fixture_id").fetchall()
        hg_raw = [r[1] for r in rows]
        ag_raw = [r[2] for r in rows]
        types_h = Counter(type(v).__name__ for v in hg_raw)
        types_a = Counter(type(v).__name__ for v in ag_raw)
        print(f"  observed Python types, home goals : {dict(types_h)}")
        print(f"  observed Python types, away goals : {dict(types_a)}")
        bad_h = [r[0] for r in rows if not is_integral(r[1])]
        bad_a = [r[0] for r in rows if not is_integral(r[2])]
        neg_h = [r[0] for r in rows if is_integral(r[1]) and r[1] < 0]
        neg_a = [r[0] for r in rows if is_integral(r[2]) and r[2] < 0]
        hg = np.array([int(v) for v in hg_raw if is_integral(v)], dtype=int)
        ag = np.array([int(v) for v in ag_raw if is_integral(v)], dtype=int)
        print(f"\n  minimum home goals : {hg.min() if len(hg) else 'n/a'}")
        print(f"  maximum home goals : {hg.max() if len(hg) else 'n/a'}")
        print(f"  minimum away goals : {ag.min() if len(ag) else 'n/a'}")
        print(f"  maximum away goals : {ag.max() if len(ag) else 'n/a'}")
        gate_val_a = report("no non-integer / non-numeric home goals", not bad_h, str(bad_h[:10]))
        gate_val_b = report("no non-integer / non-numeric away goals", not bad_a, str(bad_a[:10]))
        gate_val_c = report("no negative home goals", not neg_h, str(neg_h[:10]))
        gate_val_d = report("no negative away goals", not neg_a, str(neg_a[:10]))
        outl = [(r[0], r[1], r[2]) for r in rows
                if is_integral(r[1]) and is_integral(r[2])
                and (r[1] > GOAL_CEILING or r[2] > GOAL_CEILING)]
        report(f"no scoreline exceeds the sanity bound of {GOAL_CEILING} goals per side",
               not outl, str(outl[:5]))
        print("  (the bound is a reporting sanity check only; nothing is capped or altered)")
        print("\n  highest-scoring training fixtures observed:")
        for fid, h, a in sorted(((r[0], r[1], r[2]) for r in rows
                                 if is_integral(r[1]) and is_integral(r[2])),
                                key=lambda t: -(t[1] + t[2]))[:5]:
            print(f"    fixture {fid}  {h}-{a}  (total {h + a})")

        # =============================================================== (3a)
        rule("SECTION 3a -- INTERNAL CONSISTENCY (weak: guaranteed by construction)")
        derived, stored, mismatches = [], [], []
        for fid, h, a, res, sid, cid in rows:
            if not (is_integral(h) and is_integral(a)) or res is None:
                continue
            d = "H" if h > a else ("A" if h < a else "D")
            derived.append(d)
            stored.append(res)
            if d != res:
                mismatches.append((fid, h, a, res, d, sid, cid))
        n_cmp = len(derived)
        n_agree = sum(1 for d, s in zip(derived, stored) if d == s)
        print(f"  rows compared            : {n_cmp}")
        print(f"  exact agreement          : {n_agree}")
        print(f"  disagreements            : {len(mismatches)}")
        print(f"  disagreement percentage  : "
              f"{100 * len(mismatches) / n_cmp if n_cmp else 0:.6f}%")
        if mismatches:
            print("\n  EVERY DISAGREEMENT:")
            print("| fixture_id | home | away | stored | derived | season_id | competition_id |")
            print("|---:|---:|---:|---|---|---:|---:|")
            for fid, h, a, res, d, sid, cid in mismatches:
                print(f"| {fid} | {h} | {a} | {res} | {d} | {sid} | {cid} |")
        gate_cons_a = report("derived H/D/A agrees with label_result on every row",
                             not mismatches, f"{len(mismatches)} disagreement(s)")
        print("  Reminder: both sides of this comparison originate in the same dict")
        print("  literal, so agreement is expected. A failure here would indicate")
        print("  post-write corruption, not a labelling error.")

        # =============================================================== (3b)
        rule("SECTION 3b -- INDEPENDENT CROSS-CHECK AGAINST matches.db (the real test)")
        src = {r[0]: (r[1], r[2]) for r in mcon.execute(
            f"SELECT fixture_id, home_goals, away_goals FROM fixtures "
            f"WHERE season_id IN ({tq})")}
        print(f"  matches.db training fixtures : {len(src)}")
        missing_up, goal_diff = [], []
        for fid, h, a, res, sid, cid in rows:
            if fid not in src:
                missing_up.append(fid)
                continue
            sh, sa = src[fid]
            if (sh, sa) != (h, a):
                goal_diff.append((fid, h, a, sh, sa))
        print(f"  feature rows with no matches.db counterpart : {len(missing_up)}")
        print(f"  rows where the goal pair differs            : {len(goal_diff)}")
        if goal_diff:
            print("\n  EVERY GOAL-VALUE DISAGREEMENT (features.db vs matches.db):")
            print("| fixture_id | features h-a | matches h-a |")
            print("|---:|---|---|")
            for fid, h, a, sh, sa in goal_diff[:200]:
                print(f"| {fid} | {h}-{a} | {sh}-{sa} |")
        gate_x_a = report("every training feature row has a matches.db counterpart",
                          not missing_up, str(missing_up[:10]))
        gate_x_b = report("goal values are identical in both databases",
                          not goal_diff, f"{len(goal_diff)} differing")

        # label_result re-derived from the INDEPENDENT source
        up_mis = []
        for fid, h, a, res, sid, cid in rows:
            if fid not in src or res is None:
                continue
            sh, sa = src[fid]
            if sh is None or sa is None:
                continue
            d = "H" if sh > sa else ("A" if sh < sa else "D")
            if d != res:
                up_mis.append((fid, sh, sa, res, d))
        print(f"\n  label_result vs H/D/A derived from matches.db goals:")
        print(f"    disagreements : {len(up_mis)}")
        if up_mis:
            print("| fixture_id | matches h-a | stored result | derived from matches.db |")
            print("|---:|---|---|---|")
            for fid, sh, sa, res, d in up_mis[:200]:
                print(f"| {fid} | {sh}-{sa} | {res} | {d} |")
        gate_x_c = report("label_result agrees with the INDEPENDENT matches.db outcome",
                          not up_mis, f"{len(up_mis)} disagreement(s)")

        # =============================================================== (4)
        rule("SECTION 4 -- GOAL DISTRIBUTION (training seasons only)")
        n = len(hg)
        vh, va = float(hg.var(ddof=1)), float(ag.var(ddof=1))
        mh, ma = float(hg.mean()), float(ag.mean())
        print(f"  n                        : {n}")
        print(f"  mean home goals          : {mh:.6f}")
        print(f"  mean away goals          : {ma:.6f}")
        print(f"  variance home goals      : {vh:.6f}")
        print(f"  variance away goals      : {va:.6f}")
        print(f"  home variance/mean       : {vh / mh:.6f}")
        print(f"  away variance/mean       : {va / ma:.6f}")
        print(f"  observed draw rate       : {float((hg == ag).mean()):.6f}")
        print(f"  home advantage           : {mh - ma:+.6f} goals   (ratio {mh / ma:.6f})")
        print(f"  corr(home, away) goals   : {float(np.corrcoef(hg, ag)[0, 1]):+.6f}")
        print("\n  A variance/mean ratio near 1.0 is consistent with a Poisson")
        print("  distribution; above 1.0 indicates overdispersion. Reported as a")
        print("  descriptive property of the target -- no distribution is fitted here,")
        print("  and no modelling decision is taken in STEP 1.")
        print("\n  goal-count frequencies:")
        print("| goals | home count | home share | away count | away share |")
        print("|---:|---:|---:|---:|---:|")
        for k in range(0, int(max(hg.max(), ag.max())) + 1):
            ch, ca = int((hg == k).sum()), int((ag == k).sum())
            if ch == 0 and ca == 0:
                continue
            print(f"| {k} | {ch} | {ch / n:.4f} | {ca} | {ca / n:.4f} |")
        print("\n  outcome distribution from the goal targets:")
        for lab, cnt in Counter(derived).most_common():
            print(f"    {lab}  {cnt:5d}  ({cnt / len(derived):.4f})")
    finally:
        fcon.close()
        mcon.close()

    # ================================================================ GATE
    rule("DECISION GATE")
    gates = {
        "both goal labels complete for the intended training rows":
            gate_cov_a and gate_cov_b and gate_cov_c,
        "goals are valid non-negative integers":
            gate_val_a and gate_val_b and gate_val_c and gate_val_d,
        "derived H/D/A exactly agrees with label_result (internal)":
            gate_cons_a,
        "targets verified against the INDEPENDENT matches.db source":
            gate_x_a and gate_x_b and gate_x_c,
    }
    for k, v in gates.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    # =============================================================== INTEGRITY
    rule("POST-RUN INTEGRITY")
    after = snapshot()
    changed = [k for k in before if before[k] != after.get(k)]
    check("nothing modified anywhere", not changed, str(changed))
    print("    features created  : NONE")
    print("    models trained    : NONE")
    print("    artifacts created : NONE")
    print("    databases written : NONE")
    print("    production source : UNCHANGED")
    print("    files created     : NONE")
    for k in ("data/processed/features.db", "data/processed/matches.db",
              "data/models/v1_logreg.pkl", "src/models/train.py",
              "src/features/feature_builder.py"):
        print(f"    {k:38s} {after[k]}")
    print(f"    LOCKED_INPUTS                          {len(pins)}/{len(pins)} unchanged")

    if not all(gates.values()):
        abort("One or more decision-gate conditions failed. See the FAIL lines above. "
              "Nothing was corrected, imputed or worked around.")

    print("\n" + "=" * 122)
    print("STEP 1 COMPLETE -- GOAL TARGETS VERIFIED")
    print("=" * 122)
    print("Not proceeding to STEP 2. No Poisson model is fitted and no design decision")
    print("is taken from these measurements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
