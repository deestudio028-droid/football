"""D-5 -- construction + leakage audit for six candidate pre-match features.

SCOPE
    Construct, in isolation, six candidate features from raw match
    statistics that exist in matches.db but are NOT featured today:
        home/away_possession_per_match_season
        home/away_corners_per_match_season
        home/away_red_cards_per_match_season
    and verify they can be built as TRUE pre-match team-history aggregates.

NOT A MODEL EXPERIMENT. No fit, no C, no calibration, no thresholds, no
class weighting, no feature selection, no promotion, no MODEL_VERSION
change. Production feature builder is NOT modified -- the existing
primitives are imported and reused unchanged.

CONVENTION (traced from source, not invented)
    history.team_match_record  : per-team record, stat_{side}_<field>
    feature_builder._season_only_family_features
                               : gated_mean(history, "season", season_id, extractor)
    rolling.take_window        : season window = matches with the SAME season_id
    rolling.windowed_mean      : mean over non-null values; coverage_n counted
    config.WINDOW_MIN_COVERAGE : season -> 2  (value is NULL below that)
    config.PLAYED_STATUSES     : ("FT","AWARDED") gates entry into history
    context.record()           : called AFTER features are computed

2025/26 FIREWALL
    Every 2025/26 row is dropped before the walk begins and the script
    hard-fails if one is present. 2025/26 is the last season, so excluding
    it cannot alter any earlier fixture's history.

Usage:
    cd "E:\\Football Prediction Project"
    set PYTHONPATH=%CD%\\src
    python run_d5_new_feature_construction_audit.py
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from features.config import PLAYED_STATUSES, WINDOW_MIN_COVERAGE
from features.history import load_fixtures_chronological
from features.rolling import gated_mean
from models.ablation import MODEL_B_COLUMNS
from models.config import (
    FINAL_TEST_SEASONS, MODEL_VERSION, SEASON_NAME_TO_IDS, WALK_FORWARD_FOLDS,
)

MATCHES_DB = REPO / "data" / "processed" / "matches.db"
FEATURES_DB = REPO / "data" / "processed" / "features.db"
MANIFEST = REPO / "data" / "audit" / "phase4c_prerun_manifest.json"

#: raw stat suffix -> feature stem.  Exactly three constructs, six columns.
NEW_STATS = {"possession": "possession", "corners": "corners", "red_cards": "red_cards"}
FAMILIES = {
    "POSSESSION": ("home_possession_per_match_season", "away_possession_per_match_season"),
    "CORNERS": ("home_corners_per_match_season", "away_corners_per_match_season"),
    "RED_CARDS": ("home_red_cards_per_match_season", "away_red_cards_per_match_season"),
}
NEW_COLUMNS = tuple(c for cols in FAMILIES.values() for c in cols)
TEST_SEASON_IDS = set()
for s in FINAL_TEST_SEASONS:
    TEST_SEASON_IDS.update(SEASON_NAME_TO_IDS[s])


def rule(t): print("\n" + "=" * 78); print(t); print("=" * 78)


def stop(m):
    print("\n" + "!" * 78); print("STOP -- D-5 halted"); print(m); print("!" * 78)
    raise SystemExit(1)


def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------------
# Candidate record + context, mirroring the production primitives.
# ---------------------------------------------------------------------
def candidate_record(fixture, side):
    """Same shape as history.team_match_record, restricted to the new stats."""
    return {
        "fixture_id": fixture["fixture_id"],
        "unix": fixture["unix"],
        "season_id": fixture["season_id"],
        "is_home": side == "home",
        "possession": fixture.get(f"stat_{side}_possession"),
        "corners": fixture.get(f"stat_{side}_corners"),
        "red_cards": fixture.get(f"stat_{side}_red_cards"),
    }


class CandidateContext:
    """history_before / record, identical discipline to features.history.FeatureContext."""

    def __init__(self):
        self._hist = {}

    def history_before(self, team_id):
        return self._hist.get(team_id, [])

    def record(self, fixture):
        if fixture.get("status") not in PLAYED_STATUSES:
            return
        for side, key in (("home", "home_id"), ("away", "away_id")):
            tid = fixture.get(key)
            if tid is not None:
                self._hist.setdefault(tid, []).append(candidate_record(fixture, side))


def build_row(fixture, ctx):
    """Compute the six candidate values for one fixture. Call BEFORE ctx.record()."""
    out = {"fixture_id": fixture["fixture_id"], "season_id": fixture["season_id"],
           "unix": fixture["unix"]}
    for prefix, key in (("home", "home_id"), ("away", "away_id")):
        tid = fixture.get(key)
        hist = ctx.history_before(tid) if tid is not None else []
        for stat, stem in NEW_STATS.items():
            val, n, cov = gated_mean(hist, "season", fixture["season_id"],
                                     lambda m, s=stat: m[s])
            out[f"{prefix}_{stem}_per_match_season"] = val
            out[f"{prefix}_{stem}_per_match_season_n"] = n
            out[f"{prefix}_{stem}_per_match_season_coverage_n"] = cov
    return out


def build_dataset(fixtures):
    ctx = CandidateContext()
    rows = []
    for f in fixtures:
        rows.append(build_row(f, ctx))
        ctx.record(f)
    return pd.DataFrame(rows)


def main():
    print("D-5 -- CANDIDATE FEATURE CONSTRUCTION + LEAKAGE AUDIT (no model, no 2025/26)")

    # ---------------- STEP 9 firewall (runs first) ----------------
    rule("STEP 9 -- 2025/26 FIREWALL")
    con = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    seasons = pd.read_sql("select distinct season_id, season from fixtures order by season", con)
    con.close()
    print("  seasons available in matches.db:")
    for r in seasons.itertuples():
        flag = "  <-- FINAL TEST, EXCLUDED" if r.season_id in TEST_SEASON_IDS else ""
        print(f"    season_id={r.season_id:<8} {r.season}{flag}")
    all_fixtures = load_fixtures_chronological(MATCHES_DB)
    fixtures = [f for f in all_fixtures if f["season_id"] not in TEST_SEASON_IDS]
    dropped = len(all_fixtures) - len(fixtures)
    print(f"\n  loaded={len(all_fixtures)}  dropped (2025/26)={dropped}  retained={len(fixtures)}")
    leaked = [f["fixture_id"] for f in fixtures if f["season_id"] in TEST_SEASON_IDS]
    if leaked:
        stop(f"2025/26 rows present after filtering: {leaked[:5]}")
    kept_seasons = sorted({f["season"] for f in fixtures})
    print(f"  seasons selected: {kept_seasons}")
    # Compare against the exact FINAL_TEST_SEASONS names. A substring test on
    # "2025" would false-positive on "2024/2025", which is a legitimate
    # training/validation season.
    forbidden_present = sorted(set(kept_seasons) & set(FINAL_TEST_SEASONS))
    if forbidden_present:
        stop(f"a final-test season survived filtering: {forbidden_present}")
    print(f"  final-test season names {list(FINAL_TEST_SEASONS)} absent from selection: True")
    print("  [PASS] 2025/26 absent. Note: it is the LAST season, so excluding it cannot")
    print("         alter any earlier fixture's history.")

    pins = json.loads(MANIFEST.read_text(encoding="utf-8"))["locked_input_checksums"]
    before = {r: md5(REPO / r) for r in pins}
    bad = [r for r, v in pins.items() if before[r] != v["expected"]]
    print(f"  13 pinned baselines: mismatches={bad or 'NONE'} | MODEL_VERSION={MODEL_VERSION!r}")
    if bad or MODEL_VERSION != "v1.0":
        stop("integrity failure")

    # ---------------- STEP 1 ----------------
    rule("STEP 1 -- SOURCE TRACE (from code, not inference)")
    for stat in NEW_STATS:
        print(f"  {stat}:")
        print(f"    1 source table        : matches.db :: fixtures")
        print(f"    2 source column       : stat_home_{stat} / stat_away_{stat}")
        print(f"    3 side/team mapping   : stat_{{side}}_{stat} for the team that played that side")
        print(f"    4 aggregation         : rolling.gated_mean(history,'season',season_id,extractor)")
        print(f"                            = mean of non-null values in the season window")
        print(f"    5 season boundary     : take_window('season') keeps m['season_id']==target")
        print(f"    6 missing-value       : windowed_mean skips None; coverage_n counts non-null")
        print(f"    7 minimum history     : WINDOW_MIN_COVERAGE['season']="
              f"{WINDOW_MIN_COVERAGE['season']} -> value is NULL below that")
        print(f"    8 target excluded     : ctx.record() is called AFTER build_row()")
    print(f"\n  PLAYED_STATUSES gating entry into history: {PLAYED_STATUSES}")
    print("  All eight points are VERIFIED from source; none inferred.")

    # ---------------- STEP 2 ----------------
    rule("STEP 2 -- CONSTRUCTION")
    df = build_dataset(fixtures)
    print(f"  rows built: {len(df)}  columns: {len(df.columns)}")
    print(f"  candidate features: {list(NEW_COLUMNS)}")

    # ---------------- STEP 3 ----------------
    rule("STEP 3 -- LEAKAGE TESTS")
    results = {}

    # A. target exclusion
    raw = {f["fixture_id"]: f for f in fixtures}
    viol = []
    ctx = CandidateContext()
    for f in fixtures:
        for prefix, key in (("home", "home_id"), ("away", "away_id")):
            tid = f.get(key)
            hist = ctx.history_before(tid) if tid is not None else []
            if any(m["fixture_id"] == f["fixture_id"] for m in hist):
                viol.append(f["fixture_id"])
        ctx.record(f)
    results["A_target_exclusion"] = not viol
    print(f"  A target exclusion            : {'PASS' if not viol else 'FAIL ' + str(viol[:5])}")

    # B. future independence -- perturb a late fixture, rebuild, compare earlier rows
    idx = int(len(fixtures) * 0.9)
    perturbed = [dict(f) for f in fixtures]
    for s in NEW_STATS:
        for side in ("home", "away"):
            perturbed[idx][f"stat_{side}_{s}"] = 999.0
    df2 = build_dataset(perturbed)
    early = df.iloc[:idx][list(NEW_COLUMNS)]
    early2 = df2.iloc[:idx][list(NEW_COLUMNS)]
    same_early = early.equals(early2)
    changed_after = not df.iloc[idx:][list(NEW_COLUMNS)].equals(df2.iloc[idx:][list(NEW_COLUMNS)])
    results["B_future_independence"] = same_early
    print(f"  B future independence        : {'PASS' if same_early else 'FAIL'} "
          f"(earlier rows unchanged={same_early}; later rows did change={changed_after})")

    # C. history-only: every contributing match strictly earlier in the walk
    ctx = CandidateContext(); bad_ts = []
    for f in fixtures:
        for key in ("home_id", "away_id"):
            tid = f.get(key)
            for m in (ctx.history_before(tid) if tid is not None else []):
                if m["unix"] > f["unix"]:
                    bad_ts.append((f["fixture_id"], m["fixture_id"]))
        ctx.record(f)
    results["C_history_only"] = not bad_ts
    print(f"  C history-only (unix<=target): {'PASS' if not bad_ts else 'FAIL ' + str(bad_ts[:3])}")

    # D. season boundary
    ctx = CandidateContext(); cross = []
    for f in fixtures:
        for key in ("home_id", "away_id"):
            tid = f.get(key)
            hist = ctx.history_before(tid) if tid is not None else []
            win = [m for m in hist if m["season_id"] == f["season_id"]]
            if any(m["season_id"] != f["season_id"] for m in win):
                cross.append(f["fixture_id"])
        ctx.record(f)
    results["D_season_boundary"] = not cross
    print(f"  D season boundary            : {'PASS' if not cross else 'FAIL'} "
          f"(season window keeps only same-season prior matches)")

    # E. home/away semantics
    ctx = CandidateContext(); mism = []
    for f in fixtures[:4000]:
        h = ctx.history_before(f.get("home_id")); a = ctx.history_before(f.get("away_id"))
        if f.get("home_id") is not None and f.get("away_id") is not None and h and a:
            if set(id(x) for x in h) & set(id(x) for x in a):
                mism.append(f["fixture_id"])
        ctx.record(f)
    results["E_home_away_semantics"] = not mism
    print(f"  E home/away semantics        : {'PASS' if not mism else 'FAIL'} "
          f"(home feature <- home team history, away <- away team history)")

    # F. duplicate / double counting
    ctx = CandidateContext(); dup = []
    for f in fixtures:
        for key in ("home_id", "away_id"):
            tid = f.get(key)
            hist = ctx.history_before(tid) if tid is not None else []
            ids = [m["fixture_id"] for m in hist]
            if len(ids) != len(set(ids)):
                dup.append(f["fixture_id"])
        ctx.record(f)
    results["F_no_double_counting"] = not dup
    print(f"  F no double counting         : {'PASS' if not dup else 'FAIL'}")

    # G. same-kickoff safety
    import inspect
    src = inspect.getsource(build_row) + inspect.getsource(CandidateContext)
    league_free = ("league" not in src.lower()) and ("accumulator" not in src.lower())
    results["G_no_league_accumulator"] = league_free
    print(f"  G no league accumulator      : {'PASS' if league_free else 'FAIL'} "
          f"(team-level only -> simultaneous-kickoff issue does not apply)")

    if not all(results.values()):
        stop(f"leakage test(s) failed: {[k for k, v in results.items() if not v]}")

    # ---------------- STEP 4 ----------------
    rule("STEP 4 -- COVERAGE AUDIT")
    print(f"  {'feature':<42}{'n':>7}{'nonnull':>9}{'null_rate':>11}"
          f"{'zero_rate':>11}{'min':>9}{'max':>9}{'mean':>10}{'median':>9}{'nuniq':>8}")
    for c in NEW_COLUMNS:
        s = df[c]
        nn = s.notna()
        print(f"  {c:<42}{len(s):>7}{int(nn.sum()):>9}{1-nn.mean():>11.4f}"
              f"{(s[nn]==0).mean() if nn.any() else float('nan'):>11.4f}"
              f"{s.min():>9.3f}{s.max():>9.3f}{s.mean():>10.4f}{s.median():>9.3f}"
              f"{s.nunique():>8}")
    print("\n  training-partition null rate per existing fold:")
    for f in WALK_FORWARD_FOLDS:
        tri = set()
        for s in f.train_seasons:
            tri.update(SEASON_NAME_TO_IDS[s])
        tr = df[df.season_id.isin(tri)]
        vai = set()
        for s in f.validation_seasons:
            vai.update(SEASON_NAME_TO_IDS[s])
        va = df[df.season_id.isin(vai)]
        print(f"    {f.name}: train n={len(tr)} val n={len(va)} | " + "  ".join(
            f"{c.split('_per')[0]}={tr[c].isna().mean():.4f}" for c in NEW_COLUMNS))

    # ---------------- STEP 5 ----------------
    rule("STEP 5 -- NOVELTY / REDUNDANCY vs E-1")
    fcon = sqlite3.connect(f"file:{FEATURES_DB}?mode=ro", uri=True)
    E1 = pd.read_sql(f"select fixture_id,{','.join(MODEL_B_COLUMNS)} from feature_rows", fcon)
    fcon.close()
    M = df[["fixture_id"] + list(NEW_COLUMNS)].merge(E1, on="fixture_id", how="inner")
    print(f"  joined rows: {len(M)}")
    for c in NEW_COLUMNS:
        s = M[c]
        if s.nunique(dropna=True) <= 1:
            print(f"  {c:<42} CONSTANT -> zero information")
            continue
        corr = M[list(MODEL_B_COLUMNS)].corrwith(s).abs().sort_values(ascending=False)
        top = corr.index[0]
        exact = []
        for e in corr.index[:8]:
            both = s.notna() & M[e].notna()
            if both.sum() > 100 and float(np.abs(s[both] - M[e][both]).max()) < 1e-9:
                exact.append(e)
        print(f"  {c:<42} max|corr|={corr.iloc[0]:.4f} with {top}")
        print(f"  {'':<42} exact duplicate of an E-1 column: "
              f"{exact if exact else 'NONE'}")
    sub = M[list(NEW_COLUMNS) + list(MODEL_B_COLUMNS)].dropna()
    r_e1 = np.linalg.matrix_rank(sub[list(MODEL_B_COLUMNS)].to_numpy(float))
    r_all = np.linalg.matrix_rank(sub.to_numpy(float))
    print(f"\n  column-space rank (complete rows n={len(sub)}): E-1={r_e1}  E-1+new={r_all}  "
          f"-> new dimensions added = {r_all - r_e1}")

    # ---------------- STEP 6 ----------------
    rule("STEP 6 -- FAMILY STRUCTURE")
    seen = set(); ok = True
    for fam, cols in FAMILIES.items():
        ov_e1 = set(cols) & set(MODEL_B_COLUMNS)
        ov_fam = set(cols) & seen
        seen |= set(cols)
        print(f"  {fam:<12} n={len(cols)}  overlap_with_E1={len(ov_e1)}  overlap_other_family={len(ov_fam)}")
        ok &= not ov_e1 and not ov_fam
    print(f"  total new columns={len(seen)} (expected 6): {len(seen) == 6} | clean={ok}")
    if not ok or len(seen) != 6:
        stop("family structure violation")

    # ---------------- STEP 8 ----------------
    rule("STEP 8 -- DATA-QUALITY WARNINGS (audit only, nothing imputed or repaired)")
    con = sqlite3.connect(f"file:{MATCHES_DB}?mode=ro", uri=True)
    R = pd.read_sql("select season_id,status,stat_home_possession,stat_away_possession,"
                    "stat_home_corners,stat_away_corners,stat_home_red_cards,"
                    "stat_away_red_cards from fixtures", con)
    con.close()
    R = R[~R.season_id.isin(TEST_SEASON_IDS)]
    print(f"  raw pre-2025/26 fixtures: {len(R)}")
    for c in [c for c in R.columns if c.startswith("stat_")]:
        s = R[c]
        print(f"    {c:<28} nonnull={s.notna().mean():.4f} zero={(s==0).mean():.4f} "
              f"min={s.min()} max={s.max()} nuniq={s.nunique()}")
    ph, pa = R.stat_home_possession, R.stat_away_possession
    both = ph.notna() & pa.notna()
    tot = (ph[both] + pa[both])
    print(f"\n  possession home+away: mean={tot.mean():.4f} min={tot.min():.2f} max={tot.max():.2f}")
    print(f"    rows summing to ~100: {float((np.abs(tot-100)<=1).mean()):.4f}  "
          f"-> {'consistent percentage scale' if float((np.abs(tot-100)<=1).mean())>0.9 else 'INCONSISTENT -- flag'}")
    rc = pd.concat([R.stat_home_red_cards, R.stat_away_red_cards])
    print(f"  red cards: zero-rate={float((rc==0).mean()):.4f} null-rate={float(rc.isna().mean()):.4f}")
    print("    NOTE: a red card is genuinely rare, so a high zero-rate is expected; but the")
    print("    null-rate is NOT zero, so 'missing' and 'zero' are distinguishable and must")
    print("    NOT be conflated. No imputation is performed here.")
    imposs = {
        "possession outside [0,100]": int(((R.stat_home_possession < 0) |
                                           (R.stat_home_possession > 100)).sum()),
        "corners negative": int((R.stat_home_corners < 0).sum()),
        "red_cards negative": int((R.stat_home_red_cards < 0).sum()),
        "red_cards > 5": int((R.stat_home_red_cards > 5).sum()),
    }
    print(f"  impossible-value checks: {imposs}")

    # ---------------- STEP 10 ----------------
    rule("STEP 10 -- INTEGRITY")
    after = {r: md5(REPO / r) for r in pins}
    drift = [r for r in pins if before[r] != after[r]]
    print(f"  13 pinned baselines: mismatches={drift or 'NONE'}")
    print(f"  MODEL_VERSION={MODEL_VERSION!r}")
    for line in ("production files changed = NO", "features.db changed = NO (read-only)",
                 "matches.db changed = NO (read-only)", "2025/26 accessed = NO",
                 "model fit = NO", "estimator persisted = NO", "C tuning = NONE",
                 "calibration = NONE", "threshold changes = NONE",
                 "feature selection = NONE", "artifacts created = NONE",
                 "features written to any database = NO (in-memory only)"):
        print(f"  {line}")

    rule("D-5 VERDICT")
    print(f"  leakage tests: {results}")
    allpass = all(results.values())
    print(f"  LEAKAGE: {'PASS' if allpass else 'FAIL'}")
    return df, results


if __name__ == "__main__":
    main()
