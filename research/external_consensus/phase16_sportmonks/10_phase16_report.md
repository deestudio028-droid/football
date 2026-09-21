# Phase 16 — V4 + Sportmonks Research Arm Report

> ⚠️ **RESEARCH ONLY — NOT PRODUCTION**  
> V4 model is frozen and unchanged. Sportmonks predictions are RESEARCH only.
> Sportmonks temporal status: **TEMPORAL_UNKNOWN** (fetched post-kickoff, API has no publication timestamp)

**Cohort:** 2026-09-18 → 2026-09-21 | **Generated:** 2026-09-21 10:34 UTC  
**V4 SHA256:** `1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`  
**Phase 15 Baseline:** 1X2=56.2%, Brier=0.5785, LogLoss=0.9732 (full 48)

---

## 1. Sportmonks Coverage

| Metric | Value |
|--------|-------|
| Total target fixtures | 48 |
| Sportmonks available | **48** |
| Sportmonks unavailable | 0 |
| Coverage % | **100.0%** |
| Temporal status | TEMPORAL_UNKNOWN: 48, PRE_KICKOFF_VERIFIED: 0 |

> ⚠️ **All Sportmonks data is TEMPORAL_UNKNOWN** — data was retrieved after kickoff.
> The Sportmonks API does not expose a publication timestamp.
> These results cannot be used as clean prospective benchmarks.

**Coverage by league:**

| League | Total | SM Available | Coverage |
|--------|-------|-------------|---------|
| Bundesliga | 9 | 9 | 100% |
| La Liga | 10 | 10 | 100% |
| Ligue 1 | 9 | 9 | 100% |
| Premier League | 10 | 10 | 100% |
| Serie A | 10 | 10 | 100% |

## 2. V4 vs Sportmonks Agreement

| Metric | Count | Rate |
|--------|-------|------|
| V4 / SM agree on argmax | 46 | 95.8% |
| V4 / SM disagree | 2 | 4.2% |
| SM increases P(D) vs V4 | 25 | 52% |
| SM decreases P(D) vs V4 | 23 | 48% |
| SM changes argmax to DRAW | 0 | — |
| SM changes argmax FROM DRAW | 0 | — |

## 3. Method Comparison Results

> ⚠️ All Sportmonks blends are **TEMPORAL_UNKNOWN** — diagnostic only, not prospective.

| Method | N | 1X2 Acc | Exact Acc | Brier | Log Loss | Draw Preds |
|--------|---|---------|-----------|-------|----------|------------|
| V4 Only (Full 48) | 48 | 56.2% | 8.3% | 0.5785 | 0.9732 | 0 |
| V4 Only (SM-covered subset) | 48 | 56.2% | 8.3% | 0.5785 | 0.9732 | 0 |
| Sportmonks Only | 48 | 52.1% | 8.3% | 0.5804 | 0.9756 | 0 |
| 50/50 Blend | 48 | 52.1% | 8.3% | 0.5794 | 0.9743 | 0 |
| 70/30 V4-Heavy | 48 | 52.1% | 8.3% | 0.5791 | 0.9739 | 0 |
| 30/70 SM-Heavy (Diagnostic) | 48 | 52.1% | 8.3% | 0.5798 | 0.9748 | 0 |

## 4. Draw Analysis

10 actual draws occurred in Phase 15 (of 48 = 20.8%).
V4 predicted **0 draws** (0.0% predicted draw rate).

**Draw calibration by method (SM-covered fixtures only):**

| Method | N | Actual Draws | Pred Draws | Actual Rate | Pred Rate | Gap |
|--------|---|------------|-----------|------------|----------|-----|
| V4 Only | 48 | 10 | 0 | 20.8% | 0.0% | +20.8% |
| Sportmonks Only | 48 | 10 | 0 | 20.8% | 0.0% | +20.8% |
| 50/50 Blend | 48 | 10 | 0 | 20.8% | 0.0% | +20.8% |
| 70/30 V4-Heavy | 48 | 10 | 0 | 20.8% | 0.0% | +20.8% |
| 30/70 SM-Heavy | 48 | 10 | 0 | 20.8% | 0.0% | +20.8% |

**Actual draw fixtures — P(D) comparison:**

| Fixture | Score | V4 P(D) | SM P(D) | V4 Pred | SM Pred | League |
|---------|-------|---------|---------|---------|---------|--------|
| Osasuna vs Rayo Vallecano | 1-1 | 0.244 | 0.243 | H | H | La Liga |
| Bologna vs Torino | 1-1 | 0.279 | 0.275 | H | H | Serie A |
| Eintracht Frankfurt vs SC Freiburg | 2-2 | 0.239 | 0.250 | A | A | Bundesliga |
| Athletic Club vs Deportivo Alavés | 0-0 | 0.265 | 0.259 | H | H | La Liga |
| Roma vs Inter | 2-2 | 0.262 | 0.263 | H | H | Serie A |
| Fiorentina vs Napoli | 1-1 | 0.280 | 0.277 | A | A | Serie A |
| Leeds United vs Crystal Palace | 0-0 | 0.257 | 0.262 | H | H | Premier League |
| Schalke 04 vs Elversberg | 0-0 | 0.247 | 0.254 | H | H | Bundesliga |
| Fulham vs Manchester United | 1-1 | 0.258 | 0.254 | A | A | Premier League |
| Deportivo A Coruña vs Real Betis | 1-1 | 0.255 | 0.267 | A | A | La Liga |

## 5. Signal Analysis

### 2-0 Profile Fixtures

| Method | N | Outcome Acc | Exact Acc |
|--------|---|------------|----------|
| V4 Only | 2 | 100.0% | 0.0% |
| Sportmonks Only | 2 | 100.0% | 0.0% |
| 50/50 Blend | 2 | 100.0% | 0.0% |

### Strong Home Profile

| Method | N | Outcome Acc |
|--------|---|------------|
| V4 Only | 3 | 100.0% |
| Sportmonks Only | 3 | 100.0% |
| 50/50 Blend | 3 | 100.0% |

## 6. League Breakdown (SM-Covered Fixtures)

| League | N | V4 Acc | SM Acc | 50/50 Acc | 70/30 Acc |
|--------|---|--------|--------|-----------|-----------|
| Bundesliga | 9 | 66.7% | 55.6% | 55.6% | 55.6% |
| La Liga | 10 | 60.0% | 50.0% | 50.0% | 50.0% |
| Ligue 1 | 9 | 77.8% | 77.8% | 77.8% | 77.8% |
| Premier League | 10 | 40.0% | 40.0% | 40.0% | 40.0% |
| Serie A | 10 | 40.0% | 40.0% | 40.0% | 40.0% |

## 7. Answers to Research Questions

1. **Sportmonks coverage:** 48/48 = 100% of fixtures
2. **Temporally verified:** 0 / 48 — all TEMPORAL_UNKNOWN (retrieved post-kickoff)
3. **V4 / SM agreement rate:** 95.8%
4. **V4 / SM disagreement rate:** 4.2%
5. **Sportmonks-only accuracy:** 52.1% (on 48 SM-covered fixtures)
6. **V4 accuracy on same fixtures:** 56.2%
7. **50/50 blend accuracy:** 52.1%
8. **70/30 V4-heavy blend accuracy:** 52.1%
9. **30/70 diagnostic blend accuracy:** 52.1%
10. **Does SM improve DRAW prediction?** SM predicted 0 draws; V4 predicted 0. See draw calibration table.
11. **SM vs V4 Brier:** 0.5804 vs 0.5785
12. **SM vs V4 Log Loss:** 0.9756 vs 0.9732
13. **SM changes 2-0 signals?** SM predicted 2 2-0 profile fixtures with 100.0% accuracy
14. **SM changes Strong Home?** 100.0% vs V4 100.0%
15. **Evidence for production integration?** INSUFFICIENT — all data is TEMPORAL_UNKNOWN. Requires prospective pre-kickoff capture.

## 8. Governance Decision

> **Sportmonks remains RESEARCH ONLY.**

All Phase 16 Sportmonks data is marked `TEMPORAL_UNKNOWN` because:
- Data was retrieved after all Phase 15 matches have concluded (FT status)
- Sportmonks API does not expose a publication timestamp for predictions
- We cannot verify whether the probabilities shown are identical to pre-kickoff values

**Recommendation:** Continue prospective V4 vs V4+Sportmonks validation by capturing
Sportmonks predictions BEFORE kickoff for the next upcoming cohort.

> Do NOT declare Sportmonks better or worse based on this TEMPORAL_UNKNOWN retrospective sample.

---

## Appendix — Frozen V4 Model

```
model_v4_sha256: 1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5
```

*Phase 16 is research-only. V4 model was NOT modified, retrained, or re-parameterized.*
*Production predictions remain unchanged. Sportmonks is NOT promoted to production.*