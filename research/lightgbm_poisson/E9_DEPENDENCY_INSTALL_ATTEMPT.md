# E9 — Dependency Installation Attempt (approved) — OUTCOME: FAILED

**Date:** 2026-08-20
**Authorisation:** Installation of LightGBM + scikit-learn into the research environment approved.
**Constraint honoured:** production `requirements.txt` was **NOT** modified. No production file was touched.
**Outcome:** **Installation is impossible in this environment. STOPPING per §3.**

---

## Pre-installation package state (recorded as required)

| Item | Value |
|---|---|
| Python | **3.10.12** |
| Installed packages | **136** (full list captured via `pip freeze`) |
| lightgbm | MISSING |
| scikit-learn | MISSING |
| scipy | MISSING |
| numpy | 2.2.6 |
| pandas | 2.3.3 |

---

## Installation attempts — six, all failed

### Attempt 1 — pip, standard index

```
$ pip install lightgbm scikit-learn --break-system-packages
ProxyError('Cannot connect to proxy.', OSError('Tunnel connection failed: 403 Forbidden'))
ERROR: Could not find a version that satisfies the requirement lightgbm (from versions: none)
```

Note `(from versions: none)` — pip could not even enumerate available versions, so the index itself is unreachable rather than the specific package being blocked.

### Attempt 2 — diagnostic: is *any* index reachable?

```
$ pip download --no-deps --dest /tmp/dl requests
ERROR: Could not find a version that satisfies the requirement requests (from versions: none)
```

**This is the decisive test.** `requests` is *already installed* in this environment. pip still cannot reach it on the index. The failure is total network unreachability of PyPI, not a package-specific restriction.

### Attempt 3 — locally cached or vendored wheels

```
$ find / -iname "lightgbm*" -o -iname "scikit_learn*"
/sessions/.../research/lightgbm_poisson          <- the directory I created for E9
```

No wheels, no sdists, no vendored copies anywhere on the filesystem.

### Attempt 4 — alternative package manager

`conda` / `mamba`: absent. `apt-get`: present.

### Attempt 5 — apt (`python3-sklearn` is a Debian package)

```
$ apt-get install -y --no-install-recommends python3-sklearn
E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
E: Unable to acquire the dpkg frontend lock, are you root?
```

No root privileges. (This would only have provided scikit-learn in any case, never LightGBM.)

### Attempt 6 — direct reachability check

```python
urllib.request.urlopen('https://pypi.org/simple/lightgbm/')
→ URLError: <urlopen error Tunnel connection failed: 403 Forbidden>
```

The proxy at `localhost:3128` rejects the CONNECT tunnel to PyPI with HTTP 403. This is enforced infrastructure policy, not a misconfiguration I can correct: `pip config list` is empty, so no bad local setting is responsible.

---

## Post-attempt verification

| Check | Result |
|---|---|
| Packages before | 136 |
| Packages after | **136** |
| `diff` of `pip freeze` | **no differences — environment unchanged** |
| lightgbm | **STILL MISSING** |
| scikit-learn | **STILL MISSING** |
| scipy | **STILL MISSING** |

No partial or broken installation was left behind.

### Protected files — unchanged throughout

| File | Expected | Actual | Status |
|---|---|---|---|
| `v2_poisson_venue.pkl` | `25935b4e93fc4074f67f16e3181ed4df` | `25935b4e93fc4074f67f16e3181ed4df` | **IDENTICAL** |
| `v1_logreg.pkl` | `5e504427712b35778bb8a62a8496c7cd` | `5e504427712b35778bb8a62a8496c7cd` | **IDENTICAL** |
| `v3_poisson_venue_elo_candidate.pkl` | `a2850a7687822a5916663301f5ccc96c` | `a2850a7687822a5916663301f5ccc96c` | **IDENTICAL** |
| `features.db` | `e7ebe7fc07040a5927683c35b6371e63` | `e7ebe7fc07040a5927683c35b6371e63` | **IDENTICAL** |
| `matches.db` | `fdeed042096fa1c851aaee6c84995247` | `fdeed042096fa1c851aaee6c84995247` | **IDENTICAL** |

`requirements.txt` unmodified — still `requests>=2.31` and `pytest>=7.4`.

---

## What I did NOT do

Per §3, none of the following was attempted or considered:

- Substituting XGBoost, CatBoost, or sklearn's `HistGradientBoostingRegressor`
- Implementing a hand-rolled gradient booster to stand in for LightGBM
- Modifying production `requirements.txt`
- Altering the E9 hypothesis to fit an available library
- Fetching wheels through non-pip channels

§8 of the approved decisions is explicit: implementation may not begin until **LightGBM import succeeds** and **sklearn import succeeds**. Neither does. The gate is closed.

---

## STOP

**E9 cannot proceed in this environment.** The approved dependency decision removed the *authorisation* barrier but not the *infrastructure* one.

The block is a sandbox network policy. It is the same barrier that left E3, E4, E5 and E7 unmeasured, and it is not something I can route around without violating §3.

---

## What is ready, and what E9 still needs

**Ready now (from the Phase 1 audit):**

| Item | Status |
|---|---|
| V3 87-feature contract and ordering | Documented |
| V3 preprocessing (median-impute → scale → one-hot) | Documented |
| Targets, objective, score grid, conversion | Documented |
| Walk-forward folds and universe (8,983 / 5,331 OOS) | Verified |
| `competition_id` handling decision | Native categorical, justified |
| **Missing-data policy** | **Median-impute both arms (approved)** |
| **Shared adaptive K constraint** | **Identified and specified** |
| 16 pre-registered hyperparameter configurations | Drafted |
| Honest benchmarks on both universes | Recorded |

**Still required, in order, once the environment permits:**

1. Install and record versions; rerun availability audit
2. Freeze the configuration grid
3. Implement `lightgbm_poisson.py`
4. Implement `test_lightgbm_poisson.py`
5. Run the full suite (must pass before proceeding)
6. Synthetic LightGBM sanity test
7. Leakage and zero/identity controls
8. Market-separation verification
9. Walk-forward
10. Deterministic repeat
11. Quarantine and MD5 POST
12. Report

---

## Options

**A. Run E9 locally.** LightGBM and scikit-learn install normally on a standard machine. This is the direct path, and it also unblocks the four other complete-but-unmeasured harnesses (E3, E4, E5, E7).

**B. Build the E9 module, tests and harness here anyway**, so E9 becomes one local command. This is what E3–E7 did. Nothing executes here and the decision stays open. Say the word and I'll build it.

**C. Defer E9 and proceed to E10.** E10 (Match Importance + Squad) is a feature-engineering experiment and may be partly testable with numpy/pandas alone, as E8 was.

I'd suggest **B now, A when you're at a local machine** — but E9 has no measurable outcome from this environment either way, and I will not manufacture one.

---

## Roadmap status — unchanged

```
E0 — V2 Baseline                  COMPLETE
E1 — Causal Elo -> V3             PASS
E2 — Market Odds                  PASS
E3 — Dixon-Coles                  INCONCLUSIVE
E4 — Exponential Time Decay       INCONCLUSIVE
E5 — Quadratic Elo -> xG          INCONCLUSIVE
E6 — Online Attack/Defense        PASS
E7 — Best Proven Combination      FAIL
E8 — Temperature Scaling          FAIL
E9 — LightGBM Poisson             BLOCKED — dependency unavailable
E10 — Match Importance + Squad    QUEUED
E11 — Advanced Tactical Efficiency QUEUED
E12 — Final Champion Selection    PENDING
E13 — Final 5-League Walk-Forward PENDING
E14 — Production Inference        PENDING
E15 — V2 -> V4/V5 Promotion       PENDING
E16 — Production Monitoring       PENDING
```

**E9 is BLOCKED, not FAILED.** A FAIL would assert a measured result. Nothing was measured.

**Nothing promoted. No production file modified. No substitute library used.**
