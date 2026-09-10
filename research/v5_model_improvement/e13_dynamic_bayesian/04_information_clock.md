# 04 — Dynamic Bayesian Information Clock & State Isolation Audit

## 1. Two-Pass Causal Architecture

To strictly eliminate lookahead contamination and simultaneous match leakage:

```
Timestamp T (Simultaneous Kickoffs)
 ├── PASS 1: Extract pre-match latent states & variances for all matches kicking off at T.
 └── PASS 2: After all predictions at T are locked, apply observation updates from completed matches at T.
```

---

## 2. Automated Causal Invariant Tests

```python
# Causal Safety Invariants:
assert last_update_time < target_kickoff_time, "FATAL: Future match observation used in state transition"
assert "home_goals" not in pre_match_feature_row, "FATAL: Target match result leaked into features"
assert all(weights >= 0.0) and math.isclose(sum(weights), 1.0), "FATAL: Invalid probability simplex"
```

- [x] Pre-match states extracted strictly before score updates.
- [x] Inter-season shrinkage applied before first kickoff of new season.
- [x] All 20 protected baseline assets 100% bit-identical.
