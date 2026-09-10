# 04 — Regime Detection Information Clock & Causal Safety Audit

## 1. Information Clock & State Isolation

Every feature vector and regime indicator is extracted strictly prior to match kickoff.

```
       Past Matches M_1, ..., M_{k}          Target Match M_{k+1}
 [------------------------------------] | [---------------------------]
            t < t_cutoff                |          t_cutoff (KO - 15m)
           Feature State                |       Prediction Frozen
```

---

## 2. Automated Assertions Verified

- [x] `assert "home_goals" not in regime_feature_dict`
- [x] `assert "away_goals" not in regime_feature_dict`
- [x] All unsupervised clustering fitted strictly on training folds.
- [x] Gating models fitted strictly on training folds.
- [x] All 20 baseline protected assets 100% bit-identical.
