# 04 — Orthogonal Signal Causal Leakage & Governance Audit

## 1. Information Clock & State Isolation

Every feature vector in `data/research/e12_orthogonal_features.parquet` is extracted before match kickoff.

```
       Past Matches M_1, ..., M_{k}          Target Match M_{k+1}
 [------------------------------------] | [---------------------------]
            t < t_cutoff                |          t_cutoff (KO - 15m)
           Feature State                |       Prediction Frozen
```

---

## 2. Automated Assertions Verified

- [x] `assert "target_home_goals" not in feat_dict`
- [x] `assert "target_away_goals" not in feat_dict`
- [x] `assert "stat_home_xg" not in feat_dict`
- [x] `assert "xg_home" not in feat_dict`
- [x] Residual targets $y_{\text{res}}$ zero-centered on training folds only.
- [x] All 20 baseline protected assets 100% bit-identical.
