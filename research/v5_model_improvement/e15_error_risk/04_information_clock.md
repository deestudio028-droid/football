# 04 — Pre-Match Information Clock & Causal Safety Audit

## 1. Information Clock Lifecycle

```
    Match History t < t_cutoff               Kickoff t = 0
 [------------------------------------] | [---------------------------]
       1. Base V4 Lambdas Extracted     |
       2. E10 Probabilities Generated   |
       3. Risk Model Features Built     |
       4. Risk Score / Alert Emitted    |      Actual Match Played
       ----------------------------------       Actual Outcome Observed
       (Strictly Pre-Kickoff Boundary)
```

---

## 2. Leakage Guardrails

- [x] Base model and risk model receive only causal features prior to match kickoff.
- [x] Error targets computed exclusively on training folds during meta-training.
- [x] Test fold fixtures scored prior to observing actual match outcomes.
- [x] All 20 baseline protected assets 100% bit-identical.
