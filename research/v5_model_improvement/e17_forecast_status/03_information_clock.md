# 03 — Causal Information Clock & Decision Timing Assertions

## 1. Information Flow Lifecycle

```
    Historical State t < t_cutoff            Kickoff t = 0
 [------------------------------------] | [---------------------------]
       1. Base V4 Lambdas Extracted     |
       2. E10 Probabilities Generated   |
       3. Status Features Computed      |
       4. Actionable Status Assigned    |      Actual Match Played
          (STRONG/LEAN/CAUTION/AVOID)   |       Actual Outcome Observed
       ----------------------------------
       (Strictly Pre-Kickoff Boundary)
```

---

## 2. Timing Guardrails & Causal Guarantees

- [x] Status model fitted exclusively on training history folds.
- [x] Thresholds $\tau_{75}, \tau_{50}, \tau_{25}$ determined strictly from training folds.
- [x] Test fold fixtures assigned status tags prior to observing match kickoff.
- [x] All 20 baseline protected assets 100% bit-identical.
