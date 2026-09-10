# 01 — Prospective Forecast Lock Protocol

## 1. Absolute Scientific Rules & Invariants

1. **Immutable Model Lock**:
   - Model file: `data/models/v4_1_prospective_candidate_2025_26.pkl`
   - MD5: `145f918d933eb343c0f63ca342b10289`
   - SHA256: `cbb00b32c3ce4dbf600564a95d11b67a0c236f6b34cd184a9ca8abcc8ab94ac8`
   - No retraining, parameter modification, or probability alteration is permitted.
2. **Pre-Kickoff Timing**:
   - Every prediction is generated and cryptographically locked strictly prior to kickoff ($t_{	ext{pred}} < t_{	ext{kickoff}}$).
   - Any fixture that kicked off prior to prediction lock is excluded from the pre-match ledger.
3. **Information-Clock Firewall**:
   - Features access strictly historical data through May 24, 2026.
   - Zero 2026 match outcomes enter the feature vector or model inputs.
4. **Probability Simplex**:
   - For every locked match: $P(H) + P(D) + P(A) = 1.000000$ and $P \ge 0$.
5. **Advisory Decoupling**:
   - E16 Reliability bands and E17 Decision Status tags are strictly advisory metadata.
   - $P_{	ext{FINAL}} = P_{	ext{V4.1}}$ is strictly maintained.
