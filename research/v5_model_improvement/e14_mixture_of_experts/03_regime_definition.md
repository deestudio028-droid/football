# 03 — Causal Pre-Match Regime Definitions

## 1. Discrete Regime Taxonomies

1. **Early Season Regimes**:
   - `First_1_Match_Of_Season`: Matchweek 1 for competition season.
   - `First_3_Matches_Of_Season`: Matchweeks 1 through 3.
   - `First_5_Matches_Of_Season`: Matchweeks 1 through 5.
   - `First_10_Matches_Of_Season`: Matchweeks 1 through 10.
   - `Remaining_Season_Matches`: Matchweeks > 10.

2. **Team Maturity & Promotion Status**:
   - `Promoted_Sparse`: $\min(n_H, n_A) < 5$ historical matches.
   - `Developing`: $5 \le \min(n_H, n_A) \le 10$ matches.
   - `Established`: $\min(n_H, n_A) > 10$ matches.

3. **Elo Parity Regimes**:
   - `High_Parity`: $|\Delta \text{Elo}| \le 25.0$.
   - `Moderate_Parity`: $25.0 < |\Delta \text{Elo}| \le 75.0$.
   - `Large_Mismatch`: $|\Delta \text{Elo}| > 75.0$.

4. **Expected Total Scoring Regimes**:
   - `Low_Scoring`: $\lambda_H + \lambda_A < 2.40$.
   - `Medium_Scoring`: $2.40 \le \lambda_H + \lambda_A \le 3.00$.
   - `High_Scoring`: $\lambda_H + \lambda_A > 3.00$.
