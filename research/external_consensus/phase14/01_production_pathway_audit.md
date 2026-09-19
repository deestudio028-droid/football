# Production Prediction Pathway Audit — Phase 14

## Executive Summary
This audit provides a comprehensive forensic breakdown of the production scoreline generation and presentation pathways across the Football Prediction Project codebase. 

In Phase 12 forensics and Phase 13 prospective validation, it was discovered that the frozen V4.0 model binary (`data/models/v4_poisson_venue_elo_online_ad.pkl`, SHA256: `1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`) was mathematically intact and consistently generated `2-0` bivariate Poisson modal scorelines for strong home favorites. However, legitimate `2-0` outputs had completely vanished from the September 10–16, 2026 recorded prediction ledger (`reports/upcoming_batch_2026_09_10_to_2026_09_16_ledger.jsonl`) and were subject to distortion in several dashboard rendering layers.

This audit details the exact code locations, architectural mechanisms, and root causes responsible for this divergence.

---

## 1. Frozen V4 Model Scoreline Specification

In the canonical V4 mathematical model (`src/models/poisson.py`), match expected goals $\lambda_h$ and $\lambda_a$ define independent Poisson probability mass functions over goals $(i, j)$:

$$P(i, j) = \frac{\lambda_h^i e^{-\lambda_h}}{i!} \cdot \frac{\lambda_a^j e^{-\lambda_a}}{j!}$$

The theoretical mode of an independent Poisson distribution with rate $\lambda$ is $\lfloor \lambda \rfloor$ (when $\lambda$ is non-integer). Therefore, for bivariate independent Poisson goals:
$$\text{mode}(i, j) = (\lfloor \lambda_h \rfloor, \lfloor \lambda_a \rfloor)$$

In `src/models/poisson.py`:
```python
def modal_scoreline(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    grid_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the modal (most probable) scoreline for each match."""
    # Computes PMF grid and takes argmax over (i, j)
```

For strong home favorites with heavy offensive advantage and superior defensive suppression (e.g. $\lambda_h \approx 2.16$, $\lambda_a \approx 0.73$):
- $\lfloor \lambda_h \rfloor = 2$
- $\lfloor \lambda_a \rfloor = 0$
- **Canonical Model Mode**: `2-0` with $P(2, 0) \approx 14.5\%$.

---

## 2. Forensic Analysis of Production Pipelines

The investigation uncovered **four distinct failure mechanisms** in downstream pipelines where the canonical output was bypassed, flattened, or altered.

### Mechanism 1: Simplified Batch Ledger Routine (`reports/upcoming_batch_...ledger.jsonl`)
In the batch generation runner used to create `reports/upcoming_batch_2026_09_10_to_2026_09_16_ledger.jsonl`, a simplified score-recording routine was used:
- Every `PRED == "H"` fixture was assigned `"Predicted Score": "2-1"`
- Every `PRED == "A"` fixture was assigned `"Predicted Score": "1-2"`

**Impact**:
Out of 55 fixtures in the September 10–16 ledger:
- 27 Home predictions were recorded as `2-1` (100.0%)
- 22 Away predictions were recorded as `1-2` (100.0%)
- Exactly 0 matches were recorded as `2-0`.

Two fixtures with genuine canonical V4 `2-0` modes (**LOSC Lille vs Troyes**, fid: 420637600; **FC Barcelona vs Racing Santander**, fid: 420644693) were flattened to `2-1`.

### Mechanism 2: Nearest-Integer Rounding with Tie-Breaking (`src/dashboard/fixture_service.py:840-848`)
In `src/dashboard/fixture_service.py`, historical evaluation records were synthesized using:
```python
lh = float(snap.expected_home_goals)
la = float(snap.expected_away_goals)
lh_r = max(0, int(round(lh)))
la_r = max(0, int(round(la)))
if lh_r == la_r and dec == "H":
    lh_r += 1
elif lh_r == la_r and dec == "A":
    la_r += 1
pred_score = f"{lh_r}-{la_r}"
```
**Why this breaks mathematical mode**:
- For $\lambda_a = 0.73$, `round(0.73)` evaluates to `1`.
- For $\lambda_h = 2.16$, `round(2.16)` evaluates to `2`.
- Resulting scoreline: `2-1`, even though the bivariate Poisson mode is strictly `2-0` ($P(2, 0) > P(2, 1)$ because $0.73^0 / 0! = 1 > 0.73^1 / 1! = 0.73$).

### Mechanism 3: Dashboard UI Presentation Layer (`src/dashboard/app.py:636-644`)
In the Streamlit user interface, fixture scorelines were rendered on the fly:
```python
if pred.lambda_home is not None and pred.lambda_away is not None:
    lh_round = max(0, int(round(pred.lambda_home)))
    la_round = max(0, int(round(pred.lambda_away)))
    if lh_round == la_round and sel_dec == "H":
        lh_round += 1
    elif lh_round == la_round and sel_dec == "A":
        la_round += 1
    goal_pred_display = f"{lh_round}-{la_round}"
```
This mirrored Mechanism 2 and prevented the UI from displaying genuine `2-0` predictions whenever $\lambda_a \in [0.50, 1.00)$.

### Mechanism 4: Performance Monitor Baseline Heuristic (`src/dashboard/performance_monitor_service.py:218-225`)
In `performance_monitor_service.py`:
```python
if dec == "H":
    base_score = "2-1" if p_h < 0.60 else "2-0"
elif dec == "A":
    base_score = "1-2" if p_a < 0.60 else "0-2"
else:
    base_score = "1-1"
```
This hardcoded heuristic assigned `2-0` purely based on $P(H) \ge 0.60$ without querying the Poisson $\lambda$ parameters or `modal_scoreline`. While it accidentally produced `2-0` for some matches, it ignored cases where $\lambda_a \ge 1.00$ (where `2-1` or `3-1` was the true mode).

---

## 3. Data Integrity & Invariant Requirements

1. **Model Immutability**:
   `data/models/v4_poisson_venue_elo_online_ad.pkl` SHA256 must remain:
   `1c7f1e69587f0520edc371f62942d6269541410e7687525ba13c3c26c4acfbc5`.
2. **Mathematical Source of Truth**:
   The sole authoritative source of scoreline predictions is `src/models/poisson.py:modal_scoreline`.
3. **No Synthetic Volume Inflation**:
   Do not force 8–10 selections per week. Preserve the natural operating volume (2–3 strict 2-0 / week, 3–5 strong home / week).
4. **Explicit Signal Labeling**:
   - `strong_home_profile`: $P(H) \ge 0.60 \land \text{Draw Risk} == \text{'LOW'}$
   - `is_2_0_profile`: $\text{canonical\_score} == \text{'2-0'} \land \text{Draw Risk} == \text{'LOW'}$
   - `signal_profile`: `'2-0_PROFILE'` | `'STRONG_HOME_PROFILE'` | `None`

---

## 4. Conclusion
The V4.0 model was never broken. The disappearance of `2-0` was entirely an artifact of downstream rounding, heuristic fallbacks, and batch script flattening. By replacing these layers with direct calls to `modal_scoreline` and preserving canonical scoreline metadata in snapshots and batch generators, the production pathway is restored to complete mathematical consistency.
