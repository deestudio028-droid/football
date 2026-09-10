"""Reconciliation script for FPP Elo Experiment Metrics."""
import json
from pathlib import Path
import numpy as np

p = Path(r"E:\Football Prediction Project\research\worldcup_elo\experiment_results.json")
data = json.loads(p.read_text(encoding="utf-8"))

print("=== EXPERIMENT RESULTS JSON SUMMARY ===")
for arm in data["summary"]:
    print(f"{arm['Arm']}: Acc={arm['accuracy']*100:.4f}%, LL={arm['log_loss']:.6f}, Brier={arm['brier']:.6f}, RPS={arm['rps']:.6f}")

print("\n=== PER FOLD DETAILS FOR CHAMPION V2 ===")
f_champ = data["per_fold"]["Champion V2 (84 cols)"]
for fname, metrics in f_champ.items():
    print(f"{fname}: Acc={metrics['accuracy']*100:.4f}%, LL={metrics['log_loss']:.6f}, Brier={metrics['brier']:.6f}, RPS={metrics['rps']:.6f}")

print("\n=== PER FOLD DETAILS FOR CHALLENGER + 3 ELO (87 cols) ===")
f_elo87 = data["per_fold"]["Challenger + 3 Elo (87 cols)"]
for fname, metrics in f_elo87.items():
    print(f"{fname}: Acc={metrics['accuracy']*100:.4f}%, LL={metrics['log_loss']:.6f}, Brier={metrics['brier']:.6f}, RPS={metrics['rps']:.6f}")

# Calculate both unweighted mean and fixture-weighted mean
# Fold counts: Fold 1 = 1827, Fold 2 = 1752, Fold 3 = 1752 (Total = 5331)
weights = [1827/5331, 1752/5331, 1752/5331]

champ_ll_folds = [f_champ["fold_1"]["log_loss"], f_champ["fold_2"]["log_loss"], f_champ["fold_3"]["log_loss"]]
champ_acc_folds = [f_champ["fold_1"]["accuracy"], f_champ["fold_2"]["accuracy"], f_champ["fold_3"]["accuracy"]]
champ_brier_folds = [f_champ["fold_1"]["brier"], f_champ["fold_2"]["brier"], f_champ["fold_3"]["brier"]]
champ_rps_folds = [f_champ["fold_1"]["rps"], f_champ["fold_2"]["rps"], f_champ["fold_3"]["rps"]]

champ_unweighted_ll = np.mean(champ_ll_folds)
champ_weighted_ll = np.sum(np.array(champ_ll_folds) * np.array(weights))
champ_unweighted_acc = np.mean(champ_acc_folds)
champ_weighted_acc = np.sum(np.array(champ_acc_folds) * np.array(weights))
champ_unweighted_brier = np.mean(champ_brier_folds)
champ_weighted_brier = np.sum(np.array(champ_brier_folds) * np.array(weights))
champ_unweighted_rps = np.mean(champ_rps_folds)
champ_weighted_rps = np.sum(np.array(champ_rps_folds) * np.array(weights))

elo87_ll_folds = [f_elo87["fold_1"]["log_loss"], f_elo87["fold_2"]["log_loss"], f_elo87["fold_3"]["log_loss"]]
elo87_acc_folds = [f_elo87["fold_1"]["accuracy"], f_elo87["fold_2"]["accuracy"], f_elo87["fold_3"]["accuracy"]]
elo87_brier_folds = [f_elo87["fold_1"]["brier"], f_elo87["fold_2"]["brier"], f_elo87["fold_3"]["brier"]]
elo87_rps_folds = [f_elo87["fold_1"]["rps"], f_elo87["fold_2"]["rps"], f_elo87["fold_3"]["rps"]]

elo87_unweighted_ll = np.mean(elo87_ll_folds)
elo87_weighted_ll = np.sum(np.array(elo87_ll_folds) * np.array(weights))
elo87_unweighted_acc = np.mean(elo87_acc_folds)
elo87_weighted_acc = np.sum(np.array(elo87_acc_folds) * np.array(weights))
elo87_unweighted_brier = np.mean(elo87_brier_folds)
elo87_weighted_brier = np.sum(np.array(elo87_brier_folds) * np.array(weights))
elo87_unweighted_rps = np.mean(elo87_rps_folds)
elo87_weighted_rps = np.sum(np.array(elo87_rps_folds) * np.array(weights))

print("\n=== RECONCILIATION SUMMARY ===")
print("Champion V2:")
print(f"  Unweighted Mean across 3 Folds: Accuracy={champ_unweighted_acc*100:.4f}%, LL={champ_unweighted_ll:.6f}, Brier={champ_unweighted_brier:.6f}, RPS={champ_unweighted_rps:.6f}")
print(f"  Fixture-Weighted (5331 rows):   Accuracy={champ_weighted_acc*100:.4f}%, LL={champ_weighted_ll:.6f}, Brier={champ_weighted_brier:.6f}, RPS={champ_weighted_rps:.6f}")

print("\nChallenger 87 (Poisson+Venue+Elo):")
print(f"  Unweighted Mean across 3 Folds: Accuracy={elo87_unweighted_acc*100:.4f}%, LL={elo87_unweighted_ll:.6f}, Brier={elo87_unweighted_brier:.6f}, RPS={elo87_unweighted_rps:.6f}")
print(f"  Fixture-Weighted (5331 rows):   Accuracy={elo87_weighted_acc*100:.4f}%, LL={elo87_weighted_ll:.6f}, Brier={elo87_weighted_brier:.6f}, RPS={elo87_weighted_rps:.6f}")

print("\nDeltas (Challenger 87 - Champion V2):")
print(f"  Unweighted Delta: Accuracy={(elo87_unweighted_acc - champ_unweighted_acc)*100:+.4f}pp, LL={elo87_unweighted_ll - champ_unweighted_ll:+.6f}, Brier={elo87_unweighted_brier - champ_unweighted_brier:+.6f}, RPS={elo87_unweighted_rps - champ_unweighted_rps:+.6f}")
print(f"  Weighted Delta:   Accuracy={(elo87_weighted_acc - champ_weighted_acc)*100:+.4f}pp, LL={elo87_weighted_ll - champ_weighted_ll:+.6f}, Brier={elo87_weighted_brier - champ_weighted_brier:+.6f}, RPS={elo87_weighted_rps - champ_weighted_rps:+.6f}")