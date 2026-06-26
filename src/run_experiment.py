# =============================================================
# src/run_experiment.py
# =============================================================
# PURPOSE:
#   Master pipeline script — runs the complete experiment from
#   data loading to attack evaluation in one command.
#
# WHAT THIS FILE DOES:
#   Step 1 → Load real NSL-KDD data (network_flows.csv)
#   Step 2 → Train standard IDS model (no defence)
#   Step 3 → Train adversarially-hardened IDS model
#   Step 4 → Prepare attack samples from test set
#   Step 5 → Run FGSM + PGD attacks on STANDARD model
#   Step 6 → Run FGSM + PGD attacks on ADVERSARIAL model
#   Step 7 → Compare results and save summary report
#
# HOW TO RUN:
#   cd ~/A-IDS/adversarial_ids/src
#   python3 run_experiment.py
#
# PRE-REQUISITE:
#   prepare_kdd.py must be run first — only once:
#   cd ~/A-IDS/adversarial_ids
#   python3 prepare_kdd.py
#
# FILES USED:
#   reads  → ../data/network_flows.csv   (prepared by prepare_kdd.py)
#   writes → ../models/                  (trained model files)
#   writes → ../results/                 (attack result JSONs)
#
# KEY DESIGN DECISIONS:
#   1. Loads real NSL-KDD CSV — never generates synthetic data
#      Synthetic data was too perfectly separable (100% accuracy,
#      0% evasion) making the experiment scientifically invalid.
#
#   2. fgsm_epsilon=0.80 for adversarial training
#      Previous value (0.15) was too small — model only learned
#      to resist tiny perturbations but became MORE vulnerable
#      to larger ones. 0.80 matches the realistic attack range.
#
#   3. Epsilon range [0.05 → 2.00]
#      NSL-KDD trained model has very high confidence (sigmoid
#      output ≈ 0.9999). Small epsilons (< 0.75) cannot push
#      predictions across the 0.5 threshold. Range up to 2.0
#      shows the full evasion curve and defence gap.
#
#   4. Hard stop if CSV missing
#      No silent fallback to synthetic data. If CSV is missing,
#      user must run prepare_kdd.py first.
#
# CDAC ITISS — Adversarial Input Attack on ML-based IDS
# =============================================================

import os
import sys
import json
import numpy as np
import pandas as pd

# ── Allow Python to find src/ modules when running from src/ ──
sys.path.insert(0, os.path.dirname(__file__))

# ── Import model and attack modules ───────────────────────────
# ids_model.py      → IDSTrainer (trains, saves, loads model)
#                   → NUMERIC_FEATURES (38 NSL-KDD feature names)
# fgsm_pgd_attack.py → FGSMAttacker (FGSM + PGD attack logic)
#                    → epsilon_sweep (runs attack at many epsilons)
from ids_model       import IDSTrainer, NUMERIC_FEATURES
from fgsm_pgd_attack import FGSMAttacker, epsilon_sweep


# ─────────────────────────────────────────────────────────────
#  PATH CONFIGURATION
#  All paths relative to src/ (where this script runs from)
# ─────────────────────────────────────────────────────────────
DATA_DIR    = '../data'
MODELS_DIR  = '../models'
RESULTS_DIR = '../results'
CSV_PATH    = f'{DATA_DIR}/network_flows.csv'


# ─────────────────────────────────────────────────────────────
#  EPSILON VALUES for attack sweep
#
#  WHY THIS RANGE:
#    ε < 0.75  → too small to cross 0.5 threshold on NSL-KDD
#    ε = 1.00  → first meaningful evasion appears (~10%)
#    ε = 1.50  → strong evasion (~90%)
#    ε = 2.00  → maximum evasion (~100% on standard model)
#
#  In standardised feature space (after StandardScaler).
#  ±5σ clipping in attack functions keeps values valid.
# ─────────────────────────────────────────────────────────────
EPSILONS = [0.05, 0.10, 0.20, 0.50, 0.75, 1.00, 1.50, 2.00]


def banner(title: str):
    """Print a section banner to terminal for readability."""
    width = 62
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def run_full_experiment():
    """
    Run the complete adversarial IDS experiment pipeline.

    Trains two models (standard + adversarial), attacks both
    with FGSM and PGD, compares results and saves summary.
    """

    # ── Create output directories ──────────────────────────────
    os.makedirs(MODELS_DIR,                      exist_ok=True)
    os.makedirs(RESULTS_DIR,                     exist_ok=True)
    os.makedirs(f'{RESULTS_DIR}/standard',       exist_ok=True)
    os.makedirs(f'{RESULTS_DIR}/adversarial',    exist_ok=True)
    os.makedirs(f'{RESULTS_DIR}/plots',          exist_ok=True)

    # ──────────────────────────────────────────────────────────
    #  STEP 1 — Load Real NSL-KDD Dataset
    #
    #  WHY HARD STOP:
    #    No silent fallback to synthetic data.
    #    Synthetic data produces 100% accuracy + 0% evasion
    #    making the entire experiment meaningless.
    #    User must run prepare_kdd.py first (one-time setup).
    # ──────────────────────────────────────────────────────────
    banner("STEP 1 — Loading Real NSL-KDD Dataset")

    if not os.path.exists(CSV_PATH):
        print(f"\n  ERROR: {CSV_PATH} not found!")
        print(f"  Run prepare_kdd.py first:")
        print(f"    cd ~/A-IDS/adversarial_ids")
        print(f"    python3 prepare_kdd.py\n")
        sys.exit(1)

    print(f"  Loading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # Verify CSV has required columns
    missing = [f for f in NUMERIC_FEATURES if f not in df.columns]
    if missing:
        print(f"\n  ERROR: CSV missing columns: {missing}")
        print(f"  Re-run prepare_kdd.py to regenerate CSV.\n")
        sys.exit(1)

    if 'label' not in df.columns:
        print(f"\n  ERROR: CSV missing 'label' column.")
        print(f"  Re-run prepare_kdd.py to regenerate CSV.\n")
        sys.exit(1)

    # Print dataset summary
    n_normal = (df['label'] == 0).sum()
    n_attack = (df['label'] == 1).sum()
    print(f"  ✓ Dataset loaded successfully")
    print(f"  Total samples  : {len(df):,}")
    print(f"  Normal traffic : {n_normal:,} ({n_normal/len(df)*100:.1f}%)")
    print(f"  Attack traffic : {n_attack:,} ({n_attack/len(df)*100:.1f}%)")
    print(f"  Features       : {len(NUMERIC_FEATURES)}")
    print(f"  Source         : Real NSL-KDD (KDDTrain+.txt)")

    # ──────────────────────────────────────────────────────────
    #  STEP 2 — Train Standard IDS Model
    #
    #  Standard model = trained on clean data only.
    #  No adversarial augmentation.
    #  This is the BASELINE — represents a typical IDS.
    # ──────────────────────────────────────────────────────────
    banner("STEP 2 — Training Standard IDS Model (No Defence)")

    standard_trainer = IDSTrainer(model_dir=MODELS_DIR)
    std_metrics = standard_trainer.train(
        df,
        epochs      = 30,
        batch_size  = 256,
        lr          = 1e-3,
        adversarial = False    # no FGSM augmentation
    )
    print(f"\n  ✓ Standard model accuracy: "
          f"{std_metrics['accuracy']*100:.2f}%")

    # ──────────────────────────────────────────────────────────
    #  STEP 3 — Train Adversarially-Hardened IDS Model
    #
    #  Adversarial model = trained on clean + FGSM examples.
    #  During each batch, FGSM examples are generated on the fly
    #  and mixed with clean examples (doubles effective batch size).
    #
    #  WHY fgsm_epsilon=0.80:
    #    Training epsilon must match the attack range we test.
    #    If too small (0.15): model only resists tiny perturbations
    #    and actually becomes MORE vulnerable to large ones.
    #    0.80 covers the meaningful attack range (0.75-2.0).
    # ──────────────────────────────────────────────────────────
    banner("STEP 3 — Training Adversarially-Hardened IDS Model")

    adv_trainer = IDSTrainer(model_dir=MODELS_DIR)
    adv_metrics = adv_trainer.train(
        df,
        epochs       = 30,
        batch_size   = 256,
        lr           = 1e-3,
        adversarial  = True,   # enable FGSM augmentation
        fgsm_epsilon = 0.80    # matches realistic attack range
    )
    print(f"\n  ✓ Adversarial model accuracy: "
          f"{adv_metrics['accuracy']*100:.2f}%")

    # ──────────────────────────────────────────────────────────
    #  STEP 4 — Prepare Attack Samples
    #
    #  We attack using ATTACK samples only (label=1).
    #  Goal: make the model classify them as Normal (0).
    #  We sample 1000 attack records from the dataset.
    # ──────────────────────────────────────────────────────────
    banner("STEP 4 — Preparing Attack Samples")

    attack_df = df[df['label'] == 1].sample(
        n            = min(1000, int((df['label'] == 1).sum())),
        random_state = 99
    )
    X_attacks = attack_df[NUMERIC_FEATURES].values
    print(f"  Using {len(X_attacks):,} real NSL-KDD attack samples")
    print(f"  Epsilon range  : {EPSILONS}")

    # ──────────────────────────────────────────────────────────
    #  STEP 5 — Attack Standard Model with FGSM + PGD
    #
    #  FGSMAttacker is initialised with the trained model and
    #  its scaler. epsilon_sweep runs the attack at each epsilon
    #  value and saves results to JSON.
    # ──────────────────────────────────────────────────────────
    banner("STEP 5a — FGSM Attack on STANDARD Model")
    std_attacker     = FGSMAttacker(
        standard_trainer.model,
        standard_trainer.scaler
    )
    std_fgsm_results = epsilon_sweep(
        attacker    = std_attacker,
        X_attacks   = X_attacks,
        epsilons    = EPSILONS,
        attack_type = "FGSM",
        save_dir    = f'{RESULTS_DIR}/standard'
    )

    banner("STEP 5b — PGD Attack on STANDARD Model")
    std_pgd_results = epsilon_sweep(
        attacker    = std_attacker,
        X_attacks   = X_attacks,
        epsilons    = EPSILONS,
        attack_type = "PGD",
        save_dir    = f'{RESULTS_DIR}/standard'
    )

    # ──────────────────────────────────────────────────────────
    #  STEP 6 — Attack Adversarial Model with FGSM + PGD
    #
    #  Same attacks on the hardened model.
    #  We expect LOWER evasion rates here — defence is working
    #  if adversarial evasion < standard evasion at same epsilon.
    # ──────────────────────────────────────────────────────────
    banner("STEP 6a — FGSM Attack on ADVERSARIAL Model")
    adv_attacker     = FGSMAttacker(
        adv_trainer.model,
        adv_trainer.scaler
    )
    adv_fgsm_results = epsilon_sweep(
        attacker    = adv_attacker,
        X_attacks   = X_attacks,
        epsilons    = EPSILONS,
        attack_type = "FGSM",
        save_dir    = f'{RESULTS_DIR}/adversarial'
    )

    banner("STEP 6b — PGD Attack on ADVERSARIAL Model")
    adv_pgd_results = epsilon_sweep(
        attacker    = adv_attacker,
        X_attacks   = X_attacks,
        epsilons    = EPSILONS,
        attack_type = "PGD",
        save_dir    = f'{RESULTS_DIR}/adversarial'
    )

    # ──────────────────────────────────────────────────────────
    #  STEP 7 — Summary Report
    #
    #  Compare standard vs adversarial model at each epsilon.
    #  Defence gain = standard evasion% - adversarial evasion%
    #  Positive gain = adversarial training helped.
    #  Negative gain = adversarial training hurt (epsilon mismatch)
    # ──────────────────────────────────────────────────────────
    banner("STEP 7 — Experiment Summary")

    report = {
        'dataset'        : 'NSL-KDD (Real) — KDDTrain+.txt',
        'total_samples'  : len(df),
        'attack_samples' : len(X_attacks),
        'epsilon_range'  : EPSILONS,
        'model_accuracy' : {
            'standard_model'   : round(std_metrics['accuracy'] * 100, 2),
            'adversarial_model': round(adv_metrics['accuracy'] * 100, 2)
        },
        'evasion_comparison': []
    }

    # Print comparison table
    print(f"\n  {'ε':>6} | {'Std FGSM':>10} | {'Std PGD':>9} | "
          f"{'Adv FGSM':>10} | {'Adv PGD':>9} | "
          f"{'Gain (FGSM)':>12}")
    print(f"  {'-'*68}")

    for i, eps in enumerate(EPSILONS):
        sf   = std_fgsm_results[i].evasion_rate  * 100
        sp   = std_pgd_results[i].evasion_rate   * 100
        af   = adv_fgsm_results[i].evasion_rate  * 100
        ap   = adv_pgd_results[i].evasion_rate   * 100
        gain = sf - af    # positive = defence helped

        print(f"  {eps:>6.2f} | {sf:>9.1f}% | {sp:>8.1f}% | "
              f"{af:>9.1f}% | {ap:>8.1f}% | {gain:>+11.1f}%")

        report['evasion_comparison'].append({
            'epsilon'                   : eps,
            'standard_fgsm_evasion_pct' : round(sf, 1),
            'standard_pgd_evasion_pct'  : round(sp, 1),
            'adversarial_fgsm_evasion_pct': round(af, 1),
            'adversarial_pgd_evasion_pct' : round(ap, 1),
            'defence_gain_fgsm_pct'     : round(gain, 1)
        })

    # Save summary report
    report_path = f'{RESULTS_DIR}/summary_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # ── Final summary ─────────────────────────────────────────
    banner("EXPERIMENT COMPLETE")
    print(f"  Dataset                    : NSL-KDD (Real)")
    print(f"  Standard Model Accuracy    : "
          f"{std_metrics['accuracy']*100:.2f}%")
    print(f"  Adversarial Model Accuracy : "
          f"{adv_metrics['accuracy']*100:.2f}%")
    print(f"  Best FGSM evasion (std)    : "
          f"{max(r.evasion_rate for r in std_fgsm_results)*100:.1f}%")
    print(f"  Best FGSM evasion (adv)    : "
          f"{max(r.evasion_rate for r in adv_fgsm_results)*100:.1f}%")
    print(f"  Best PGD  evasion (std)    : "
          f"{max(r.evasion_rate for r in std_pgd_results)*100:.1f}%")
    print(f"  Best PGD  evasion (adv)    : "
          f"{max(r.evasion_rate for r in adv_pgd_results)*100:.1f}%")
    print(f"\n  Results saved → {RESULTS_DIR}/")
    print(f"  Summary  saved → {report_path}")
    print(f"\n  Next step: python3 visualise_results.py")
    print()


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    run_full_experiment()