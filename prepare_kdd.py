# =============================================================
# prepare_kdd.py
# =============================================================
# PURPOSE:
#   Converts raw NSL-KDD dataset files into a clean CSV file
#   that the model can directly train on.
#
# WHY THIS FILE EXISTS:
#   Raw NSL-KDD has 43 columns including categorical features
#   (protocol_type, service, flag) and a difficulty score.
#   Our neural network only works with numeric features.
#   This script extracts the 38 numeric features + binary label.
#
# INPUT:
#   data/kdd/KDDTrain+.txt  — 125,973 raw NSL-KDD records
#
# OUTPUT:
#   data/network_flows.csv  — 20,000 clean numeric records
#                             ready for training
#
# RUN ONCE:
#   python prepare_kdd.py
#   (No need to run again unless you delete network_flows.csv)
#
# CDAC ITISS — Adversarial Input Attack on ML-based IDS
# =============================================================

import pandas as pd
import os
import sys

# ─────────────────────────────────────────────────────────────
#  NSL-KDD COLUMN NAMES
#  The raw file has no header row — we assign names manually.
#  43 total: 41 features + label + difficulty score
# ─────────────────────────────────────────────────────────────
ALL_COLUMNS = [
    # Basic connection features (1-9)
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',

    # Content features (10-22)
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login',

    # Traffic features — same host (23-31)
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate',

    # Traffic features — destination host (32-41)
    'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',

    # Target + metadata
    'label',       # attack type string e.g. 'normal', 'neptune'
    'difficulty'   # NSL-KDD difficulty score — not used, dropped
]

# ─────────────────────────────────────────────────────────────
#  38 NUMERIC FEATURES
#  We drop: protocol_type, service, flag (categorical)
#           difficulty (not relevant to detection)
#  These 38 features are what the neural network trains on.
#  Order here = exact column order in network_flows.csv
# ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
N_SAMPLES   = 20000                    # Total samples in output CSV
INPUT_PATH  = 'data/kdd/KDDTrain+.txt' # Raw NSL-KDD input
OUTPUT_PATH = 'data/network_flows.csv'  # Clean CSV output


def main():

    # ── Verify input file exists ──────────────────────────────
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: {INPUT_PATH} not found!")
        print("Run this first:")
        print("  wget https://raw.githubusercontent.com/jmnwong/"
              "NSL-KDD-Dataset/master/KDDTrain%2B.txt "
              "-O data/kdd/KDDTrain+.txt")
        sys.exit(1)

    # ── Step 1: Load raw NSL-KDD file ────────────────────────
    print("=" * 55)
    print("  NSL-KDD Preprocessing")
    print("=" * 55)
    print(f"\n[1/5] Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH, header=None, names=ALL_COLUMNS)
    print(f"      Rows loaded : {len(df):,}")
    print(f"      Columns     : {len(df.columns)}")

    # ── Step 2: Show original attack type distribution ───────
    print(f"\n[2/5] Original label distribution (top 10):")
    label_counts = df['label'].value_counts().head(10)
    for label, count in label_counts.items():
        bar = '█' * (count // 2000)
        print(f"      {label:<20} {count:>6,}  {bar}")

    # ── Step 3: Convert multi-class → binary label ───────────
    # NSL-KDD has 39 attack types + normal traffic.
    # We simplify to binary:
    #   normal  → 0  (legitimate traffic)
    #   anything else → 1  (attack traffic)
    print(f"\n[3/5] Converting to binary labels...")
    print(f"      normal → 0  (legitimate)")
    print(f"      attack → 1  (all 38 attack types combined)")
    df['label'] = (df['label'] != 'normal').astype(int)

    n_normal = (df['label'] == 0).sum()
    n_attack = (df['label'] == 1).sum()
    print(f"      Normal : {n_normal:,} ({n_normal/len(df)*100:.1f}%)")
    print(f"      Attack : {n_attack:,} ({n_attack/len(df)*100:.1f}%)")

    # ── Step 4: Select numeric features only ─────────────────
    # Drop categorical: protocol_type, service, flag
    # Drop metadata  : difficulty
    # Keep           : 38 numeric features + binary label
    print(f"\n[4/5] Selecting {len(NUMERIC_FEATURES)} numeric features...")
    df_numeric = df[NUMERIC_FEATURES + ['label']]

    # ── Step 5: Stratified sampling to N_SAMPLES ─────────────
    # Keep same class ratio as full dataset (53.5% normal / 46.5% attack)
    # so training data reflects real NSL-KDD distribution
    print(f"\n[5/5] Stratified sampling to {N_SAMPLES:,} records...")

    normal_df = df_numeric[df_numeric['label'] == 0]
    attack_df = df_numeric[df_numeric['label'] == 1]

    # Calculate how many of each class to sample
    n_normal_sample = int(N_SAMPLES * len(normal_df) / len(df_numeric))
    n_attack_sample = N_SAMPLES - n_normal_sample

    # Sample from each class
    normal_sample = normal_df.sample(
        n=min(n_normal_sample, len(normal_df)), random_state=42
    )
    attack_sample = attack_df.sample(
        n=min(n_attack_sample, len(attack_df)), random_state=42
    )

    # Combine and shuffle
    df_final = pd.concat([normal_sample, attack_sample])
    df_final = df_final.sample(frac=1, random_state=42)
    df_final = df_final.reset_index(drop=True)

    # ── Save to CSV ───────────────────────────────────────────
    df_final.to_csv(OUTPUT_PATH, index=False)

    # ── Summary ───────────────────────────────────────────────
    n_out_normal = (df_final['label'] == 0).sum()
    n_out_attack = (df_final['label'] == 1).sum()

    print(f"\n{'=' * 55}")
    print(f"  Preprocessing Complete")
    print(f"{'=' * 55}")
    print(f"  Output file : {OUTPUT_PATH}")
    print(f"  Total rows  : {len(df_final):,}")
    print(f"  Normal      : {n_out_normal:,} ({n_out_normal/len(df_final)*100:.1f}%)")
    print(f"  Attack      : {n_out_attack:,} ({n_out_attack/len(df_final)*100:.1f}%)")
    print(f"  Features    : {len(NUMERIC_FEATURES)}")
    print(f"\n  Next step: run src/run_experiment.py")
    print(f"{'=' * 55}\n")


if __name__ == '__main__':
    main()