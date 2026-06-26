# =============================================================
# src/ids_model.py
# =============================================================
# PURPOSE:
#   Defines the Neural Network architecture (IDSNet) and the
#   training/loading/prediction logic (IDSTrainer) for the
#   Intrusion Detection System.
#
# WHAT THIS FILE DOES:
#   1. Defines NUMERIC_FEATURES — the 38 NSL-KDD features the
#      model trains on (same order as network_flows.csv columns)
#   2. IDSNet — 4-layer neural network: 38→256→128→64→1
#   3. IDSTrainer — handles training, saving, loading, predicting
#
# WHAT THIS FILE DOES NOT DO:
#   - Does NOT load data (that is run_experiment.py's job)
#   - Does NOT know about file paths of datasets
#   - Does NOT generate synthetic data
#   IDSTrainer.train() accepts any DataFrame passed to it.
#   This keeps model logic separate from data pipeline logic.
#
# BUGS FIXED VS ORIGINAL:
#   1. squeeze(-1) used everywhere instead of squeeze()
#      → bare squeeze() collapses tensor to scalar when
#        batch_size=1, breaking BCELoss and numpy conversion
#        This matters in live_ids.py (one packet at a time)
#   2. self.model.zero_grad() added in _fgsm_batch()
#      → without it, model gradients accumulated across batches
#        silently corrupting adversarial training loss
#   3. weights_only=True added in torch.load()
#      → weights_only=False allows arbitrary code execution
#        via pickle — serious security vulnerability
#
# CDAC ITISS — Adversarial Input Attack on ML-based IDS
# =============================================================

import os
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib


# ─────────────────────────────────────────────────────────────
#  NUMERIC FEATURES — 38 NSL-KDD features used for training
#
#  WHY THESE 38:
#    NSL-KDD has 41 features total. We drop 3 categorical ones:
#    protocol_type (tcp/udp/icmp), service (http/ftp/..),
#    flag (SF/REJ/..) — neural networks need numeric input only.
#
#  ORDER MATTERS:
#    This exact order must match:
#    1. network_flows.csv column order (set by prepare_kdd.py)
#    2. Feature vector built in live_ids.py (packet → features)
#    Any mismatch = model receives wrong features = wrong predictions
# ─────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    # Basic connection features
    'duration',             # 0  length of connection in seconds
    'src_bytes',            # 1  bytes sent from source to dest
    'dst_bytes',            # 2  bytes sent from dest to source
    'land',                 # 3  1 if src/dst host:port are same
    'wrong_fragment',       # 4  number of wrong fragments
    'urgent',               # 5  number of urgent packets

    # Content features (application layer)
    'hot',                  # 6  number of hot indicators
    'num_failed_logins',    # 7  number of failed login attempts
    'logged_in',            # 8  1 if successfully logged in
    'num_compromised',      # 9  number of compromised conditions
    'root_shell',           # 10 1 if root shell obtained
    'su_attempted',         # 11 1 if su root command attempted
    'num_root',             # 12 number of root accesses
    'num_file_creations',   # 13 number of file creation operations
    'num_shells',           # 14 number of shell prompts
    'num_access_files',     # 15 number of operations on access control files
    'num_outbound_cmds',    # 16 number of outbound commands in ftp session
    'is_host_login',        # 17 1 if login is a host login
    'is_guest_login',       # 18 1 if login is a guest login

    # Traffic features — same host (last 2 seconds)
    'count',                # 19 connections to same host
    'srv_count',            # 20 connections to same service
    'serror_rate',          # 21 % connections with SYN errors
    'srv_serror_rate',      # 22 % connections with SYN errors (service)
    'rerror_rate',          # 23 % connections with REJ errors
    'srv_rerror_rate',      # 24 % connections with REJ errors (service)
    'same_srv_rate',        # 25 % connections to same service
    'diff_srv_rate',        # 26 % connections to different services
    'srv_diff_host_rate',   # 27 % connections to different hosts

    # Traffic features — destination host (last 100 connections)
    'dst_host_count',           # 28 connections to same dest host
    'dst_host_srv_count',       # 29 connections to same dest host/service
    'dst_host_same_srv_rate',   # 30 % same service connections to dest host
    'dst_host_diff_srv_rate',   # 31 % different service connections
    'dst_host_same_src_port_rate', # 32 % same src port connections
    'dst_host_srv_diff_host_rate', # 33 % different host connections
    'dst_host_serror_rate',     # 34 % SYN error connections to dest host
    'dst_host_srv_serror_rate', # 35 % SYN error connections (service)
    'dst_host_rerror_rate',     # 36 % REJ error connections to dest host
    'dst_host_srv_rerror_rate', # 37 % REJ error connections (service)
]


# ─────────────────────────────────────────────────────────────
#  IDSNet — Neural Network Architecture
#
#  TYPE    : Multi-layer Perceptron (MLP)
#  TASK    : Binary classification — Normal (0) vs Attack (1)
#  INPUT   : 38 scaled numeric NSL-KDD features
#  OUTPUT  : Single probability score [0.0 → 1.0]
#            >= 0.5 = Attack, < 0.5 = Normal
#
#  ARCHITECTURE:
#    Input(38) → BatchNorm → Dense(256) → ReLU → Dropout(0.3)
#             → Dense(128) → ReLU → Dropout(0.3)
#             → Dense(64)  → ReLU
#             → Dense(1)   → Sigmoid
#
#  WHY BATCHNORM FIRST:
#    Normalises input features even if scaler is imperfect.
#    Stabilises training, especially with adversarial examples.
#
#  WHY DROPOUT:
#    Prevents overfitting — important for adversarial robustness.
#    Forces model to learn distributed representations.
# ─────────────────────────────────────────────────────────────
class IDSNet(nn.Module):

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Number of input features (38 for NSL-KDD)
        """
        super(IDSNet, self).__init__()

        self.network = nn.Sequential(
            # Input normalisation
            nn.BatchNorm1d(input_dim),

            # Layer 1: 38 → 256
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 2: 256 → 128
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Layer 3: 128 → 64
            nn.Linear(128, 64),
            nn.ReLU(),

            # Output layer: 64 → 1 probability
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)


# ─────────────────────────────────────────────────────────────
#  IDSTrainer — Handles Training, Saving, Loading, Predicting
#
#  USAGE:
#    trainer = IDSTrainer(model_dir='../models')
#    trainer.train(df, adversarial=False)   # standard training
#    trainer.train(df, adversarial=True)    # adversarial training
#    trainer.load('standard')               # load saved model
#    trainer.predict(X_raw)                 # 0/1 predictions
#    trainer.predict_proba(X_raw)           # probability scores
# ─────────────────────────────────────────────────────────────
class IDSTrainer:

    def __init__(self, model_dir: str = '../models'):
        """
        Args:
            model_dir: Directory to save/load model files
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        # StandardScaler: transforms features to mean=0, std=1
        # Fitted on training data only to prevent data leakage
        self.scaler    = StandardScaler()
        self.model     = None
        self.input_dim = None

        # Training history for debugging
        self.history = {
            'train_loss': [],
            'val_loss'  : [],
            'val_acc'   : []
        }

    def preprocess(self, df: pd.DataFrame):
        """
        Extract features and labels from DataFrame.

        Args:
            df: DataFrame with NUMERIC_FEATURES columns + 'label' column
                Loaded from data/network_flows.csv

        Returns:
            X: Feature array, shape (N, 38), dtype float32
            y: Label array, shape (N,), dtype float32, values 0 or 1
        """
        X = df[NUMERIC_FEATURES].values.astype(np.float32)
        y = df['label'].values.astype(np.float32)
        return X, y

    def train(self,
              df         : pd.DataFrame,
              epochs     : int   = 30,
              batch_size : int   = 256,
              lr         : float = 1e-3,
              adversarial: bool  = False,
              fgsm_epsilon: float = 0.80):
        """
        Train the IDS neural network.

        Args:
            df          : DataFrame from data/network_flows.csv
            epochs      : Number of training epochs
            batch_size  : Mini-batch size for SGD
            lr          : Adam learning rate
            adversarial : If True → mix clean + FGSM adversarial
                          batches during training (hardens model)
            fgsm_epsilon: Perturbation budget for adversarial
                          augmentation. Set to 0.80 to match the
                          realistic attack range (0.75-2.0).
                          Too small (0.15) = model only resists
                          tiny perturbations = fails on real attacks

        Returns:
            metrics dict with accuracy, type, input_dim, epochs
        """
        # ── Split data ────────────────────────────────────────
        X, y = self.preprocess(df)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y          # maintain class balance in both sets
        )

        # ── Scale features ────────────────────────────────────
        # FIT on training data only — never on validation data
        # Prevents data leakage (val data influencing scaling)
        X_train_s = self.scaler.fit_transform(X_train)
        X_val_s   = self.scaler.transform(X_val)

        # ── Build model ───────────────────────────────────────
        self.input_dim = X_train_s.shape[1]    # should be 38
        self.model     = IDSNet(self.input_dim)

        # ── Optimiser + Loss + Scheduler ─────────────────────
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4   # L2 regularisation
        )
        # Reduce LR by 50% every 10 epochs
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.5
        )

        # ── DataLoader ────────────────────────────────────────
        train_dataset = TensorDataset(
            torch.tensor(X_train_s, dtype=torch.float32),
            torch.tensor(y_train,   dtype=torch.float32)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True        # shuffle each epoch
        )

        # Validation tensors (used as-is, no DataLoader needed)
        X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
        y_val_t = torch.tensor(y_val,   dtype=torch.float32)

        # ── Training header ───────────────────────────────────
        mode = 'Adversarially-Hardened' if adversarial else 'Standard'
        print(f"\n{'='*60}")
        print(f"  Training {mode} IDS Model")
        print(f"  Data         : Real NSL-KDD")
        print(f"  Train samples: {len(X_train):,}")
        print(f"  Val samples  : {len(X_val):,}")
        print(f"  Architecture : {self.input_dim}→256→128→64→1")
        print(f"  Epochs       : {epochs}")
        if adversarial:
            print(f"  FGSM epsilon : {fgsm_epsilon} (adversarial aug)")
        print(f"{'='*60}")

        # ── Training loop ─────────────────────────────────────
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()

                if adversarial:
                    # Generate adversarial versions of this batch
                    X_adv = self._fgsm_batch(
                        X_batch, y_batch, criterion, fgsm_epsilon
                    )
                    # Mix original + adversarial (doubles batch size)
                    # Model learns to classify both correctly
                    X_combined = torch.cat([X_batch, X_adv], dim=0)
                    y_combined = torch.cat([y_batch, y_batch], dim=0)

                    # FIX: squeeze(-1) not squeeze()
                    # squeeze() with no dim breaks when batch_size=1
                    out  = self.model(X_combined).squeeze(-1)
                    loss = criterion(out, y_combined)
                else:
                    # FIX: squeeze(-1) not squeeze()
                    out  = self.model(X_batch).squeeze(-1)
                    loss = criterion(out, y_batch)

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()

            # ── Validation ────────────────────────────────────
            self.model.eval()
            with torch.no_grad():
                # FIX: squeeze(-1) not squeeze()
                val_out   = self.model(X_val_t).squeeze(-1)
                val_loss  = criterion(val_out, y_val_t).item()
                val_preds = (val_out >= 0.5).float()
                val_acc   = (val_preds == y_val_t).float().mean().item()

            avg_loss = epoch_loss / len(train_loader)
            self.history['train_loss'].append(avg_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch [{epoch+1:3d}/{epochs}] "
                      f"Loss: {avg_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Val Acc: {val_acc*100:.2f}%")

        # ── Final evaluation ──────────────────────────────────
        self.model.eval()
        with torch.no_grad():
            # FIX: squeeze(-1) not squeeze()
            val_out     = self.model(X_val_t).squeeze(-1)
            final_preds = (val_out >= 0.5).numpy()

        print(f"\n{'='*60}")
        print("  Final Validation Report:")
        print(classification_report(
            y_val, final_preds,
            target_names=['Normal', 'Attack'],
            digits=4
        ))

        # ── Save model, scaler, metrics ───────────────────────
        tag = 'adversarial' if adversarial else 'standard'

        # Save model weights
        torch.save(
            self.model.state_dict(),
            f'{self.model_dir}/ids_{tag}.pth'
        )

        # Save scaler (must use same scaler at inference time)
        joblib.dump(
            self.scaler,
            f'{self.model_dir}/scaler_{tag}.pkl'
        )

        # Save metrics for reference
        metrics = {
            'accuracy' : float(accuracy_score(y_val, final_preds)),
            'type'     : tag,
            'input_dim': self.input_dim,
            'epochs'   : epochs,
            'dataset'  : 'NSL-KDD'
        }
        with open(f'{self.model_dir}/metrics_{tag}.json', 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"  Model saved  → {self.model_dir}/ids_{tag}.pth")
        print(f"  Scaler saved → {self.model_dir}/scaler_{tag}.pkl")
        return metrics

    def _fgsm_batch(self,
                    X_batch  : torch.Tensor,
                    y_batch  : torch.Tensor,
                    criterion: nn.Module,
                    epsilon  : float) -> torch.Tensor:
        """
        Generate adversarial examples for one training batch.

        HOW FGSM WORKS:
          x_adv = x + ε · sign(∇_x Loss(model(x), y))
          We nudge each feature in the direction that increases
          the loss — making the model more likely to misclassify.

        Used during adversarial training to harden the model.
        The model then trains on both clean + adversarial examples.

        FIX 1: self.model.zero_grad() added
               Without this, model parameter gradients accumulate
               across batches, corrupting adversarial training loss.

        FIX 2: squeeze(-1) instead of squeeze()
               Safe for any batch size including 1.

        Args:
            X_batch  : Clean input batch
            y_batch  : True labels for batch
            criterion: Loss function (BCELoss)
            epsilon  : Perturbation budget

        Returns:
            X_adv: Adversarial version of X_batch (same shape)
        """
        # FIX: clear model gradients before adversarial forward pass
        self.model.zero_grad()

        # Clone so we don't modify original batch
        X_adv = X_batch.clone().detach().requires_grad_(True)

        # Forward pass to compute loss
        # FIX: squeeze(-1) not squeeze()
        out  = self.model(X_adv).squeeze(-1)
        loss = criterion(out, y_batch)

        # Backward pass — compute gradient w.r.t. INPUT (not weights)
        loss.backward()

        # FGSM step: move in direction of gradient sign
        perturbation = epsilon * X_adv.grad.sign()

        # Return adversarial example (detached from computation graph)
        return (X_adv + perturbation).detach()

    def load(self, tag: str = 'standard'):
        """
        Load a saved model and scaler from disk.

        Used by live_ids.py for real-time inference.

        FIX: weights_only=True
             Old default (False) uses pickle which allows arbitrary
             code execution when loading untrusted model files.
             weights_only=True only loads tensor data — secure.

        Args:
            tag: 'standard' or 'adversarial'
        """
        # Load scaler
        self.scaler = joblib.load(
            f'{self.model_dir}/scaler_{tag}.pkl'
        )

        # Load metadata
        with open(f'{self.model_dir}/metrics_{tag}.json') as f:
            meta = json.load(f)

        # Rebuild model with same architecture
        self.input_dim = meta['input_dim']
        self.model     = IDSNet(self.input_dim)

        # Load saved weights
        # FIX: weights_only=True — secure loading
        self.model.load_state_dict(
            torch.load(
                f'{self.model_dir}/ids_{tag}.pth',
                map_location='cpu',
                weights_only=True
            )
        )
        self.model.eval()
        print(f"  Loaded {tag} model "
              f"(accuracy={meta['accuracy']*100:.2f}%)")

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Predict class labels for raw (unscaled) feature arrays.

        FIX: squeeze(-1) — critical for live_ids.py which passes
             one packet at a time (batch_size=1).
             bare squeeze() would collapse output to scalar,
             breaking numpy conversion.

        Args:
            X_raw: Raw feature array, shape (N, 38)

        Returns:
            Array of 0/1 predictions, shape (N,)
            0 = Normal, 1 = Attack
        """
        X_scaled = self.scaler.transform(X_raw.astype(np.float32))
        tensor   = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            # FIX: squeeze(-1) not squeeze()
            out = self.model(tensor).squeeze(-1)

        return (out >= 0.5).numpy().astype(int)

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        """
        Return raw attack probability scores.

        Used by live_ids.py to show confidence percentage in alerts.
        e.g. 0.92 → displayed as "92% attack probability"

        FIX: squeeze(-1) — same single-sample safety fix as predict()

        Args:
            X_raw: Raw feature array, shape (N, 38)

        Returns:
            Probability array, shape (N,), values in [0.0, 1.0]
        """
        X_scaled = self.scaler.transform(X_raw.astype(np.float32))
        tensor   = torch.tensor(X_scaled, dtype=torch.float32)

        with torch.no_grad():
            # FIX: squeeze(-1) not squeeze()
            out = self.model(tensor).squeeze(-1)

        return out.numpy()