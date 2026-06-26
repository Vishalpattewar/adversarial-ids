# =============================================================
# src/fgsm_pgd_attack.py
# =============================================================
# PURPOSE:
#   Implements two white-box adversarial attacks against IDSNet:
#
#   1. FGSM — Fast Gradient Sign Method (Goodfellow et al. 2014)
#      Single-step attack. Fast but less powerful.
#
#   2. PGD — Projected Gradient Descent (Madry et al. 2017)
#      Multi-step iterative FGSM. Slower but much stronger.
#
# HOW ADVERSARIAL ATTACKS WORK HERE:
#   The attacker has a network flow classified as Attack (1).
#   Goal: perturb its features slightly so the model classifies
#   it as Normal (0) → attack evades detection.
#
#   FGSM formula:
#     x_adv = x + ε · sign(∇_x Loss(model(x), y=1))
#
#   We ADD gradient (not subtract) because we want to MAXIMISE
#   the loss for label y=1 (attack). When loss is maximised, the
#   model's confidence in "attack" drops below 0.5 → predicts
#   "normal" → evasion achieved.
#
# WHITE-BOX ASSUMPTION:
#   Attacker knows model weights (worst case scenario).
#   This gives upper bound on how vulnerable the model is.
#
# REALISTIC CONSTRAINT:
#   Not all features can be modified by an attacker in real traffic.
#   e.g. dst_bytes is set by the server, not the attacker.
#   MUTABLE_FEATURES = features attacker can realistically control.
#   Gradients for immutable features are zeroed out.
#
# BUGS FIXED VS ORIGINAL:
#   1. squeeze(-1) everywhere instead of squeeze()
#      → bare squeeze() breaks when batch_size=1
#   2. non_neg_idx fixed in attack_fgsm()
#      → original wrongly excluded valid non-negative rate
#        features from the clip, allowing them to go negative
#        producing physically invalid feature values
#      → fix: clip ALL 38 features to >= 0 (consistent with PGD)
#
# CDAC ITISS — Adversarial Input Attack on ML-based IDS
# =============================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ─────────────────────────────────────────────────────────────
#  NUMERIC FEATURES — must match ids_model.py exactly
#
#  We redefine here (instead of importing) to keep this file
#  self-contained and usable independently of ids_model.py.
#  Both lists must always be identical.
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
#  MUTABLE FEATURES — features attacker can realistically modify
#
#  WHY ONLY THESE:
#    In real network traffic, some features are determined by
#    the SERVER (e.g. dst_bytes) or by the NETWORK (e.g. land).
#    An attacker crafting malicious packets can only control
#    features on the CLIENT/SENDER side.
#
#  IMMUTABLE (excluded):
#    dst_bytes       — set by server response
#    land            — determined by routing
#    num_compromised — result of actual exploitation
#    root_shell      — result of actual exploitation
#    su_attempted    — result of actual exploitation
#    num_root        — result of actual exploitation
#    num_file_creations, num_shells — result of exploitation
# ─────────────────────────────────────────────────────────────
MUTABLE_FEATURE_NAMES = [
    'duration', 'src_bytes', 'wrong_fragment',
    'hot', 'num_failed_logins', 'logged_in',
    'num_access_files', 'num_outbound_cmds', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate'
]

# Pre-compute indices of mutable features in NUMERIC_FEATURES list
# Used to build a mask that zeroes out immutable feature gradients
MUTABLE_INDICES = [
    NUMERIC_FEATURES.index(f) for f in MUTABLE_FEATURE_NAMES
]


# ─────────────────────────────────────────────────────────────
#  AttackResult — stores results from one attack run
#
#  Stores all metrics for one (model, attack_type, epsilon) combo.
#  to_dict() converts to JSON-serialisable format for saving.
# ─────────────────────────────────────────────────────────────
@dataclass
class AttackResult:
    epsilon            : float  # perturbation budget used
    n_original         : int    # samples originally detected as attack
    n_evaded           : int    # samples that evaded detection
    evasion_rate       : float  # n_evaded / n_original
    avg_perturbation_l2: float  # mean L2 norm of perturbations
    avg_perturbation_linf: float # mean L∞ norm of perturbations
    confidence_drop    : float  # mean drop in attack probability
    iterations         : int    # 1 for FGSM, 10 for PGD
    attack_type        : str    # "FGSM" or "PGD"

    def to_dict(self) -> dict:
        """Convert to JSON-serialisable dict for saving to file."""
        return {
            'epsilon'              : self.epsilon,
            'n_original'           : self.n_original,
            'n_evaded'             : self.n_evaded,
            'evasion_rate'         : round(self.evasion_rate * 100, 2),
            'avg_perturbation_l2'  : round(self.avg_perturbation_l2, 4),
            'avg_perturbation_linf': round(self.avg_perturbation_linf, 4),
            'confidence_drop'      : round(self.confidence_drop, 4),
            'iterations'           : self.iterations,
            'attack_type'          : self.attack_type
        }


# ─────────────────────────────────────────────────────────────
#  FGSMAttacker — main attack class
#
#  Implements both FGSM and PGD attacks.
#  Operates in SCALED feature space (what the model sees).
#  Applies perturbations only to MUTABLE features.
#
#  USAGE:
#    attacker = FGSMAttacker(model, scaler, epsilon=0.5)
#    X_adv, _ = attacker.attack_fgsm(X_raw)
#    X_adv, _ = attacker.attack_pgd(X_raw, n_steps=10)
#    result   = attacker.evaluate_evasion(X_raw, X_adv, eps)
# ─────────────────────────────────────────────────────────────
class FGSMAttacker:

    def __init__(self,
                 model  : nn.Module,
                 scaler ,
                 epsilon: float = 0.5):
        """
        Args:
            model  : Trained IDSNet model (in eval mode)
            scaler : Fitted StandardScaler from IDSTrainer
            epsilon: Default perturbation budget
        """
        self.model     = model
        self.scaler    = scaler
        self.epsilon   = epsilon
        self.criterion = nn.BCELoss()

        # Always set to eval mode — we do not want dropout during attack
        self.model.eval()

    def _mask_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Zero out gradients for immutable features.

        WHY:
          We only want to perturb features the attacker can
          realistically modify. Perturbing dst_bytes (set by
          server) is not physically possible in real attacks.

        Args:
            grad: Raw gradient tensor, shape (N, 38)

        Returns:
            Masked gradient — immutable feature gradients = 0
        """
        mask = torch.zeros_like(grad)
        mask[:, MUTABLE_INDICES] = 1.0
        return grad * mask

    def attack_fgsm(self,
                    X_raw  : np.ndarray,
                    epsilon: Optional[float] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-step FGSM attack.

        FORMULA:
          x_adv = x + ε · sign(∇_x Loss(model(x), y=1))

        WHY WE ADD (not subtract) THE GRADIENT:
          We want to MAXIMISE loss for y=1 (attack label).
          Maximising loss for "attack" → model confidence in
          "attack" drops → model predicts "normal" → evasion.

        STEPS:
          1. Scale raw features with scaler
          2. Forward pass → compute loss for y=1 (attack)
          3. Backward pass → get gradient w.r.t. input
          4. Mask gradient (mutable features only)
          5. Add ε · sign(gradient) to input
          6. Clip to ±5σ (valid standardised range)
          7. Inverse transform back to original scale
          8. Clip all features to >= 0 (non-negative constraint)

        FIX: non_neg_idx now clips ALL 38 features to >= 0.
             Original code excluded rate features (serror_rate etc.)
             from clipping, allowing them to go negative.
             All NSL-KDD features are >= 0 by definition.

        Args:
            X_raw  : Raw (unscaled) attack samples, shape (N, 38)
            epsilon: Perturbation budget (uses self.epsilon if None)

        Returns:
            X_adv_raw          : Adversarial examples in original space
            perturbation_linf  : L∞ perturbation per sample, shape (N,)
        """
        eps = epsilon if epsilon is not None else self.epsilon

        # Scale features to standardised space
        X_scaled = self.scaler.transform(X_raw.astype(np.float32))

        # Convert to tensor and enable gradient tracking on INPUT
        X_tensor = torch.tensor(
            X_scaled, dtype=torch.float32, requires_grad=True
        )

        # True labels = 1 (all are attack samples)
        y_tensor = torch.ones(len(X_raw), dtype=torch.float32)

        # Forward pass
        # FIX: squeeze(-1) not squeeze()
        out  = self.model(X_tensor).squeeze(-1)
        loss = self.criterion(out, y_tensor)

        # Backward pass — gradient w.r.t. INPUT features
        self.model.zero_grad()
        loss.backward()

        # Apply mutable-only mask to gradient
        masked_grad = self._mask_gradient(X_tensor.grad.data)

        # FGSM step — add ε · sign(gradient) to maximise loss
        perturbation = eps * masked_grad.sign()
        X_adv_scaled = (X_tensor + perturbation).detach().numpy()

        # Clip to ±5σ — keeps features in valid standardised range
        X_adv_scaled = np.clip(X_adv_scaled, -5.0, 5.0)

        # Inverse transform back to original feature space
        X_adv_raw = self.scaler.inverse_transform(X_adv_scaled)

        # FIX: Clip ALL features to >= 0
        # All NSL-KDD numeric features are non-negative by definition.
        # Original code wrongly excluded rate features from this clip.
        X_adv_raw = np.maximum(X_adv_raw, 0)

        # Compute L∞ perturbation magnitude per sample
        perturbation_linf = np.abs(X_adv_scaled - X_scaled).max(axis=1)

        return X_adv_raw, perturbation_linf

    def attack_pgd(self,
                   X_raw    : np.ndarray,
                   epsilon  : Optional[float] = None,
                   n_steps  : int   = 10,
                   step_size: float = None
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """
        PGD — Projected Gradient Descent attack.

        HOW PGD DIFFERS FROM FGSM:
          FGSM = 1 large step of size ε
          PGD  = n_steps small steps of size α, projected back
                 into ε-ball after each step.
          PGD is strictly stronger — finds better adversarial
          examples within the same ε budget.

        FORMULA (each step):
          x_adv = Proj(x_adv + α · sign(∇_x Loss(x_adv, y=1)))
          where Proj clips x_adv to stay within ε-ball of x_orig.

        Args:
            X_raw    : Raw attack samples, shape (N, 38)
            epsilon  : Total perturbation budget
            n_steps  : Number of PGD iterations (default 10)
            step_size: Per-step size α (default: ε/n_steps * 2)

        Returns:
            X_adv_raw         : Adversarial examples in original space
            perturbation_linf : L∞ perturbation per sample, shape (N,)
        """
        eps   = epsilon   if epsilon   is not None else self.epsilon
        alpha = step_size if step_size is not None else (eps / n_steps * 2)

        # Scale to standardised space
        X_scaled_orig = self.scaler.transform(X_raw.astype(np.float32))

        # Start from original point
        X_adv_scaled = X_scaled_orig.copy()

        # Iterative attack steps
        for step in range(n_steps):

            # Convert current adversarial point to tensor
            X_tensor = torch.tensor(
                X_adv_scaled, dtype=torch.float32, requires_grad=True
            )
            y_tensor = torch.ones(len(X_raw), dtype=torch.float32)

            # Forward + backward pass
            # FIX: squeeze(-1) not squeeze()
            out  = self.model(X_tensor).squeeze(-1)
            loss = self.criterion(out, y_tensor)
            self.model.zero_grad()
            loss.backward()

            # Mask gradient — mutable features only
            masked_grad = self._mask_gradient(
                X_tensor.grad.data
            ).numpy()

            # PGD step — move in gradient sign direction
            X_adv_scaled = X_adv_scaled + alpha * np.sign(masked_grad)

            # Project back into ε-ball around original point
            # Ensures total perturbation never exceeds ε
            delta        = np.clip(X_adv_scaled - X_scaled_orig, -eps, eps)
            X_adv_scaled = np.clip(X_scaled_orig + delta, -5.0, 5.0)

        # Inverse transform to original feature space
        X_adv_raw = self.scaler.inverse_transform(X_adv_scaled)

        # Clip all features to >= 0 (non-negative constraint)
        X_adv_raw = np.maximum(X_adv_raw, 0)

        # Compute L∞ perturbation magnitude per sample
        perturbation_linf = np.abs(
            X_adv_scaled - X_scaled_orig
        ).max(axis=1)

        return X_adv_raw, perturbation_linf

    def evaluate_evasion(self,
                         X_raw     : np.ndarray,
                         X_adv_raw : np.ndarray,
                         epsilon   : float,
                         attack_type: str = "FGSM",
                         n_steps   : int  = 1
                         ) -> AttackResult:
        """
        Measure how effectively the attack evades detection.

        EVASION DEFINITION:
          A sample successfully evades if:
            - BEFORE attack: model predicts Attack (prob >= 0.5)
            - AFTER  attack: model predicts Normal (prob <  0.5)

          We only count samples that were correctly detected
          originally. If the model already missed a sample,
          it does not count as "evaded".

        Args:
            X_raw     : Original raw samples, shape (N, 38)
            X_adv_raw : Adversarial samples,  shape (N, 38)
            epsilon   : Epsilon value used in this attack
            attack_type: "FGSM" or "PGD"
            n_steps   : 1 for FGSM, 10 for PGD

        Returns:
            AttackResult with all evasion metrics
        """
        # Scale both original and adversarial to model input space
        X_scaled     = self.scaler.transform(X_raw.astype(np.float32))
        X_adv_scaled = self.scaler.transform(X_adv_raw.astype(np.float32))

        # Get predictions for both — no gradient needed
        with torch.no_grad():
            # FIX: squeeze(-1) not squeeze()
            orig_proba = self.model(
                torch.tensor(X_scaled, dtype=torch.float32)
            ).squeeze(-1).numpy()

            adv_proba  = self.model(
                torch.tensor(X_adv_scaled, dtype=torch.float32)
            ).squeeze(-1).numpy()

        # Binary predictions (threshold = 0.5)
        orig_preds = (orig_proba >= 0.5).astype(int)
        adv_preds  = (adv_proba  >= 0.5).astype(int)

        # Originally correctly detected as attack
        originally_detected = (orig_preds == 1)

        # Successfully evaded: was attack → now predicted normal
        evaded = (orig_preds == 1) & (adv_preds == 0)

        n_original   = originally_detected.sum()
        n_evaded     = evaded.sum()
        evasion_rate = n_evaded / n_original if n_original > 0 else 0.0

        # Perturbation metrics
        perturbations = np.abs(X_adv_scaled - X_scaled)
        avg_l2        = np.linalg.norm(perturbations, axis=1).mean()
        avg_linf      = perturbations.max(axis=1).mean()

        # Confidence drop — how much did attack probability fall
        # Only measured on originally-detected samples
        if originally_detected.sum() > 0:
            conf_drop = float(
                (orig_proba[originally_detected] -
                 adv_proba[originally_detected]).mean()
            )
        else:
            conf_drop = 0.0

        return AttackResult(
            epsilon             = epsilon,
            n_original          = int(n_original),
            n_evaded            = int(n_evaded),
            evasion_rate        = float(evasion_rate),
            avg_perturbation_l2 = float(avg_l2),
            avg_perturbation_linf = float(avg_linf),
            confidence_drop     = conf_drop,
            iterations          = n_steps,
            attack_type         = attack_type
        )


# ─────────────────────────────────────────────────────────────
#  epsilon_sweep — run attack across multiple epsilon values
#
#  Runs FGSM or PGD attack at each epsilon in a given list.
#  Prints a live table to terminal.
#  Saves results to JSON file.
#
#  Used by run_experiment.py to generate the full evasion curve.
# ─────────────────────────────────────────────────────────────
def epsilon_sweep(attacker   : FGSMAttacker,
                  X_attacks  : np.ndarray,
                  epsilons   : List[float],
                  attack_type: str = "FGSM",
                  save_dir   : str = '../results'
                  ) -> List[AttackResult]:
    """
    Run attack at each epsilon value and collect results.

    Args:
        attacker   : FGSMAttacker instance with loaded model
        X_attacks  : Raw attack samples, shape (N, 38)
        epsilons   : List of epsilon values to sweep
        attack_type: "FGSM" or "PGD"
        save_dir   : Directory to save JSON results

    Returns:
        List of AttackResult — one per epsilon value
    """
    os.makedirs(save_dir, exist_ok=True)
    results = []

    # Print table header
    print(f"\n{'='*62}")
    print(f"  ε-Sweep: {attack_type} on {len(X_attacks):,} samples")
    print(f"{'='*62}")
    print(f"  {'ε':>8} | {'Evaded':>8} | {'Evasion%':>10} | "
          f"{'L∞ Pert':>9} | {'Conf Drop':>10}")
    print(f"  {'-'*57}")

    for eps in epsilons:

        # Run selected attack type
        if attack_type == "FGSM":
            X_adv, _ = attacker.attack_fgsm(X_attacks, epsilon=eps)
            result   = attacker.evaluate_evasion(
                X_attacks, X_adv, eps, "FGSM", n_steps=1
            )
        else:
            # PGD — 10 iterative steps, stronger than FGSM
            X_adv, _ = attacker.attack_pgd(
                X_attacks, epsilon=eps, n_steps=10
            )
            result   = attacker.evaluate_evasion(
                X_attacks, X_adv, eps, "PGD", n_steps=10
            )

        results.append(result)

        # Print result row
        print(f"  {eps:>8.3f} | {result.n_evaded:>8d} | "
              f"{result.evasion_rate*100:>9.1f}% | "
              f"{result.avg_perturbation_linf:>9.4f} | "
              f"{result.confidence_drop:>10.4f}")

    # Save results to JSON
    tag           = attack_type.lower()
    output_path   = f'{save_dir}/attack_results_{tag}.json'
    results_dicts = [r.to_dict() for r in results]

    with open(output_path, 'w') as f:
        json.dump(results_dicts, f, indent=2)

    print(f"\n  Saved → {output_path}")
    return results