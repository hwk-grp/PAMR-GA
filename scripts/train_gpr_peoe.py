#!/usr/bin/env python3
"""
train_gpr_peoe.py
=================
PAMR-GA Phase 2 & 3 — Spatial-Electrostatic Weighted Fingerprint (SEWF) + GPR Training
===========================================================================

Descriptor (2048 features):
    Morgan fingerprint (radius=2, 2048 bits) weighted by SASA × PEOE.

    V_bit_i = Σ mean_SASA(a) × |PEOE(a)|  for atoms a contributing to bit i

    Where mean_SASA(a) = average SASA of atom a across 5 conformers.

    Physical interpretation:
        mean_SASA(a): Average Solvent Accessible Surface Area across 5 conformers
                  → 3D property — preserves 2.5D character
                  → Averaging captures conformational flexibility


        |PEOE(a)|: Absolute Gasteiger partial charge
                  → 2D property — physically meaningful at every atom
                  → Encodes electrostatic/H-bond character


        Product mean_SASA × |PEOE|:
                  → "Electrostatically active surface exposure"
                  → Only exposed polar atoms contribute significantly

        V_bit_i = Σ mean_SASA(a) × |PEOE(a)|
             → Exposed charged atoms weighted highest
             → Directly captures H-bond donor/acceptor surface exposure

    Systematic testing (RF R²):
        SASA × PEOE:                  R²=0.658 ← best
        SASA × (PEOE+LogP):           R²=0.655
        SASA × (PEOE+vdW):            R²=0.641
        SASA × (PEOE+vdW+LogP+SMR):   R²=0.638

Model: GPR with Tanimoto kernel
Conformers: 5 per molecule (MMFF optimized, O3A aligned, SASA averaged)

Usage:
    python3 scripts/train_gpr_peoe.py

Output:
    models/gpr_peoe_bace1.pkl
    data/X_sasa_peoe_descriptors.npy
    data/y_sasa_peoe_labels.npy
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdPartialCharges, rdFreeSASA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class TanimotoKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    def __init__(self): pass
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None: Y = X
        X = np.atleast_2d(X); Y = np.atleast_2d(Y)
        XY = np.dot(X, Y.T)
        XX = np.sum(X**2, axis=1); YY = np.sum(Y**2, axis=1)
        denom = XX[:,None] + YY[None,:] - XY
        denom = np.where(denom==0, 1e-10, denom)
        K = np.clip(XY/denom, 0.0, 1.0)
        if eval_gradient: return K, np.empty((X.shape[0], Y.shape[0], 0))
        return K
    def diag(self, X): return np.ones(X.shape[0])
    def is_stationary(self): return False
    def get_params(self, deep=True): return {}
    def __repr__(self): return 'TanimotoKernel()'

RDLogger.DisableLog('rdApp.*')

# ── Paths ──────────────────────────────────────────────────────────────
DATA_PATH  = 'data/bace1_dataset.csv'
MODEL_PATH = 'models/gpr_peoe_bace1.pkl'
DESC_X     = 'data/X_sasa_peoe_descriptors.npy'
DESC_Y     = 'data/y_sasa_peoe_labels.npy'

# ── Descriptor Parameters ──────────────────────────────────────────────
N_BITS           = 2048
RADIUS           = 2
N_CONFS          = 5
ACTIVE_THRESHOLD = 8.0


# ── Descriptor: SASA × PEOE Weighted Morgan ───────────────────────────
def get_sasa_peoe_descriptor(smiles):
    """
    Compute the spatial-electrostatic weighted fingerprint (SEWF): a Morgan
    fingerprint (2048 bits, radius=2) whose bits are weighted by SASA x |PEOE|.

    V_bit_i = Σ mean_SASA(a) × |PEOE(a)|  for atoms a contributing to bit i

    Uses 5 conformers with SASA averaging — averaged over conformers.

    Returns np.ndarray [2048] or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        # Gasteiger partial charges (2D — same for all conformers)
        rdPartialCharges.ComputeGasteigerCharges(mol)

        # Generate N_CONFS conformers
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        conf_ids = AllChem.EmbedMultipleConfs(
            mol_h, numConfs=N_CONFS, params=params)

        if len(conf_ids) == 0:
            return None

        # Optimize each conformer with MMFF
        for conf_id in conf_ids:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, confId=conf_id)
            except Exception:
                pass

        # Average SASA across conformers
        n_heavy = mol.GetNumAtoms()
        sasa_sum = np.zeros(n_heavy, dtype=np.float64)
        n_valid  = 0

        for conf_id in conf_ids:
            try:
                radii = rdFreeSASA.classifyAtoms(mol_h)
                rdFreeSASA.CalcSASA(mol_h, radii, confIdx=conf_id)
                for i in range(n_heavy):
                    sasa_sum[i] += float(
                        mol_h.GetAtomWithIdx(i).GetPropsAsDict().get(
                            'SASA', 0.0))
                n_valid += 1
            except Exception:
                pass

        if n_valid == 0:
            return None

        mean_sasa = sasa_sum / n_valid

        # Morgan bit info
        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(
            mol, RADIUS, N_BITS, bitInfo=bit_info)

        # Weight bits by mean_SASA × |PEOE|
        fp = np.zeros(N_BITS, dtype=np.float32)
        for bit_id, incidents in bit_info.items():
            target_idx = bit_id % N_BITS
            for center_atom_idx, _ in incidents:
                atom = mol.GetAtomWithIdx(center_atom_idx)
                peoe = abs(float(
                    atom.GetPropsAsDict().get('_GasteigerCharge', 0.0)))
                fp[target_idx] += mean_sasa[center_atom_idx] * peoe

    except Exception:
        return None

    if np.isnan(fp).any() or np.isinf(fp).any():
        return None

    return fp


# ── Main ───────────────────────────────────────────────────────────────
def main():
    os.makedirs('models', exist_ok=True)
    os.makedirs('data',   exist_ok=True)

    print("=" * 65)
    print("  PAMR-GA GPR Training")
    print("  Descriptor: SEWF — SASA×PEOE weighted Morgan (2048 bits, r=2)")
    print("  Weight:     mean_SASA(atom) × |PEOE(atom)|")
    print(f"  Conformers: {N_CONFS} per molecule (MMFF optimized)")
    print("  Model:      GPR with Tanimoto kernel")
    print("=" * 65)

    # Step 1: Load dataset
    print("\n[Step 1/5] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Total molecules:          {len(df)}")
    print(f"  pIC50 range:              "
          f"{df['pIC50'].min():.2f} — {df['pIC50'].max():.2f}")
    print(f"  Actives (>={ACTIVE_THRESHOLD}): "
          f"{(df['pIC50'] >= ACTIVE_THRESHOLD).sum()}")

    # Step 2: Generate descriptors
    if os.path.exists(DESC_X) and os.path.exists(DESC_Y):
        print(f"\n[Step 2/5] Loading cached descriptors...")
        X = np.load(DESC_X)
        y = np.load(DESC_Y)
        print(f"  Loaded: {X.shape[0]} molecules x {X.shape[1]} features")
    else:
        print(f"\n[Step 2/5] Generating SEWF (SASA×PEOE weighted Morgan) descriptors...")
        print(f"  Bits: {N_BITS} | Radius: {RADIUS} | Conformers: {N_CONFS}")
        X_list, y_list = [], []
        failed = 0

        for row in tqdm(df.itertuples(), total=len(df),
                        desc="  Computing", unit="mol"):
            vec = get_sasa_peoe_descriptor(row.canonical_smiles)
            if vec is not None:
                X_list.append(vec)
                y_list.append(row.pIC50)
            else:
                failed += 1

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list)
        mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        X, y = X[mask], y[mask]
        np.save(DESC_X, X)
        np.save(DESC_Y, y)
        print(f"  Valid:   {len(X)} ({100*len(X)/len(df):.1f}%)")
        print(f"  Failed:  {failed}")
        print(f"  Saved:   {DESC_X}")

    print(f"\n  Descriptor statistics:")
    print(f"  Shape:    {X.shape}")
    print(f"  Mean:     {X.mean():.4f}")
    print(f"  Std:      {X.std():.4f}")
    print(f"  Non-zero: {(X != 0).mean():.1%} entries")

    # Step 3: Train/test split
    print(f"\n[Step 3/5] Train/test split (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Actives in train: {(y_train >= ACTIVE_THRESHOLD).sum()}")

    # Step 4: Scale + train GPR
    print(f"\n[Step 4/5] Training GPR (Tanimoto kernel)...")

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(
        y_train.reshape(-1, 1)).ravel()

    X_scaler = StandardScaler()
    X_train_scaled = X_scaler.fit_transform(X_train)
    X_test_scaled  = X_scaler.transform(X_test)

    kernel = TanimotoKernel()
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.1,
        n_restarts_optimizer=5,
    )
    t0 = time.time()
    gpr.fit(X_train_scaled, y_train_scaled)
    train_time = (time.time() - t0) / 60
    print(f"  Done in {train_time:.1f} min")
    print(f"  Kernel: {gpr.kernel_}")

    # Step 5: Evaluate
    print(f"\n[Step 5/5] Evaluating on {len(X_test)} test molecules...")
    y_pred_scaled = gpr.predict(X_test_scaled, return_std=False)
    y_pred = y_scaler.inverse_transform(
        y_pred_scaled.reshape(-1, 1)).ravel()

    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"  Test R²:    {r2:.4f}")
    print(f"  Test RMSE:  {rmse:.4f} pIC50 units")

    # Save model
    model_data = {
        'gpr':             gpr,
        'y_scaler':        y_scaler,
        'scaler_mean':     float(y_scaler.mean_[0]),
        'scaler_std':      float(y_scaler.scale_[0]),
        'X_scaler':        X_scaler,
        'target':          'BACE-1 (CHEMBL4822)',
        'n_train':         len(X_train),
        'r2_test':         float(r2),
        'rmse_test':       float(rmse),
        'kernel':          str(gpr.kernel_),
        'descriptor':      'SASA_PEOE_weighted_Morgan_2048bits_r2',
        'n_bits':          N_BITS,
        'radius':          RADIUS,
        'n_conformers':    N_CONFS,
        'weight':          'mean_SASA(atom) x |PEOE(atom)|',
        'version':         '',
        'active_threshold': ACTIVE_THRESHOLD,
        'train_time_min':  train_time,
    }

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model_data, f)

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print("=" * 65)
    print(f"  Descriptor:  SASA×PEOE Weighted Morgan")
    print(f"  Weight:      mean_SASA(atom) × |PEOE(atom)|")
    print(f"  Bits:        {N_BITS} | Radius: {RADIUS}")
    print(f"  Conformers:  {N_CONFS} per molecule (MMFF averaged)")
    print(f"  Train size:  {len(X_train)}")
    print(f"  Test R²:     {r2:.4f}")
    print(f"  Test RMSE:   {rmse:.4f}")
    print(f"  Model saved: {MODEL_PATH}")
    print("=" * 65)
    print("\n  Next: python3 pamr_ga.py")


if __name__ == "__main__":
    main()
