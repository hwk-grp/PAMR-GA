#!/usr/bin/env python
"""
space_ga.py
=======================
2.5D SPACE-GA: Full Pipeline — Spatially-Constrained Genetic Algorithm
=======================================================================


Prerequisites:
    1. python3 scripts/pocket_map.py
    2. python3 scripts/train_gpr_peoe.py

Usage:
    python3 space_ga.py

Output:
    results/steps/INDEX_N.dat
    results/steps/top_mols_step_N.csv
    results/all_top_molecules.csv
"""

import os
import shutil
import time
import random
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdFreeSASA, rdPartialCharges

import crossover as co
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════
# Tanimoto Kernel (required for loading GPR model)
# ══════════════════════════════════════════════════════════════════════
from sklearn.gaussian_process.kernels import StationaryKernelMixin, NormalizedKernelMixin, Kernel

class TanimotoKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    """Generalized Tanimoto kernel for sparse fingerprint vectors."""
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

# ══════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════

INITPOOL_PATH  = 'data/bace1_initpool.dat'
POOL_DESC_PATH = 'data/pool_sasa_peoe_descriptors.npy'
POOL_SMILES_PATH = 'data/pool_smiles.npy'
MODEL_PATH     = 'models/gpr_peoe_bace1.pkl'
POCKET_PATH    = 'data/pocket_map.pkl'
MUTATION_PATH  = 'mutate_reaction.dat'
RESULTS_DIR    = 'results'

# GA Parameters
NSTEP          = 50
TARGET_POOL    = 1000
TARGET_VALUE   = 10
GAU_SIGMA      = 0.001
MU_PROB        = 0.3
SIM_THRESHOLD  = 0.8     # FIXED — no decay
MAX_ATOMS      = 41
MAX_MW         = 500

# Spatial fitness
STERIC_PENALTY = 1.0
VOL_PENALTY    = 0.05    # quadratic: VOL_PENALTY * (vol - vmax)^2
HBOND_DIST     = 3.5

# Crossover
CO_AVERAGE_SIZE = 30
CO_SIZE_STDEV   = 40

# Per-step saving
TOP_PER_STEP   = 20

# ══════════════════════════════════════════════════════════════════════
# Reactive group filter
# ══════════════════════════════════════════════════════════════════════

_REACTIVE_SMARTS = [
    '[SH]',        # free thiol
    'OF',          # hypofluorite
    '[C-]#[N+]',   # isocyanide
    'C=CO',        # enol
    'O-O',         # peroxide
    '[N-]',        # nitrene
]
_REACTIVE_QUERIES = [Chem.MolFromSmarts(s) for s in _REACTIVE_SMARTS]
MAX_ALKYNES = 2

def has_reactive_groups(mol, smiles):
    if smiles.count('C#C') > MAX_ALKYNES:
        return True
    for q in _REACTIVE_QUERIES:
        if q and mol.HasSubstructMatch(q):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════
# Phase 2: SASA×PEOE Weighted Morgan Descriptor
# ══════════════════════════════════════════════════════════════════════

N_BITS   = 2048
RADIUS   = 2
N_CONFS  = 5

def get_sasa_peoe_descriptor(smiles):
    """
    SASA×PEOE weighted Morgan fingerprint.

    V_bit_i = sum(mean_SASA(a) × |PEOE(a)|) for atoms a in bit i

    - SASA: 3D property — preserves 2.5D novelty (5 conformers, averaged)
    - PEOE: 2D atomic charge — captures H-bond donor/acceptor character
    - Weighting emphasizes electrostatically active, solvent-exposed atoms

    Returns np.ndarray [2048] or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol)
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=N_CONFS, params=params)
        if len(conf_ids) == 0:
            return None
        for conf_id in conf_ids:
            try:
                AllChem.MMFFOptimizeMolecule(mol_h, confId=conf_id)
            except Exception:
                pass
        n_heavy = mol.GetNumAtoms()
        sasa_sum = np.zeros(n_heavy, dtype=np.float64)
        n_valid = 0
        for conf_id in conf_ids:
            try:
                radii = rdFreeSASA.classifyAtoms(mol_h)
                rdFreeSASA.CalcSASA(mol_h, radii, confIdx=conf_id)
                for i in range(n_heavy):
                    sasa_sum[i] += float(
                        mol_h.GetAtomWithIdx(i).GetPropsAsDict().get('SASA', 0.0))
                n_valid += 1
            except Exception:
                pass
        if n_valid == 0:
            return None
        mean_sasa = sasa_sum / n_valid
        bit_info = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, RADIUS, N_BITS, bitInfo=bit_info)
        fp = np.zeros(N_BITS, dtype=np.float32)
        for bit_id, incidents in bit_info.items():
            target_idx = bit_id % N_BITS
            for center_atom_idx, _ in incidents:
                atom = mol.GetAtomWithIdx(center_atom_idx)
                peoe = abs(float(atom.GetPropsAsDict().get('_GasteigerCharge', 0.0)))
                fp[target_idx] += mean_sasa[center_atom_idx] * peoe
    except Exception:
        return None
    if np.isnan(fp).any() or np.isinf(fp).any():
        return None
    return fp


# ══════════════════════════════════════════════════════════════════════
# Feature scaling (StandardScaler, matches training script)
# ══════════════════════════════════════════════════════════════════════

def scale_X_single(vec, X_scaler):
    """Scale a single descriptor vector [2048]."""
    return X_scaler.transform(vec.reshape(1, -1)).astype(np.float32)[0]


def scale_X_batch(matrix, X_scaler):
    """Scale a batch descriptor matrix [N x 2048]."""
    return X_scaler.transform(matrix).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# Batch pool scoring (KEY FIX — replaces 1000 individual GPR calls)
# ══════════════════════════════════════════════════════════════════════

def score_pool_batch(pool, gpr_model, scaler, desc_cache=None):
    """
    Score all molecules using cached descriptors + single batch GPR call.

    desc_cache: dict {smiles: descriptor_array} — avoids recomputing
                conformers for molecules already seen.

    Returns list of float scores, -999.0 for failures.
    """
    if desc_cache is None:
        desc_cache = {}

    scores    = [-999.0] * len(pool)
    descs     = []
    valid_idx = []

    for i, smi in enumerate(pool):
        if smi not in desc_cache:
            desc = get_sasa_peoe_descriptor(smi)
            if desc is not None:
                desc_cache[smi] = desc
        if smi in desc_cache:
            descs.append(desc_cache[smi])
            valid_idx.append(i)

    if not descs:
        return scores

    X_batch = scale_X_batch(np.array(descs), scaler['X_scaler'])
    mus_scaled = gpr_model.predict(X_batch)
    mus = mus_scaled * scaler['std'] + scaler['mean']

    for idx, mu in zip(valid_idx, mus):
        scores[idx] = float(mu)

    return scores


# ══════════════════════════════════════════════════════════════════════
# Phase 4: Spatial Fitness
# ══════════════════════════════════════════════════════════════════════

def get_3d_mol(smiles):
    """Generate 3D conformer for spatial fitness evaluation."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=50) < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        return None
    return mol

def calculate_spatial_fitness(mol_3d, base_score, pocket_map):
    """
    S_final = S_GPR - Penalty(Volume) - Penalty(Steric) + Bonus(H-bond)
    """
    vol = AllChem.ComputeMolVolume(mol_3d)
    pen_vol = VOL_PENALTY * (vol - pocket_map['v_max']) ** 2 \
              if vol > pocket_map['v_max'] else 0.0

    positions  = mol_3d.GetConformer().GetPositions()
    pen_steric = 0.0
    for center, radius in pocket_map['exclusion_spheres']:
        c = np.asarray(center, dtype=np.float64)
        for pos in positions:
            p = np.asarray(pos, dtype=np.float64)
            if np.sqrt(np.sum((p - c)**2)) < radius:
                pen_steric += STERIC_PENALTY

    bonus_hbond = 0.0
    for hotspot in pocket_map['hotspots']:
        h = np.asarray(hotspot, dtype=np.float64)
        for i, pos in enumerate(positions):
            if mol_3d.GetAtomWithIdx(i).GetAtomicNum() in [7, 8]:
                p = np.asarray(pos, dtype=np.float64)
                if np.sqrt(np.sum((p - h)**2)) < HBOND_DIST:
                    bonus_hbond += 1.0

    return base_score - pen_vol - pen_steric + bonus_hbond


# ══════════════════════════════════════════════════════════════════════
# Offspring Evaluation
# ══════════════════════════════════════════════════════════════════════

def evaluate_offspring(smiles, gpr_model, scaler, pocket_map, desc_cache=None):
    """
    Returns (S_final, S_GPR) or (-999.0, -999.0) on rejection.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -999.0, -999.0

    # 1. Reactive filter
    if has_reactive_groups(mol, smiles):
        return -999.0, -999.0

    # 2. SASA×PEOE descriptor (with cache)
    if desc_cache is not None and smiles in desc_cache:
        desc = desc_cache[smiles]
    else:
        desc = get_sasa_peoe_descriptor(smiles)
        if desc is None:
            return -999.0, -999.0
        if desc_cache is not None:
            desc_cache[smiles] = desc

    # 3. GPR prediction (single molecule — acceptable here)
    try:
        desc_scaled = scale_X_single(desc, scaler['X_scaler'])
        mu = float(gpr_model.predict(desc_scaled.reshape(1, -1))[0]) \
             * scaler['std'] + scaler['mean']
    except Exception:
        return -999.0, -999.0

    # 4. 3D conformer for spatial fitness
    mol_3d = get_3d_mol(smiles)
    if mol_3d is None:
        return -999.0, -999.0

    # 5. Hard volume culling
    vol = AllChem.ComputeMolVolume(mol_3d)
    if vol > pocket_map['v_max'] * 1.2:
        return -999.0, -999.0

    # 6. Hard steric culling
    positions = mol_3d.GetConformer().GetPositions()
    steric_clashes = 0
    for center, radius in pocket_map['exclusion_spheres']:
        c = np.asarray(center, dtype=np.float64)
        for pos in positions:
            p = np.asarray(pos, dtype=np.float64)
            if np.sqrt(np.sum((p - c)**2)) < radius:
                steric_clashes += 1
    if steric_clashes > 3:
        return -999.0, -999.0

    s_final = calculate_spatial_fitness(mol_3d, mu, pocket_map)
    return s_final, mu


# ══════════════════════════════════════════════════════════════════════
# Main GA Loop
# ══════════════════════════════════════════════════════════════════════

def GB_GA(gpr_model, scaler, GenPool, istep, pocket_map,
          rxn_list, results_dir, desc_cache=None):
    if desc_cache is None:
        desc_cache = {}

    # Pool scoring — BATCH (key fix)
    scores = score_pool_batch(GenPool, gpr_model, scaler, desc_cache)

    # Selection — Gaussian noise around TARGET_VALUE (R2GB-GA)
    sort_a = []
    for s in scores:
        gau_target = TARGET_VALUE + np.random.normal(0, GAU_SIGMA)
        sort_a.append(abs(gau_target - s))

    sort_a_sorted = sorted(sort_a)
    cut = sort_a_sorted[int(len(sort_a_sorted) * 0.2)] \
          if sort_a_sorted else 0.0

    x_parents1 = []
    x_parents2 = []
    mol_list   = []
    over_cut   = f"High-score molecules (cutoff {cut:.5f}):\n"
    less_cut   = f"\nLow-score molecules (cutoff {cut:.5f}):\n"

    if istep == 0:
        x_parents1 = list(GenPool)
        mol_list   = [s for s in scores if s > -999.0]
    else:
        for i, smi in enumerate(GenPool):
            if sort_a[i] < cut:
                x_parents1.append(smi)
                over_cut += f"  mol {i:5d}: {smi}  score={scores[i]:.5f}\n"
                mol_list.append(scores[i])
            else:
                less_cut += f"  mol {i:5d}: {smi}  score={scores[i]:.5f}\n"
                if random.random() >= 0.8:
                    x_parents1.append(smi)

        step_dir = os.path.join(results_dir, 'steps')
        with open(os.path.join(step_dir, f'INDEX_{istep}.dat'), 'w') as f:
            f.write(over_cut)
            f.write(less_cut)

    mol_arr  = np.array(mol_list) if mol_list else np.array([0.0])
    mean_val = float(mol_arr.mean())
    std_val  = float(mol_arr.std()) if len(mol_list) > 1 else 0.0

    pool_fps = {}
    for smi in x_parents1:
        m = Chem.MolFromSmiles(smi)
        if m:
            pool_fps[smi] = AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048)

    ncross = 0
    nmut   = 0
    need   = max(0, TARGET_POOL - len(x_parents1))
    offspring_scored = []

    while ncross < need:
        smi1 = random.choice(x_parents1)
        smi2 = random.choice(x_parents1)
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            continue

        try:
            child = co.crossover(mol1, mol2)
        except Exception:
            continue
        if child is None:
            continue

        l_mut = False
        if random.random() < MU_PROB:
            l_mut = True
            mutate_candidates = []
            for rxn_smarts in rxn_list:
                try:
                    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
                    products = rxn.RunReactants((child,))
                    for mols in products:
                        mutate_candidates.append(mols[0])
                except Exception:
                    continue
            if not mutate_candidates:
                continue
            child = np.random.choice(mutate_candidates)

        if child is None:
            continue
        child_smi = Chem.MolToSmiles(child)
        if not child_smi:
            continue
        child = Chem.MolFromSmiles(child_smi, sanitize=True)
        if child is None:
            continue

        if child.GetNumAtoms() > MAX_ATOMS:
            continue
        if Descriptors.MolWt(child) > MAX_MW:
            continue

        cano_child   = Chem.MolToSmiles(child, True)
        child_fp_rdk = AllChem.GetMorganFingerprintAsBitVect(child, 3, 2048)

        # Evaluate offspring
        s_final, s_gpr = evaluate_offspring(
            cano_child, gpr_model, scaler, pocket_map, desc_cache)
        if s_final <= -999.0:
            continue

        # Diversity — fixed threshold
        too_similar = any(
            DataStructs.TanimotoSimilarity(child_fp_rdk, fp) > SIM_THRESHOLD
            for fp in pool_fps.values()
        )
        if too_similar:
            continue

        ncross += 1
        if l_mut:
            nmut += 1
        x_parents2.append(cano_child)
        offspring_scored.append((s_final, s_gpr, cano_child))

    # Deduplicate
    seen, GenPool_new = set(), []
    for smi in x_parents1 + x_parents2:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        canon = Chem.MolToSmiles(m, True)
        if canon not in seen:
            seen.add(canon)
            GenPool_new.append(canon)

    # Save top molecules this step
    if offspring_scored:
        offspring_scored.sort(key=lambda x: x[0], reverse=True)
        step_csv = os.path.join(results_dir, 'steps',
                                f'top_mols_step_{istep+1}.csv')
        pd.DataFrame(offspring_scored[:TOP_PER_STEP],
                     columns=['S_final', 'S_GPR', 'SMILES']).to_csv(
            step_csv, index=False)

    print(
        f"{istep+1:6d}  {cut:10.5f} / {mean_val:8.5f} / {std_val:8.5f}"
        f"  {len(GenPool_new):6d} ({len(x_parents1):6d} +"
        f" {len(x_parents2):6d})"
        f"  {ncross:6d}  {nmut:6d}",
        flush=True
    )

    return GenPool_new


# ══════════════════════════════════════════════════════════════════════
# Post-run: Aggregate top molecules
# ══════════════════════════════════════════════════════════════════════

def collect_top_molecules(results_dir, top_n=200):
    step_dir = os.path.join(results_dir, 'steps')
    all_rows = []
    for step in range(1, NSTEP + 1):
        csv_path = os.path.join(step_dir, f'top_mols_step_{step}.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['step'] = step
            all_rows.append(df)
    if not all_rows:
        print("  No step CSVs found.")
        return
    combined = pd.concat(all_rows, ignore_index=True)
    combined['canon'] = combined['SMILES'].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), True)
        if Chem.MolFromSmiles(s) else None)
    combined = combined.dropna(subset=['canon'])
    combined = combined.sort_values('S_final', ascending=False)
    combined = combined.drop_duplicates(subset='canon', keep='first')
    top = combined.head(top_n)[['SMILES', 'S_final', 'S_GPR', 'step']]
    out_path = os.path.join(results_dir, 'all_top_molecules.csv')
    top.to_csv(out_path, index=False)
    print(f"\n  Top molecules saved: {out_path}")
    print(f"  Unique candidates:   {len(combined)}")
    print(f"  Best S_final:        {top['S_final'].max():.3f}")
    print(f"  Mean S_final:        {top['S_final'].mean():.3f}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    print("=" * 65)
    print("  2.5D SPACE-GA")
    print("  Descriptor: SASA×PEOE Weighted Morgan (2048 bits, r=2)")
    print("  Weight: mean_SASA(atom) × |PEOE(atom)|")
    print("  Tanimoto kernel — fingerprint-appropriate GPR")
    print("=" * 65)

    step_dir = os.path.join(RESULTS_DIR, 'steps')
    if os.path.exists(step_dir):
        old_dir = step_dir + '_old'
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir)
        shutil.move(step_dir, old_dir)
    os.makedirs(step_dir, exist_ok=True)

    print("\n[Step 1/5] Loading pocket map...")
    with open(POCKET_PATH, 'rb') as f:
        pocket_map = pickle.load(f)
    print(f"  V_max:             {pocket_map['v_max']:.2f} Angstrom^3")
    print(f"  Hotspots:          {len(pocket_map['hotspots'])}")
    print(f"  Exclusion spheres: {len(pocket_map['exclusion_spheres'])}")

    print("\n[Step 2/5] Loading GP model (VSA descriptor)...")
    with open(MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
    gpr_model = model_data['gpr']
    scaler    = {'mean':     model_data['scaler_mean'],
                 'std':      model_data['scaler_std'],
                 'X_scaler': model_data['X_scaler']}
    print(f"  Target:      {model_data['target']}")
    print(f"  Train size:  {model_data['n_train']}")
    print(f"  Test R2:     {model_data['r2_test']:.4f}")
    print(f"  Descriptor:  SASA×PEOE Weighted Morgan ({N_BITS} bits)")

    print("\n[Step 3/5] Loading initial pool...")
    GenPool = pd.read_csv(INITPOOL_PATH)['SMILES'].tolist()
    print(f"  Pool size: {len(GenPool)} BACE-1 inhibitors")
    with open(MUTATION_PATH, 'r') as f:
        rxn_list = [line.rstrip() for line in f if line.strip()]
    print(f"  Mutation reactions: {len(rxn_list)}")

    if len(GenPool) > TARGET_POOL:
        print(f"  Trimming pool to {TARGET_POOL}...")
        # Cache pool descriptors to avoid recomputation on subsequent runs
        if os.path.exists(POOL_DESC_PATH) and os.path.exists(POOL_SMILES_PATH):
            print(f"  Loading cached pool descriptors...")
            pool_descs = np.load(POOL_DESC_PATH)
            pool_smiles = list(np.load(POOL_SMILES_PATH))
            # Score using cached descriptors
            init_scores = []
            valid_pool = []
            for smi, desc in zip(pool_smiles, pool_descs):
                try:
                    desc_scaled = scale_X_single(desc, scaler['X_scaler'])
                    mu = float(gpr_model.predict(desc_scaled.reshape(1,-1))[0]) * scaler['std'] + scaler['mean']
                    init_scores.append(mu)
                    valid_pool.append(smi)
                except:
                    init_scores.append(-999.0)
                    valid_pool.append(smi)
            GenPool = valid_pool
        else:
            print(f"  Computing pool descriptors (first run, will cache)...")
            pool_descs, pool_smiles_valid, init_scores = [], [], []
            for smi in tqdm(GenPool, desc="  Pool descriptors"):
                desc = get_sasa_peoe_descriptor(smi)
                if desc is not None:
                    try:
                        desc_scaled = scale_X_single(desc, scaler['X_scaler'])
                        mu = float(gpr_model.predict(desc_scaled.reshape(1,-1))[0]) * scaler['std'] + scaler['mean']
                        pool_descs.append(desc)
                        pool_smiles_valid.append(smi)
                        init_scores.append(mu)
                    except:
                        pool_descs.append(desc)
                        pool_smiles_valid.append(smi)
                        init_scores.append(-999.0)
            np.save(POOL_DESC_PATH, np.array(pool_descs))
            np.save(POOL_SMILES_PATH, np.array(pool_smiles_valid))
            GenPool = pool_smiles_valid
            print(f"  Cached pool descriptors: {len(pool_descs)} molecules")
        sorted_pool = [s for _, s in sorted(zip(init_scores, GenPool), reverse=True)]
        GenPool = sorted_pool[:TARGET_POOL]
        print(f"  Pool trimmed to {len(GenPool)}")

    co.average_size = CO_AVERAGE_SIZE
    co.size_stdev   = CO_SIZE_STDEV
    co.string_type  = 'SMILES'

    print(f"\n[Step 4/5] Running GA ({NSTEP} steps)...")
    print(f"  Descriptor:       SASA×PEOE Weighted Morgan")
    print(f"  Sim threshold:    {SIM_THRESHOLD} FIXED")
    print(f"  Reactive filter:  ON (max {MAX_ALKYNES} alkynes)")
    print(f"  Vol penalty:      quadratic {VOL_PENALTY} x (excess)^2")
    print(f"  H-bond dist:      {HBOND_DIST} Angstrom")
    print(f"  Top per step:     {TOP_PER_STEP} saved")
    print()
    print(f"{'#step':>6}  {'cut/mean/std':>30}  "
          f"{'pool(sel+off)':>20}  {'ncross':>6}  {'nmut':>6}")

    # Pre-populate descriptor cache from saved pool descriptors
    desc_cache = {}
    if os.path.exists(POOL_DESC_PATH) and os.path.exists(POOL_SMILES_PATH):
        pool_descs_cached = np.load(POOL_DESC_PATH)
        pool_smiles_cached = list(np.load(POOL_SMILES_PATH))
        for smi, desc in zip(pool_smiles_cached, pool_descs_cached):
            desc_cache[smi] = desc
        print(f'  Pre-loaded {len(desc_cache)} descriptors from cache')
    n0_time = start_time
    for istep in range(NSTEP):
        GenPool = GB_GA(
            gpr_model, scaler, GenPool, istep,
            pocket_map, rxn_list, RESULTS_DIR, desc_cache
        )
        n1_time = time.time()
        print(f"  Step {istep+1} wall time: {n1_time - n0_time:.1f}s",
              flush=True)
        n0_time = n1_time

    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.1f}s")

    print("\n[Step 5/5] Collecting top molecules across all steps...")
    collect_top_molecules(RESULTS_DIR, top_n=1000)

    print(f"\nResults: {RESULTS_DIR}/")
    print("=" * 65)
    print("\n  Next: python3 scripts/validate_molecules.py"
          " --input results/all_top_molecules.csv")


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    main()
