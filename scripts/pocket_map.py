"""
pocket_map.py
==================
PAMR-GA: Phase 1 — Pocket Inverse-Engineering
=====================================================


Steps:
    1. Load dataset
    2. Find activity cliff pairs (Tc > 0.7, ΔpIC50 > 2.0)
    3. For each pair: align inactive to active in 3D
    4. Find atoms in inactive that protrude beyond active volume
    5. Cluster protrusion positions → exclusion spheres
    6. Extract V_max, hotspots (iterative pharmacophore)
    7. Save pocket map

Usage:
    python3 scripts/pocket_map.py --force-regenerate
    sbatch run_pocket_map.sh

Output:
    data/pocket_map.pkl
    data/reference_mol.pkl
    data/active_conformers.pkl
    data/pharmacophore_features.pkl
    data/activity_cliff_pairs.pkl

Reference:
    PAMR-GA idea note:
    "By analyzing Activity Cliff data, the protruding volumes
    of inactive analogs are defined as steric walls."
"""

import numpy as np
import pandas as pd
import pickle
import os
import argparse
from tqdm import tqdm
from rdkit import Chem, RDLogger, RDConfig
from rdkit.Chem import AllChem, rdMolAlign, ChemicalFeatures, DataStructs
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

RDLogger.DisableLog('rdApp.*')

# ── Argument Parser ────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description='Phase 1: Proper Activity Cliff Analysis'
    )
    parser.add_argument('--data', type=str,
        default='data/bace1_dataset.csv')
    parser.add_argument('--data-dir', type=str,
        default='data')
    parser.add_argument('--threshold', type=float, default=8.0,
        help='Active threshold pIC50 >= threshold')
    parser.add_argument('--vmax-percentile', type=float, default=95.0)
    parser.add_argument('--hotspot-consensus', type=float, default=0.3,
        help='Min fraction of actives sharing a pharmacophore feature for it to be kept as a hotspot')
    parser.add_argument('--cliff-tanimoto', type=float, default=0.7,
        help='Tanimoto threshold for activity cliff pairs (default: 0.7)')
    parser.add_argument('--cliff-delta', type=float, default=2.0,
        help='pIC50 difference for activity cliff (default: 2.0)')
    parser.add_argument('--cluster-radius', type=float, default=2.0)
    parser.add_argument('--exclusion-radius', type=float, default=1.5)
    parser.add_argument('--max-actives', type=int, default=500)
    parser.add_argument('--max-cliff-pairs', type=int, default=500,
        help='Keep this many strongest cliff pairs after ranking by delta-pIC50')
    parser.add_argument('--n-iterations', type=int, default=5)
    parser.add_argument('--force-regenerate', action='store_true')
    return parser.parse_args()

# ── Helper: Generate 3D conformer ─────────────────────────────────────
def get_3d_mol(smiles, seed=42):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=seed, maxAttempts=10) < 0:
            return None
    except Exception:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    return mol

# ── Extract pharmacophore features ────────────────────────────────────
def extract_features(mol, factory, key_families={'Donor', 'Acceptor', 'Aromatic'}):
    features = {}
    try:
        feats = factory.GetFeaturesForMol(mol)
        for f in feats:
            family = f.GetFamily()
            if family in key_families:
                pos = f.GetPos()
                coords = np.array([pos.x, pos.y, pos.z])
                if family not in features:
                    features[family] = []
                features[family].append(coords)
    except Exception:
        pass
    return features

# ── Compute consensus centroids ────────────────────────────────────────
def compute_consensus_centroids(all_features, key_families, threshold=0.5):
    n_mols = len(all_features)
    centroids = {}
    for family in key_families:
        all_positions, mol_membership = [], []
        for i, mol_feat in enumerate(all_features):
            for pos in mol_feat.get(family, []):
                all_positions.append(pos)
                mol_membership.append(i)
        if len(all_positions) < 3:
            continue
        all_positions  = np.array(all_positions)
        mol_membership = np.array(mol_membership)
        try:
            distances = pdist(all_positions)
            Z      = linkage(distances, method='average')
            labels = fcluster(Z, t=2.0, criterion='distance')
            best_centroid, best_consensus = None, 0
            for cluster_id in np.unique(labels):
                mask      = labels == cluster_id
                n_unique  = len(np.unique(mol_membership[mask]))
                consensus = n_unique / n_mols
                if consensus > best_consensus:
                    best_consensus = consensus
                    best_centroid  = all_positions[mask].mean(axis=0)
            if best_consensus >= threshold and best_centroid is not None:
                centroids[family] = best_centroid
        except Exception:
            pass
    return centroids

# ═════════════════════════════════════════════════════════════════════
# NEW: Activity Cliff Analysis (from idea note)
# ═════════════════════════════════════════════════════════════════════

def find_activity_cliff_pairs(df, tc_threshold=0.7, delta_threshold=2.0,
                               max_pairs=500):
    """
    Find activity cliff pairs as specified in idea note:
    - Tanimoto similarity > tc_threshold (default 0.7)
    - |ΔpIC50| > delta_threshold (default 2.0)

    Returns list of (active_smiles, inactive_smiles, tc, delta_pic50)
    """
    print(f"\n  Finding activity cliff pairs...")
    print(f"  Tanimoto threshold: Tc > {tc_threshold}")
    print(f"  pIC50 difference:   ΔpIC50 > {delta_threshold}")

    # Compute Morgan fingerprints for all molecules
    fps   = []
    smiles_list = []
    pics  = []

    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="  Computing FPs", unit="mol"):
        mol = Chem.MolFromSmiles(row['canonical_smiles'])
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
            fps.append(fp)
            smiles_list.append(row['canonical_smiles'])
            pics.append(row['pIC50'])

    print(f"  Valid molecules: {len(fps)}")

    # Find cliff pairs
    cliff_pairs = []
    n = len(fps)

    # EXHAUSTIVE scan, then rank, then truncate. Previously the loops broke at
    # max_pairs and sorted afterwards, so the pairs kept were the first found
    # in dataset order rather than the strongest cliffs (see docs/FIXES.md).
    # The inner loop is vectorised because a full scan is ~32M pairs for BACE-1.
    pics_arr = np.asarray(pics, dtype=np.float64)

    for i in tqdm(range(n - 1), desc="  Finding cliffs", unit="mol"):
        tcs    = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fps[i], fps[i+1:]),
            dtype=np.float64)
        deltas = np.abs(pics_arr[i] - pics_arr[i+1:])
        hits   = np.nonzero((tcs >= tc_threshold) &
                            (deltas >= delta_threshold))[0]

        for h in hits:
            j     = i + 1 + int(h)
            tc    = float(tcs[h])
            delta = float(deltas[h])
            # active = higher pIC50, inactive = lower pIC50
            if pics[i] > pics[j]:
                cliff_pairs.append((smiles_list[i], smiles_list[j],
                                    pics[i], pics[j], tc, delta))
            else:
                cliff_pairs.append((smiles_list[j], smiles_list[i],
                                    pics[j], pics[i], tc, delta))

    n_qualifying = len(cliff_pairs)
    cliff_pairs.sort(key=lambda x: x[5], reverse=True)  # sort by delta
    cliff_pairs = cliff_pairs[:max_pairs]

    print(f"  Qualifying pairs in dataset: {n_qualifying}")
    print(f"  Retained (top {max_pairs} by ΔpIC50): {len(cliff_pairs)}")
    if cliff_pairs:
        print(f"  Best cliff:  Tc={cliff_pairs[0][4]:.3f}, "
              f"ΔpIC50={cliff_pairs[0][5]:.2f}")
        print(f"  Worst kept:  Tc={cliff_pairs[-1][4]:.3f}, "
              f"ΔpIC50={cliff_pairs[-1][5]:.2f}")

    return cliff_pairs


def extract_protrusion_positions(active_smi, inactive_smi, ref_mol,
                                  protrusion_cutoff=1.5):
    """
    From idea note:
    "For each activity cliff pair, the inactive molecule's atoms
    that protrude beyond the active molecule's volume define
    exclusion spheres."

    Method:
        1. Generate 3D conformers for both molecules
        2. Align both to reference
        3. For each inactive atom: find minimum distance to any active atom
        4. If distance > protrusion_cutoff → this atom protrudes
        5. Return protrusion positions
    """
    active_mol   = get_3d_mol(active_smi)
    inactive_mol = get_3d_mol(inactive_smi)

    if active_mol is None or inactive_mol is None:
        return []

    # Align both to reference
    for mol in [active_mol, inactive_mol]:
        try:
            o3a = rdMolAlign.GetO3A(mol, ref_mol)
            if o3a:
                o3a.Align()
        except Exception:
            pass

    # Get heavy atom positions
    active_conf = active_mol.GetConformer()
    inactive_conf = inactive_mol.GetConformer()

    active_pos = []
    for i in range(active_mol.GetNumAtoms()):
        atom = active_mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() > 1:  # heavy atoms only
            pos = active_conf.GetAtomPosition(i)
            active_pos.append(np.array([pos.x, pos.y, pos.z]))

    if not active_pos:
        return []

    active_pos = np.array(active_pos)

    # Find inactive atoms that protrude beyond active molecule
    protrusions = []
    for i in range(inactive_mol.GetNumAtoms()):
        atom = inactive_mol.GetAtomWithIdx(i)
        if atom.GetAtomicNum() <= 1:  # skip hydrogen
            continue
        pos = inactive_conf.GetAtomPosition(i)
        inactive_atom_pos = np.array([pos.x, pos.y, pos.z])

        # Minimum distance to any active atom
        dists = np.linalg.norm(active_pos - inactive_atom_pos, axis=1)
        min_dist = dists.min()

        # If this atom is far from all active atoms → protrusion
        if min_dist > protrusion_cutoff:
            protrusions.append(inactive_atom_pos)

    return protrusions


def build_exclusion_spheres_from_cliffs(cliff_pairs, ref_mol,
                                         cluster_radius=2.0,
                                         exclusion_radius=1.5,
                                         n_spheres=5):
    """
    Build exclusion spheres from activity cliff protrusions.
    Following idea note exactly.
    """
    print(f"\n  Extracting protrusion positions from {len(cliff_pairs)} cliff pairs...")

    all_protrusions = []
    for active_smi, inactive_smi, active_pic50, inactive_pic50, tc, delta \
            in tqdm(cliff_pairs, desc="  Processing cliffs", unit="pair"):
        protrusions = extract_protrusion_positions(
            active_smi, inactive_smi, ref_mol)
        all_protrusions.extend(protrusions)

    print(f"  Total protrusion positions: {len(all_protrusions)}")

    if len(all_protrusions) < 3:
        print("  WARNING: Not enough protrusions found")
        return []

    all_protrusions = np.array(all_protrusions)

    # Cluster protrusion positions
    try:
        distances = pdist(all_protrusions)
        Z      = linkage(distances, method='average')
        labels = fcluster(Z, t=cluster_radius, criterion='distance')
        unique, counts = np.unique(labels, return_counts=True)

        # Take top n_spheres largest clusters
        top_clusters = unique[np.argsort(counts)[::-1][:n_spheres]]
        exclusion_spheres = []
        for cid in top_clusters:
            centroid = all_protrusions[labels==cid].mean(axis=0)
            exclusion_spheres.append((centroid, exclusion_radius))
            print(f"  Exclusion sphere: [{centroid[0]:.2f}, "
                  f"{centroid[1]:.2f}, {centroid[2]:.2f}] "
                  f"(n={counts[unique==cid][0]} protrusions)")

        return exclusion_spheres
    except Exception as e:
        print(f"  WARNING: Clustering failed: {e}")
        return []

# ── Main ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.data_dir, exist_ok=True)

    fdef_file = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory   = ChemicalFeatures.BuildFeatureFactory(fdef_file)
    key_families = {'Donor', 'Acceptor', 'Aromatic'}

    print("=" * 60)
    print("  PAMR-GA: Phase 1 — Pocket Inverse-Engineering")
    print("  Method: True Activity Cliff Analysis (from idea note)")
    print("=" * 60)
    print(f"  Activity cliff: Tc > {args.cliff_tanimoto}, "
          f"ΔpIC50 > {args.cliff_delta}")
    print(f"  Iterations: {args.n_iterations}")
    print(f"  Max actives: {args.max_actives}")

    # ── Step 1: Load data ──────────────────────────────────────────────
    print("\n[Step 1/8] Loading dataset...")
    df     = pd.read_csv(args.data)
    active = df[df['pIC50'] >= args.threshold].sort_values(
        'pIC50', ascending=False).head(args.max_actives)
    print(f"  Total molecules: {len(df)}")
    print(f"  Active (pIC50 >= {args.threshold}): {len(active)}")

    # ── Step 2: Reference molecule ─────────────────────────────────────
    print("\n[Step 2/8] Getting reference molecule...")
    ref_mol, ref_smi, ref_pic50 = None, None, None
    for _, row in active.iterrows():
        mol = get_3d_mol(row['canonical_smiles'])
        if mol is not None:
            ref_mol   = mol
            ref_smi   = row['canonical_smiles']
            ref_pic50 = row['pIC50']
            break

    print(f"  Reference: {ref_smi[:60]}...")
    print(f"  pIC50: {ref_pic50:.2f}")

    ref_path = os.path.join(args.data_dir, 'reference_mol.pkl')
    with open(ref_path, 'wb') as f:
        pickle.dump({'mol': ref_mol, 'smiles': ref_smi, 'pIC50': ref_pic50}, f)

    # ── Step 3: Generate conformers ────────────────────────────────────
    print("\n[Step 3/8] Generating 3D conformers...")
    mols, smiles_list, picic50s = [], [], []
    for _, row in tqdm(active.iterrows(), total=len(active),
                       desc="  Conformers", unit="mol"):
        mol = get_3d_mol(row['canonical_smiles'])
        if mol is None:
            continue
        try:
            o3a = rdMolAlign.GetO3A(mol, ref_mol)
            if o3a:
                o3a.Align()
        except Exception:
            pass
        mols.append(mol)
        smiles_list.append(row['canonical_smiles'])
        picic50s.append(row['pIC50'])
    print(f"  Generated: {len(mols)} conformers")

    # ── Step 4: Iterative pharmacophore alignment ──────────────────────
    print(f"\n[Step 4/8] Iterative pharmacophore alignment ({args.n_iterations} iter)...")
    for iteration in range(args.n_iterations):
        print(f"  Iteration {iteration+1}/{args.n_iterations}...")
        all_features = [extract_features(mol, factory, key_families)
                        for mol in mols]
        consensus_centroids = compute_consensus_centroids(
            all_features, key_families, args.hotspot_consensus)
        print(f"  Consensus features: {list(consensus_centroids.keys())}")
        mols_new = []
        for mol in mols:
            try:
                o3a = rdMolAlign.GetO3A(mol, ref_mol)
                if o3a:
                    o3a.Align()
            except Exception:
                pass
            mols_new.append(mol)
        mols = mols_new

    # ── Step 5: Extract hotspots ───────────────────────────────────────
    print("\n[Step 5/8] Extracting consensus hotspots...")
    all_features = []
    for mol, smi in zip(mols, smiles_list):
        feats = extract_features(mol, factory, key_families)
        all_features.append({'smiles': smi, 'features': feats})

    hotspots = []
    n_mols   = len(mols)
    for family in ['Donor', 'Acceptor']:
        all_positions, mol_membership = [], []
        for i, feat_dict in enumerate(all_features):
            for pos in feat_dict['features'].get(family, []):
                all_positions.append(pos)
                mol_membership.append(i)
        if len(all_positions) < 3:
            continue
        all_positions  = np.array(all_positions)
        mol_membership = np.array(mol_membership)
        try:
            distances = pdist(all_positions)
            Z      = linkage(distances, method='average')
            labels = fcluster(Z, t=args.cluster_radius, criterion='distance')
            for cluster_id in np.unique(labels):
                mask      = labels == cluster_id
                n_unique  = len(np.unique(mol_membership[mask]))
                consensus = n_unique / n_mols
                if consensus >= args.hotspot_consensus:
                    centroid = all_positions[mask].mean(axis=0)
                    hotspots.append({
                        'coords':    centroid,
                        'family':    family,
                        'consensus': consensus,
                        'n_mols':    n_unique,
                    })
        except Exception as e:
            print(f"  Warning: {e}")

    hotspots.sort(key=lambda x: x['consensus'], reverse=True)
    print(f"  Found {len(hotspots)} hotspots:")
    for i, hs in enumerate(hotspots):
        print(f"    [{i+1}] {hs['family']:10s} consensus={hs['consensus']*100:.1f}%")

    # ── Step 6: V_max ──────────────────────────────────────────────────
    print(f"\n[Step 6/8] Estimating V_max...")
    volumes = []
    for mol in mols:
        try:
            volumes.append(AllChem.ComputeMolVolume(mol))
        except Exception:
            pass
    v_max = float(np.percentile(volumes, args.vmax_percentile))
    print(f"  V_max = {v_max:.2f} Angstrom^3 ({args.vmax_percentile}th percentile)")

    # ── Step 7: Activity cliff exclusion spheres ──────────────
    print(f"\n[Step 7/8] Activity cliff analysis (from idea note)...")

    # Use full dataset for cliff finding (not just actives)
    cliff_pairs = find_activity_cliff_pairs(
        df,
        tc_threshold=args.cliff_tanimoto,
        delta_threshold=args.cliff_delta,
        max_pairs=args.max_cliff_pairs)

    # Save cliff pairs
    cliff_path = os.path.join(args.data_dir, 'activity_cliff_pairs.pkl')
    with open(cliff_path, 'wb') as f:
        pickle.dump(cliff_pairs, f)
    print(f"  Saved: {cliff_path}")

    # Build exclusion spheres from protrusions
    exclusion_spheres = build_exclusion_spheres_from_cliffs(
        cliff_pairs, ref_mol,
        cluster_radius=args.cluster_radius,
        exclusion_radius=args.exclusion_radius,
        n_spheres=5)

    # Fallback if not enough cliff pairs found
    if len(exclusion_spheres) < 3:
        print(f"  WARNING: Only {len(exclusion_spheres)} spheres from cliffs")
        print(f"  Falling back to inactive molecule approach for remaining spheres")
        inactive_sample = df[df['pIC50'] <= 6.0].head(100)
        inactive_positions = []
        for _, row in inactive_sample.iterrows():
            mol = get_3d_mol(row['canonical_smiles'])
            if mol is None:
                continue
            try:
                o3a = rdMolAlign.GetO3A(mol, ref_mol)
                if o3a:
                    o3a.Align()
                conf = mol.GetConformer()
                for i in range(mol.GetNumAtoms()):
                    if mol.GetAtomWithIdx(i).GetAtomicNum() > 1:
                        pos = conf.GetAtomPosition(i)
                        inactive_positions.append(
                            np.array([pos.x, pos.y, pos.z]))
            except Exception:
                pass
        if len(inactive_positions) >= 3:
            inactive_positions = np.array(inactive_positions)
            try:
                distances = pdist(inactive_positions)
                Z      = linkage(distances, method='average')
                labels = fcluster(Z, t=args.exclusion_radius,
                                  criterion='distance')
                unique, counts = np.unique(labels, return_counts=True)
                n_needed = 5 - len(exclusion_spheres)
                top = unique[np.argsort(counts)[::-1][:n_needed]]
                for cid in top:
                    centroid = inactive_positions[labels==cid].mean(axis=0)
                    exclusion_spheres.append((centroid, args.exclusion_radius))
            except Exception:
                pass

    print(f"  Final exclusion spheres: {len(exclusion_spheres)}")

    # ── Step 8: Save ───────────────────────────────────────────────────
    print(f"\n[Step 8/8] Saving outputs...")

    conf_path = os.path.join(args.data_dir, 'active_conformers.pkl')
    with open(conf_path, 'wb') as f:
        pickle.dump({'mols': mols, 'smiles': smiles_list,
                     'picic50s': picic50s}, f)

    feat_path = os.path.join(args.data_dir, 'pharmacophore_features.pkl')
    with open(feat_path, 'wb') as f:
        pickle.dump(all_features, f)

    pocket_map = {
        'v_max':               v_max,
        'hotspots':            [hs['coords'] for hs in hotspots],
        'hotspot_details':     hotspots,
        'exclusion_spheres':   exclusion_spheres,
        'ref_smiles':          ref_smi,
        'n_actives_used':      len(mols),
        'active_threshold':    args.threshold,
        'vmax_percentile':     args.vmax_percentile,
        'consensus_threshold': args.hotspot_consensus,
        'n_iterations':        args.n_iterations,
        'cliff_tanimoto':      args.cliff_tanimoto,
        'cliff_delta':         args.cliff_delta,
        'n_cliff_pairs':       len(cliff_pairs),
        'method':              'activity_cliff',
    }

    pocket_path = os.path.join(args.data_dir, 'pocket_map.pkl')
    with open(pocket_path, 'wb') as f:
        pickle.dump(pocket_map, f)

    print("\n" + "=" * 60)
    print("  POCKET MAP COMPLETE")
    print("=" * 60)
    print(f"  Method:          True activity cliff analysis")
    print(f"  Activity cliffs: {len(cliff_pairs)} pairs "
          f"(Tc>{args.cliff_tanimoto}, ΔpIC50>{args.cliff_delta})")
    print(f"  V_max:           {v_max:.2f} Angstrom^3")
    print(f"  Hotspots:        {len(hotspots)}")
    print(f"  Excl. spheres:   {len(exclusion_spheres)} (from cliffs)")
    print(f"  Active mols:     {len(mols)}")
    print(f"\n  Saved:")
    print(f"    {pocket_path}")
    print(f"    {conf_path}")
    print(f"    {cliff_path}")
    print("=" * 60)
    print("\n  Next: python3 scripts/train_gpr_peoe.py")

if __name__ == "__main__":
    main()
