#!/usr/bin/env python3
"""
validate_molecules.py
=====================
PAMR-GA Post-GA Molecule Validation
===============================================

Validates top molecules from GA output against drug-likeness criteria,
spatial fitness, and QED.

Pipeline:
    1. Load top molecules from results/all_top_molecules.csv
    2. Reactive group filter (same as GA)
    3. Lipinski Rule of Five filter (hard: <=1 violation)
    4. QED drug-likeness filter (hard: QED >= 0.3)
    5. Spatial fitness re-evaluation on a fresh 3D conformer
    6. Save validated candidates

Usage:
    python3 scripts/validate_molecules.py
    python3 scripts/validate_molecules.py --input results/all_top_molecules.csv
    python3 scripts/validate_molecules.py --top 822

Output:
    results/validated_molecules.csv
    results/validation_report.txt
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

# ── Paths ──────────────────────────────────────────────────────────────
POCKET_PATH = 'data/pocket_map.pkl'
DEFAULT_INPUT = 'results/all_top_molecules.csv'
OUTPUT_CSV  = 'results/validated_molecules.csv'
OUTPUT_TXT  = 'results/validation_report.txt'

# ── Thresholds ─────────────────────────────────────────────────────────
LIPINSKI_MW       = 500
LIPINSKI_LOGP     = 5
LIPINSKI_HBD      = 5
LIPINSKI_HBA      = 10
QED_THRESHOLD     = 0.3
HBOND_DIST        = 3.5
STERIC_PENALTY    = 1.0
VOL_PENALTY       = 0.05

# ── Reactive SMARTS (same as pamr_ga.py) ────────────────────────
_REACTIVE_SMARTS = [
    '[SH]', 'OF', '[C-]#[N+]', 'C=CO', 'O-O', '[N-]',
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


def lipinski_filter(mol):
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = Descriptors.NumHDonors(mol)
    hba  = Descriptors.NumHAcceptors(mol)
    violations = sum([
        mw   > LIPINSKI_MW,
        logp > LIPINSKI_LOGP,
        hbd  > LIPINSKI_HBD,
        hba  > LIPINSKI_HBA,
    ])
    return violations <= 1, violations, mw, logp, hbd, hba


def get_3d_mol(smiles):
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
    vol = AllChem.ComputeMolVolume(mol_3d)
    pen_vol = VOL_PENALTY * (vol - pocket_map['v_max']) ** 2 \
              if vol > pocket_map['v_max'] else 0.0
    positions = mol_3d.GetConformer().GetPositions()
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
    return base_score - pen_vol - pen_steric + bonus_hbond, vol, pen_vol, pen_steric, bonus_hbond


def parse_args():
    parser = argparse.ArgumentParser(description='Validate GA molecules')
    parser.add_argument('--input',  default=DEFAULT_INPUT,
                        help='Input CSV (default: results/all_top_molecules.csv)')
    parser.add_argument('--top',    type=int, default=822,
                        help='Number of top molecules to validate (default: 822)')
    parser.add_argument('--output', default=OUTPUT_CSV,
                        help='Output CSV path')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs('results', exist_ok=True)

    print("=" * 65)
    print("  PAMR-GA Molecule Validation")
    print("=" * 65)

    print("\n[Step 1/5] Loading pocket map...")
    with open(POCKET_PATH, 'rb') as f:
        pocket_map = pickle.load(f)
    print(f"  Pocket V_max: {pocket_map['v_max']:.2f} A^3")
    print(f"  Hotspots: {len(pocket_map['hotspots'])}")

    print(f"\n[Step 2/5] Loading GA results from {args.input}...")
    if not os.path.exists(args.input):
        print(f"  ERROR: {args.input} not found. Run GA first.")
        return
    df = pd.read_csv(args.input)
    df = df.head(args.top)
    print(f"  Loaded: {len(df)} molecules")
    print(f"  S_final range: {df['S_final'].min():.3f} — {df['S_final'].max():.3f}")

    print(f"\n[Step 3/5] Validating molecules...")
    results = []
    stats = {
        'total': len(df),
        'invalid_smiles': 0,
        'reactive': 0,
        'lipinski_fail': 0,
        'qed_fail': 0,
        'conformer_fail': 0,
        'passed': 0,
    }

    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Validating"):
        smiles   = row['SMILES']
        s_final  = row['S_final']
        s_gpr    = row['S_GPR']
        step     = row.get('step', -1)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            stats['invalid_smiles'] += 1
            continue

        canon = Chem.MolToSmiles(mol, True)

        if has_reactive_groups(mol, smiles):
            stats['reactive'] += 1
            continue

        lip_pass, lip_viol, mw, logp, hbd, hba = lipinski_filter(mol)
        if not lip_pass:
            stats['lipinski_fail'] += 1
            continue

        qed_score = QED.qed(mol)
        if qed_score < QED_THRESHOLD:
            stats['qed_fail'] += 1
            continue

        mol_3d = get_3d_mol(canon)
        if mol_3d is None:
            stats['conformer_fail'] += 1
            s_final_rescore = np.nan
            vol = np.nan
            pen_vol = np.nan
            pen_steric = np.nan
            bonus_hbond = np.nan
        else:
            s_final_rescore, vol, pen_vol, pen_steric, bonus_hbond = \
                calculate_spatial_fitness(mol_3d, s_gpr, pocket_map)

        stats['passed'] += 1
        results.append({
            'SMILES':          canon,
            'step':            step,
            'S_final_GA':      s_final,
            'S_GPR_GA':        s_gpr,
            'S_final_rescore': s_final_rescore,
            'MolWt':           mw,
            'MolLogP':         logp,
            'NumHDonors':      hbd,
            'NumHAcceptors':   hba,
            'NumRotBonds':     Descriptors.NumRotatableBonds(mol),
            'QED':             qed_score,
            'LipinskiViol':    lip_viol,
            'Volume_A3':       vol,
            'Pen_Vol':         pen_vol,
            'Pen_Steric':      pen_steric,
            'Bonus_Hbond':     bonus_hbond,
            'HeavyAtoms':      mol.GetNumHeavyAtoms(),
        })

    print(f"\n[Step 4/5] Saving results...")
    out_df = pd.DataFrame(results)
    if len(out_df) > 0:
        out_df = out_df.sort_values('S_final_GA', ascending=False)
    out_df.to_csv(args.output, index=False)
    print(f"  Saved: {args.output}")

    print(f"\n[Step 5/5] Computing summary statistics...")
    print(f"\n{'='*65}")
    print(f"  VALIDATION SUMMARY")
    print(f"{'='*65}")
    print(f"  Input molecules:      {stats['total']}")
    print(f"  Invalid SMILES:       {stats['invalid_smiles']}")
    print(f"  Reactive groups:      {stats['reactive']}")
    print(f"  Lipinski failures:    {stats['lipinski_fail']}")
    print(f"  QED failures:         {stats['qed_fail']}")
    print(f"  Conformer failures:   {stats['conformer_fail']}")
    print(f"  Passed all filters:   {stats['passed']} "
          f"({100*stats['passed']/stats['total']:.1f}%)")

    if len(out_df) > 0:
        print(f"\n  Property distributions (passed molecules):")
        print(f"  MolWt:      {out_df['MolWt'].mean():.1f} ± {out_df['MolWt'].std():.1f}")
        print(f"  MolLogP:    {out_df['MolLogP'].mean():.2f} ± {out_df['MolLogP'].std():.2f}")
        print(f"  NumHDonors: {out_df['NumHDonors'].mean():.1f} ± {out_df['NumHDonors'].std():.1f}")
        print(f"  NumHAccept: {out_df['NumHAcceptors'].mean():.1f} ± {out_df['NumHAcceptors'].std():.1f}")
        print(f"  QED:        {out_df['QED'].mean():.3f} ± {out_df['QED'].std():.3f}")
        print(f"  S_final:    {out_df['S_final_GA'].mean():.3f} ± {out_df['S_final_GA'].std():.3f}")
        print(f"  S_GPR:      {out_df['S_GPR_GA'].mean():.3f} ± {out_df['S_GPR_GA'].std():.3f}")
        if not out_df['Bonus_Hbond'].isna().all():
            print(f"  H-bond bonus: {out_df['Bonus_Hbond'].mean():.2f} ± {out_df['Bonus_Hbond'].std():.2f}")

    print(f"\n  Writing validation report...")
    with open(OUTPUT_TXT, 'w') as f:
        f.write("PAMR-GA — Validation Report\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Input:  {args.input}\n")
        f.write(f"Output: {args.output}\n\n")
        f.write(f"Filter summary:\n")
        for k, v in stats.items():
            f.write(f"  {k:25s}: {v}\n")
        if len(out_df) > 0:
            f.write(f"\nTop 10 molecules by S_final:\n")
            f.write("-" * 65 + "\n")
            for i, row in out_df.head(10).iterrows():
                f.write(f"  S_final={row['S_final_GA']:.3f}  "
                        f"S_GPR={row['S_GPR_GA']:.3f}  "
                        f"QED={row['QED']:.3f}  "
                        f"MW={row['MolWt']:.1f}\n")
                f.write(f"  {row['SMILES']}\n\n")
    print(f"  Saved: {OUTPUT_TXT}")
    print(f"\n  Next: python3 scripts/check_novelty.py")
    print("=" * 65)


if __name__ == "__main__":
    main()
