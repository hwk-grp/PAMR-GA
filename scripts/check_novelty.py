#!/usr/bin/env python3
"""
check_novelty.py
================
PAMR-GA Novelty Analysis
=====================================

Computes Tanimoto similarity of validated GA molecules against the
ChEMBL BACE-1 training set to assess novelty.

Novelty definition (Tc < threshold vs all training molecules):
    Tc < 0.4  — novel (structurally distinct)
    Tc < 0.6  — moderately novel
    Tc >= 0.6 — similar to known inhibitor

Also reports:
    - Nearest neighbour in training set for each molecule
    - Distribution of max Tc scores
    - Scaffold novelty (Murcko scaffold analysis)

Usage:
    python3 scripts/check_novelty.py
    python3 scripts/check_novelty.py --input results/validated_molecules.csv
    python3 scripts/check_novelty.py --threshold 0.4

Output:
    results/novelty_analysis.csv
    results/novelty_report.txt
"""

import os
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

RDLogger.DisableLog('rdApp.*')

# ── Paths ──────────────────────────────────────────────────────────────
TRAINING_DATA   = 'data/bace1_dataset.csv'
DEFAULT_INPUT   = 'results/validated_molecules.csv'
OUTPUT_CSV      = 'results/novelty_analysis.csv'
OUTPUT_TXT      = 'results/novelty_report.txt'

NOVELTY_THRESH  = 0.4   # Tc < 0.4 → novel
MOD_THRESH      = 0.6   # Tc < 0.6 → moderately novel


def get_morgan_fp(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)


def get_murcko_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, True)
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description='Novelty analysis for GA molecules')
    parser.add_argument('--input',     default=DEFAULT_INPUT,
                        help='Input CSV (default: results/validated_molecules.csv)')
    parser.add_argument('--threshold', type=float, default=NOVELTY_THRESH,
                        help='Tanimoto novelty threshold (default: 0.4)')
    parser.add_argument('--output',    default=OUTPUT_CSV)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs('results', exist_ok=True)

    print("=" * 65)
    print("  PAMR-GA Novelty Analysis")
    print(f"  Novelty threshold: Tc < {args.threshold}")
    print("=" * 65)

    # Load training set
    print(f"\n[Step 1/4] Loading ChEMBL training set...")
    train_df = pd.read_csv(TRAINING_DATA)
    print(f"  Training molecules: {len(train_df)}")

    print(f"  Computing training set fingerprints...")
    train_fps = []
    train_smiles = []
    for smi in tqdm(train_df['canonical_smiles'], desc="  Train FPs"):
        fp = get_morgan_fp(smi)
        if fp is not None:
            train_fps.append(fp)
            train_smiles.append(smi)
    print(f"  Valid training FPs: {len(train_fps)}")

    # Load validated molecules
    print(f"\n[Step 2/4] Loading validated molecules from {args.input}...")
    if not os.path.exists(args.input):
        # Fall back to all_top_molecules.csv
        fallback = 'results/all_top_molecules.csv'
        if os.path.exists(fallback):
            print(f"  WARNING: {args.input} not found, using {fallback}")
            args.input = fallback
        else:
            print(f"  ERROR: Neither {args.input} nor {fallback} found.")
            return
    gen_df = pd.read_csv(args.input)
    print(f"  Generated molecules: {len(gen_df)}")

    # Compute Tanimoto similarity
    print(f"\n[Step 3/4] Computing Tanimoto similarities...")
    results = []

    for _, row in tqdm(gen_df.iterrows(), total=len(gen_df), desc="  Similarity"):
        smiles = row['SMILES']
        gen_fp = get_morgan_fp(smiles)
        if gen_fp is None:
            continue

        # Bulk Tanimoto against all training fps
        similarities = DataStructs.BulkTanimotoSimilarity(gen_fp, train_fps)
        max_tc   = float(np.max(similarities))
        mean_tc  = float(np.mean(similarities))
        nn_idx   = int(np.argmax(similarities))
        nn_smiles = train_smiles[nn_idx]

        # Novelty classification
        if max_tc < args.threshold:
            novelty = 'novel'
        elif max_tc < MOD_THRESH:
            novelty = 'moderate'
        else:
            novelty = 'similar'

        # Murcko scaffold
        scaffold = get_murcko_scaffold(smiles)

        entry = {
            'SMILES':          smiles,
            'MaxTc':           max_tc,
            'MeanTc':          mean_tc,
            'Novelty':         novelty,
            'NearestNeighbour': nn_smiles,
            'Scaffold':        scaffold,
        }
        # Carry over columns from input if present
        for col in ['S_final_GA', 'S_GPR_GA', 'S_final', 'S_GPR',
                    'MolWt', 'MolLogP', 'QED', 'step']:
            if col in row:
                entry[col] = row[col]

        results.append(entry)

    # Summary
    print(f"\n[Step 4/4] Summary...")
    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values('MaxTc')  # most novel first

    n_total    = len(out_df)
    n_novel    = (out_df['Novelty'] == 'novel').sum()
    n_moderate = (out_df['Novelty'] == 'moderate').sum()
    n_similar  = (out_df['Novelty'] == 'similar').sum()

    # Scaffold analysis
    unique_scaffolds_gen   = out_df['Scaffold'].nunique()
    train_scaffolds        = set()
    for smi in tqdm(train_smiles, desc="  Train scaffolds"):
        s = get_murcko_scaffold(smi)
        if s:
            train_scaffolds.add(s)
    gen_scaffolds = set(out_df['Scaffold'].dropna())
    novel_scaffolds = gen_scaffolds - train_scaffolds

    print(f"\n{'='*65}")
    print(f"  NOVELTY ANALYSIS SUMMARY")
    print(f"{'='*65}")
    print(f"  Generated molecules:    {n_total}")
    print(f"  Novel      (Tc < {args.threshold:.1f}): {n_novel:4d} ({100*n_novel/n_total:.1f}%)")
    print(f"  Moderate   (Tc < {MOD_THRESH:.1f}): {n_moderate:4d} ({100*n_moderate/n_total:.1f}%)")
    print(f"  Similar    (Tc >= {MOD_THRESH:.1f}): {n_similar:4d} ({100*n_similar/n_total:.1f}%)")
    print(f"\n  Tanimoto statistics:")
    print(f"  Mean MaxTc:  {out_df['MaxTc'].mean():.3f}")
    print(f"  Median MaxTc:{out_df['MaxTc'].median():.3f}")
    print(f"  Min MaxTc:   {out_df['MaxTc'].min():.3f}")
    print(f"  Max MaxTc:   {out_df['MaxTc'].max():.3f}")
    print(f"\n  Scaffold analysis:")
    print(f"  Unique scaffolds (generated): {unique_scaffolds_gen}")
    print(f"  Novel scaffolds (not in train): {len(novel_scaffolds)}")

    # Save
    out_df.to_csv(args.output, index=False)
    print(f"\n  Results saved: {args.output}")

    # Write report
    with open(OUTPUT_TXT, 'w') as f:
        f.write("PAMR-GA — Novelty Analysis Report\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Training set:  {TRAINING_DATA} ({len(train_fps)} molecules)\n")
        f.write(f"Generated set: {args.input} ({n_total} molecules)\n")
        f.write(f"Threshold:     Tc < {args.threshold}\n\n")
        f.write(f"Novelty breakdown:\n")
        f.write(f"  Novel      (Tc < {args.threshold:.1f}): {n_novel} ({100*n_novel/n_total:.1f}%)\n")
        f.write(f"  Moderate   (Tc < {MOD_THRESH:.1f}): {n_moderate} ({100*n_moderate/n_total:.1f}%)\n")
        f.write(f"  Similar    (Tc >= {MOD_THRESH:.1f}): {n_similar} ({100*n_similar/n_total:.1f}%)\n\n")
        f.write(f"Tanimoto statistics:\n")
        f.write(f"  Mean:   {out_df['MaxTc'].mean():.4f}\n")
        f.write(f"  Median: {out_df['MaxTc'].median():.4f}\n")
        f.write(f"  Min:    {out_df['MaxTc'].min():.4f}\n")
        f.write(f"  Max:    {out_df['MaxTc'].max():.4f}\n\n")
        f.write(f"Scaffold analysis:\n")
        f.write(f"  Unique scaffolds (generated): {unique_scaffolds_gen}\n")
        f.write(f"  Novel scaffolds:              {len(novel_scaffolds)}\n\n")
        f.write(f"Top 10 most novel molecules:\n")
        f.write("-" * 65 + "\n")
        for _, row in out_df.head(10).iterrows():
            f.write(f"  MaxTc={row['MaxTc']:.3f}  Novelty={row['Novelty']}\n")
            f.write(f"  {row['SMILES']}\n")
            f.write(f"  Nearest: {row['NearestNeighbour']}\n\n")

    print(f"  Report saved: {OUTPUT_TXT}")
    print("=" * 65)


if __name__ == "__main__":
    main()
