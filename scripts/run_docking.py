#!/usr/bin/env python3
"""
run_docking.py
==============
2.5D SPACE-GA: AutoDock Vina Docking Validation
"""

import os
import subprocess
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')

BASE    = '.'
DOCK    = 'docking'
VINA    = os.environ.get('VINA_PATH', 'vina')
RESULTS = f'{DOCK}/results'
POSES   = f'{RESULTS}/poses'

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(POSES,   exist_ok=True)

# Binding site (from co-crystallized ligand 3BN in 2B8V)
CENTER_X = 29.79
CENTER_Y = 39.28
CENTER_Z = 1.94
SIZE_X   = 20.0
SIZE_Y   = 20.0
SIZE_Z   = 20.0

print("=" * 55)
print("  AutoDock Vina: BACE-1 Docking Validation")
print("=" * 55)

# ── Step 1: Verify protein ─────────────────────────────────────────────
print("\n[Step 1/4] Verifying protein PDBQT...")
prot_pdbqt = f'{DOCK}/protein.pdbqt'
if not os.path.exists(prot_pdbqt):
    print("  ERROR: protein.pdbqt not found!")
    exit(1)

# Check no ROOT/BRANCH tags
bad_tags = subprocess.run(
    f"grep -c 'ROOT\\|BRANCH\\|TORSDOF' {prot_pdbqt}",
    shell=True, capture_output=True, text=True)
n_bad = int(bad_tags.stdout.strip()) if bad_tags.stdout.strip().isdigit() else 0
if n_bad > 0:
    print(f"  Removing {n_bad} flexible receptor tags...")
    subprocess.run(
        f"grep -v '^ROOT$\\|^ENDROOT$\\|^BRANCH\\|^ENDBRANCH\\|^TORSDOF' "
        f"{prot_pdbqt} > /tmp/prot_clean.pdbqt && cp /tmp/prot_clean.pdbqt {prot_pdbqt}",
        shell=True)
print(f"  Protein ready: {prot_pdbqt} ✓")

# ── Step 2: Load molecules ─────────────────────────────────────────────
print("\n[Step 2/4] Loading top molecules...")
val_df = pd.read_csv(f'{BASE}/results/validated_molecules.csv')
val_df = val_df.dropna(subset=['SMILES'])
top_mols = val_df.reset_index(drop=True)
print(f"  Loaded {len(top_mols)} molecules")

# ── Step 3: Convert + Dock ─────────────────────────────────────────────
print(f"\n[Step 3/4] Converting and docking...")
print(f"  Center: ({CENTER_X}, {CENTER_Y}, {CENTER_Z})")
print(f"  Box: {SIZE_X} x {SIZE_Y} x {SIZE_Z} Angstrom\n")

docking_results = []

for i, row in top_mols.iterrows():
    smi = row['SMILES']
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        continue

    mol = Chem.AddHs(mol)
    try:
        if AllChem.EmbedMolecule(mol, randomSeed=42) < 0:
            continue
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        continue

    # Save SDF
    sdf_path   = f'{DOCK}/mol_{i:03d}.sdf'
    pdbqt_path = f'{DOCK}/mol_{i:03d}.pdbqt'
    out_path   = f'{POSES}/pose_{i:03d}.pdbqt'
    log_path   = f'{POSES}/log_{i:03d}.txt'

    writer = Chem.SDWriter(sdf_path)
    writer.write(mol)
    writer.close()

    # Convert to PDBQT
    subprocess.run(
        f"obabel {sdf_path} -O {pdbqt_path} --partialcharge gasteiger -h 2>/dev/null",
        shell=True)

    if not os.path.exists(pdbqt_path) or os.path.getsize(pdbqt_path) == 0:
        print(f"  Mol {i:03d}: PDBQT conversion failed")
        continue

    # Run Vina
    cmd = (f"{VINA} "
           f"--receptor {prot_pdbqt} "
           f"--ligand {pdbqt_path} "
           f"--center_x {CENTER_X} --center_y {CENTER_Y} --center_z {CENTER_Z} "
           f"--size_x {SIZE_X} --size_y {SIZE_Y} --size_z {SIZE_Z} "
           f"--out {out_path} "
           f"--exhaustiveness 8 --num_modes 5 --cpu 4 "
           f"> {log_path} 2>&1")

    subprocess.run(cmd, shell=True)

    # Parse score from log file
    best_score = None
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0] == '1':
                    try:
                        best_score = float(parts[1])
                        break
                    except ValueError:
                        pass

    status = f"{best_score:.3f} kcal/mol" if best_score else "Failed"
    print(f"  Mol {i:03d}: pIC50={row['S_GPR_GA']:.2f} | Score={status}")

    lipinski_viol = row.get('LipinskiViol', np.nan)
    ro5_pass = bool(lipinski_viol <= 1) if pd.notna(lipinski_viol) else ''

    docking_results.append({
        'SMILES':        smi,
        'S_GPR_GA':      row['S_GPR_GA'],
        'MolWt':         row.get('MolWt', ''),
        'MolLogP':       row.get('MolLogP', ''),
        'Ro5_Pass':      ro5_pass,
        'Docking_Score': best_score,
    })

# ── Step 4: Save ──────────────────────────────────────────────────────
print(f"\n[Step 4/4] Saving results...")
df = pd.DataFrame(docking_results)
df_success = df.dropna(subset=['Docking_Score']).sort_values('Docking_Score')

csv_path = f'{RESULTS}/docking_results.csv'
df.to_csv(csv_path, index=False)

print("\n" + "=" * 55)
print("  DOCKING COMPLETE")
print("=" * 55)
print(f"  Total attempted: {len(docking_results)}")
print(f"  Successful:      {len(df_success)}")

if not df_success.empty:
    print(f"  Best score:      {df_success['Docking_Score'].min():.3f} kcal/mol")
    print(f"  Mean score:      {df_success['Docking_Score'].mean():.3f} kcal/mol")
    print(f"\n  Top 5:")
    print(f"  {'#':<4} {'pIC50':>6} {'Score':>8} {'Ro5':>5}  SMILES")
    print(f"  {'-'*60}")
    for j, (_, r) in enumerate(df_success.head(5).iterrows()):
        ro5 = 'Pass' if r['Ro5_Pass'] else 'Fail'
        print(f"  {j+1:<4} {r['S_GPR_GA']:>6.2f} {r['Docking_Score']:>8.3f} {ro5:>5}  {r['SMILES'][:35]}")

print(f"\n  Saved: {csv_path}")
print("=" * 55)
