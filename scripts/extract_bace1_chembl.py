#!/usr/bin/env python3
"""
extract_bace1_chembl.py
=======================
Dataset Extraction Script for PAMR-GA
Extracts BACE-1 inhibitor bioactivity data from ChEMBL 36 SQLite database.

Target  : BACE-1 (Beta-secretase 1), ChEMBL ID: CHEMBL4822
Disease : Alzheimer's Disease
Output  : data/bace1_dataset.csv

Usage:
    python3 scripts/extract_bace1_chembl.py --db /path/to/chembl_36.db

Requirements:
    pip install pandas numpy rdkit

ChEMBL 36 SQLite database can be downloaded from:
    https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_36/
"""

import argparse
import sqlite3
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

# ── Settings ───────────────────────────────────────────────────────────────
TARGET_CHEMBL_ID = 'CHEMBL4822'       # BACE-1
PIC50_MIN        = 4.0                # Below this = negligible inhibition
PIC50_MAX        = 12.0               # Above this = beyond assay detection limit
OUTPUT_PATH      = 'data/bace1_dataset.csv'

# ── SQL Query ──────────────────────────────────────────────────────────────
QUERY = """
SELECT
    md.chembl_id,
    cs.canonical_smiles,
    a.standard_value,
    td.chembl_id   AS target_id,
    td.pref_name   AS target_name
FROM activities a
JOIN assays            ass ON a.assay_id   = ass.assay_id
JOIN target_dictionary td  ON ass.tid      = td.tid
JOIN molecule_dictionary md ON a.molregno  = md.molregno
JOIN compound_structures cs ON md.molregno = cs.molregno
WHERE
    td.chembl_id          = '{target}'
    AND a.standard_type   = 'IC50'
    AND a.standard_units  = 'nM'
    AND a.standard_relation = '='
    AND a.standard_value  IS NOT NULL
    AND cs.canonical_smiles IS NOT NULL
""".format(target=TARGET_CHEMBL_ID)


def extract(db_path: str) -> pd.DataFrame:
    print(f"Connecting to ChEMBL database: {db_path}")
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql_query(QUERY, conn)
    conn.close()
    print(f"  Raw records fetched  : {len(df)}")

    # ── 1. Basic cleaning ──────────────────────────────────────────────
    df = df.dropna(subset=['canonical_smiles', 'standard_value'])
    df = df[df['standard_value'] > 0]
    print(f"  After NaN/zero drop  : {len(df)}")

    # ── 2. Canonicalize SMILES with RDKit ─────────────────────────────
    canon = []
    for s in df['canonical_smiles']:
        mol = Chem.MolFromSmiles(s)
        canon.append(Chem.MolToSmiles(mol) if mol else None)
    df['canonical_smiles'] = canon
    df = df.dropna(subset=['canonical_smiles'])
    print(f"  After invalid SMILES : {len(df)}")

    # ── 3. Remove duplicates (keep first occurrence) ───────────────────
    before = len(df)
    df = df.drop_duplicates(subset='canonical_smiles', keep='first')
    print(f"  Duplicates removed   : {before - len(df)}")

    # ── 4. Convert IC50 (nM) → pIC50 ──────────────────────────────────
    df['pIC50'] = -np.log10(df['standard_value'].astype(float) * 1e-9)

    # ── 5. Filter pIC50 range ──────────────────────────────────────────
    df = df[(df['pIC50'] >= PIC50_MIN) & (df['pIC50'] <= PIC50_MAX)]
    print(f"  After pIC50 filter   : {len(df)}")

    return df[['chembl_id', 'canonical_smiles', 'pIC50']].reset_index(drop=True)


def summarize(df: pd.DataFrame):
    print("\n── Dataset Summary ───────────────────────────────────────")
    print(f"  Total molecules      : {len(df)}")
    print(f"  pIC50 range          : {df['pIC50'].min():.2f} – {df['pIC50'].max():.2f}")
    print(f"  pIC50 mean ± std     : {df['pIC50'].mean():.2f} ± {df['pIC50'].std():.2f}")
    print(f"  pIC50 ≥ 7 (pool)     : {len(df[df['pIC50'] >= 7.0])} molecules")
    print(f"  pIC50 ≥ 8 (potent)   : {len(df[df['pIC50'] >= 8.0])} molecules")
    print("──────────────────────────────────────────────────────────")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract BACE-1 dataset from ChEMBL 36')
    parser.add_argument('--db', required=True,
                        help='Path to chembl_36.db SQLite file')
    args = parser.parse_args()

    df = extract(args.db)
    summarize(df)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset saved to: {OUTPUT_PATH}")
