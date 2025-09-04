#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug script to find the exact pKa format issue
"""

import sys
import pandas as pd
import numpy as np
from rdkit import Chem

# Add paths
sys.path.append("/work/u5066474/NTHU/LiveTransForM-main/OMGNN/src/")
sys.path.append("/work/u5066474/NTHU/LiveTransForM-main/MetalGCN/src/")

def create_proton_format_csv_debug(metal_csv_path: str, output_csv_path: str, num_samples: int = 5):
    """
    Create a CSV file in proton model format from metal complex data (debug version)
    """
    print(f"Converting metal data to proton format...")
    df_metal = pd.read_csv(metal_csv_path)
    
    # Sample some data
    df_sample = df_metal.head(num_samples)
    
    proton_data = []
    for _, row in df_sample.iterrows():
        smiles = row['SMILES']
        pka_values = eval(row['pKa_value'])
        
        print(f"\n--- Processing SMILES: {smiles} ---")
        print(f"pKa values: {pka_values}")
        
        # Create simplified pKa matrix for proton model
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"  ERROR: Can't parse SMILES")
            continue
            
        num_atoms = mol.GetNumAtoms()
        print(f"  Number of atoms: {num_atoms}")
        
        # Find potential binding sites
        binding_sites = []
        for i, atom in enumerate(mol.GetAtoms()):
            print(f"  Atom {i}: {atom.GetSymbol()}")
            if atom.GetSymbol() in ['O', 'N', 'S', 'P']:
                binding_sites.append(i)
        
        print(f"  Binding sites: {binding_sites}")
        
        # Create pKa matrix in the expected format: [(atom_idx, pka_value), ...]
        pka_matrix = []
        for i, pka in enumerate(pka_values):
            if i < len(binding_sites):
                pka_matrix.append((binding_sites[i], pka))
                print(f"  Added: ({binding_sites[i]}, {pka})")
            else:
                # If no more binding sites available, assign to arbitrary atoms
                if i < num_atoms:
                    pka_matrix.append((i, pka))
                    print(f"  Added fallback: ({i}, {pka})")
        
        print(f"  Final pKa matrix: {pka_matrix}")
        print(f"  pKa matrix as string: {str(pka_matrix)}")
        
        # Test the eval immediately
        try:
            eval_result = eval(str(pka_matrix))
            print(f"  Eval result: {eval_result}")
            print(f"  Eval type: {type(eval_result)}")
            
            # Test iteration
            for j, item in enumerate(eval_result):
                print(f"    Item {j}: {item}, Type: {type(item)}")
                if isinstance(item, tuple) and len(item) == 2:
                    atom_idx, pka_value = item
                    print(f"      atom_idx: {atom_idx}, pka_value: {pka_value}")
                else:
                    print(f"      ERROR: Not a tuple or wrong length!")
        except Exception as e:
            print(f"  ERROR during eval: {e}")
        
        proton_data.append({
            'smiles': smiles,
            'pka_values': str(pka_values),
            'pka_matrix': str(pka_matrix),
            'original_metal_ion': row['metal_ion']
        })
    
    # Save to CSV
    df_proton = pd.DataFrame(proton_data)
    df_proton.to_csv(output_csv_path, index=False)
    print(f"\n✓ Proton format data saved to: {output_csv_path}")
    
    # Test reading back
    print(f"\n--- Testing CSV read-back ---")
    df_read = pd.read_csv(output_csv_path)
    for _, row in df_read.iterrows():
        smiles = row['smiles']
        pka_matrix_str = row['pka_matrix']
        print(f"\nSMILES: {smiles}")
        print(f"pKa matrix string: {pka_matrix_str}")
        
        try:
            pka_matrix_eval = eval(pka_matrix_str)
            print(f"Eval result: {pka_matrix_eval}")
            
            # Test the exact failing line
            for atom_idx, pka_value in pka_matrix_eval:
                print(f"  atom_idx={atom_idx}, pka_value={pka_value}")
                
        except Exception as e:
            print(f"ERROR during read-back: {e}")
    
    return df_proton

def main():
    # Paths
    metal_csv_path = "/work/u5066474/NTHU/LiveTransForM-main/MetalGCN/data/pre_metal_t15_T25_I0.1_E3_m10_clean.csv"
    proton_csv_path = "/work/u5066474/NTHU/LiveTransForM-main/MetalGCN/model/debug_proton_format_data.csv"
    
    # Create debug version
    create_proton_format_csv_debug(metal_csv_path, proton_csv_path, num_samples=3)

if __name__ == "__main__":
    main()