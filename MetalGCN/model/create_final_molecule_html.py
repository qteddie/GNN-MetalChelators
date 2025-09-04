#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Final Molecule HTML Generator - Working version
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import sys
import os

def mol_with_atom_idx(smiles):
    """Create molecule with atom indices as labels"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add atom indices as labels
    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    
    return mol

def create_final_molecule_html(csv_path, output_path, num_samples=20):
    """Create final HTML with molecule structures"""
    
    # Load results
    df = pd.read_csv(csv_path)
    if len(df) > num_samples:
        df = df.head(num_samples)
    
    print(f"Creating final HTML with {len(df)} samples...")
    
    # Create a new dataframe for HTML output
    html_df = df.copy()
    
    # Add molecule column
    html_df['Molecule'] = html_df['smiles'].apply(mol_with_atom_idx)
    
    # Reorder columns for better display
    columns_order = ['sample_id', 'smiles', 'Molecule', 'metal_ion', 'true_pka_values', 
                     'predicted_positions', 'binding_sites', 'overlap_ratio', 'atom_mapping']
    
    # Only include columns that exist
    available_columns = [col for col in columns_order if col in html_df.columns]
    html_df = html_df[available_columns]
    
    # Custom CSS
    custom_css = """
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background-color: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        td {
            padding: 12px;
            border-bottom: 1px solid #ddd;
            vertical-align: top;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .smiles-col {
            font-family: monospace;
            font-size: 12px;
            word-break: break-all;
            max-width: 200px;
        }
        .atom-mapping {
            font-family: monospace;
            font-size: 10px;
            line-height: 1.4;
            max-width: 300px;
            word-wrap: break-word;
        }
        .overlap-excellent { 
            color: #27ae60; 
            font-weight: bold;
        }
        .overlap-good { 
            color: #f39c12; 
            font-weight: bold;
        }
        .overlap-poor { 
            color: #e74c3c; 
            font-weight: bold;
        }
        .summary {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
    </style>
    """
    
    # Generate HTML using pandas to_html with custom formatting
    html_table = html_df.to_html(
        index=False,
        escape=False,
        table_id="analysis_table",
        classes="table table-striped table-hover",
        formatters={
            'overlap_ratio': lambda x: f'<span class="overlap-{"excellent" if x >= 0.8 else "good" if x >= 0.5 else "poor"}">{x:.2%}</span>',
            'smiles': lambda x: f'<div class="smiles-col">{x}</div>',
            'atom_mapping': lambda x: f'<div class="atom-mapping">{x}</div>' if pd.notna(x) else ''
        }
    )
    
    # Calculate summary statistics
    total_samples = len(df)
    avg_overlap = df['overlap_ratio'].mean()
    perfect_overlap = sum(df['overlap_ratio'] == 1.0)
    
    # Create complete HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Metal Dataset Sample Position Analysis - Final Version</title>
        <meta charset="UTF-8">
        {custom_css}
    </head>
    <body>
        <div class="container">
            <h1>🧪 Metal Dataset Sample Position Analysis</h1>
            <p style="text-align: center; color: #666; margin-bottom: 30px;">
                Detailed analysis of predicted pKa sites with molecule structures and atom indices
            </p>
            
            <div class="summary">
                <h3>📊 Summary Statistics</h3>
                <p><strong>Total samples analyzed:</strong> {total_samples}</p>
                <p><strong>Average overlap ratio:</strong> {avg_overlap:.2%}</p>
                <p><strong>Samples with perfect overlap:</strong> {perfect_overlap}</p>
                <p><strong>Features:</strong> Molecule structures show atom indices using atom.GetIdx()</p>
            </div>
            
            <div style="overflow-x: auto;">
                {html_table}
            </div>
            
            <div style="margin-top: 30px; padding: 15px; background-color: #e8f5e8; border-radius: 8px;">
                <h4>🔍 How to interpret the results:</h4>
                <ul>
                    <li><strong>Molecule column:</strong> Shows chemical structure with atom indices</li>
                    <li><strong>Predicted positions:</strong> Atoms selected by the model as pKa sites</li>
                    <li><strong>Binding sites:</strong> Chemically expected binding sites (O, N, S, P)</li>
                    <li><strong>Overlap ratio:</strong> How well predictions match chemical expectations</li>
                    <li><strong>Atom mapping:</strong> Detailed atom-by-atom analysis</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Final molecule HTML saved to: {output_path}")
    print(f"✓ Contains {total_samples} samples with molecule structures")
    print(f"✓ Average overlap ratio: {avg_overlap:.2%}")

def main():
    """Main function"""
    num_samples = 20
    
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            print("Error: Please provide a valid number of samples")
            sys.exit(1)
    
    # Use existing CSV results
    csv_path = f"/work/u5066474/NTHU/LiveTransForM-main/MetalGCN/model/metal_dataset_sample_position_results_{num_samples}samples.csv"
    
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        print("Please run generate_molecule_html.py first to generate the CSV results.")
        sys.exit(1)
    
    output_path = f"/work/u5066474/NTHU/LiveTransForM-main/MetalGCN/model/final_molecule_analysis_{num_samples}samples.html"
    
    create_final_molecule_html(csv_path, output_path, num_samples)
    
    print(f"\n🎉 Final molecule HTML created successfully!")
    print(f"📁 File: {output_path}")
    print(f"🧪 Open in browser to view molecules with atom indices")

if __name__ == "__main__":
    main()