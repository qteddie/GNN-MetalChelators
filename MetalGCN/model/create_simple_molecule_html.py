#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple Molecule HTML Generator - Alternative version for better compatibility
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import base64
import io
import sys
import os
import torch

# Add paths
sys.path.append("/work/u5066474/NTHU/LiveTransForM-main/OMGNN/src/")
sys.path.append("/work/u5066474/NTHU/LiveTransForM-main/MetalGCN/src/")

from self_pka_models import pka_GNN_ver3

def load_proton_model(model_path: str, device: str = "cuda"):
    """Load the pretrained proton pKa model"""
    print(f"Loading proton model from: {model_path}")
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    model_params = {
        'node_dim': 153,
        'bond_dim': 18,
        'hidden_dim': 384,
        'output_dim': 1,
        'dropout': 0.1,
        'depth': 4,
        'heads': 8,
        'pe_dim': 8,
        'max_dist': 6,
        'pka_weight_path': "/work/u5066474/NTHU/LiveTransForM-main/MetalGCN/data/pka_weight_table.npz"
    }
    
    model = pka_GNN_ver3(**model_params).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model = checkpoint.to(device)
    
    model.eval()
    print("✓ Model loaded successfully")
    return model

def mol_with_atom_idx(smiles):
    """Create molecule with atom indices as labels"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add atom indices as labels
    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    
    return mol

def mol_to_image_data(mol, size=(300, 300)):
    """Convert molecule to base64 image data"""
    if mol is None:
        return "<p>Invalid molecule</p>"
    
    try:
        # Generate PNG image
        img = Draw.MolToImage(mol, size=size)
        
        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = base64.b64encode(buffer.getvalue()).decode()
        
        return f'<img src="data:image/png;base64,{img_data}" alt="Molecule structure" style="max-width: 100%; height: auto;">'
    except Exception as e:
        return f"<p>Error generating molecule: {e}</p>"

def create_simple_molecule_html(csv_path, output_path, num_samples=20):
    """Create simple HTML with molecule structures"""
    
    # Load results
    df = pd.read_csv(csv_path)
    if len(df) > num_samples:
        df = df.head(num_samples)
    
    print(f"Creating HTML with {len(df)} samples...")
    
    # Generate HTML
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Metal Dataset Sample Position Analysis with Molecules</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1400px;
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
            .sample-card {
                border: 1px solid #ddd;
                border-radius: 8px;
                margin: 20px 0;
                padding: 20px;
                background-color: #fafafa;
            }
            .sample-header {
                background-color: #3498db;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .content-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }
            .molecule-container {
                text-align: center;
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #ddd;
            }
            .info-section {
                background-color: white;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #ddd;
            }
            .info-item {
                margin: 10px 0;
                padding: 8px;
                background-color: #f8f9fa;
                border-radius: 4px;
            }
            .info-label {
                font-weight: bold;
                color: #2c3e50;
            }
            .atom-analysis {
                font-family: monospace;
                font-size: 12px;
                line-height: 1.6;
                background-color: #f8f9fa;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
            }
            .predicted { background-color: #e8f5e8; color: #2e7d32; font-weight: bold; }
            .binding-site { background-color: #fff3e0; color: #f57c00; }
            .other { color: #757575; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 Metal Dataset Sample Position Analysis</h1>
            <p style="text-align: center; color: #666; margin-bottom: 30px;">
                Molecule structures with atom indices and predicted pKa sites
            </p>
    """
    
    # Add each sample
    for i, row in df.iterrows():
        smiles = row['smiles']
        mol = mol_with_atom_idx(smiles)
        img_html = mol_to_image_data(mol)
        
        predicted_positions = eval(row['predicted_positions'])
        binding_sites = eval(row['binding_sites'])
        
        # Create atom analysis
        atom_analysis = []
        if mol:
            for atom in mol.GetAtoms():
                atom_idx = atom.GetIdx()
                atom_symbol = atom.GetSymbol()
                
                if atom_idx in predicted_positions:
                    status = "🎯 PREDICTED"
                    status_class = "predicted"
                elif atom_idx in binding_sites:
                    status = "🔗 BINDING SITE"
                    status_class = "binding-site"
                else:
                    status = "⚫ OTHER"
                    status_class = "other"
                
                atom_analysis.append(f'<span class="{status_class}">[{atom_idx}]{atom_symbol} {status}</span>')
        
        overlap_ratio = row['overlap_ratio']
        overlap_class = "color: #27ae60;" if overlap_ratio >= 0.8 else "color: #f39c12;" if overlap_ratio >= 0.5 else "color: #e74c3c;"
        
        html_content += f"""
        <div class="sample-card">
            <div class="sample-header">
                <h3>Sample {row['sample_id']} - {row['metal_ion']} Complex</h3>
                <p style="margin: 5px 0;">SMILES: {smiles}</p>
            </div>
            
            <div class="content-grid">
                <div class="molecule-container">
                    <h4>Molecule Structure</h4>
                    {img_html}
                </div>
                
                <div class="info-section">
                    <h4>Analysis Results</h4>
                    <div class="info-item">
                        <span class="info-label">True pKa Values:</span> {row['true_pka_values']}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Predicted Positions:</span> {row['predicted_positions']}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Binding Sites:</span> {row['binding_sites']}
                    </div>
                    <div class="info-item">
                        <span class="info-label">Overlap Ratio:</span> 
                        <span style="{overlap_class} font-weight: bold;">{overlap_ratio:.2%}</span>
                    </div>
                    
                    <div class="atom-analysis">
                        <strong>Atom-by-Atom Analysis:</strong><br>
                        {' '.join(atom_analysis)}
                    </div>
                </div>
            </div>
        </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ Simple molecule HTML saved to: {output_path}")

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
    
    output_path = f"/work/u5066474/NTHU/LiveTransForM-main/MetalGCN/model/simple_molecule_analysis_{num_samples}samples.html"
    
    create_simple_molecule_html(csv_path, output_path, num_samples)
    
    print(f"\n🎉 Simple molecule HTML created!")
    print(f"📁 File: {output_path}")
    print(f"🔍 Features: Molecule structures with atom indices and predicted sites")

if __name__ == "__main__":
    main()