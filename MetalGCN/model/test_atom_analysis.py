#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Atom Analysis Tool for SMILES

This script provides functions to analyze atoms in SMILES strings,
showing their IDs, properties, and potential binding sites.
"""

import sys
from rdkit import Chem

# Add paths
sys.path.append("../src/")
sys.path.append("./src/")

from MetalGCN.model.new_metal_pka_transfer import binding_atom


def get_atom_info(smiles: str):
    """
    Get detailed atom information from SMILES string
    
    Args:
        smiles (str): SMILES string
        
    Returns:
        dict: Information about each atom including ID, symbol, and properties
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Cannot parse SMILES '{smiles}'")
        return None
    
    Chem.SanitizeMol(mol)
    
    atom_info = {
        'smiles': smiles,
        'num_atoms': mol.GetNumAtoms(),
        'atoms': []
    }
    
    print(f"SMILES: {smiles}")
    print(f"Number of atoms: {mol.GetNumAtoms()}")
    print("-" * 60)
    print(f"{'ID':<3} {'Symbol':<8} {'Charge':<6} {'Valence':<8} {'Neighbors':<10} {'Hybridization':<12}")
    print("-" * 60)
    
    for atom in mol.GetAtoms():
        atom_id = atom.GetIdx()
        symbol = atom.GetSymbol()
        formal_charge = atom.GetFormalCharge()
        valence = atom.GetTotalValence()
        num_neighbors = len(atom.GetNeighbors())
        hybridization = str(atom.GetHybridization())
        
        atom_data = {
            'id': atom_id,
            'symbol': symbol,
            'formal_charge': formal_charge,
            'valence': valence,
            'num_neighbors': num_neighbors,
            'hybridization': hybridization
        }
        
        atom_info['atoms'].append(atom_data)
        
        print(f"{atom_id:<3} {symbol:<8} {formal_charge:<6} {valence:<8} {num_neighbors:<10} {hybridization:<12}")
    
    return atom_info


def analyze_binding_sites(smiles: str):
    """
    Analyze potential binding sites in a molecule
    
    Args:
        smiles (str): SMILES string
        
    Returns:
        dict: Binding site analysis
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Cannot parse SMILES '{smiles}'")
        return None
    
    # Get binding atom indices using the function from metal_pka_transfer
    binding_indices = binding_atom(mol)
    
    print(f"\nBinding Site Analysis:")
    print("-" * 50)
    print(f"Potential binding atoms: {binding_indices}")
    
    if binding_indices:
        print(f"{'Atom ID':<8} {'Symbol':<8} {'Binding Type':<15} {'Environment':<20}")
        print("-" * 55)
        
        for idx in binding_indices:
            atom = mol.GetAtomWithIdx(idx)
            symbol = atom.GetSymbol()
            
            # Determine binding type based on atom properties
            if symbol == 'O':
                binding_type = "Oxygen donor"
            elif symbol == 'N':
                binding_type = "Nitrogen donor"
            elif symbol == 'S':
                binding_type = "Sulfur donor"
            elif symbol == 'P':
                binding_type = "Phosphorus donor"
            else:
                binding_type = "Other"
            
            # Get environment info
            neighbors = [mol.GetAtomWithIdx(n.GetIdx()).GetSymbol() for n in atom.GetNeighbors()]
            environment = f"Connected to: {', '.join(neighbors)}"
                
            print(f"{idx:<8} {symbol:<8} {binding_type:<15} {environment:<20}")
    
    return {
        'smiles': smiles,
        'binding_indices': binding_indices,
        'num_binding_sites': len(binding_indices)
    }


def visualize_molecule_with_ids(smiles: str):
    """
    Complete analysis of a molecule showing atom IDs and binding sites
    
    Args:
        smiles (str): SMILES string
    """
    print("=" * 80)
    print("MOLECULE ANALYSIS")
    print("=" * 80)
    
    # Get basic atom information
    atom_info = get_atom_info(smiles)
    
    if atom_info:
        # Analyze binding sites
        binding_info = analyze_binding_sites(smiles)
        
        # Show detailed atom mapping
        print(f"\nATOM MAPPING:")
        print("-" * 40)
        for atom_data in atom_info['atoms']:
            binding_status = "BINDING SITE" if atom_data['id'] in binding_info['binding_indices'] else "non-binding"
            print(f"  Atom {atom_data['id']}: {atom_data['symbol']} ({binding_status})")
        
        # Summary
        print(f"\nSUMMARY:")
        print(f"- Total atoms: {atom_info['num_atoms']}")
        print(f"- Binding sites: {binding_info['num_binding_sites']}")
        print(f"- Binding atom IDs: {binding_info['binding_indices']}")
        
        return {
            'atom_info': atom_info,
            'binding_info': binding_info
        }
    
    return None


if __name__ == "__main__":
    # Test with the specific example from the user
    print("SPECIFIC TEST: NCCO")
    print("=" * 80)
    
    test_smiles = "NCCO"
    result = visualize_molecule_with_ids(test_smiles)
    
    if result:
        print(f"\nFor SMILES '{test_smiles}' (Ethanolamine):")
        print(f"Each atom corresponds to the following IDs:")
        for atom_data in result['atom_info']['atoms']:
            print(f"  ID {atom_data['id']}: {atom_data['symbol']} atom")
    