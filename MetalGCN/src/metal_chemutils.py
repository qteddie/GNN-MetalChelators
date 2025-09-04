#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metal Chemistry Utilities
=====================

處理分子和金屬離子的工具模組
"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
import logging

# 配置logger
logger = logging.getLogger('metal_chemutils')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)

# 讀取RDKit特徵工廠
try:
    fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
except Exception as e:
    logger.warning(f"無法加載RDKit特徵工廠: {e}")
    chem_feature_factory = None


# 原子特徵計算
def atom_features(atom, oxidation_state=None):
    """
    計算原子特徵
    
    Args:
        atom: RDKit原子對象
        oxidation_state: 氧化態（可選，金屬離子使用）
        
    Returns:
        torch.Tensor: 原子特徵張量
    """
    # 原子類型的one-hot編碼 (常見元素)
    elements = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 
                'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'Cu', 'Zn', 'B', 'Si', 'Se']
    element_type = [atom.GetSymbol() == e for e in elements]
    
    # 如果不是列表中的元素，添加一個標記
    if not any(element_type):
        element_type.append(1)
    else:
        element_type.append(0)
    
    # 雜化狀態
    hybridization_types = [
        Chem.rdchem.HybridizationType.SP, 
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, 
        Chem.rdchem.HybridizationType.SP3D, 
        Chem.rdchem.HybridizationType.SP3D2
    ]
    hybridization = [atom.GetHybridization() == h for h in hybridization_types]
    if not any(hybridization):
        hybridization.append(1)
    else:
        hybridization.append(0)
    
    # 原子位置信息
    is_in_ring = [atom.IsInRing()]
    is_aromatic = [atom.GetIsAromatic()]
    
    # 原子形式電荷
    formal_charge = [0] * 5  # -2, -1, 0, +1, +2
    charge = atom.GetFormalCharge()
    if -2 <= charge <= 2:
        formal_charge[charge+2] = 1
    
    # 原子度信息
    atom_degree = [0] * 6  # 0, 1, 2, 3, 4, 5+
    degree = min(atom.GetDegree(), 5)
    atom_degree[degree] = 1
    
    # 隱式氫原子數
    implicit_h = [0] * 5  # 0, 1, 2, 3, 4+
    num_h = min(atom.GetTotalNumHs(), 4)
    implicit_h[num_h] = 1
    
    # 手性
    chirality = [0] * 3  # R, S, 無
    if atom.HasProp('_CIPCode'):
        cip_code = atom.GetProp('_CIPCode')
        if cip_code == 'R':
            chirality[0] = 1
        elif cip_code == 'S':
            chirality[1] = 1
    else:
        chirality[2] = 1
    
    # 酸鹼性
    acidity = [0] * 4  # 強酸, 弱酸, 中性, 鹼性
    acidic_groups = ['CO2H', 'SO3H', 'PO3H2']
    basic_groups = ['NH2', 'NHR', 'NR2']
    
    if oxidation_state is not None:
        # 氧化態 (用於金屬離子)
        oxidation = [0] * 9  # -4, -3, -2, -1, 0, +1, +2, +3, +4
        if -4 <= oxidation_state <= 4:
            oxidation[oxidation_state+4] = 1
    else:
        oxidation = [0] * 9
        # 默認將中性原子標記為0氧化態
        oxidation[4] = 1
    
    # 電負性 (用於更好地估計pKa影響)
    electronegativity = [0] * 4  # 低(<2.0), 中(2.0-2.5), 高(2.6-3.0), 很高(>3.0)
    en_values = {
        'H': 2.2, 'Li': 0.98, 'Na': 0.93, 'K': 0.82, 'Rb': 0.82, 'Cs': 0.79,
        'Be': 1.57, 'Mg': 1.31, 'Ca': 1.0, 'Sr': 0.95, 'Ba': 0.89,
        'B': 2.04, 'Al': 1.61, 'Ga': 1.81, 'In': 1.78, 'Tl': 2.04,
        'C': 2.55, 'Si': 1.9, 'Ge': 2.01, 'Sn': 1.96, 'Pb': 2.33,
        'N': 3.04, 'P': 2.19, 'As': 2.18, 'Sb': 2.05, 'Bi': 2.02,
        'O': 3.44, 'S': 2.58, 'Se': 2.55, 'Te': 2.1, 'Po': 2.0,
        'F': 3.98, 'Cl': 3.16, 'Br': 2.96, 'I': 2.66, 'At': 2.2,
        'Cu': 1.9, 'Ag': 1.93, 'Au': 2.54,
        'Zn': 1.65, 'Cd': 1.69, 'Hg': 2.0,
        'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91
    }
    
    element = atom.GetSymbol()
    en = en_values.get(element, 2.0)  # 默認值為2.0
    
    if en < 2.0:
        electronegativity[0] = 1
    elif en < 2.6:
        electronegativity[1] = 1
    elif en < 3.1:
        electronegativity[2] = 1
    else:
        electronegativity[3] = 1
    
    # 結合所有特徵
    features = element_type + hybridization + is_in_ring + is_aromatic + \
               formal_charge + atom_degree + implicit_h + chirality + \
               acidity + oxidation + electronegativity
    
    return torch.tensor(features, dtype=torch.float)


# 鍵特徵計算
def bond_features(bond):
    """
    計算鍵特徵
    
    Args:
        bond: RDKit鍵對象
        
    Returns:
        numpy.ndarray: 鍵特徵數組
    """
    # 鍵類型
    bond_type_list = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    bond_type = [bond.GetBondType() == b for b in bond_type_list]
    
    # 鍵位置信息
    is_in_ring = [bond.IsInRing()]
    is_conjugated = [bond.GetIsConjugated()]
    
    # 鍵立體化學
    stereo_list = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE
    ]
    stereo = [bond.GetStereo() == s for s in stereo_list]
    
    # 結合所有特徵
    features = bond_type + is_in_ring + is_conjugated + stereo
    
    return np.array(features, dtype=np.float32)


# 將分子轉換為圖格式
def tensorize_for_pka(smiles):
    """
    將分子SMILES轉換為圖格式的張量
    
    Args:
        smiles: 分子SMILES字符串
        
    Returns:
        tuple: (原子特徵張量, 邊索引, 邊特徵張量)
    """
    try:
        # 解析分子
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"無法解析SMILES: {smiles}")
        
        # 添加氫原子
        mol = Chem.AddHs(mol)
        
        # 生成3D構象
        try:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.UFFOptimizeMolecule(mol)
        except:
            pass
            # logger.warning(f"無法生成3D構象: {smiles}")
        
        # 計算原子特徵
        num_atoms = mol.GetNumAtoms()
        atom_feature_list = []
        for atom_idx in range(num_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)
            atom_feature = atom_features(atom)
            atom_feature_list.append(atom_feature)
        atom_features_tensor = torch.stack(atom_feature_list)
        
        # 計算鍵特徵和邊索引
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # 雙向邊
            edge_indices.append((i, j))
            edge_indices.append((j, i))
            
            # 計算鍵特徵
            bond_feat = bond_features(bond)
            edge_attrs.append(bond_feat)
            edge_attrs.append(bond_feat)  # 雙向使用相同特徵
        
        # 如果沒有鍵，添加自循環
        if len(edge_indices) == 0:
            for i in range(num_atoms):
                edge_indices.append((i, i))
                edge_attrs.append(np.zeros(11, dtype=np.float32))  # 使用全零特徵
        
        # 轉換為張量
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.float)
        
        return atom_features_tensor, edge_index, edge_attr
    except Exception as e:
        logger.error(f"處理SMILES時出錯: {e}")
        # 返回一個單節點圖作為回退
        atom_feat = torch.zeros(1, 78)  # 使用與正常原子特徵相同的維度
        edge_index = torch.tensor([[0], [0]])
        edge_attr = torch.zeros(1, 11)
        return atom_feat, edge_index, edge_attr


# 為金屬-配位鍵生成特徵
def create_coordination_bond_features(metal_atom, ligand_atom, bond_length=None):
    """
    為金屬-配位鍵生成特徵
    
    Args:
        metal_atom: 金屬原子對象
        ligand_atom: 配體原子對象
        bond_length: 鍵長（可選）
        
    Returns:
        numpy.ndarray: 配位鍵特徵數組
    """
    # 配位鍵基本特徵（與普通鍵不同）
    bond_type = [1, 0, 0, 0]  # 視為單鍵
    is_in_ring = [0]  # 通常不在環中
    is_conjugated = [0]  # 通常不共軛
    stereo = [1, 0, 0]  # 無立體化學
    
    # 配位鍵特有特徵
    is_coordination = [1]  # 標記為配位鍵
    
    # 配位原子類型
    coordination_atom_types = {
        'O': [1, 0, 0, 0],  # 氧配位
        'N': [0, 1, 0, 0],  # 氮配位
        'S': [0, 0, 1, 0],  # 硫配位
        'P': [0, 0, 0, 1],  # 磷配位
    }
    ligand_symbol = ligand_atom.GetSymbol()
    coordination_type = coordination_atom_types.get(ligand_symbol, [0, 0, 0, 0])
    
    # 合併所有特徵
    features = bond_type + is_in_ring + is_conjugated + stereo + is_coordination + coordination_type
    
    return np.array(features, dtype=np.float32)


# 測試函數
def test_featurization():
    """測試分子特徵化功能"""
    # 測試分子
    test_smiles = [
        'CC(=O)OC1=CC=CC=C1C(=O)O',  # 阿司匹林
        'OC1=C(O)C=CC=C1',            # 間苯二酚
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'  # 咖啡因
    ]
    
    for smiles in test_smiles:
        try:
            print(f"\n處理SMILES: {smiles}")
            atoms, edges, edge_feats = tensorize_for_pka(smiles)
            print(f"  原子數: {atoms.shape[0]}")
            print(f"  特徵維度: {atoms.shape[1]}")
            print(f"  邊數: {edges.shape[1]}")
            print(f"  邊特徵維度: {edge_feats.shape[1]}")
        except Exception as e:
            print(f"  處理出錯: {e}")


if __name__ == "__main__":
    # 運行測試
    test_featurization()
