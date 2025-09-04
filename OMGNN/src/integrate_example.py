
# 示範如何在pka_prediction_kmeans.py中整合functional_group_labeler.py

import sys
import os
import numpy as np
from rdkit import Chem
from typing import List, Dict

# 導入原始的功能模組
sys.path.append('../')
from other.chemutils_old import atom_features

# 導入新的標記器
from functional_group_labeler import extract_functional_group_features, integrate_with_kmeans

# 測試SMILES
test_smiles = [
    "CC(=O)O",                # 乙酸
    "CCN",                    # 乙胺
    "c1ccccc1O",              # 酚
    "c1ccccn1",               # 吡啶
    "N[C@@H](C)C(=O)O",       # 丙氨酸
    "O=S(=O)(O)c1ccccc1",     # 苯磺酸
]

def original_extract_feature(smiles: str):
    """模擬原始的特徵提取函數"""
    print(f"使用原始方法提取: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    
    # 模擬原始程式碼的邏輯
    from src.pka_prediction_kmeans import extract_molecule_feature
    return extract_molecule_feature(smiles)

def new_extract_feature(smiles: str):
    """使用新的關鍵原子特徵提取函數"""
    print(f"使用新方法提取: {smiles}")
    return integrate_with_kmeans(smiles)

def compare_methods():
    """比較兩種方法的結果"""
    for smiles in test_smiles:
        print("\n" + "="*60)
        print(f"分子: {smiles}")
        print("="*60)
        
        # 原始方法
        try:
            original_features = original_extract_feature(smiles)
            print(f"原始方法找到 {len(original_features)} 個官能基")
            for i, feat in enumerate(original_features):
                print(f"  官能基 {i+1}: {feat['group_type']} at 位置 {feat['match_indices']}")
        except Exception as e:
            print(f"原始方法錯誤: {e}")
        
        print("-"*40)
        
        # 新方法
        try:
            new_features = new_extract_feature(smiles)
            print(f"新方法找到 {len(new_features)} 個關鍵官能基原子")
            for i, feat in enumerate(new_features):
                print(f"  官能基 {i+1}: {feat['group_type']} at 關鍵原子位置 {feat['key_atom_index']}")
        except Exception as e:
            print(f"新方法錯誤: {e}")

# 主函數
if __name__ == "__main__":
    compare_methods()
