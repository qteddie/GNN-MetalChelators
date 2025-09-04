import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import warnings
from tqdm import tqdm

# 添加正確的路徑
sys.path.append('src/')
from pka_prediction_kmeans import extract_molecule_feature, FUNCTIONAL_GROUPS_SMARTS

# 忽略RDKit的警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')
warnings.filterwarnings("ignore", category=FutureWarning)

def draw_molecule_with_pka(smiles, pka_values=None, output_file=None):
    """
    繪製分子結構，並標註pKa值（如果有）
    
    Args:
        smiles: 分子的SMILES字串
        pka_values: pKa值列表(可選)
        output_file: 輸出檔案路徑
    
    Returns:
        bool: 是否成功繪製
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # 生成2D座標
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        
        # 設置繪圖選項
        drawer = Draw.MolDraw2DCairo(400, 300)
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().addStereoAnnotation = True
        
        # 添加pKa標籤（如果有）
        if pka_values is not None:
            if isinstance(pka_values, (int, float)):
                pka_text = f"pKa = {pka_values}"
            else:
                pka_text = f"pKa = {pka_values}"
            drawer.AddText(pka_text, (0.05, 0.05))
        
        # 繪製分子
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        # 保存圖片
        if output_file:
            with open(output_file, 'wb') as f:
                f.write(drawer.GetDrawingText())
        
        return True
    except Exception as e:
        print(f"繪製SMILES時出錯 {smiles}: {str(e)}")
        return False

def draw_molecule_with_highlight(smiles, pattern, pka_values=None, output_file=None):
    """
    繪製分子結構，高亮顯示指定的子結構模式
    
    Args:
        smiles: 分子的SMILES字串
        pattern: 要高亮顯示的子結構模式(RDKit Mol對象)
        pka_values: pKa值列表(可選)
        output_file: 輸出檔案路徑
    
    Returns:
        bool: 是否成功繪製
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or pattern is None:
            return False
        
        # 生成2D座標
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        
        # 查找匹配的原子
        matches = mol.GetSubstructMatches(pattern)
        if not matches:
            # 如果沒有找到匹配，直接使用原函數繪製
            return draw_molecule_with_pka(smiles, pka_values, output_file)
        
        # 設置繪圖選項
        drawer = Draw.MolDraw2DCairo(400, 300)
        drawer.drawOptions().addAtomIndices = False
        drawer.drawOptions().addStereoAnnotation = True
        
        # 準備高亮原子和鍵
        highlight_atoms = []
        highlight_bonds = []
        highlight_atom_colors = {}
        highlight_bond_colors = {}
        
        # 收集所有匹配的原子
        for match in matches:
            for atom_idx in match:
                highlight_atoms.append(atom_idx)
                highlight_atom_colors[atom_idx] = (0.8, 0.2, 0.2)  # 紅色高亮
                
                # 處理與該原子相連的鍵
                atom = mol.GetAtomWithIdx(atom_idx)
                for bond in atom.GetBonds():
                    bond_idx = bond.GetIdx()
                    begin_atom = bond.GetBeginAtomIdx()
                    end_atom = bond.GetEndAtomIdx()
                    
                    # 如果鍵的兩端都在高亮范圍內，則高亮該鍵
                    if begin_atom in match and end_atom in match:
                        highlight_bonds.append(bond_idx)
                        highlight_bond_colors[bond_idx] = (0.8, 0.2, 0.2)  # 紅色高亮
        
        # 添加pKa標籤（如果有）
        if pka_values is not None:
            if isinstance(pka_values, (int, float)):
                pka_text = f"pKa = {pka_values}"
            else:
                pka_text = f"pKa = {pka_values}"
            drawer.AddText(pka_text, (0.05, 0.05))
        
        # 繪製高亮分子
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightBonds=highlight_bonds,
            highlightAtomColors=highlight_atom_colors,
            highlightBondColors=highlight_bond_colors
        )
        drawer.FinishDrawing()
        
        # 保存圖片
        if output_file:
            with open(output_file, 'wb') as f:
                f.write(drawer.GetDrawingText())
        
        return True
    except Exception as e:
        print(f"繪製高亮SMILES時出錯 {smiles}: {str(e)}")
        return False

def find_molecules_without_functional_groups(csv_file, output_dir):
    """
    找出CSV文件中無法識別官能基的分子，並將它們繪製存放到指定目錄
    
    Args:
        csv_file: 輸入的CSV文件路徑
        output_dir: 輸出圖像的目錄路徑
    """
    # 創建輸出目錄（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 讀取CSV文件
    df = pd.read_csv(csv_file)
    total_molecules = len(df)
    print(f"總共發現 {total_molecules} 個分子")
    
    # 準備統計資訊
    no_functional_group_count = 0
    no_functional_group_smiles = []
    
    # 使用tqdm創建進度條
    for index, row in tqdm(df.iterrows(), total=total_molecules, desc="分析分子"):
        smiles = row['SMILES']
        pka_value = row.get('pKa_value', None)
        
        # 提取官能基特徵
        features = extract_molecule_feature(smiles)
        
        # 如果沒有找到官能基
        if not features:
            no_functional_group_count += 1
            no_functional_group_smiles.append(smiles)
            
            # 生成輸出文件名
            output_file = os.path.join(output_dir, f"no_func_group_{index+1}.png")
            
            # 繪製分子結構
            success = draw_molecule_with_pka(smiles, pka_value, output_file)
            if not success:
                print(f"無法繪製 SMILES: {smiles}")
    
    # 輸出統計信息
    print("\n分析完成！")
    print(f"找到 {no_functional_group_count} 個無法識別官能基的分子")
    
    # 將無法識別官能基的SMILES保存到文件
    if no_functional_group_smiles:
        error_file = os.path.join(output_dir, "no_functional_group_smiles.txt")
        with open(error_file, "w") as f:
            f.write("無法識別官能基的SMILES列表：\n")
            for i, smiles in enumerate(no_functional_group_smiles):
                f.write(f"{i+1}. {smiles}\n")
        print(f"無法識別官能基的SMILES已保存到: {error_file}")
        
    return no_functional_group_smiles

def analyze_phosphonic_acid_molecules(csv_file, output_dir):
    """
    分析並繪製含有磷酸官能基的分子
    
    Args:
        csv_file: 輸入的CSV文件路徑
        output_dir: 輸出圖像的目錄路徑
    """
    # 創建輸出目錄（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 讀取CSV文件
    df = pd.read_csv(csv_file)
    total_molecules = len(df)
    print(f"總共發現 {total_molecules} 個分子")
    
    # 準備統計資訊
    phosphonic_count = 0
    phosphonic_smiles = []
    
    # 獲取磷酸模式
    phosphonic_pattern = FUNCTIONAL_GROUPS_SMARTS.get("PhosphonicAcid")
    if phosphonic_pattern is None:
        print("錯誤：無法獲取磷酸官能基模式")
        return []
    
    # 創建通用磷酸模式（嘗試匹配更廣泛的磷酸結構）
    generic_phosphonic_pattern = Chem.MolFromSmarts('[P]')
    
    # 使用tqdm創建進度條
    for index, row in tqdm(df.iterrows(), total=total_molecules, desc="分析磷酸分子"):
        smiles = row['SMILES']
        pka_value = row.get('pKa_value', None)
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            # 檢查是否包含任何磷原子
            has_phosphorus = False
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'P':
                    has_phosphorus = True
                    break
            
            if has_phosphorus:
                # 檢查是否包含標準磷酸官能基
                matches = mol.GetSubstructMatches(phosphonic_pattern)
                is_standard_phosphonic = len(matches) > 0
                
                # 儲存並繪製分子
                phosphonic_count += 1
                phosphonic_smiles.append((smiles, is_standard_phosphonic))
                
                # 生成輸出文件名
                prefix = "std_" if is_standard_phosphonic else "non_std_"
                output_file = os.path.join(output_dir, f"{prefix}phosphonic_{index+1}.png")
                
                # 繪製分子結構，高亮磷酸部分
                if is_standard_phosphonic:
                    success = draw_molecule_with_highlight(smiles, phosphonic_pattern, pka_value, output_file)
                else:
                    success = draw_molecule_with_highlight(smiles, generic_phosphonic_pattern, pka_value, output_file)
                
                if not success:
                    print(f"無法繪製磷酸SMILES: {smiles}")
        except Exception as e:
            print(f"分析SMILES時出錯 {smiles}: {str(e)}")
            continue
    
    # 輸出統計信息
    print("\n分析完成！")
    print(f"找到 {phosphonic_count} 個含有磷原子的分子")
    
    # 計算符合標準模式的數量
    std_count = sum(1 for _, is_std in phosphonic_smiles if is_std)
    print(f"其中 {std_count} 個符合標準磷酸模式，{phosphonic_count - std_count} 個不符合")
    
    # 將含磷酸官能基的SMILES保存到文件
    if phosphonic_smiles:
        phosphonic_file = os.path.join(output_dir, "phosphonic_acid_smiles.txt")
        with open(phosphonic_file, "w") as f:
            f.write("含磷酸官能基的SMILES列表：\n")
            for i, (smiles, is_std) in enumerate(phosphonic_smiles):
                status = "標準模式" if is_std else "非標準模式"
                f.write(f"{i+1}. {smiles} ({status})\n")
        print(f"含磷酸官能基的SMILES已保存到: {phosphonic_file}")
        
    return phosphonic_smiles

def test_specific_smiles():
    """
    測試特定的SMILES分子
    """
    test_dir = "../output/test_smiles"
    os.makedirs(test_dir, exist_ok=True)
    
    # 特定的SMILES用來測試
    test_cases = [
        "OCC(O)CO[P](O)(O)=O",  # 圖中所示的磷酸分子
        "CC[P](=O)(O)O",        # 另一個磷酸分子
        "O=P(O)(O)O",           # 磷酸
        "C[P](=O)(O)O"          # 甲基磷酸
    ]
    
    # 獲取磷酸模式
    phosphonic_pattern = FUNCTIONAL_GROUPS_SMARTS.get("PhosphonicAcid")
    
    # 測試自定義的模式
    custom_patterns = [
        ("標準模式", phosphonic_pattern),
        ("簡化模式-1", Chem.MolFromSmarts('P(=O)(O)(O)(O)')),
        ("簡化模式-2", Chem.MolFromSmarts('[P](=O)([O])([O])[O]')),
        ("簡化模式-3", Chem.MolFromSmarts('[P](=O)(O)(O)O')),
        ("簡化模式-4", Chem.MolFromSmarts('[PX4](=[OX1])([O])([O])([O])')),
    ]
    
    # 測試每個SMILES
    for i, smiles in enumerate(test_cases):
        print(f"\n測試 SMILES {i+1}: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            print(f"無法解析SMILES: {smiles}")
            continue
        
        # 測試每個模式
        for pattern_name, pattern in custom_patterns:
            if pattern is None:
                print(f"模式 {pattern_name} 無效")
                continue
                
            matches = mol.GetSubstructMatches(pattern)
            match_count = len(matches)
            print(f"模式 '{pattern_name}' 找到 {match_count} 個匹配")
            
            # 繪製高亮圖像
            if match_count > 0:
                output_file = os.path.join(test_dir, f"test_{i+1}_{pattern_name}.png")
                draw_molecule_with_highlight(smiles, pattern, None, output_file)
                print(f"圖像已保存到: {output_file}")

if __name__ == "__main__":
    # 設置輸入和輸出路徑
    csv_file = "../data/processed_pka_data.csv"
    no_func_output_dir = "../output/no_functional_group_images"
    phosphonic_output_dir = "../output/phosphonic_acid_images"
    
    # 測試特定SMILES
    print("===== 測試特定SMILES =====")
    test_specific_smiles()
    
    # 分析無法識別官能基的分子
    print("\n===== 分析無法識別官能基的分子 =====")
    no_func_smiles = find_molecules_without_functional_groups(csv_file, no_func_output_dir)
    
    # 分析含磷酸官能基的分子
    print("\n===== 分析含磷酸官能基的分子 =====")
    phosphonic_smiles = analyze_phosphonic_acid_molecules(csv_file, phosphonic_output_dir)
    
    print("\n所有分析完成!") 