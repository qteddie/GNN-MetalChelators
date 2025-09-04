import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
import warnings
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from tqdm import tqdm  # 添加進度條

# 添加正確的路徑
sys.path.append('src/')
# 從functional_group_labeler導入新的標記器
from functional_group_labeler import FunctionalGroupLabeler
from pka_smiles_query import map_pka_to_functional_groups, display_pka_mapping_results, visualize_pka_mapping
# 忽略RDKit的警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')
warnings.filterwarnings("ignore", category=FutureWarning)

def process_all_molecules(csv_file, output_dir):
    """
    處理CSV文件中的所有分子，為每個分子生成pKa映射圖
    
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
    
    # 創建進度條
    success_count = 0
    error_count = 0
    error_smiles = []
    
    # 初始化FunctionalGroupLabeler
    labeler = FunctionalGroupLabeler()
    
    # 使用tqdm創建進度條
    for index, row in tqdm(df.iterrows(), total=total_molecules, desc="處理分子"):
        smiles = row['SMILES']
        try:
            # 生成輸出文件名
            output_file = os.path.join(output_dir, f"molecule_{index+1}.png")
            
            # 標記分子官能基
            mol, atom_to_group = labeler.label_molecule(smiles)
            
            if mol and atom_to_group:
                # 使用新的函數進行映射和可視化
                # 將atom_to_group轉換為pka_smiles_query需要的格式
                functional_groups = []
                for atom_idx, group_type in atom_to_group.items():
                    atom = mol.GetAtomWithIdx(atom_idx)
                    functional_groups.append({
                        'group_type': group_type,
                        'match_indices': [atom_idx],
                        'key_atom_index': atom_idx
                    })
                
                # 映射pKa值，使用轉換後的functional_groups
                results = map_pka_to_functional_groups(smiles, csv_file, functional_groups)
                
                if results:
                    # 設置matplotlib後端為Agg以避免顯示圖形
                    plt.switch_backend('Agg')
                    
                    # 生成可視化結果
                    visualize_pka_mapping(results, output_file)
                    
                    success_count += 1
                else:
                    print(f"\n警告: 無法處理SMILES: {smiles}")
                    error_count += 1
                    error_smiles.append(smiles)
            else:
                print(f"\n警告: 無法標記SMILES中的官能基: {smiles}")
                error_count += 1
                error_smiles.append(smiles)
                
        except Exception as e:
            print(f"\n處理SMILES時出錯 {smiles}: {str(e)}")
            error_count += 1
            error_smiles.append(smiles)
            continue
    
    # 輸出統計信息
    print("\n處理完成！")
    print(f"成功處理: {success_count} 個分子")
    print(f"處理失敗: {error_count} 個分子")
    
    # 將錯誤的SMILES保存到文件
    if error_smiles:
        error_file = os.path.join(output_dir, "error_smiles.txt")
        with open(error_file, "w") as f:
            f.write("處理失敗的SMILES列表：\n")
            for smiles in error_smiles:
                f.write(f"{smiles}\n")
        print(f"失敗的SMILES已保存到: {error_file}")

if __name__ == "__main__":
    # 設置輸入和輸出路徑
    csv_file = "../data/processed_pka_data.csv"
    output_dir = "../output/pka_mapping_images"
    
    # 處理所有分子
    process_all_molecules(csv_file, output_dir)
    
    