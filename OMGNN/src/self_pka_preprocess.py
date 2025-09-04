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

# import ast 
# df['pKa_value'] = df['pKa_value'].apply(ast.literal_eval)
# 添加正確的路徑
sys.path.append('src/')
# 從functional_group_labeler導入新的標記器
from functional_group_labeler import FunctionalGroupLabeler
from pka_smiles_query import map_pka_to_functional_groups, display_pka_mapping_results, visualize_pka_mapping
# 忽略RDKit的警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Configuration ---
INPUT_CSV_PATH = '../data/NIST_database_onlyH_6TypeEq_pos_match_max_fg_other.csv'
OUTPUT_CSV_PATH = '../data/processed_pka_data.csv'
OUTPUT_COLUMNS = ['SMILES', 'pKa_num', 'pKa_value']


# --- Main Processing Logic ---
def process_pka_data(input_path: str, output_path: str):
    """
    Reads the NIST pKa data, filters by temperature (25°C) and ionic strength (0.1),
    and creates a new CSV with max_eq_num and sorted pKa values as list.
    """
    try:
        df = pd.read_csv(input_path)
    except:
        print(f"Error: Failed to read input data from {input_path}")
        return

    # --- Validate required columns ---
    required_cols = ['SMILES', 'Equilibrium', 'Value', 'max_eq_num', 'Temperature (C)', 'Ionic strength']
    if not all(col in df.columns for col in required_cols):
        return

    # --- Data Cleaning ---
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.dropna(subset=['SMILES', 'Value'], inplace=True)

    # Filter data by temperature and ionic strength
    filtered_df = df[(df['Temperature (C)'] == 25) & (df['Ionic strength'] == 0.1)]
    
    if filtered_df.empty:
        filtered_df = df
    
    # 修改：按SMILES和Equilibrium分組，計算每個平衡類型的平均值
    grouped_eq = filtered_df.groupby(['SMILES', 'Equilibrium'])['Value'].mean().reset_index()
    
    # 然後按SMILES分組，獲取不同平衡類型的值列表
    grp = grouped_eq.groupby('SMILES')['Value'].agg(list).reset_index()
    
    # Function to sort pKa values
    def aggregate_pka(values):
        return sorted(values)

    # Apply the aggregation function
    grp['pKa_value'] = grp['Value'].apply(aggregate_pka)
    grp.drop('Value', axis=1, inplace=True)

    # Get the max_eq_num (should be consistent per SMILES, take the first)
    df_max_eq = filtered_df.groupby('SMILES', as_index=False)['max_eq_num'].first()
    try:
        df_max_eq['max_eq_num'] = df_max_eq['max_eq_num'].astype(int)
    except:
        print(f"Error: Failed to convert max_eq_num to int")
        pass
    df_max_eq.rename(columns={'max_eq_num': 'pKa_num'}, inplace=True)

    # Merge the aggregated pKa lists with the max_eq_num
    df_final = pd.merge(df_max_eq, grp, on='SMILES', how='inner')

    # Ensure correct column order
    df_final = df_final[OUTPUT_COLUMNS]
    df_final = df_final.sort_values(by='pKa_num')

    count = 0
    # 確保pKa_value的數量與pKa_num一致，如果不足則記錄並刪除
    for index, row in df_final.iterrows():
        if len(row['pKa_value']) < row['pKa_num']:
            df_final = df_final.drop(index)
            count += 1
            # print(f"警告: SMILES {row['SMILES']} 的pKa_value數量不足，應為{row['pKa_num']}，但只有{len(row['pKa_value'])}個")
    print(f"刪除 {count} 個pKa_value數量不足的分子")
    # --- Write Output ---
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except:
        print(f"Error: Failed to save processed data to {output_path}")
        pass

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
    
    # 新增: 記錄因功能基數量不足的錯誤
    insufficient_fg_count = 0
    insufficient_fg_smiles = []
    
    # 初始化FunctionalGroupLabeler
    labeler = FunctionalGroupLabeler()
    
    # 準備儲存結果的列表
    results_list = []
    
    # 使用tqdm創建進度條
    for index, row in tqdm(df.iterrows(), total=total_molecules, desc="處理分子"):
        smiles = row['SMILES']
        try:
            
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
                    # 新增: 檢查官能基數量是否小於pKa數量
                    if len(functional_groups) < len(results['pka_values']):
                        print(f"\n警告: SMILES {smiles} 的官能基數量({len(functional_groups)})小於pKa數量({len(results['pka_values'])})")
                        insufficient_fg_count += 1
                        insufficient_fg_smiles.append(smiles)
                        error_count += 1
                        error_smiles.append(smiles)
                        continue
                    
                    # 生成pKa矩陣表示
                    pka_matrix = []
                    for pka_value, group_info in results['mapping'].items():
                        for atom_idx in group_info['indices']:
                            pka_matrix.append((atom_idx, float(pka_value)))
                    
                    # 將結果添加到列表中
                    result_dict = {
                        'smiles': smiles,
                        'pka_values': results['pka_values'],
                        'functional_groups': [{'type': group['type'], 'indices': group['indices']} 
                                             for group in results['functional_groups']],
                        'mapping': {str(k): v for k, v in results['mapping'].items()},  # 將pKa值轉為字串作為key
                        'pka_matrix': pka_matrix
                    }
                    
                    results_list.append(result_dict)
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
    
    # 儲存結果到CSV文件
    if results_list:
        results_df = pd.DataFrame(results_list)
        
        # 將複雜的列轉為字符串表示
        for col in ['pka_values', 'functional_groups', 'mapping', 'pka_matrix']:
            results_df[col] = results_df[col].apply(lambda x: str(x))
        
        # 保存到CSV
        csv_output_path = os.path.join(output_dir, "pka_mapping_results.csv")
        results_df.to_csv(csv_output_path, index=False)
        print(f"結果已保存到: {csv_output_path}")
    
    # 輸出統計信息
    print("\n處理完成！")
    print(f"成功處理: {success_count} 個分子")
    print(f"處理失敗: {error_count} 個分子")
    print(f"官能基數量不足: {insufficient_fg_count} 個分子")
    
    # 將錯誤的SMILES保存到文件
    if error_smiles:
        error_file = os.path.join(output_dir, "error_smiles.txt")
        with open(error_file, "w") as f:
            f.write("處理失敗的SMILES列表：\n")
            for smiles in error_smiles:
                f.write(f"{smiles}\n")
        print(f"失敗的SMILES已保存到: {error_file}")
    
    # 新增: 將官能基數量不足的SMILES保存到文件
    if insufficient_fg_smiles:
        insufficient_file = os.path.join(output_dir, "insufficient_fg_smiles.txt")
        with open(insufficient_file, "w") as f:
            f.write("官能基數量不足的SMILES列表：\n")
            for smiles in insufficient_fg_smiles:
                f.write(f"{smiles}\n")
        print(f"官能基數量不足的SMILES已保存到: {insufficient_file}")
    
    return results_list

if __name__ == "__main__":
    # 設置輸入和輸出路徑
    csv_file = "../data/processed_pka_data.csv"
    output_dir = "../output"
    process_pka_data(INPUT_CSV_PATH, OUTPUT_CSV_PATH) 
    # 處理所有分子並獲取結果
    results_data = process_all_molecules(csv_file, output_dir)
    
    