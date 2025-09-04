# 格式： 
# Metal ion,SMILES,pKa_num, pKa_value
# Cu²⁺,NCC(=O)O, 2, "3,9"
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm  # 添加進度條

# 添加正確的路徑
sys.path.append('../src/')
# 從functional_group_labeler導入新的標記器
from functional_group_labeler import FunctionalGroupLabeler
# 忽略RDKit的警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')
warnings.filterwarnings("ignore", category=FutureWarning)


# --- 配置 ---
INPUT_CSV_PATH = '../data/Metal_t15_T25_I0.1_E3_m11.csv'  # 修正路徑
OUTPUT_CSV_PATH = '../data/pre_metal_t15_T25_I0.1_E3_m11.csv'  # 修正路徑
OUTPUT_COLUMNS = ['metal_ion', 'SMILES', 'pKa_num', 'pKa_value']


# --- 金屬離子數據處理邏輯 ---
def process_metal_data(input_path: str, output_path: str):
    """
    讀取金屬離子數據，過濾溫度和離子強度，
    創建一個新的CSV文件，包含metal ion資訊、SMILES、pKa數量和排序後的pKa值。
    確保對不同的metal_ion進行單獨處理。
    """
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"無法讀取文件: {e}")
        return

    # --- 驗證必要的列 ---
    required_cols = ['metal_ion', 'SMILES', 'Value', 'Equilibrium', 'Temperature', 'Ionic_strength']
    if not all(col in df.columns for col in required_cols):
        print(f"缺少必要的列: {[col for col in required_cols if col not in df.columns]}")
        return

    # --- 數據清理 ---
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.dropna(subset=['SMILES', 'Value', 'metal_ion'], inplace=True)

    # 過濾溫度和離子強度
    filtered_df = df[df['Temperature'] == 0.0]
    
    if filtered_df.empty:
        filtered_df = df
        print("警告: 沒有找到溫度為0的數據，使用所有數據")
    
    # 修改：按Metal ion、SMILES和Equilibrium分組，計算每個平衡類型的平均值
    grouped_eq = filtered_df.groupby(['metal_ion', 'SMILES', 'Equilibrium'])['Value'].mean().reset_index()
    
    # 然後按Metal ion和SMILES分組，獲取不同平衡類型的值列表
    grp = grouped_eq.groupby(['metal_ion', 'SMILES'])['Value'].agg(list).reset_index()
    
    # 排序pKa值的函數
    def aggregate_pka(values):
        return sorted(values)

    # 應用聚合函數
    grp['pKa_value'] = grp['Value'].apply(aggregate_pka)
    grp.drop('Value', axis=1, inplace=True)

    # 獲取每個分子的平衡類型數量
    eq_counts = filtered_df.groupby(['metal_ion', 'SMILES'])['Equilibrium'].nunique().reset_index()
    eq_counts.rename(columns={'Equilibrium': 'pKa_num'}, inplace=True)

    # 合併pKa列表與平衡類型數量
    df_final = pd.merge(eq_counts, grp, on=['metal_ion', 'SMILES'], how='inner')

    # 確保正確的列順序
    df_final = df_final[OUTPUT_COLUMNS]
    df_final = df_final.sort_values(by=['metal_ion', 'pKa_num'])

    count = 0
    # 確保pKa_value的數量與pKa_num一致，如果不足則記錄並刪除
    for index, row in df_final.iterrows():
        if len(row['pKa_value']) < row['pKa_num']:
            df_final = df_final.drop(index)
            count += 1
    print(f"刪除 {count} 個pKa_value數量不足的分子")
    
    # --- 寫入輸出 ---
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)
        print(f"處理後的數據已保存到 {output_path}")
    except Exception as e:
        print(f"無法保存文件: {e}")
    
    # 收集不同解離態的binding constants數據
    # 用於繪圖
    return df_final


def plot_binding_constant_by_eq(df, output_dir):
    """
    為每種金屬離子繪製不同解離態下的binding constant分佈圖
    
    Args:
        df: 處理後的DataFrame，包含metal_ion, pKa_num, pKa_value等信息
        output_dir: 輸出圖像的目錄路徑
    """
    # 創建輸出目錄
    plot_dir = os.path.join(output_dir, "eq_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    # 獲取所有不同的金屬離子
    unique_metal_ions = df['metal_ion'].unique()
    print(f"開始為 {len(unique_metal_ions)} 種金屬離子繪製解離態分佈圖...")
    
    # 直接設置Matplotlib參數
    plt.rcParams.update({
        'figure.figsize': (16, 10),
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # 為每種金屬離子生成一個單獨的圖表
    for metal_ion in unique_metal_ions:
        # 篩選特定金屬離子的數據
        metal_df = df[df['metal_ion'] == metal_ion]
        
        # 確定這種金屬離子的最大pKa_num
        max_pka_num = metal_df['pKa_num'].max()
        
        if max_pka_num == 0:
            print(f"警告: {metal_ion} 沒有有效的pKa數據，跳過繪圖")
            continue
        
        # 創建一個圖形和軸
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # 為每個解離態準備數據
        eq_data = [[] for _ in range(max_pka_num)]
        eq_labels = [f"Eq{i+1}" for i in range(max_pka_num)]
        eq_counts = [0] * max_pka_num
        
        # 收集每個解離態的binding constants
        for _, row in metal_df.iterrows():
            pka_num = row['pKa_num']
            pka_values = row['pKa_value']
            
            # 檢查pka_values是否為字符串，如果是則轉換為列表
            if isinstance(pka_values, str):
                try:
                    # 假設格式為 [1.2, 3.4, 5.6]
                    if pka_values.startswith('[') and pka_values.endswith(']'):
                        pka_values = pka_values.strip('[]').split(',')
                    # 假設格式為 "1.2,3.4,5.6"
                    else:
                        pka_values = pka_values.split(',')
                    
                    pka_values = [float(val.strip()) for val in pka_values]
                except:
                    print(f"警告: 無法解析pKa值 '{pka_values}'，跳過")
                    continue
            
            # 確保pka_values是列表且長度符合pka_num
            if not isinstance(pka_values, list):
                pka_values = [pka_values]
            
            if len(pka_values) < pka_num:
                print(f"警告: pKa值數量({len(pka_values)})小於pKa_num({pka_num})，跳過")
                continue
            
            # 將每個解離態的值加入相應的列表
            for i in range(pka_num):
                eq_data[i].append(pka_values[i])
                eq_counts[i] += 1
        
        # 檢查是否有足夠的數據繪圖
        if all(len(data) == 0 for data in eq_data):
            print(f"警告: {metal_ion} 沒有有效的解離態數據，跳過繪圖")
            continue
        
        # 繪製箱形圖
        positions = list(range(1, max_pka_num + 1))
        box_plot = ax.boxplot([data for data in eq_data if len(data) > 0], 
                             positions=[pos for pos, data in zip(positions, eq_data) if len(data) > 0],
                             patch_artist=True, 
                             labels=[label for label, data in zip(eq_labels, eq_data) if len(data) > 0])
        
        # 美化箱形圖顏色
        colors = plt.cm.tab10(np.linspace(0, 1, len([data for data in eq_data if len(data) > 0])))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        # 添加數據點
        for i, data in enumerate([data for data in eq_data if len(data) > 0]):
            # 在箱形圖上稍微偏移以避免重疊
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax.scatter(x, data, alpha=0.5, s=20, color='black')
        
        # 添加標題和標籤
        ax.set_title(f'Distribution of Binding Constants for {metal_ion} by Equilibrium State', fontsize=16)
        ax.set_ylabel('Binding Constant Value', fontsize=14)
        
        # 將x軸標題移到最下方，避免與數據重疊
        # 首先設置x軸標籤為空
        ax.set_xlabel('', fontsize=14)
        # 然後在適當的位置添加文字作為x軸標題
        # 獲取x軸範圍
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        # 計算適當的y位置 (在y軸最小值下方添加足夠空間)
        text_y_pos = y_min - (y_max - y_min) * 0.15
        
        # 添加x軸標題文字
        ax.text((x_min + x_max) / 2, text_y_pos, 'Equilibrium State', 
               ha='center', va='center', fontsize=14)
        
        # 增加底部空白以容納移動的x軸標題
        plt.subplots_adjust(bottom=0.15)
        
        # 添加每組數據的樣本數量
        valid_counts = [count for count, data in zip(eq_counts, eq_data) if len(data) > 0]
        for i, count in enumerate(valid_counts):
            ax.text(i+1, y_min + (y_max - y_min)*0.05, f"n={count}", 
                   ha='center', va='center', fontsize=12)
        
        # 添加每組數據的平均值和中位數
        valid_data = [data for data in eq_data if len(data) > 0]
        for i, values in enumerate(valid_data):
            mean_val = np.mean(values)
            median_val = np.median(values)
            ax.text(i+1, y_min - (y_max - y_min)*0.07, 
                   f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}", 
                   ha='center', va='top', fontsize=10)
        
        # 調整布局和保存圖像
        plt.tight_layout()
        output_path = os.path.join(plot_dir, f"{metal_ion.replace('²⁺', '2+').replace('³⁺', '3+').replace('+', 'p')}_eq_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"為 {metal_ion} 生成的解離態分佈圖已保存到 {output_path}")
    
    print(f"所有金屬離子的解離態分佈圖已保存到 {plot_dir}")

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
        metal_ion = row['metal_ion']  # 記錄金屬離子信息
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
                
                # 根據SMILES和metal_ion過濾dataframe獲取相應數據
                df_filtered = df[(df['SMILES'] == smiles) & (df['metal_ion'] == metal_ion)]
                
                if df_filtered.empty:
                    print(f"\n警告: 無法找到SMILES: {smiles}與金屬離子: {metal_ion}的對應記錄")
                    error_count += 1
                    error_smiles.append(f"{smiles} (metal_ion: {metal_ion})")
                    continue
                
                # 不再生成臨時CSV文件，而是直接處理數據
                pka_values = []
                # 從row中獲取pKa值
                if isinstance(row['pKa_value'], str):
                    try:
                        # 解析pKa字符串
                        if row['pKa_value'].startswith('[') and row['pKa_value'].endswith(']'):
                            pka_str = row['pKa_value'].strip('[]').split(',')
                        else:
                            pka_str = row['pKa_value'].split(',')
                        pka_values = [float(val.strip()) for val in pka_str]
                    except:
                        print(f"\n警告: 無法解析pKa值: {row['pKa_value']}")
                        error_count += 1
                        error_smiles.append(smiles)
                        continue
                
                # 使用解析後的pKa值創建結果對象
                results = {
                    'pka_values': pka_values,
                    'functional_groups': functional_groups,
                    'mapping': {}
                }
                
                # 嘗試映射pKa值到官能團
                if len(functional_groups) < len(pka_values):
                    print(f"\n警告: SMILES {smiles} 的官能基數量({len(functional_groups)})小於pKa數量({len(pka_values)})")
                    insufficient_fg_count += 1
                    insufficient_fg_smiles.append(smiles)
                    error_count += 1
                    error_smiles.append(smiles)
                    continue
                
                # 簡單映射: 按官能團順序分配pKa值
                for i, pka in enumerate(pka_values):
                    if i < len(functional_groups):
                        fg = functional_groups[i]
                        results['mapping'][str(pka)] = {
                            'indices': [fg['key_atom_index']],
                            'type': fg['group_type']
                        }
                
                if results:
                    # 生成pKa矩陣表示
                    pka_matrix = []
                    for pka_value, group_info in results['mapping'].items():
                        for atom_idx in group_info['indices']:
                            pka_matrix.append((atom_idx, float(pka_value)))
                    
                    # 將結果添加到列表中，加入metal_ion信息
                    result_dict = {
                        'metal_ion': metal_ion,
                        'smiles': smiles,
                        'pka_values': results['pka_values'],
                        'functional_groups': [{'type': group['group_type'], 'indices': [group['key_atom_index']]} 
                                             for group in functional_groups[:len(pka_values)]],
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
        csv_output_path = os.path.join(output_dir, "Metal_pka_mapping_results.csv")
        results_df.to_csv(csv_output_path, index=False)
        print(f"結果已保存到: {csv_output_path}")
    
    # 輸出統計信息
    print("\n處理完成！")
    print(f"成功處理: {success_count} 個分子")
    print(f"處理失敗: {error_count} 個分子")
    print(f"官能基數量不足: {insufficient_fg_count} 個分子")
    
    # 將錯誤的SMILES保存到文件
    if error_smiles:
        error_file = os.path.join(output_dir, "Metal_error_smiles.txt")
        with open(error_file, "w") as f:
            f.write("處理失敗的SMILES列表：\n")
            for smiles in error_smiles:
                f.write(f"{smiles}\n")
        print(f"失敗的SMILES已保存到: {error_file}")
    
    # 新增: 將官能基數量不足的SMILES保存到文件
    if insufficient_fg_smiles:
        insufficient_file = os.path.join(output_dir, "Metal_insufficient_fg_smiles.txt")
        with open(insufficient_file, "w") as f:
            f.write("官能基數量不足的SMILES列表：\n")
            for smiles in insufficient_fg_smiles:
                f.write(f"{smiles}\n")
        print(f"官能基數量不足的SMILES已保存到: {insufficient_file}")
    
    return results_list

if __name__ == "__main__":
    # 設置輸入和輸出路徑
    csv_file = "../data/pre_metal_t15_T25_I0.1_E3_m11.csv"  # 修正路徑
    output_dir = "../output/Metal"  # 修正路徑
    
    # 處理金屬離子數據
    print("開始處理金屬離子數據...")
    result_df = process_metal_data(INPUT_CSV_PATH, OUTPUT_CSV_PATH) 
    print("金屬離子數據處理完成")
    
    # 繪製metal ion與equilibrium分佈圖
    print("開始繪製金屬離子解離態分佈圖...")
    plot_binding_constant_by_eq(result_df, output_dir)
    print("金屬離子解離態分佈圖繪製完成")
    
    # 對處理後的數據進行官能團標記和pKa映射
    print("開始進行官能團標記和pKa映射...")
    results_data = process_all_molecules(csv_file, output_dir) 
    print("官能團標記和pKa映射完成") 