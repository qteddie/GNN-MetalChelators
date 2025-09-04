# 格式： 
# Metal ion,SMILES,pKa_num, pKa_value
# Cu²⁺,NCC(=O)O, 2, "3,9"
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# --- 配置 ---
INPUT_CSV_PATH = '../data/pre_metal_Eq6_Mnum10.csv'
OUTPUT_CSV_PATH = '../data/pre_metal_Eq6_Mnum10_temp_ion.csv'
# OUTPUT_COLUMNS = ['metal_ion', 'SMILES', 'pKa_num', 'pKa_value']
OUTPUT_COLUMNS = [
    "metal_ion",          # ⬅ 改小寫
    "SMILES",
    "pKa_num",
    "pKa_value",
    "Temperature (C)",
    "Ionic strength"
]
# --- 主要處理邏輯 ---
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
        return None

    required_cols = [
        "metal_ion", "SMILES", "Equilibrium",
        "Temperature (C)", "Ionic strength"        # ← h 補回來
    ]
    if not all(col in df.columns for col in required_cols):
        missing = [c for c in required_cols if c not in df.columns]
        raise KeyError(f"缺少必要欄位: {missing}")
    
    # ❷ 直接一次 groupby 四個欄位
    final_results = []
    key_cols = ["Metal ion", "SMILES", "Temperature (C)", "Ionic strength"]

    for (metal_ion, smiles, temp, ionic), grp in df.groupby(key_cols):
        vals = sorted(grp["Value"].tolist())

        final_results.append({
            "metal_ion"        : metal_ion,
            "SMILES"           : smiles,
            "pKa_num"          : grp["Equilibrium"].nunique(),   # 或 len(vals)
            "pKa_value"        : ",".join(f"{v:.2f}" for v in vals),
            "Temperature (C)"  : float(temp),                    # 保證唯一
            "Ionic strength"   : float(ionic)
        })

    # ❸ 輸出
    df_final = (pd.DataFrame(final_results)[OUTPUT_COLUMNS]
                  .sort_values(by=["metal_ion", "SMILES", "Temperature (C)", "Ionic strength"]))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    print(f"已保存 {len(df_final)} 筆資料至 {output_path}")

    return df_final

def plot_binding_constant_distribution(metal_binding_constants, output_dir):
    """
    繪製金屬離子結合常數分佈圖
    
    Args:
        metal_binding_constants: 每種金屬離子的結合常數值字典
        output_dir: 輸出圖像的目錄路徑
    """
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 直接設置Matplotlib參數，不使用預設樣式
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
    
    # 創建圖形和軸
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 準備數據
    data = []
    labels = []
    counts = []
    
    # 保持金屬離子順序一致
    metal_ions = sorted(metal_binding_constants.keys())
    
    for metal_ion in metal_ions:
        values = metal_binding_constants[metal_ion]
        if values:  # 確保有數據
            data.append(values)
            labels.append(metal_ion)
            counts.append(len(values))
    
    # 繪製箱形圖
    box_plot = ax.boxplot(data, patch_artist=True, labels=labels, showfliers=True)
    
    # 美化箱形圖顏色 - 使用不同的顏色生成方法
    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    # 添加數據點
    for i, d in enumerate(data):
        # 在箱形圖上稍微偏移以避免重疊
        x = np.random.normal(i+1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=20, color='black')
    
    # 添加標題和標籤
    ax.set_title('Distribution of Binding Constant Values by Metal Ion', fontsize=16)
    ax.set_ylabel('Binding Constant Value', fontsize=14)
    ax.set_xlabel('Metal Ion', fontsize=14)
    
    # 添加每組數據的樣本數量
    y_min, y_max = ax.get_ylim()
    for i, count in enumerate(counts):
        ax.text(i+1, y_min + (y_max - y_min)*0.05, f"n={count}", 
                ha='center', va='center', fontsize=12)
    
    # 添加每組數據的平均值和中位數
    for i, values in enumerate(data):
        mean_val = np.mean(values)
        median_val = np.median(values)
        ax.text(i+1, y_min - (y_max - y_min)*0.05, 
                f"Mean: {mean_val:.2f}\nMedian: {median_val:.2f}", 
                ha='center', va='top', fontsize=10)
    
    # 調整布局和保存圖像
    plt.tight_layout()
    output_path = os.path.join(output_dir, "metal_binding_constant_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"金屬離子結合常數分佈圖已保存到 {output_path}")

# --- 執行處理 ---
if __name__ == "__main__":
    # 創建輸出目錄
    output_dir = "/work/u5066474/MetalGCN/output/Metal"
    os.makedirs(output_dir, exist_ok=True)
    
    # 處理金屬離子數據並獲取結合常數
    metal_binding_constants = process_metal_data(INPUT_CSV_PATH, OUTPUT_CSV_PATH)
    
    # if metal_binding_constants:
    #     # 繪製結合常數分佈圖
    #     plot_binding_constant_distribution(metal_binding_constants, output_dir) 