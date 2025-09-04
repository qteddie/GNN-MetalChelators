#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pka_functional_group_analysis.py

此腳本用於分析CSV檔案中每個官能基對應的pKa分佈，並繪製箱型圖和直方圖。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import warnings


# 添加正確的路徑
sys.path.append('src/')
import importlib
import pka_smiles_query
importlib.reload(pka_smiles_query)
from pka_smiles_query import map_pka_to_functional_groups

# 忽略RDKit的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 設置中文字體支援（如果有需要）
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def analyze_functional_group_pka_distribution(csv_file, output_dir):
    """
    分析每個官能基對應的pKa分佈
    
    Args:
        csv_file: 輸入的CSV文件路徑
        output_dir: 輸出圖像的目錄路徑
    """
    # 創建輸出目錄（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 讀取CSV文件
    df = pd.read_csv(csv_file)
    print(f"從CSV檔案載入了 {len(df)} 筆數據")
    
    # 準備存儲每個官能基對應的pKa值
    functional_group_pka = defaultdict(list)
    
    # 處理每個分子
    print("正在分析每個分子的官能基與pKa關係...")
    
    functional_group_count_smaller_than_pka_count_list = []
    
    # 使用tqdm創建進度條
    for index, row in tqdm(df.iterrows(), total=len(df), desc="分析進度"):
        smiles = row['SMILES']
        
        try:
            # 映射pKa值到官能基
            results = map_pka_to_functional_groups(smiles, csv_file)
            
            # 這邊對於pka數量大於找到官能基的數量，需要進行處理
            if len(results['pka_values']) > len(results['functional_groups']):
                functional_group_count_smaller_than_pka_count_list.append(smiles)
                continue
            
            if results and 'mapping' in results:
                # 處理每個pKa與對應的官能基
                for pka, group_info in results['mapping'].items():
                    group_type = group_info['type']
                    # 移除可能的猜測標記
                    if " (guess)" in group_type:
                        group_type = group_type.replace(" (guess)", "")
                    
                    # 將pKa值添加到相應的官能基列表中
                    functional_group_pka[group_type].append(float(pka))
        except Exception as e:
            print(f"\n處理SMILES時出錯 {smiles}: {str(e)}")
            continue
    
    # 過濾掉少於5個數據點的官能基
    filtered_data = {k: v for k, v in functional_group_pka.items() if len(v) >= 5}
    
    print(f"pka數量大於找到官能基的數量(error): {len(functional_group_count_smaller_than_pka_count_list)}")
    with open(os.path.join(output_dir, "functional_group_count_smaller_than_pka_count_list.txt"), "w") as f:
        for smiles in functional_group_count_smaller_than_pka_count_list:
            f.write(f"{smiles}\n")
    print(f"已儲存到 {os.path.join(output_dir, 'functional_group_count_smaller_than_pka_count_list.txt')}")
    
    print(f"找到了 {len(filtered_data)} 種有效官能基進行分析")
    
    # 輸出每個官能基的數據點數量和pKa範圍
    print("\n每個官能基的統計資訊:")
    
    # 新增：創建統計資訊檔案
    stats_file_path = os.path.join(output_dir, "functional_group_stats.csv")
    
    # 創建統計數據列表
    stats_data = []
    
    for group, pka_values in filtered_data.items():
        # 計算統計數據
        count = len(pka_values)
        min_val = min(pka_values)
        max_val = max(pka_values)
        mean_val = np.mean(pka_values)
        median_val = np.median(pka_values)
        std_val = np.std(pka_values)
        # 輸出到控制台
        print(f"{group:<20}: {count:>4} 個數據點, pKa範圍: {min_val:>5.2f} - {max_val:<5.2f}, 均值: {mean_val:>6.2f}, 中位數: {median_val:>6.2f}")
        # 添加到數據列表
        stats_data.append({
            'Functional Group': group,
            'Data Points': count,
            'pKa_Min': min_val,
            'pKa_Max': max_val,
            'pKa_Range': max_val - min_val,
            'pKa_Mean': mean_val,
            'pKa_Median': median_val,
            'pKa_StdDev': std_val
        })
    
    # 創建DataFrame並保存為CSV
    stats_df = pd.DataFrame(stats_data)
    # 按數據點數量排序
    stats_df = stats_df.sort_values(by='Data Points', ascending=False)
    stats_df.to_csv(stats_file_path, index=False, encoding='utf-8-sig')
    print(f"\nStatistics saved to: {stats_file_path}")
    
    # 1. 繪製箱型圖
    draw_boxplot(filtered_data, output_dir)
    
    # 2. 繪製分佈直方圖
    draw_histograms(filtered_data, output_dir)
    
    # 3. 繪製小提琴圖
    draw_violinplot(filtered_data, output_dir)
    
    # 4. 繪製組合圖 (箱型圖 + 散點圖)
    draw_combined_plot(filtered_data, output_dir)
    
    print(f"分析完成！所有圖形已保存到 {output_dir}")

def draw_boxplot(data, output_dir):
    """繪製官能基pKa分佈的箱型圖"""
    plt.figure(figsize=(14, 8))
    
    # 創建整理好的數據框
    all_data = []
    for group, pka_values in data.items():
        for pka in pka_values:
            all_data.append({'Functional Group': group, 'pKa': pka})
    
    df = pd.DataFrame(all_data)
    
    # 按照中位數排序
    median_order = df.groupby('Functional Group')['pKa'].median().sort_values().index
    
    # 繪製箱型圖
    sns.boxplot(x='Functional Group', y='pKa', data=df, order=median_order, palette='viridis')
    
    # 添加數據點
    sns.stripplot(x='Functional Group', y='pKa', data=df, order=median_order, 
                 size=4, color='.3', alpha=0.5)
    
    # 格式化圖表
    plt.title('Distribution of pKa Values by Functional Group', fontsize=16)
    plt.xlabel('Functional Group', fontsize=14)
    plt.ylabel('pKa Value', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存圖表
    plt.savefig(os.path.join(output_dir, 'functional_group_pka_boxplot.png'), dpi=300)
    plt.close()

def draw_histograms(data, output_dir):
    """繪製官能基pKa分佈的直方圖"""
    # 計算需要的子圖數量
    n = len(data)
    if n <= 3:
        rows, cols = 1, n
    else:
        cols = 3
        rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    
    # 將軸轉換為1D數組以便迭代
    if n > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # 獲取所有pKa值的範圍以設置統一的x軸範圍
    all_pka = [pka for pka_list in data.values() for pka in pka_list]
    min_pka = min(all_pka) - 1
    max_pka = max(all_pka) + 1
    
    # 為每個官能基繪製直方圖
    for i, (group, pka_values) in enumerate(data.items()):
        if i < len(axes):
            ax = axes[i]
            
            # 繪製帶有KDE的直方圖
            sns.histplot(pka_values, kde=True, ax=ax, bins=20, color='skyblue', edgecolor='black')
            
            # 添加垂直線表示均值和中位數
            mean_pka = np.mean(pka_values)
            median_pka = np.median(pka_values)
            ax.axvline(mean_pka, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_pka:.2f}')
            ax.axvline(median_pka, color='green', linestyle='--', linewidth=2, label=f'Median: {median_pka:.2f}')
            
            # 設置標題和標籤
            ax.set_title(f'{group} (n={len(pka_values)})')
            ax.set_xlabel('pKa Value')
            ax.set_ylabel('Frequency')
            ax.set_xlim(min_pka, max_pka)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
    
    # 隱藏未使用的子圖
    for i in range(len(data), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'functional_group_pka_histograms.png'), dpi=300)
    plt.close()

def draw_violinplot(data, output_dir):
    """繪製官能基pKa分佈的小提琴圖"""
    plt.figure(figsize=(14, 8))
    
    # 創建整理好的數據框
    all_data = []
    for group, pka_values in data.items():
        for pka in pka_values:
            all_data.append({'Functional Group': group, 'pKa': pka})
    
    df = pd.DataFrame(all_data)
    
    # 按照中位數排序
    median_order = df.groupby('Functional Group')['pKa'].median().sort_values().index
    
    # 繪製小提琴圖
    sns.violinplot(x='Functional Group', y='pKa', data=df, order=median_order, palette='muted', 
                  inner='box', scale='width')
    
    # 格式化圖表
    plt.title('Distribution of pKa Values by Functional Group', fontsize=16)
    plt.xlabel('Functional Group', fontsize=14)
    plt.ylabel('pKa Value', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # 保存圖表
    plt.savefig(os.path.join(output_dir, 'functional_group_pka_violinplot.png'), dpi=300)
    plt.close()

def draw_combined_plot(data, output_dir):
    """繪製組合圖表(箱型圖與散點圖)"""
    plt.figure(figsize=(20, 10))
    
    # 創建包含所有數據的DataFrame
    all_data = []
    colors = {}
    
    # 準備顏色映射
    color_palette = sns.color_palette("hsv", len(data))
    for i, group in enumerate(data.keys()):
        colors[group] = color_palette[i]
    
    # 準備數據
    for group, pka_values in data.items():
        count = len(pka_values)
        mean_pka = np.mean(pka_values)
        median_pka = np.median(pka_values)
        std_pka = np.std(pka_values)
        
        for pka in pka_values:
            all_data.append({
                'Functional Group': group,
                'pKa': pka,
                'Count': count,
                'Mean': mean_pka,
                'Median': median_pka,
                'StdDev': std_pka
            })
    
    df = pd.DataFrame(all_data)
    
    # 按照中位數排序
    group_stats = df.groupby('Functional Group')['pKa'].agg(['median', 'count']).reset_index()
    group_stats = group_stats.sort_values('median')
    ordered_groups = group_stats['Functional Group'].tolist()
    
    # 繪製箱型圖
    ax = sns.boxplot(x='Functional Group', y='pKa', data=df, order=ordered_groups, 
                    palette='pastel', width=0.5, fliersize=0)
    
    # 在箱型圖上添加散點圖，使用抖動
    sns.stripplot(x='Functional Group', y='pKa', data=df, order=ordered_groups,
                 size=5, alpha=0.5, palette='dark', jitter=True)
    
    # 儲存每個官能基的箱型圖上下四分位數(Q1, Q3)
    boxplot_stats = []
    
    # 添加數據標籤
    for i, group in enumerate(ordered_groups):
        group_data = df[df['Functional Group'] == group]
        count = group_data['Count'].iloc[0]
        mean = group_data['Mean'].iloc[0]
        median = group_data['Median'].iloc[0]
        
        # 計算箱型圖的上下四分位數
        q1 = np.percentile(group_data['pKa'], 25)
        q3 = np.percentile(group_data['pKa'], 75)
        iqr = q3 - q1
        
        # 儲存統計資料
        boxplot_stats.append({
            'Functional Group': group,
            'Count': count,
            'Q1 (25%)': q1,
            'Median (50%)': median,
            'Q3 (75%)': q3,
            'IQR': iqr,
            'Mean': mean,
            'Min': group_data['pKa'].min(),
            'Max': group_data['pKa'].max()
        })
        
        # 在箱型圖上方添加數據點數量
        plt.text(i, df['pKa'].max() + 0.5, f"n={count}", 
                ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # 在箱型圖下方添加均值和中位數
        plt.text(i, df['pKa'].min() - 1.0, f"Mean: {mean:.2f}\nMedian: {median:.2f}", 
                ha='center', va='top', fontsize=12)
    
    # 將箱型圖統計資料儲存為CSV
    boxplot_stats_df = pd.DataFrame(boxplot_stats)
    boxplot_stats_path = os.path.join(output_dir, 'functional_group_boxplot_stats.csv')
    boxplot_stats_df.to_csv(boxplot_stats_path, index=False, encoding='utf-8-sig')
    print(f"箱型圖統計資料已儲存至: {boxplot_stats_path}")
    
    # 格式化圖表
    plt.title('Distribution of pKa Values by Functional Group', fontsize=18, pad=20)
    plt.xlabel('Functional Group', fontsize=16)
    plt.ylabel('pKa Value', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=16)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 增加y軸範圍以容納標籤
    ymin, ymax = plt.ylim()
    plt.ylim(ymin - 2, ymax + 1)
    
    plt.tight_layout()
    
    # 保存圖表
    plt.savefig(os.path.join(output_dir, 'functional_group_pka_combined_plot.png'), dpi=300)
    plt.close()


if __name__ == "__main__":
    # 設置輸入和輸出路徑
    csv_file = "../data/processed_pka_data.csv"
    output_dir = "../src/pka_distribution"
    
    # 分析官能基pKa分佈
    analyze_functional_group_pka_distribution(csv_file, output_dir) 