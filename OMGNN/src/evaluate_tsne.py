#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate_tsne.py

該腳本用於載入已保存的t-SNE結果，並對給定的SMILES進行評估，
顯示其在t-SNE降維空間中的位置。
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from typing import List, Dict, Tuple, Optional
import warnings
from sklearn.cluster import KMeans
import math
# 添加正確的路徑
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)
sys.path.append('src/')
from pka_prediction_kmeans import extract_molecule_feature
# 忽略RDKit的警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')
warnings.filterwarnings("ignore", category=FutureWarning)
# 使用更具區分度的色彩方案，避免相近顏色
colors = [
    '#1E90FF',  # 道奇藍
    '#32CD32',  # 萊姆綠 
    '#9932CC',  # 深蘭紫色
    '#8B0000',  # 深紅色
    '#FFD700',  # 金色
    '#00CED1',  # 深青色
    '#FF1493',  # 深粉色
    '#006400',  # 深綠色
    '#4682B4',  # 鋼藍色
    '#2E8B57',  # 海洋綠
    '#FF4500',  # 橙紅色
    '#8A2BE2',  # 藍紫色
    '#FF6347',  # 番茄色
    '#40E0D0',  # 青綠色
    '#FFDAB9',  # 桃色
    '#ADFF2F',  # 黃綠色
    '#FF69B4',  # 熱情粉紅
    '#CD5C5C',  # 印度紅
    '#F0E68C',  # 卡其色
    '#B22222',  # 鮮紅色
    '#7FFF00',  # 查特茲綠
]  # 如果聚類數超過10，將循環使用這些顏色

# 為官能基類型創建標記形狀
group_markers = {
    'COOH': 'o',           # 圓形
    'SulfonicAcid': 's',   # 方形
    'PhosphonicAcid': 'p', # 五邊形
    'PrimaryAmine': '^',   # 上三角
    'SecondaryAmine': '<', # 左三角
    'TertiaryAmine': '>',  # 右三角
    'Phenol': 'D',         # 鑽石
    'Alcohol': 'v',        # 下三角
    'Imidazole': 'd',      # 薄鑽石
    'Pyridine': '*',       # 星號
    'Thiol': 'h',          # 六邊形
    # 'Nitro': 'x',          # 叉號
    'NHydroxylamine': 'x', # 加號
    'PyrazoleNitrogen': '*' # 星號
}
# 默認標記為圓形
default_marker = 'o'

def load_tsne_results(tsne_cache_file: str) -> dict:
    """
    載入已保存的t-SNE結果
    
    Args:
        tsne_cache_file: t-SNE結果缓存文件路徑
        
    Returns:
        包含t-SNE結果的字典
    """
    print(f"正在載入t-SNE結果: {tsne_cache_file}")
    try:
        with open(tsne_cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"成功載入t-SNE結果, 形狀: {cached_data['X_tsne'].shape}")
        print(f"原始維度: 樣本={cached_data['n_samples']}, 特徵={cached_data['n_features']}")
        print(f"聚類數量: k={cached_data['n_clusters']}")
        
        # 檢查是否有保存的聚類標籤
        if 'cluster_labels' in cached_data:
            print(f"已載入保存的聚類標籤，形狀: {cached_data['cluster_labels'].shape}")
            # 打印每個群集的樣本數量
            unique_labels, counts = np.unique(cached_data['cluster_labels'], return_counts=True)
            print("群集分布情況:")
            for label, count in zip(unique_labels, counts):
                percentage = count/len(cached_data['cluster_labels'])*100
                print(f"- 群集 {label}: {count} 個樣本 ({percentage:.1f}%)")
        
        return cached_data
    except Exception as e:
        print(f"載入t-SNE結果失敗: {e}")
        return None

def load_feature_data(data_path: str) -> tuple:
    """
    從原始數據中提取特徵和標籤信息
    
    Args:
        data_path: 原始數據CSV文件路徑
        
    Returns:
        特徵列表, SMILES列表, pKa值列表, 官能基類型列表
    """
    print(f"正在載入原始數據: {data_path}")
    df = pd.read_csv(data_path)
    print(f"原始數據行數: {len(df)}")
    
    # 處理可能包含多個 pKa 值的數據
    expanded_data = []
    
    # 使用進度計數器
    total_rows = len(df)
    print(f"處理 pKa 值和官能基特徵 (總共 {total_rows} 行)...")
    

 
    # 提取特徵
    all_features = []
    smiles_to_features = {}
    feature_groups = []
    feature_smiles = []
    y_pka = []
    
    # 使用更簡單的方法提取特徵
    for index, row in df.iterrows():
        if index % 100 == 0:
            print(f"  處理進度: {index}/{len(df)} 行")
            
        smiles = row['SMILES']
        try:
            pka = float(row['pKa_value'])
        except (ValueError, TypeError):
            continue
            
        feature_list = extract_molecule_feature(smiles)
        
        if feature_list:
            for feature_info in feature_list:
                all_features.append(feature_info['feature'])
                feature_groups.append(feature_info['group_type'])
                feature_smiles.append(smiles)
                y_pka.append(pka)
    
    print(f"提取了 {len(all_features)} 個官能基特徵來自 {len(set(feature_smiles))} 個唯一SMILES")
    
    # 確保數據格式正確
    X = np.array(all_features)
    y_pka = np.array(y_pka)
    
    return X, feature_smiles, y_pka, feature_groups

def get_radial_position(center_x, center_y, index, total, base_distance=10.0):
    """計算圓形排列的位置，將標籤以圓形方式分布在中心點周圍"""
    # 如果只有一個標籤，則直接放在右上方
    if total == 1:
        return center_x + base_distance, center_y + base_distance
    
    # 計算角度（均勻分布在圓上）
    angle = 2 * math.pi * index / total
    
    # 計算新位置（圓上的點）
    x = center_x + base_distance * math.cos(angle)
    y = center_y + base_distance * math.sin(angle)
    
    return x, y

def visualize_smiles_in_tsne(smiles_list: List[str], tsne_data: dict, 
                           feature_smiles: List[str], feature_groups: List[str],
                           output_filename: str = None):
    """
    在t-SNE圖中可視化指定SMILES的位置
    
    Args:
        smiles_list: 要可視化的SMILES列表
        tsne_data: t-SNE結果數據
        feature_smiles: 特徵對應的SMILES列表
        feature_groups: 特徵對應的官能基類型列表
        output_filename: 輸出文件名
    """
    X_tsne = tsne_data['X_tsne']
    n_clusters = tsne_data['n_clusters']
    
    # 檢查維度匹配
    if len(feature_groups) != len(X_tsne):
        print(f"警告: feature_groups 長度 ({len(feature_groups)}) 與 X_tsne 長度 ({len(X_tsne)}) 不匹配")
        print("使用pickle文件中保存的原始數據...")
        # 如果維度不匹配，只使用t-SNE結果中的數據，不使用新載入的數據
        if 'feature_groups' in tsne_data:
            feature_groups = tsne_data['feature_groups']
            print(f"使用保存的feature_groups，長度: {len(feature_groups)}")
        else:
            print("pickle文件中沒有保存feature_groups，僅顯示聚類而不顯示官能基...")
            feature_groups = ['Unknown'] * len(X_tsne)
    
    # 創建一個更寬的圖，為圖例預留足夠空間
    fig, ax = plt.subplots(figsize=(20, 16), dpi=300)
    
    # 檢查是否已有保存的聚類標籤
    if 'cluster_labels' in tsne_data:
        print("使用已保存的聚類標籤...")
        cluster_labels_tsne = tsne_data['cluster_labels']
    else:
        print("未找到保存的聚類標籤，重新聚類...")
        # 使用K-means重新聚類
        kmeans_tsne = KMeans(n_clusters=n_clusters, random_state=42, init='k-means++')
        cluster_labels_tsne = kmeans_tsne.fit_predict(X_tsne)
    
    # 為每個點分配一個顏色基於其群集
    point_colors = [colors[label % len(colors)] for label in cluster_labels_tsne]
    
    # 繪製背景點，使用分配的顏色
    for i in range(n_clusters):
        cluster_mask = (cluster_labels_tsne == i)
        if not np.any(cluster_mask):
            continue
        
        # 獲取該群集的點
        cluster_points = X_tsne[cluster_mask]
        cluster_groups = np.array(feature_groups)[cluster_mask]
        
        # 繪製群集點
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                 color=colors[i % len(colors)], 
                 label=f'Cluster {i} (n={np.sum(cluster_mask)})', 
                 alpha=0.5, s=40, edgecolors='w', linewidths=0.5)
        
        # 繪製群集中心
        center_x = np.mean(cluster_points[:, 0])
        center_y = np.mean(cluster_points[:, 1])
        ax.scatter(center_x, center_y, marker='D', s=120, 
                 color=colors[i % len(colors)], edgecolors='black', 
                 linewidths=1.5, zorder=10)
                 
        # 使用annotate替代text
        ax.annotate(f'{i}', 
                   xy=(center_x, center_y),
                   xytext=(center_x, center_y),
                   fontsize=14, fontweight='bold',
                   ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.8, 
                           edgecolor='none', boxstyle='round,pad=0.2'),
                   zorder=11)
        
        # 為每種官能基添加形狀標記
        for group in np.unique(cluster_groups):
            group_mask = cluster_groups == group
            if np.any(group_mask):
                marker = group_markers.get(group, default_marker)
                ax.scatter(cluster_points[group_mask][:, 0], 
                         cluster_points[group_mask][:, 1],
                         marker=marker, s=80, facecolors='none', 
                         edgecolors='black', linewidths=1.0, 
                         alpha=0.9, zorder=11)
    
    # 添加官能基圖例項目
    existing_groups = set(feature_groups)
    for group, marker in group_markers.items():
        if group in existing_groups:
            ax.scatter([], [], marker=marker, s=80, facecolors='none', 
                     edgecolors='black', linewidths=1.0, 
                     label=f'{group}', alpha=0.9)
    
    # 創建字典來追踪每個位置的標籤數量
    position_labels = {}  # 格式: {(x,y): [標籤數量, 當前索引]}

    # 對每個要評估的SMILES，提取特徵並找到對應的點
    for test_smiles in smiles_list:
        print(f"\n評估SMILES: {test_smiles}")
        target_features = extract_molecule_feature(test_smiles)
        
        if not target_features:
            print(f"  未找到任何官能基")
            continue
            
        print(f"  找到 {len(target_features)} 個官能基")
        
        # 檢查是否有任何官能基被找到
        any_match_found = False
        
        # 針對每個官能基，尋找對應的匹配點
        for feature_info in target_features:
            group_type = feature_info['group_type']
            print(f"  檢查官能基: {group_type}")
            
            # 在原始數據中查找匹配的SMILES和官能基類型
            matching_indices = [i for i, (s, g) in enumerate(zip(feature_smiles, feature_groups)) 
                              if s == test_smiles and g == group_type]
            
            if matching_indices:
                any_match_found = True
                print(f"  在t-SNE數據中找到 {len(matching_indices)} 個匹配點 (類型: {group_type})")
                
                # 為每個匹配點繪製大標記
                for idx in matching_indices:
                    marker = group_markers.get(group_type, default_marker)
                    cluster = cluster_labels_tsne[idx]
                    
                    # 用大號標記突出顯示這些點
                    ax.scatter(X_tsne[idx, 0], X_tsne[idx, 1], 
                             color='red', marker=marker, s=300, alpha=1.0,
                             edgecolors='yellow', linewidths=3.0, zorder=15,
                             label=f'{test_smiles}: {group_type}' if idx == matching_indices[0] else "")
                    
                    # 處理標籤位置（將位置四捨五入到1位小數作為字典鍵）
                    pos_key = (round(X_tsne[idx, 0], 1), round(X_tsne[idx, 1], 1))
                    
                    # 如果該位置尚未記錄，初始化
                    if pos_key not in position_labels:
                        position_labels[pos_key] = [1, 0]  # [總數, 當前索引]
                    else:
                        position_labels[pos_key][0] += 1  # 增加總數
                    
                    # 獲取當前標籤是該位置的第幾個
                    current_index = position_labels[pos_key][1]
                    total_at_position = position_labels[pos_key][0]
                    
                    # 計算放射狀位置
                    label_x, label_y = get_radial_position(
                        X_tsne[idx, 0], X_tsne[idx, 1], 
                        current_index, total_at_position, 
                        base_distance=50.0  # 調整基本距離
                    )
                    
                    # 更新索引，用於下一個相同位置的標籤
                    position_labels[pos_key][1] += 1
                    
                    # 使用計算出的位置進行標註
                    ax.annotate(f'{group_type}\n{test_smiles}',
                              xy=(X_tsne[idx, 0], X_tsne[idx, 1]),
                              xytext=(label_x, label_y),
                              fontsize=11, fontweight='bold',
                              color='black', ha='center', va='center',
                              zorder=16,
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'),
                              arrowprops=dict(
                                  arrowstyle='->', 
                                  color='black',     # 黑色箭頭
                                  lw=1.5,            # 增加線寬
                                  alpha=1.0,         # 完全不透明
                                  shrinkA=0,
                                  connectionstyle='arc3,rad=0.0',  # 直線連接
                                  mutation_scale=15  # 箭頭大小
                              ))
            else:
                print(f"  在t-SNE數據中未找到匹配的 {group_type} 官能基點")
                
                # 嘗試找到相似的官能基並估計位置
                print(f"  嘗試匹配相似的 {group_type} 官能基並估計位置...")
                
                # 尋找相同官能基類型的所有點
                similar_type_indices = [i for i, g in enumerate(feature_groups) if g == group_type]
                
                if similar_type_indices:
                    print(f"  找到 {len(similar_type_indices)} 個相同類型的官能基 ({group_type})")
                    
                    # 從相同類型中計算平均位置作為代表
                    similar_points = X_tsne[similar_type_indices]
                    representative_x = np.mean(similar_points[:, 0])
                    representative_y = np.mean(similar_points[:, 1])
                    
                    # 確定這個類型在哪個群集中最常見
                    cluster_counts = {}
                    for idx in similar_type_indices:
                        cluster = cluster_labels_tsne[idx]
                        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
                    
                    most_common_cluster = max(cluster_counts.items(), key=lambda x: x[1])[0]
                    print(f"  該官能基類型最常出現在群集 {most_common_cluster}")
                    
                    # 用特殊標記顯示這個新分子的位置
                    marker = group_markers.get(group_type, default_marker)
                    ax.scatter(representative_x, representative_y, 
                              color='red', marker=marker, s=300, alpha=0.7,
                              edgecolors='black', linewidths=2.0, zorder=15, 
                              label=f'{test_smiles}: {group_type} (estimated)')
                    
                    # 處理標籤位置
                    pos_key = (round(representative_x, 1), round(representative_y, 1))
                    
                    if pos_key not in position_labels:
                        position_labels[pos_key] = [1, 0]
                    else:
                        position_labels[pos_key][0] += 1
                    
                    current_index = position_labels[pos_key][1]
                    total_at_position = position_labels[pos_key][0]
                    
                    # 計算放射狀位置
                    label_x, label_y = get_radial_position(
                        representative_x, representative_y, 
                        current_index, total_at_position, 
                        base_distance=50.0
                    )
                    
                    position_labels[pos_key][1] += 1
                    
                    # 使用計算出的位置進行標註
                    ax.annotate(f'{group_type}\n{test_smiles}\n(estimated)',
                              xy=(representative_x, representative_y),
                              xytext=(label_x, label_y),
                              fontsize=10, fontweight='bold',
                              color='darkred', ha='center', va='center',
                              zorder=16,
                              bbox=dict(facecolor='white', alpha=0.7, edgecolor='darkred', boxstyle='round,pad=0.2'),
                              arrowprops=dict(
                                  arrowstyle='->', 
                                  color='darkred',   # 暗紅色箭頭
                                  lw=1.5,            # 增加線寬
                                  alpha=1.0,         # 完全不透明
                                  shrinkA=0,
                                  connectionstyle='arc3,rad=0.0',  # 直線連接
                                  mutation_scale=15  # 箭頭大小
                              ))
                    
                    # 使用虛線連接到群集中心
                    center_indices = np.where(cluster_labels_tsne == most_common_cluster)[0]
                    if len(center_indices) > 0:
                        center_points = X_tsne[center_indices]
                        center_x = np.mean(center_points[:, 0])
                        center_y = np.mean(center_points[:, 1])
                        ax.plot([representative_x, center_x], [representative_y, center_y], 
                              'k--', alpha=0.5, zorder=14)
                else:
                    print(f"  未找到相同類型的官能基 ({group_type})，無法估計位置")
        
        if not any_match_found:
            print(f"  在t-SNE數據中未找到任何匹配的官能基點")
    
    # 美化圖表
    ax.set_xlabel("t-SNE Dimension 1", fontsize=14)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=14)
    ax.set_title(f"Functional Group t-SNE Distribution with Highlighted SMILES", 
               fontsize=16, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # 分開顯示兩種圖例
    handles, labels = ax.get_legend_handles_labels()
    
    # 找出群集與官能基圖例的分界
    cluster_handles = []
    cluster_labels = []
    group_handles = []
    group_labels = []
    highlight_handles = []
    highlight_labels = []
    
    for h, label in zip(handles, labels):
        if label.startswith('Cluster'):
            cluster_handles.append(h)
            cluster_labels.append(label)
        elif ':' in label:  # 高亮的SMILES項目
            highlight_handles.append(h)
            highlight_labels.append(label)
        else:
            group_handles.append(h)
            group_labels.append(label)
    
    # 調整布局，為圖例預留更多空間
    fig.subplots_adjust(right=0.65)
    
    # 添加群集圖例
    if cluster_handles:
        cluster_legend = ax.legend(cluster_handles, cluster_labels,
                                 title="Clusters", bbox_to_anchor=(1.1, 1), 
                                 loc='upper left', fontsize=10, frameon=True, 
                                 title_fontsize=12)
        ax.add_artist(cluster_legend)
    
    # 添加官能基圖例
    if group_handles:
        group_legend = ax.legend(group_handles, group_labels,
                               title="Functional Groups", bbox_to_anchor=(1.1, 0.6), 
                               loc='center left', fontsize=10, frameon=True, 
                               title_fontsize=12)
        ax.add_artist(group_legend)
    
    # 添加高亮SMILES圖例
    if highlight_handles:
        highlight_legend = ax.legend(highlight_handles, highlight_labels,
                                   title="Highlighted SMILES", bbox_to_anchor=(1.1, 0.1), 
                                   loc='lower left', fontsize=10, frameon=True, 
                                   title_fontsize=12)
    
    # 保存結果
    if output_filename:
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=1.0)
        print(f"結果已保存至 {output_filename}")
    
    plt.show()





if __name__ == "__main__":
    # 設置路徑
    tsne_cache_file = "src/tsne_results.pkl"
    data_path = "data/processed_pka_data.csv"
    output_filename = "src/pka_tsne_evaluation.png"
    
    # 載入t-SNE結果
    tsne_data = load_tsne_results(tsne_cache_file)
    
    
    # ===========確認t-SNE結果===========
    if tsne_data is None:
        print("無法載入t-SNE結果，程序終止")
        sys.exit(1)
    else:
        print(f"t-SNE結果載入成功，形狀: {tsne_data['X_tsne'].shape}")
        print(f"原始維度: 樣本={tsne_data['n_samples']}, 特徵={tsne_data['n_features']}")
        print(f"聚類數量: k={tsne_data['n_clusters']}")
        
        if 'cluster_labels' in tsne_data:
            print(f"聚類標籤形狀: {tsne_data['cluster_labels'].shape}")
        
        if 'feature_groups' in tsne_data:
            print(f"已保存的官能基類型數量: {len(tsne_data['feature_groups'])}")
            # 直接使用保存的官能基類型和SMILES
            feature_groups = tsne_data['feature_groups']
            # 如果pickle文件中有保存feature_smiles，也加載它
            if 'feature_smiles' in tsne_data:
                feature_smiles = tsne_data['feature_smiles']
                print(f"使用已保存的feature_smiles，長度: {len(feature_smiles)}")
            else:
                # 否則，仍然需要加載原始數據獲取feature_smiles
                print("從原始數據加載feature_smiles...")
                _, feature_smiles, _, _ = load_feature_data(data_path)
        else:
            # 需要從原始數據加載官能基類型
            print("從原始數據加載官能基類型...")
            _, feature_smiles, _, feature_groups = load_feature_data(data_path)
            
            # 檢查長度是否匹配
            if len(feature_groups) != len(tsne_data['X_tsne']):
                print(f"警告: 原始數據中的feature_groups長度 ({len(feature_groups)}) 與t-SNE結果長度 ({len(tsne_data['X_tsne'])}) 不匹配")
                print("將使用默認的Unknown類型...")
                feature_groups = ['Unknown'] * len(tsne_data['X_tsne'])
                feature_smiles = [''] * len(tsne_data['X_tsne'])
    

    # df = pd.read_csv(data_path)
    # # 從數據集中均勻選取10個分散的樣本
    # total_samples = len(df)
    # # 隨機選取5個樣本
    # indices = np.random.choice(total_samples, 1, replace=False)
    # test_smiles_list = [df['SMILES'].tolist()[i] for i in indices]
    
    
    
    # 如果要指定也可以自定義
    test_smiles_list = ["OC(=O)[C@H](Cc1c[nH]cn1)NC(=O)CNC(=O)CS"]
    # 可視化SMILES在t-SNE空間中的位置
    visualize_smiles_in_tsne(test_smiles_list, tsne_data, 
                           feature_smiles, feature_groups,
                           output_filename)
