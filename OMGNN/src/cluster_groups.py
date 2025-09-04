import warnings
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, Dict, Tuple, List
import math

# 忽略 RDKit 的部分警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 複製自 graph_by_node.py 的定義 ---
FUNCTIONAL_GROUPS_SMARTS = {
    "COOH": Chem.MolFromSmarts('[C;X3](=[O;X1])[O;X2H1]'),
    "SulfonicAcid": Chem.MolFromSmarts('S(=O)(=O)O'),
    "PhosphonicAcid": Chem.MolFromSmarts('P(=O)(O)O'),
    "PrimaryAmine": Chem.MolFromSmarts('[NH2;!$(NC=O)]'),
    "SecondaryAmine": Chem.MolFromSmarts('[NH1;!$(NC=O)]'),
    "TertiaryAmine": Chem.MolFromSmarts('[NX3;H0;!$(NC=O)]'),
    "Phenol": Chem.MolFromSmarts('[OH1][c]'),
    "Alcohol": Chem.MolFromSmarts('[OH1][C;!c]'),
}

GROUP_LABEL_MAP = { # 這裡的標籤主要用於識別，不是 K-means 的目標
    "COOH": 0, "SulfonicAcid": 1, "PhosphonicAcid": 2,
    "PrimaryAmine": 3, "SecondaryAmine": 4, "TertiaryAmine": 5,
    "Phenol": 6, "Alcohol": 7, "Other": 8 # Other 在此腳本中不直接使用
}
LABEL_GROUP_MAP = {v: k for k, v in GROUP_LABEL_MAP.items()}
# --- 結束複製 ---


# --- 1. 提取官能基實例的特徵 ---
def extract_functional_group_features(smiles_list: list) -> tuple[List[np.ndarray], List[Dict]]:
    """
    從 SMILES 列表中提取所有已定義官能基實例的特徵。
    特徵：官能基內所有原子的平均 Gasteiger 電荷。
    返回: (特徵向量列表, 包含來源信息的字典列表)
    """
    all_group_features = []
    group_info_list = [] # {'smiles': str, 'group_name': str, 'atom_indices': tuple}

    print("正在提取官能基特徵...")
    processed_smiles = 0
    skipped_smiles = 0

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"警告：無法解析 SMILES '{smiles}'")
            skipped_smiles += 1
            continue

        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception as e:
            print(f"警告：無法計算 {smiles} 的 Gasteiger charges: {e}。跳過此分子。")
            skipped_smiles += 1
            continue

        mol_has_feature = False
        for group_name, pattern in FUNCTIONAL_GROUPS_SMARTS.items():
            if pattern is None: continue

            try:
                matches = mol.GetSubstructMatches(pattern)
            except Exception as e:
                print(f"警告：SMARTS 匹配 '{group_name}' 在 {smiles} 中出錯：{e}")
                continue

            if matches:
                # print(f"  {smiles}: Found {group_name} matches: {matches}") # 調試信息
                for match_indices in matches:
                    group_charges = []
                    valid_match = True
                    for idx in match_indices:
                        if idx >= mol.GetNumAtoms():
                            print(f"警告：匹配索引 {idx} 超出原子範圍 ({smiles})")
                            valid_match = False
                            break
                        try:
                            charge = mol.GetAtomWithIdx(idx).GetDoubleProp('_GasteigerCharge')
                            if not np.isfinite(charge):
                                 print(f"警告：原子 {idx} 在 {smiles} 中電荷無效 ({charge})，使用 0 代替。")
                                 charge = 0.0
                            group_charges.append(charge)
                        except KeyError:
                            # 這個不應該發生，因為前面檢查過電荷計算
                            print(f"警告：原子 {idx} 在 {smiles} 中缺少電荷屬性。")
                            valid_match = False
                            break

                    if valid_match and group_charges:
                        # --- 特徵向量：目前使用平均電荷 ---
                        feature_vector = np.array([np.mean(group_charges)])
                        # ---------------------------------
                        all_group_features.append(feature_vector)
                        group_info_list.append({
                            "smiles": smiles,
                            "group_name": group_name,
                            "atom_indices": match_indices
                        })
                        mol_has_feature = True
                    # elif not group_charges:
                        # print(f"警告：{group_name} 匹配 {match_indices} 在 {smiles} 未收集到有效電荷。")

        if mol_has_feature:
            processed_smiles += 1
        else:
            skipped_smiles += 1

    print(f"提取完成。處理了 {processed_smiles} 個 SMILES，跳過了 {skipped_smiles} 個。")
    print(f"共找到 {len(all_group_features)} 個官能基實例。")
    return all_group_features, group_info_list


# --- 2. 對官能基特徵進行 K-means 分群 ---
def cluster_functional_groups(
    group_features: List[np.ndarray],
    group_info: List[Dict],
    n_clusters: int = 3 # K-means n_clusters 參數保留，但視覺化不再直接使用其標籤顏色
    ):
    """
    對官能基特徵進行 K-means 分群，並根據官能基類型視覺化其分佈。
    """
    if not group_features:
        print("錯誤：沒有提取到任何官能基特徵，無法進行分群。")
        return None, None

    X = np.array(group_features)

    if not np.all(np.isfinite(X)):
        print("警告：特徵矩陣中包含 NaN 或 Inf 值。嘗試用 0 填充。")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- K-means 部分仍然執行，以供分析 (雖然不用於顏色) ---
    kmeans = None
    labels = None # K-means 標籤
    centroids = None # K-means 中心
    if X.shape[0] >= n_clusters:
        print(f"\n執行 K-means 分群 (k={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        try:
           kmeans.fit(X)
           labels = kmeans.labels_
           centroids = kmeans.cluster_centers_
           print("K-means 訓練完成。")
        except ValueError as e:
           print(f"K-means 執行時出錯：{e}")
           # 即使 K-means 出錯，仍然可以繼續視覺化原始數據分佈
    else:
         print(f"警告：官能基數量 ({X.shape[0]}) 少於 K-means 集群數 ({n_clusters})，跳過 K-means。")


    # --- 3. 分析 K-means 結果 (如果成功) ---
    cluster_summary = {}
    unique_group_names = sorted(list(FUNCTIONAL_GROUPS_SMARTS.keys()))
    if labels is not None:
        print("\nK-means 分群結果分析:")
        clusters = {} 
        for i, label in enumerate(labels):
            if label not in clusters: clusters[label] = []
            info = group_info[i].copy(); info['feature'] = X[i]
            clusters[label].append(info)

        for label in sorted(clusters.keys()):
            items = clusters[label]
            print(f"\n--- Cluster {label} (Centroid: {centroids[label]}) ---")
            group_counts = {name: 0 for name in unique_group_names}
            feature_values = [item['feature'][0] for item in items]

            for item in items:
                group_counts[item['group_name']] += 1

            print(f"  官能基分佈:")
            for name, count in group_counts.items():
                if count > 0: print(f"    {name}: {count}")
            print(f"  特徵範圍: Min={min(feature_values):.4f}, Max={max(feature_values):.4f}, Mean={np.mean(feature_values):.4f}")
            likely_group = max(group_counts, key=group_counts.get) if any(group_counts.values()) else "未知"
            cluster_summary[label] = {"centroid": centroids[label], "counts": group_counts, "likely_group": likely_group}
    else:
        print("\n未執行或 K-means 失敗，跳過分群結果分析。")


    # --- 4. 視覺化 (顏色代表官能基類型) ---
    print("\n正在繪製官能基分佈圖 (顏色代表官能基)...")
    plt.figure(figsize=(12, 6))

    # --- ★ 使用自定義顏色列表，基於官能基類型 ★ ---
    # 與 graph_by_node.py 保持一致 (確保順序對應 GROUP_LABEL_MAP)
    custom_colors_rgb_tuples = [ # 使用 RGB 元組
        (0.6, 0.8, 0.9),  # 0 COOH: 淺藍
        (1.0, 0.6, 0.2),  # 1 SulfonicAcid: 橘色
        (0.7, 0.7, 0.7),  # 2 PhosphonicAcid: 灰色
        (1.0, 0.4, 0.4),  # 3 PrimaryAmine: 紅色/粉紅
        (0.7, 0.5, 0.9),  # 4 SecondaryAmine: 紫色
        (0.4, 0.8, 0.4),  # 5 TertiaryAmine: 綠色
        (0.8, 0.8, 0.2),  # 6 Phenol: 橄欖綠/黃綠
        (0.9, 0.5, 0.8),  # 7 Alcohol: 紫紅/粉紫
        # 注意：我們不繪製 "Other"，所以顏色列表只需要前面 8 個
    ]
    # group_names_ordered = sorted(list(FUNCTIONAL_GROUPS_SMARTS.keys())) # 獲取實際存在的官能基名稱
    # 使用 LABEL_GROUP_MAP 的 keys 來確保順序和標籤對應
    group_names_ordered = sorted(LABEL_GROUP_MAP.values())[:-1] # 獲取除 Other 之外的所有名稱並排序
    group_color_map = {name: custom_colors_rgb_tuples[GROUP_LABEL_MAP[name]]
                       for name in group_names_ordered if GROUP_LABEL_MAP.get(name) is not None and GROUP_LABEL_MAP[name] < len(custom_colors_rgb_tuples)}

    # 為每個官能基類型分配一個標記符號 (marker)
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v'] # 8 個標記對應 8 個基團
    group_marker_map = {name: markers[i % len(markers)] for i, name in enumerate(group_names_ordered)}


    # 繪製散點圖
    y_jitter = np.random.rand(X.shape[0]) * 0.2 - 0.1

    legend_handles = {} 

    for i in range(X.shape[0]):
        # kmeans_label = labels[i] # 不再使用 K-means 標籤決定顏色
        group_name = group_info[i]['group_name']
        marker = group_marker_map.get(group_name, '.') 
        # ★ 顏色由官能基類型決定 ★
        color = group_color_map.get(group_name, (0,0,0)) # 獲取顏色，默認為黑色

        # 繪製點
        handle = plt.scatter(X[i, 0], y_jitter[i], c=[color], marker=marker, alpha=0.7, label=group_name if group_name not in legend_handles else "")
        if group_name not in legend_handles:
             # 使用顏色和標記創建圖例條目
             legend_handles[group_name] = plt.Line2D([0], [0], marker=marker, color='w', label=group_name,
                                                     markerfacecolor=color, markersize=8)

    # ★ 不再繪製 K-means 中心點 ★
    # plt.scatter(centroids[:, 0], np.zeros(centroids.shape[0]), ...)

    plt.xlabel("Average Gasteiger Charge (Feature)")
    plt.ylabel("Jitter")
    plt.title(f"Functional Group Distribution by Average Charge") # 更新標題
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.yticks([])

    # 創建圖例 (現在顏色和標記都代表官能基)
    # 按官能基名稱排序圖例
    sorted_legend_handles = [legend_handles[name] for name in group_names_ordered if name in legend_handles]
    plt.legend(handles=sorted_legend_handles, title="Functional Groups", bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    output_filename = "../src/functional_group_distribution_colored_by_type.png" # 新文件名
    plt.savefig(output_filename)
    print(f"官能基分佈圖 (按類型著色) 已保存至 {output_filename}")
    plt.close()

    return kmeans, cluster_summary # 仍然返回 K-means 結果以供後續使用


# --- 示例用法 ---
if __name__ == "__main__":
    # --- 讀取 SMILES 數據 ---
    try:
        df = pd.read_csv('../data/NIST_database_onlyH_6TypeEq_pos_match_max_fg_other.csv')
        smiles_list_full = df['SMILES'].unique()
        # 處理所有或部分分子
        # N_to_process = 100 # 例如處理前 100 個
        # smiles_list = smiles_list_full[:N_to_process]
        smiles_list = smiles_list_full # 處理全部
    except FileNotFoundError:
        print("錯誤：找不到 CSV 文件，將使用內建的示例列表。")
        smiles_list = [
            "CCO", "CC(=O)O", "N[C@@H](C)C(=O)O", "C1=CC=C(C=C1)C(=O)O", "CCN",
            "O=C(O)CCC(=O)O", "NCCO", "C(C(=O)O)N", "CC(N)C", "Oc1ccccc1C(=O)O",
            "CCCCN", "CCCCC(=O)O", "c1ccccc1S(=O)(=O)O", "CC(=O)N", # 加入磺酸和醯胺示例
             "N[C@@H](CC[S](O)(=O)=O)C(O)=O" # 包含磺酸的氨基酸
        ]

    print(f"將處理 {len(smiles_list)} 個 SMILES...")

    # --- 提取特徵 ---
    group_features_list, group_info_list = extract_functional_group_features(smiles_list)

    # --- 執行分群 ---
    if group_features_list:
        # --- 選擇 K 值 ---
        # 選項 1: 基於預期的化學類別數量 (例如，酸性 vs 鹼性 vs 中性)
        k_value = 3
        # 選項 2: 等於官能基種類數 (看能否按類型分開)
        # k_value = len(FUNCTIONAL_GROUPS_SMARTS)
        # ------------------
        print(f"設定 K = {k_value}")
        kmeans_model, summary = cluster_functional_groups(group_features_list, group_info_list, n_clusters=k_value)

        if kmeans_model and summary:
            print("\n分群模型訓練完成。")
            # 可以在這裡添加更多基於 summary 的分析或應用
        else:
            print("\n分群失敗。")
    else:
        print("\n未能提取任何官能基特徵，無法繼續。")
