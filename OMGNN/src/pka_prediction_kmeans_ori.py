
# print(os.getcwd())
# os.chdir('OMGNN/')
import warnings
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import sys
sys.path.append('../')
from other.chemutils_old import atom_features
# 忽略 RDKit 的警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 從 cluster_groups.py 複製的定義 ---
FUNCTIONAL_GROUPS_SMARTS = {
    "COOH": Chem.MolFromSmarts('[C;X3](=[O;X1])[O;X2H1]'),
    "SulfonicAcid": Chem.MolFromSmarts('S(=O)(=O)O'),
    "PhosphonicAcid": Chem.MolFromSmarts('P(=O)(O)O'),
    "PrimaryAmine": Chem.MolFromSmarts('[NH2;!$(NC=O)]'),
    "SecondaryAmine": Chem.MolFromSmarts('[NH1;!$(NC=O)]'),
    "TertiaryAmine": Chem.MolFromSmarts('[NX3;H0;!$(NC=O)]'),
    "Phenol": Chem.MolFromSmarts('[OH1][c]'),
    "Alcohol": Chem.MolFromSmarts('[OH1][C;!c]'),
    # 可以考慮添加其他與 pKa 相關的基團，例如 Imidazole, Pyridine 等
    "Imidazole": Chem.MolFromSmarts('c1cn[nH]c1'), # 示例：咪唑環
    "Pyridine": Chem.MolFromSmarts('c1ccccn1'),   # 示例：吡啶環
    "Thiol": Chem.MolFromSmarts('[SH]'),          # 示例：硫醇
}
FUNCTIONAL_GROUP_PRIORITY = [
    "SulfonicAcid",
    "PhosphonicAcid",
    "COOH",
    "Phenol",
    "Thiol",
    "TertiaryAmine",   # 鹼性基團
    "SecondaryAmine",
    "PrimaryAmine",
    "Pyridine",        # 雜環鹼性
    "Imidazole",
    "Alcohol"          # 通常視為較弱或中性
]

# --- 1. 特徵提取 (修改版：為每個 SMILES 提取單一特徵向量) ---
def extract_molecule_feature(smiles: str) -> Optional[np.ndarray]:
    """
    為單個 SMILES 提取特徵。
    特徵：分子中所有已定義官能基實例的平均 Gasteiger 電荷的平均值。
    如果分子沒有找到任何已定義的官能基，或計算失敗，返回 None。
    """
    
    not_found_count = 0
    print(f"提取特徵: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    group_instance_features = [] # 儲存這個分子中所有官能基實例的特徵（平均電荷）

    for group_name, pattern in FUNCTIONAL_GROUPS_SMARTS.items():
        if pattern is None: continue

        try:
            matches = mol.GetSubstructMatches(pattern)
        except Exception as e:
            # print(f"警告：SMARTS 匹配 '{group_name}' 在 {smiles} 中出錯：{e}")
            continue # 跳過這個官能基

        if matches:
            for match_indices in matches:
                # 將 match_indices 按 FUNCTIONAL_GROUP_PRIORITY 的順序排序
                # group_name
                print(f"    找到 {group_name} 的匹配索引: {match_indices}")
                group_features = []
                valid_match = True
                for idx in match_indices:
                    if idx >= mol.GetNumAtoms():
                        # print(f"警告：匹配索引 {idx} 超出原子範圍 ({smiles})")
                        valid_match = False
                        break
                    try:
                        features = atom_features(mol.GetAtomWithIdx(idx)).numpy()
                        
                        # if not np.isfinite(charge):
                        #      # print(f"警告：原子 {idx} 在 {smiles} 中電荷無效 ({charge})，使用 0 代替。")
                        #      charge = 0.0
                        group_features.append(features)
                    except KeyError:
                        valid_match = False
                        break

                if valid_match and group_features:
                    group_avg_feature = np.mean(group_features, axis=0)
                    group_instance_features.append(group_avg_feature)
    
    # 如果沒有找到官能基特徵，返回None
    if not group_instance_features:
        not_found_count += 1
        return None
        
    # 對於超過一個官能基的結果，優先保留羧酸(COOH)而非醇(Alcohol)
    if len(group_instance_features) > 1:
        # 檢查是否有羧酸官能基
        has_cooh = False
        cooh_index = -1
        
        for i, match in enumerate(matches):
            group_name = list(FUNCTIONAL_GROUPS_SMARTS.keys())[i]
            if 'COOH' in group_name or 'carboxylic' in group_name:
                has_cooh = True
                cooh_index = i
                break
        
        if has_cooh:
            # 如果有羧酸，只保留羧酸的特徵
            selected_feature = group_instance_features[cooh_index]
        else:
            # 否則只保留第一個官能基的特徵
            selected_feature = group_instance_features[0]
    else:
        # 只有一個官能基特徵
        selected_feature = group_instance_features[0]
    
    # 確保返回形狀一致的特徵向量
    return selected_feature  # 返回一個2D向量，便於後續處理



# --- 主執行流程 ---
if __name__ == "__main__":
    # --- 參數設定 ---
    smiles = "C(=O)O"
    data_path = 'data/processed_pka_data.csv'
    n_clusters_kmeans = 5 # K-means 的分群數量 (可調整)
    output_plot_filename = "src/pka_kmeans_clusters.png"
    # 定義一些目標 SMILES 來測試預測
    target_smiles_list = [
        "CC(=O)O",     # Acetic acid (預期酸性)
        "CCN",         # Ethylamine (預期鹼性)
        "CCO",         # Ethanol (預期中性/弱酸)
        "c1ccccc1O",   # Phenol (預期弱酸性)
        "c1ccccc1COOH",# Benzoic acid (預期酸性)
        "c1ccccn1",     # Pyridine (預期鹼性)
        "CN(C)C",      # Trimethylamine (預期鹼性)
        "CC(C)(C)N",   # t-Butylamine (預期鹼性)
        "N[C@@H](C)C(=O)O", # Alanine (兩性，但看哪個基團被捕捉)
        "O=S(=O)(O)c1ccccc1" # Benzenesulfonic acid (預期強酸性)
    ]

    # --- 1. 載入數據 ---
    print(f"正在從 {data_path} 載入數據...")
    df = pd.read_csv(data_path)
    # 確保 pKa 值是數字
    df['pKa_value'] = pd.to_numeric(df['pKa_value'], errors='coerce')
    df = df.dropna(subset=['SMILES', 'pKa_value'])

    # --- 2. 為數據集中的每個 SMILES 提取特徵 ---
    print("正在為數據集提取分子特徵...")
    
    features = []
    pka_values = []
    original_smiles = []
    
    for index, row in df.iterrows():
        smiles = row['SMILES']
        pka = row['pKa_value']
        # 由於 not_found_count 是作為參數傳遞，不會被更新，需要接收返回值
        feature_vector = extract_molecule_feature(smiles)
        if feature_vector is not None and np.all(np.isfinite(feature_vector)):
            features.append(feature_vector)
            pka_values.append(pka)
            original_smiles.append(smiles)
        else:
            pass
    # 打印長度
    print(f"特徵長度: {len(features)}")
    print(f"pKa數量: {len(pka_values)}")
    print(f"分子數量: {len(original_smiles)}")
    # print(f"未找到特徵的分子數量: {not_found_count}")
    
    X = np.array(features)
    y_pka = np.array(pka_values)

    # 添加整體工作流程註解
    """
    pKa 預測與分群分析工作流程 (Workflow)：

    1. 資料預處理：
       - 從 CSV 載入分子 SMILES 與實驗 pKa 值
       - 過濾無效數據

    2. 特徵提取：
       - 對每個分子提取 153 維官能團特徵
       - 統計無法提取特徵的分子數量

    3. 降維與聚類：
       - 使用 PCA 將 153 維特徵降至較低維度 (默認 50 維或樣本數的 1/10)
       - 應用 K-means 算法在降維後的空間進行分群
       - 評估聚類效果 (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index)

    4. 分群結果分析：
       - 計算每個群集的平均 pKa 值
       - 分析群集分布

    5. 視覺化：
       - 使用 PCA 降至 2 維進行聚類結果視覺化
       - 繪製群集與 pKa 分布關係圖
       - 可選 t-SNE 視覺化以展示更複雜的數據結構

    流程設計邏輯：
    - 先降維再聚類：減少高維空間中的"維度詛咒"問題，提高聚類效率
    - 保留變異性：PCA 降維保留了數據主要變異，濾除噪聲
    - 多重評估：使用多種指標評估聚類質量
    - 多角度視覺化：從不同角度展示分群結果
    """

    # --- 3. 訓練 K-means 模型 ---
    print(f"執行 K-means 分群 (k={n_clusters_kmeans})...")
    if X.shape[0] < n_clusters_kmeans:
        print(f"警告：數據點數量 ({X.shape[0]}) 少於 K 值 ({n_clusters_kmeans})。無法執行 K-means。")
        kmeans_model = None
        cluster_labels = None
    else:
        # 針對153維高維特徵進行處理
        # 工作流：特徵(153維) -> PCA降維 -> K-means聚類 -> 評估 -> 視覺化
        use_pca = True  # 啟用 PCA 降維以提高聚類效果
        pca_components = min(50, X.shape[0] // 10)  # 降維目標：樣本數的1/10或最多50維
        
        # 對高維特徵進行降維處理
        print(f"正在處理 {X.shape[1]} 維特徵...")
        if use_pca:
            from sklearn.decomposition import PCA
            print(f"使用 PCA 降維至 {pca_components} 維...")
            # 步驟 3.1: PCA降維 - 將153維特徵降至較低維度，保留主要變異
            pca = PCA(n_components=pca_components, random_state=42)
            X_transformed = pca.fit_transform(X)
            # 輸出保留的信息量
            var_ratio = sum(pca.explained_variance_ratio_)
            print(f"PCA 降維完成，保留了 {var_ratio:.2%} 的數據變異性")
        else:
            X_transformed = X
            print("使用原始 153 維特徵進行聚類")
        
        # 配置 K-means，針對高維數據優化參數
        # 步驟 3.2: 配置並執行K-means聚類
        kmeans_model = KMeans(
            n_clusters=n_clusters_kmeans,
            random_state=42,
            n_init='auto',  # 自動確定初始化次數
            max_iter=500,  # 增加迭代次數確保收斂
            tol=1e-5,  # 提高收斂精度
            algorithm='elkan'  # 對高維數據效率更高的算法
        )
        
        try:
            print(f"正在對 {X_transformed.shape[0]} 個樣本進行聚類...")
            # 執行聚類並獲取每個樣本的群集標籤
            cluster_labels = kmeans_model.fit_predict(X_transformed)
            
            # 步驟 3.3: 評估聚類質量
            from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
            try:
                sil_score = silhouette_score(X_transformed, cluster_labels)
                db_score = davies_bouldin_score(X_transformed, cluster_labels)
                ch_score = calinski_harabasz_score(X_transformed, cluster_labels)
                print(f"聚類評估指標:")
                print(f"- Silhouette Score: {sil_score:.4f} (越高越好，範圍 [-1, 1])")
                print(f"- Davies-Bouldin Index: {db_score:.4f} (越低越好)")
                print(f"- Calinski-Harabasz Index: {ch_score:.1f} (越高越好)")
            except Exception as e:
                print(f"評估指標計算失敗: {e}")
            
            # 步驟 3.4: 分析群集分布
            unique_labels, counts = np.unique(cluster_labels, return_counts=True)
            print("群集分布情況:")
            for label, count in zip(unique_labels, counts):
                percentage = count/len(cluster_labels)*100
                print(f"- 群集 {label}: {count} 個樣本 ({percentage:.1f}%)")
                
            print("K-means 訓練完成")
            
        except ValueError as e:
           print(f"K-means 執行失敗: {e}")
           kmeans_model = None
           cluster_labels = None













    # --- 4. 計算每個群集的平均 pKa 值 ---
    # 步驟 4: 分析每個群集的pKa特性
    print("計算每個群集的平均 pKa 值...")
    cluster_avg_pka = {}
    
    if kmeans_model is not None and cluster_labels is not None:
        for i in range(n_clusters_kmeans):
            cluster_mask = (cluster_labels == i)
            if np.any(cluster_mask):  # 如果該群集有樣本
                avg_pka = np.mean(y_pka[cluster_mask])
                std_pka = np.std(y_pka[cluster_mask])
                min_pka = np.min(y_pka[cluster_mask])
                max_pka = np.max(y_pka[cluster_mask])
                cluster_avg_pka[i] = avg_pka
                print(f"群集 {i} 的 pKa 統計: 平均={avg_pka:.2f}, 標準差={std_pka:.2f}, 範圍=[{min_pka:.2f}, {max_pka:.2f}]")
    else:
        print("未執行 K-means 或失敗，跳過計算分群平均 pKa。")





















    # --- 5. 視覺化分群結果 ---
    # 步驟 5: 多角度視覺化分群結果
    print("正在繪製 K-means 分群結果圖...")
    
    if kmeans_model is not None and cluster_labels is not None:
        # 對高維數據進行降維以便視覺化
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
        
        # # 創建一個包含兩張圖的圖像
        # fig, axs = plt.subplots(1, 2, figsize=(20, 8))
        
        # 使用更豐富的色彩方案，提高可視化效果
        colors = plt.cm.tab10(np.linspace(0, 1, n_clusters_kmeans))
        
        # ===== 5.1: PCA 降維視覺化 =====
        # 將降維後的結果再次映射到2D空間用於視覺化
        print("進行 PCA 降維用於視覺化...")
        # 使用與聚類相同的數據進行視覺化
        X_for_vis = X_transformed if use_pca else X
        
        # # 為視覺化創建新的 PCA 模型，總是降到 2 維
        # pca_vis = PCA(n_components=2, random_state=42)
        # X_pca = pca_vis.fit_transform(X_for_vis)
        # explained_var_ratio = pca_vis.explained_variance_ratio_
        # print(f"視覺化 PCA 解釋方差比例: [{explained_var_ratio[0]:.4f}, {explained_var_ratio[1]:.4f}]")
        
        # # 繪製 PCA 降維後的分群結果
        # for i in range(n_clusters_kmeans):
        #     cluster_mask = (cluster_labels == i)
        #     avg_pka = cluster_avg_pka.get(i)
        #     n_samples = np.sum(cluster_mask)
            
        #     # 為每個群集創建標籤
        #     label_text = f'Cluster {i} (n={n_samples})'
        #     if avg_pka is not None:
        #         label_text += f', Avg pKa: {avg_pka:.2f}'
            
        #     # 使用更大的點和更高的透明度來增強可讀性
        #     axs[0].scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1],
        #                 color=colors[i], label=label_text, alpha=0.7, s=40, 
        #                 edgecolors='w', linewidths=0.5)
        
        # # 繪製 K-means 中心點，使用更明顯的標記
        # if hasattr(kmeans_model, 'cluster_centers_'):
        #     # 將聚類中心點在相同空間中進行可視化
        #     centroids_pca = pca_vis.transform(kmeans_model.cluster_centers_)
        #     # 使用菱形標記並保持與各自群集相同的顏色
        #     for i in range(n_clusters_kmeans):
        #         axs[0].scatter(centroids_pca[i, 0], centroids_pca[i, 1], 
        #                      marker='D', s=200, c=[colors[i]], edgecolors='black', 
        #                      linewidths=2.0, zorder=10)
        #     # 添加一個單獨的圖例項目
        #     axs[0].scatter([], [], marker='D', s=200, c='gray', edgecolors='black',
        #                   linewidths=2.0, label='Centroids', zorder=10)
            
        #     # 為每個中心點添加編號標籤
        #     for i, (x, y) in enumerate(centroids_pca):
        #         axs[0].text(x + 0.05, y + 0.05, f'{i}', fontsize=12, fontweight='bold',
        #                   ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'), zorder=11)
        
        # # 美化第一張圖的標籤和標題
        # axs[0].set_xlabel(f"PCA Dimension 1 ({explained_var_ratio[0]:.2%})", fontsize=12)
        # axs[0].set_ylabel(f"PCA Dimension 2 ({explained_var_ratio[1]:.2%})", fontsize=12)
        # axs[0].set_title(f"K-means Clustering Results after PCA (k={n_clusters_kmeans})", 
        #                fontsize=14, fontweight='bold')
        # axs[0].grid(True, linestyle='--', alpha=0.5)
        
        # # 調整圖例位置和樣式
        # legend0 = axs[0].legend(title="Clusters", bbox_to_anchor=(1.02, 1), 
        #                       loc='upper left', fontsize=10, frameon=True, 
        #                       title_fontsize=12)
        # legend0.get_frame().set_alpha(0.9)
        
        # # ===== 5.2: pKa vs 主要特徵分布圖 =====
        # # 分析群集特徵與pKa的關係
        # for i in range(n_clusters_kmeans):
        #     cluster_mask = (cluster_labels == i)
        #     avg_pka = cluster_avg_pka.get(i)
        #     n_samples = np.sum(cluster_mask)
            
        #     # 為每個群集創建標籤
        #     label_text = f'Cluster {i} (n={n_samples})'
        #     if avg_pka is not None:
        #         label_text += f', Avg pKa: {avg_pka:.2f}'
            
        #     # 使用與第一張圖同樣的 PCA 結果
        #     axs[1].scatter(X_pca[cluster_mask, 0], y_pka[cluster_mask],
        #                 color=colors[i], label=label_text, alpha=0.7, s=40,
        #                 edgecolors='w', linewidths=0.5)
            
        #     # 添加每個群集的平均值標記
        #     if avg_pka is not None:
        #         mean_x = np.mean(X_pca[cluster_mask, 0])
        #         axs[1].scatter(mean_x, avg_pka, marker='D', s=100, 
        #                      color=colors[i], edgecolors='black', linewidths=1.5, zorder=10)
        #         axs[1].text(mean_x + 0.05, avg_pka + 0.2, f'{i}', fontsize=12, fontweight='bold',
        #                  ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'), zorder=11)
        
        # # 添加一條趨勢線，顯示 PCA 第一維度與 pKa 的關係
        # try:
        #     from scipy import stats
        #     slope, intercept, r_value, p_value, std_err = stats.linregress(X_pca[:, 0], y_pka)
        #     x_range = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)
        #     axs[1].plot(x_range, intercept + slope * x_range, 'k--', 
        #                label=f'Trend: r={r_value:.2f}', linewidth=1.5, alpha=0.7)
        # except:
        #     pass
        
        # # 美化第二張圖的標籤和標題
        # axs[1].set_xlabel(f"PCA Dimension 1 ({explained_var_ratio[0]:.2%})", fontsize=12)
        # axs[1].set_ylabel("Experimental pKa Values", fontsize=12)
        # axs[1].set_title(f"pKa Distribution by Cluster (k={n_clusters_kmeans})", 
        #                fontsize=14, fontweight='bold')
        # axs[1].grid(True, linestyle='--', alpha=0.5)
        
        # # 調整圖例位置和樣式
        # legend1 = axs[1].legend(title="Clusters", bbox_to_anchor=(1.02, 1), 
        #                       loc='upper left', fontsize=10, frameon=True, 
        #                       title_fontsize=12)
        # legend1.get_frame().set_alpha(0.9)
        
        # # 添加總體圖說明
        # fig.suptitle(f"K-means Clustering Analysis of {X.shape[0]} Molecules with {X.shape[1]} Features", 
        #            fontsize=16, fontweight='bold', y=0.98)
        
        # ===== 5: t-SNE 補充視覺化  =====
        # 嘗試使用 t-SNE 降維 (可選，取決於數據量和計算資源)
        if X.shape[0] <= 8000:  # 僅對較小的數據集使用 t-SNE
            try:
                print("進行 t-SNE 降維用於視覺化...")
                # 創建額外的圖用於 t-SNE 可視化
                tsne_fig = plt.figure(figsize=(14, 10), dpi=300)
                
                # 使用與聚類相同的數據進行 t-SNE
                # 先用 PCA 降到 50 維，再用 t-SNE 降到 2 維 (提高性能)
                if not use_pca:  # 如果聚類時未使用 PCA，先降維到合適維度
                    pca_tsne = PCA(n_components=min(50, X.shape[1]), random_state=42)
                    X_for_tsne = pca_tsne.fit_transform(X_for_vis)
                    print(f"t-SNE 預處理：使用 PCA 降至 {X_for_tsne.shape[1]} 維，解釋方差: {sum(pca_tsne.explained_variance_ratio_):.2f}")
                else:
                    # 如果聚類時已使用 PCA，直接使用已降維的數據
                    X_for_tsne = X_for_vis
                    print(f"t-SNE 使用已降維的 {X_for_tsne.shape[1]} 維數據")
                
                # 調整 t-SNE 參數以提高結果質量
                perplexity = min(30, X.shape[0]//5)
                print(f"使用 t-SNE, perplexity={perplexity}")
                
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                          learning_rate='auto', init='pca', n_iter=2000)
                X_tsne = tsne.fit_transform(X_for_tsne)
                
                # 繪製 t-SNE 降維後的分群結果
                for i in range(n_clusters_kmeans):
                    cluster_mask = (cluster_labels == i)
                    avg_pka = cluster_avg_pka.get(i)
                    n_samples = np.sum(cluster_mask)
                    
                    # 為每個群集創建標籤
                    label_text = f'Cluster {i} (n={n_samples})'
                    if avg_pka is not None:
                        label_text += f', Avg pKa: {avg_pka:.2f}'
                    
                    plt.scatter(X_tsne[cluster_mask, 0], X_tsne[cluster_mask, 1],
                              color=colors[i], label=label_text, alpha=0.7, s=40,
                              edgecolors='w', linewidths=0.5)
                
                # 為每個群集添加中心標記
                for i in range(n_clusters_kmeans):
                    cluster_mask = (cluster_labels == i)
                    if np.any(cluster_mask):
                        center_x = np.mean(X_tsne[cluster_mask, 0])
                        center_y = np.mean(X_tsne[cluster_mask, 1])
                        plt.scatter(center_x, center_y, marker='D', s=100, 
                                  color=colors[i], edgecolors='black', linewidths=1.5, zorder=10)
                        # 修改標籤位置，使用白色背景框避免重疊
                        plt.text(center_x, center_y, f'{i}', fontsize=12, fontweight='bold',
                               ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', 
                                                                  boxstyle='round,pad=0.2'), zorder=11)
                
                # 美化 t-SNE 圖
                plt.xlabel("t-SNE Dimension 1", fontsize=12)
                plt.ylabel("t-SNE Dimension 2", fontsize=12)
                plt.title(f"K-means Clustering Results after t-SNE (k={n_clusters_kmeans})",
                        fontsize=14, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.5)
                
                # 調整圖例
                tsne_legend = plt.legend(title="Clusters", bbox_to_anchor=(1.02, 1), 
                                      loc='upper left', fontsize=10, frameon=True, 
                                      title_fontsize=12)
                tsne_legend.get_frame().set_alpha(0.9)
                
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                
                # 保存 t-SNE 圖
                tsne_plot_filename = "t-SNE_Kmeans_Clustering_Results.png"
                plt.savefig(tsne_plot_filename, dpi=300, bbox_inches='tight')
                print(f"t-SNE 分群結果圖已保存至 {tsne_plot_filename}")
                plt.close(tsne_fig)
            except Exception as e:
                print(f"t-SNE 降維失敗：{e}")
        
    else: # 如果 K-means 失敗，只繪製原始數據
        print("K-means 失敗，跳過 t-SNE 視覺化")
        pass
        

    # 保存主圖，使用高分辨率
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_plot_filename, dpi=300, bbox_inches='tight')
    print(f"分群結果圖已保存至 {output_plot_filename}")
    plt.close()


















    # --- 6. 預測目標 SMILES 的 pKa ---
    if kmeans_model is not None and cluster_avg_pka:
        print("預測目標 SMILES 的 pKa...")
        for target_smiles in target_smiles_list:
            target_feature = extract_molecule_feature(target_smiles)
            if target_feature is not None and np.all(np.isfinite(target_feature)):
                try:
                    # target_feature已經是2D陣列了，不需要reshape
                    target_cluster = kmeans_model.predict(target_feature)[0]
                    predicted_pka = cluster_avg_pka.get(target_cluster)

                    if predicted_pka is not None:
                        print(f"  SMILES: {target_smiles}")
                        print(f"    特徵值: {target_feature[0][0]:.4f}")  # 修改這裡以訪問正確的數組元素
                        print(f"    預測分群: {target_cluster}")
                        print(f"    預測 pKa (分群平均值): {predicted_pka:.2f}")
                        # 可以顯示該分群的 pKa 分佈
                        # pka_distribution = cluster_pka_dist.get(target_cluster, [])
                        # if pka_distribution:
                        #     print(f"    分群 pKa 範圍: Min={min(pka_distribution):.2f}, Max={max(pka_distribution):.2f}")
                    else:
                        print(f"  SMILES: {target_smiles} -> 分配到分群 {target_cluster}，但該分群無平均 pKa。")

                except Exception as e:
                    print(f"  無法預測 SMILES {target_smiles}: {e}")
            else:
                print(f"  無法為目標 SMILES {target_smiles} 提取有效特徵，跳過預測。")
    else:
        print("K-means 模型未訓練或分群平均 pKa 不可用，無法進行預測。")

    print("流程結束。") 