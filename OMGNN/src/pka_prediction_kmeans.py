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
import os
import pickle
from ast import literal_eval
sys.path.append('../')
from other.chemutils_old import atom_features
# 忽略 RDKit 的警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')
warnings.filterwarnings("ignore", category=FutureWarning)


# --- 從 cluster_groups.py 複製的定義 ---
FUNCTIONAL_GROUPS_SMARTS = {
    "COOH": Chem.MolFromSmarts('[C;X3](=[O;X1])[O;X2H1]'),
    "SulfonicAcid": Chem.MolFromSmarts('S(=O)(=O)O'),
    "PhosphonicAcid": Chem.MolFromSmarts('[#15](-[#8])(=[#8])(-[#8])-[#8]'),
    "PrimaryAmine": Chem.MolFromSmarts('[NH2;!$(NC=O)]'),
    "SecondaryAmine": Chem.MolFromSmarts('[NH1;!$(NC=O)]'),
    "TertiaryAmine": Chem.MolFromSmarts('[NX3;H0;!$(NC=O)]'),
    "Phenol": Chem.MolFromSmarts('[OH1][c]'),
    "Alcohol": Chem.MolFromSmarts('[OH1][C;!c]'),
    "Imidazole": Chem.MolFromSmarts('c1[nX3]c[nX3]c1'),  # 修改為能夠匹配所有類型的咪唑環，包括N-取代的咪唑
    "Pyridine": Chem.MolFromSmarts('c1ccccn1'),   # 示例：吡啶環
    "Thiol": Chem.MolFromSmarts('[SH]'),          # 示例：硫醇
    "Nitro": Chem.MolFromSmarts('[N+](=O)[O-]'),
    "NHydroxylamine": Chem.MolFromSmarts('[NH][OH]'),
    "PyrazoleNitrogen": Chem.MolFromSmarts('c1c[nX2:1]nc1')  # 吡唑中的氮原子
}
FUNCTIONAL_GROUP_PRIORITY = [
    "SulfonicAcid",
    "PhosphonicAcid",
    "COOH",
    "Thiol",             # ↑建議放前面
    "Phenol",
    "NHydroxylamine",
    "Imidazole",         # ↑建議提前
    "PyrazoleNitrogen",  # ↑建議提前
    "Pyridine",
    "SecondaryAmine",    # ↑優於Primary
    "TertiaryAmine",
    "PrimaryAmine",
    "Alcohol"
    # "Nitro" ⚠️ 可視為電子效應，不建議列為解離官能基
]


    
# --- 1. 特徵提取 (修改版：為每個 SMILES 提取多個官能基特徵向量) ---
def extract_molecule_feature(smiles: str) -> List[np.ndarray]:
    """
    為單個 SMILES 提取特徵。
    特徵：分子中所有已定義官能基實例的特徵。
    返回一個包含所有找到的官能基實例特徵的列表，如果沒有找到，返回空列表。
    每個元素是一個特徵向量，對應一個官能基實例。
    """
    
    print(f"提取特徵: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    group_instance_features = [] # 儲存這個分子中所有官能基實例的特徵
    
    # 跟踪已匹配的原子，避免重複
    matched_atoms = set()
    
    # 按優先級順序匹配官能基
    for group_name in FUNCTIONAL_GROUP_PRIORITY:
        pattern = FUNCTIONAL_GROUPS_SMARTS.get(group_name)
        if pattern is None:
            continue

        try:
            matches = mol.GetSubstructMatches(pattern)
        except Exception as e:
            # print(f"警告：SMARTS 匹配 '{group_name}' 在 {smiles} 中出錯：{e}")
            continue  # 跳過這個官能基

        if matches:
            for match_indices in matches:
                # 檢查匹配原子是否已被更高優先級的官能基使用
                overlap = False
                for idx in match_indices:
                    if idx in matched_atoms:
                        overlap = True
                        break
                
                if overlap:
                    # 如果有重疊，跳過這個匹配
                    continue
                
                # 找到官能基匹配
                # print(f"    找到 {group_name} 的匹配索引: {match_indices}")
                group_features = []
                valid_match = True
                for idx in match_indices:
                    if idx >= mol.GetNumAtoms():
                        valid_match = False
                        break
                    try:
                        features = atom_features(mol.GetAtomWithIdx(idx)).numpy()
                        group_features.append(features)
                    except KeyError:
                        valid_match = False
                        break

                if valid_match and group_features:
                    # 將這些原子標記為已匹配
                    for idx in match_indices:
                        matched_atoms.add(idx)
                        
                    group_avg_feature = np.mean(group_features, axis=0)
                    # 為每個特徵添加官能基類型信息
                    group_instance_features.append({
                        'feature': group_avg_feature,
                        'group_type': group_name,
                        'smiles': smiles,
                        'match_indices': match_indices
                    })
    
    # 打印找到的官能基及其重要性
    if group_instance_features:
        print(f"    總共找到 {len(group_instance_features)} 個非重疊官能基")
        for i, info in enumerate(group_instance_features):
            print(f"    官能基 {i+1}: {info['group_type']} at 位置 {info['match_indices']}")
    else:
        print(f"    未找到任何官能基")
        
    # 返回所有找到的官能基特徵
    return group_instance_features


# --- 主執行流程 ---
if __name__ == "__main__":
    
    # smiles = "OCC(O)CO[P](O)(O)=O"
    # feature_list = extract_molecule_feature(smiles)
    
    # --- 參數設定 ---
    data_path = '../data/processed_pka_data.csv'
    n_clusters_kmeans = 12 # K-means 的分群數量 (可調整)
    output_plot_filename = "../src/pka_kmeans_clusters.png"
    # 定義一些目標 SMILES 來測試預測
    target_smiles_list = [
        "CC(=O)O",     # Acetic acid (預期酸性)
        "CCN",         # Ethylamine (預期鹼性)
        "CCO",         # Ethanol (預期中性/弱酸)
        "c1ccccc1O",   # Phenol (預期弱酸性)
        "c1ccccn1",     # Pyridine (預期鹼性)
        "CN(C)C",      # Trimethylamine (預期鹼性)
        "CC(C)(C)N",   # t-Butylamine (預期鹼性)
        "N[C@@H](C)C(=O)O", # Alanine (兩性，但看哪個基團被捕捉)
        "O=S(=O)(O)c1ccccc1" # Benzenesulfonic acid (預期強酸性)
    ]

    # --- 1. 載入數據 ---
    print(f"正在從 {data_path} 載入數據...")
    df = pd.read_csv(data_path)
    # df = df[df['pKa_num'] == 2]
    # 處理可能包含多個 pKa 值的數據
    print("處理 pKa 值...")
    
    # 創建一個新的 DataFrame 來存儲展開後的數據
    expanded_data = []
    
    for index, row in df.iterrows():
        smiles = row['SMILES']
        pka_num = row.get('pKa_num', 1)  # 默認為 1 如果沒有 pKa_num 列
        pka_value = literal_eval(row['pKa_value'])
        
        # 處理 pKa 值
        if pka_num == 1:
            # 單一 pKa 值情況
            try:
                pka = float(pka_value)
                expanded_data.append({'SMILES': smiles, 'pKa_value': pka,'pKa_num': int(pka_num), 'pKa_index': 0})
            except (ValueError, TypeError):
                print(f"警告: 無法解析 pKa 值 '{pka_value}' for {smiles}, 略過")
                continue
        else:
            # 多個 pKa 值情況，嘗試解析字串
            try:
                # 檢查pka_value是否已經是列表
                if isinstance(pka_value, list):
                    pka_list = pka_value  # 直接使用列表
                elif isinstance(pka_value, str):
                    # 移除引號並分割
                    pka_value = pka_value.strip('"\'')
                    if ',' in pka_value:
                        pka_list = [float(x.strip()) for x in pka_value.split(',')]
                else:
                    pka_list = [float(pka_value)]
                # 檢查解析出的 pKa 值數量是否與 pKa_num 一致
                if len(pka_list) != pka_num:
                    print(f"警告: pKa_num={pka_num} 但解析出 {len(pka_list)} 個值 for {smiles}")
                    print(f"    原始 pKa 值: {pka_value}")
                    print(f"    解析出的 pKa 值: {pka_list}")
                    print(f"    解析出的 pKa 值數量: {len(pka_list)}")
                
                # 為每個 pKa 值創建一行
                for i, pka in enumerate(pka_list):
                    expanded_data.append({'SMILES': smiles, 'pKa_value': pka,'pKa_num': int(pka_num), 'pKa_index': i})
            except Exception as e:
                print(f"警告: 解析 pKa 值 '{pka_value}' 時出錯 for {smiles}: {e}, 略過")
                continue
    
    # 創建新的 DataFrame
    expanded_df = pd.DataFrame(expanded_data)
    print(f"原始數據: {len(df)} 行, 展開後: {len(expanded_df)} 行")
    
    # 用展開後的數據替代原始 DataFrame
    df = expanded_df
    # df.to_csv("../src/test.csv", index=False)
    
    
    
    
    
    
    
    
    # --- 2. 為數據集中的每個 SMILES 提取特徵 ---
    print("正在為數據集提取分子特徵...")
    
    all_features = []  # 所有官能基特徵的列表
    smiles_to_features = {}  # 映射 SMILES 到其所有官能基特徵的索引
    
    for index, row in df.iterrows():
        smiles = row['SMILES']
        pka = row['pKa_value']
        pka_index = row.get('pKa_index', 0)  # 這個 pKa 值的索引 (對於多 pKa 情況)
        
        feature_list = extract_molecule_feature(smiles)
        
        if feature_list:  # 如果找到了至少一個官能基特徵
            # 記錄這個 SMILES 的所有官能基特徵索引
            if smiles not in smiles_to_features:
                smiles_to_features[smiles] = []
                
            # 只處理與當前 pKa_index 相關的官能基
            # 對於第 i 個 pKa 值，我們假設它與第 i 個官能基相關
            # 如果官能基數量少於 pKa 值數量，則使用最後一個官能基
            feature_count = len(feature_list)
            if feature_count > 0:
                # 選擇與 pKa_index 對應的官能基，或者最後一個
                feature_idx = min(pka_index, feature_count - 1)
                feature_info = feature_list[feature_idx]
                
                # 添加 pKa 值和索引到特徵信息中
                feature_info['pka'] = pka
                feature_info['pka_index'] = pka_index
                
                all_features.append(feature_info)
                feature_index = len(all_features) - 1
                smiles_to_features[smiles].append(feature_index)
    
    # 打印統計信息
    unique_smiles = len(smiles_to_features)
    total_features = len(all_features)
    print(f"處理了 {unique_smiles} 個唯一 SMILES")
    print(f"提取了 {total_features} 個官能基特徵")
    if unique_smiles > 0:
        print(f"平均每個分子有 {total_features/unique_smiles:.2f} 個官能基特徵")
    
    # 準備聚類數據
    if len(all_features) > 0:
        X = np.array([info['feature'] for info in all_features])
        y_pka = np.array([info['pka'] for info in all_features])
        feature_groups = [info['group_type'] for info in all_features]
        feature_smiles = [info['smiles'] for info in all_features]
        pka_indices = [info.get('pka_index', 0) for info in all_features]
        
        print(f"特徵矩陣形狀: {X.shape}, pKa 值數量: {len(y_pka)}")
    else:
        print("錯誤：沒有提取到任何有效特徵。程序終止。")
        sys.exit(1)

    # 添加整體工作流程註解
    """
    pKa 預測與分群分析工作流程 (Workflow)：

    1. 資料預處理：
       - 從 CSV 載入分子 SMILES 與實驗 pKa 值
       - 過濾無效數據

    2. 特徵提取：
       - 對每個分子提取多個官能基特徵 (每個官能基一個特徵向量)
       - 保留每個官能基的類型信息
       - 每個 SMILES 可以對應多個官能基特徵向量

    3. 降維與聚類：
       - 使用 PCA 將特徵降至較低維度 (默認 50 維或樣本數的 1/10)
       - 應用 K-means 算法在降維後的空間進行分群
       - 評估聚類效果 (Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index)

    4. 分群結果分析：
       - 計算每個群集的平均 pKa 值
       - 分析不同官能基類型的分布

    5. 視覺化：
       - 使用 PCA 降至 2 維進行聚類結果視覺化
       - 同時顯示群集分類與官能基類型
       - 繪製群集與 pKa 分布關係圖
       - 可選 t-SNE 視覺化以展示更複雜的數據結構

    6. 預測：
       - 對新分子提取多個官能基特徵
       - 針對每個官能基分別預測 pKa 值

    流程設計邏輯：
    - 支持多官能基分析：每個分子可以有多個官能基，各自具有特徵和 pKa 預測值
    - 先降維再聚類：減少高維空間中的"維度詛咒"問題，提高聚類效率
    - 保留變異性：PCA 降維保留了數據主要變異，濾除噪聲
    - 多重評估：使用多種指標評估聚類質量
    - 多角度視覺化：從不同角度展示分群結果，同時顯示官能基類型標記
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
            algorithm='elkan',  # 對高維數據效率更高的算法
              # 限制使用2個CPU核心
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
        
        # 創建一個包含兩張圖的圖像
        fig, axs = plt.subplots(1, 2, figsize=(20, 8))
        
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
        
        # ===== 5.1: PCA 降維視覺化 =====
        # 將降維後的結果再次映射到2D空間用於視覺化
        print("進行 PCA 降維用於視覺化...")
        # 使用與聚類相同的數據進行視覺化
        X_for_vis = X_transformed if use_pca else X
        
        # 為視覺化創建新的 PCA 模型，總是降到 2 維
        pca_vis = PCA(n_components=2, random_state=42)
        X_pca = pca_vis.fit_transform(X_for_vis)
        explained_var_ratio = pca_vis.explained_variance_ratio_
        print(f"視覺化 PCA 解釋方差比例: [{explained_var_ratio[0]:.4f}, {explained_var_ratio[1]:.4f}]")
        
        # 繪製 PCA 降維後的分群結果 - 同時顯示群集顏色和官能基標記
        for i in range(n_clusters_kmeans):
            cluster_mask = (cluster_labels == i)
            if not np.any(cluster_mask):
                continue
                
            avg_pka = cluster_avg_pka.get(i)
            n_samples = np.sum(cluster_mask)
            
            # 創建圖例標籤
            label_text = f'Cluster {i} (n={n_samples})'
            if avg_pka is not None:
                label_text += f', Avg pKa: {avg_pka:.2f}'
            
            # 為該群集內的每個官能基類型繪製不同標記
            # 獲取該群集的所有點
            cluster_points = X_pca[cluster_mask]
            cluster_groups = np.array(feature_groups)[cluster_mask]
            cluster_pka_indices = np.array(pka_indices)[cluster_mask]
            
            # 繪製主群集散點圖 (所有點用同一顏色但不同標記)
            axs[0].scatter(cluster_points[:, 0], cluster_points[:, 1],
                        color=colors[i], label=label_text, alpha=0.7, s=40, 
                        edgecolors='w', linewidths=0.5)
            
            # 為不同官能基分別繪製，使用不同標記
            unique_groups = np.unique(cluster_groups)
            for group in unique_groups:
                group_mask = cluster_groups == group
                if np.any(group_mask):
                    marker = group_markers.get(group, default_marker)
                    # 繪製該官能基的點 (僅作為小樣本顯示與圖例，不再繪製所有點)
                    small_sample = min(3, np.sum(group_mask))  # 最多顯示3個樣本點作為圖例
                    if i == 0:  # 只在第一個群集添加官能基圖例
                        axs[0].scatter(cluster_points[group_mask][:small_sample, 0], 
                                     cluster_points[group_mask][:small_sample, 1],
                                     marker=marker, s=80, facecolors='none', 
                                     edgecolors='black', linewidths=1.0, 
                                     label=f'{group}', alpha=0.9, zorder=11)
                        
            # 標記 pKa 索引 (可選)
            # 如果你想在圖中標記哪些點對應第一個、第二個 pKa 值等
            for pka_idx in np.unique(cluster_pka_indices):
                idx_mask = cluster_pka_indices == pka_idx
                if np.any(idx_mask) and pka_idx > 0:  # 只標記非零索引
                    idx_points = cluster_points[idx_mask]
                    # 為每個 pKa 索引添加一個特殊標記
                    axs[0].scatter(idx_points[:, 0], idx_points[:, 1],
                               marker='D', s=100, facecolors='none', 
                               edgecolors='red', linewidths=2.0,
                               alpha=0.7, zorder=12)

        # 繪製 K-means 中心點，使用更明顯的標記
        if hasattr(kmeans_model, 'cluster_centers_'):
            # 將聚類中心點在相同空間中進行可視化
            centroids_pca = pca_vis.transform(kmeans_model.cluster_centers_)
            # 使用菱形標記並保持與各自群集相同的顏色
            for i in range(n_clusters_kmeans):
                axs[0].scatter(centroids_pca[i, 0], centroids_pca[i, 1], 
                             marker='D', s=200, c=[colors[i]], edgecolors='black', 
                             linewidths=2.0, zorder=10)
            # 添加一個單獨的圖例項目
            axs[0].scatter([], [], marker='D', s=200, c='gray', edgecolors='black',
                          linewidths=2.0, label='Centroids', zorder=10)
            
            # 為每個中心點添加編號標籤
            for i, (x, y) in enumerate(centroids_pca):
                axs[0].text(x + 0.05, y + 0.05, f'{i}', fontsize=12, fontweight='bold',
                          ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'), zorder=11)
        
        # 美化第一張圖的標籤和標題
        axs[0].set_xlabel(f"PCA Dimension 1 ({explained_var_ratio[0]:.2%})", fontsize=12)
        axs[0].set_ylabel(f"PCA Dimension 2 ({explained_var_ratio[1]:.2%})", fontsize=12)
        axs[0].set_title(f"K-means Clustering Results after PCA (k={n_clusters_kmeans})", 
                       fontsize=14, fontweight='bold')
        axs[0].grid(True, linestyle='--', alpha=0.5)
        
        # 分兩列顯示圖例，提高可讀性
        handles, labels = axs[0].get_legend_handles_labels()
        cluster_handles = handles[:n_clusters_kmeans+1]  # 聚類和中心點圖例
        group_handles = handles[n_clusters_kmeans+1:]    # 官能基圖例
        
        # 先顯示聚類圖例
        legend1 = axs[0].legend(cluster_handles, labels[:n_clusters_kmeans+1], 
                               title="Clusters", bbox_to_anchor=(1.05, 1), 
                               loc='upper left', fontsize=10, frameon=True, 
                               title_fontsize=12)
        axs[0].add_artist(legend1)
        
        # 再顯示官能基圖例
        if group_handles:
            legend2 = axs[0].legend(group_handles, labels[n_clusters_kmeans+1:],
                                  title="Functional Groups", bbox_to_anchor=(1.05, 0.5), 
                                  loc='center left', fontsize=10, frameon=True, 
                                  title_fontsize=12)
            legend2.get_frame().set_alpha(0.9)
        
        # ===== 5.2: pKa vs 主要特徵分布圖 =====
        # 分析群集特徵與pKa的關係，增加官能基標記
        # 針對每個群集繪製
        for i in range(n_clusters_kmeans):
            cluster_mask = (cluster_labels == i)
            if not np.any(cluster_mask):
                continue
                
            avg_pka = cluster_avg_pka.get(i)
            n_samples = np.sum(cluster_mask)
            
            # 為每個群集創建標籤
            label_text = f'Cluster {i} (n={n_samples})'
            if avg_pka is not None:
                label_text += f', Avg pKa: {avg_pka:.2f}'
            
            # 獲取該群集的所有點
            cluster_points = X_pca[cluster_mask]
            cluster_pkas = y_pka[cluster_mask]
            cluster_groups = np.array(feature_groups)[cluster_mask]
            
            # 繪製主圖 (顏色區分群集)
            axs[1].scatter(cluster_points[:, 0], cluster_pkas,
                        color=colors[i], label=label_text, alpha=0.7, s=40,
                        edgecolors='w', linewidths=0.5)
            
            # 添加官能基標記
            unique_groups = np.unique(cluster_groups)
            for group in unique_groups:
                group_mask = cluster_groups == group
                if np.any(group_mask):
                    marker = group_markers.get(group, default_marker)
                    # 在這裡我們不添加圖例 (已在第一張圖中添加)
                    axs[1].scatter(cluster_points[group_mask][:, 0], 
                                 cluster_pkas[group_mask],
                                 marker=marker, s=80, facecolors='none', 
                                 edgecolors='black', linewidths=1.0,
                                 alpha=0.9, zorder=11)
            
            # 添加每個群集的平均值標記
            if avg_pka is not None:
                mean_x = np.mean(cluster_points[:, 0])
                axs[1].scatter(mean_x, avg_pka, marker='D', s=100, 
                             color=colors[i], edgecolors='black', linewidths=1.5, zorder=10)
                axs[1].text(mean_x + 0.05, avg_pka + 0.2, f'{i}', fontsize=12, fontweight='bold',
                         ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'), zorder=11)
        
        # 添加一條趨勢線，顯示 PCA 第一維度與 pKa 的關係
        try:
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(X_pca[:, 0], y_pka)
            x_range = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)
            axs[1].plot(x_range, intercept + slope * x_range, 'k--', 
                       label=f'Trend: r={r_value:.2f}', linewidth=1.5, alpha=0.7)
        except:
            pass
        
        # 美化第二張圖的標籤和標題
        axs[1].set_xlabel(f"PCA Dimension 1 ({explained_var_ratio[0]:.2%})", fontsize=12)
        axs[1].set_ylabel("Experimental pKa Values", fontsize=12)
        axs[1].set_title(f"pKa Distribution by Cluster and Functional Group (k={n_clusters_kmeans})", 
                       fontsize=14, fontweight='bold')
        axs[1].grid(True, linestyle='--', alpha=0.5)
        
        # 調整圖例位置和樣式
        legend1 = axs[1].legend(title="Clusters", bbox_to_anchor=(1.05, 1), 
                              loc='upper left', fontsize=10, frameon=True, 
                              title_fontsize=12)
        legend1.get_frame().set_alpha(0.9)
        
        # 調整整體布局，預留足夠空間給圖例
        fig.subplots_adjust(right=0.75)
        
        # 添加總體圖說明
        fig.suptitle(f"Multi-functional Group Analysis of {X.shape[0]} Features from {unique_smiles} Molecules", 
                   fontsize=16, fontweight='bold', y=0.98)
        
        
        
        
        
        
        fontsize = 24
        
        # ===== 5.3: t-SNE 補充視覺化 =====
        # 使用t-SNE進行降維視覺化，提供與PCA不同的視角
        if X.shape[0] <= 5000:
            try:
                print("進行t-SNE降維視覺化...")
                # 創建更寬的圖，為圖例預留足夠空間
                plt.figure(figsize=(20, 16), dpi=300)
                
                # 檢查是否有保存的t-SNE結果可以載入
                tsne_cache_file = "../src/tsne_results.pkl"
                tsne_needs_compute = True
                
                if os.path.exists(tsne_cache_file):
                    try:
                        print(f"發現緩存的t-SNE結果，嘗試載入...")
                        with open(tsne_cache_file, 'rb') as f:
                            cached_data = pickle.load(f)
                            # 檢查緩存的維度是否匹配當前數據
                            if (cached_data['n_samples'] == X.shape[0] and 
                                cached_data['n_features'] == X.shape[1] and
                                cached_data['n_clusters'] == n_clusters_kmeans):
                                X_tsne = cached_data['X_tsne']
                                print(f"成功載入緩存的t-SNE結果 (形狀: {X_tsne.shape})")
                                tsne_needs_compute = False
                            else:
                                print(f"緩存數據維度不匹配，將重新計算t-SNE")
                                print(f"緩存: 樣本={cached_data['n_samples']}, 特徵={cached_data['n_features']}, 聚類={cached_data['n_clusters']}")
                                print(f"當前: 樣本={X.shape[0]}, 特徵={X.shape[1]}, 聚類={n_clusters_kmeans}")
                    except Exception as e:
                        print(f"載入t-SNE緩存失敗: {e}")
                
                if tsne_needs_compute:
                    # 準備t-SNE輸入數據
                    if not use_pca:
                        # 若之前未用PCA，先降維到50維以提高t-SNE效率
                        pca_tsne = PCA(n_components=min(50, X.shape[1]), random_state=42)
                        X_for_tsne = pca_tsne.fit_transform(X)
                        print(f"t-SNE預處理：PCA降至{X_for_tsne.shape[1]}維，保留{sum(pca_tsne.explained_variance_ratio_):.2f}變異")
                    else:
                        # 使用已降維的數據
                        X_for_tsne = X_transformed
                        print(f"t-SNE使用已降維的{X_for_tsne.shape[1]}維數據")
                    
                    # 配置t-SNE參數
                    perplexity = min(30, X.shape[0] // 5)  # 適應數據量調整perplexity
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                              learning_rate='auto', init='pca', n_iter=2000)
                    print(f"開始計算t-SNE，這可能需要幾分鐘時間...")
                    X_tsne = tsne.fit_transform(X_for_tsne)
                    print(f"t-SNE計算完成，結果形狀: {X_tsne.shape}")
                    
                    # 將結果保存到pickle文件
                    try:
                        with open(tsne_cache_file, 'wb') as f:
                            pickle.dump({
                                'X_tsne': X_tsne,
                                'n_samples': X.shape[0],
                                'n_features': X.shape[1],
                                'n_clusters': n_clusters_kmeans,
                                'cluster_labels': cluster_labels,  # 保存聚類標籤
                                'feature_groups': feature_groups,  # 保存官能基類型
                                'feature_smiles': feature_smiles   # 保存SMILES字符串
                            }, f)
                        print(f"t-SNE結果、聚類標籤和官能基信息已保存至 {tsne_cache_file}")
                    except Exception as e:
                        print(f"保存t-SNE結果失敗: {e}")
                
                # 繪製群集
                for i in range(n_clusters_kmeans):
                    cluster_mask = (cluster_labels == i)
                    if not np.any(cluster_mask):
                        continue
                        
                    # 獲取資訊
                    avg_pka = cluster_avg_pka.get(i, None)
                    n_samples = np.sum(cluster_mask)
                    label = f'Cluster {i} (n={n_samples})'
                    if avg_pka is not None:
                        label += f', Avg pKa: {avg_pka:.2f}'
                    
                    # 繪製群集點
                    cluster_points = X_tsne[cluster_mask]
                    cluster_groups = np.array(feature_groups)[cluster_mask]
                    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                               color=colors[i], label=label, alpha=0.7, s=40,
                               edgecolors='w', linewidths=0.5)
                    
                    # 繪製群集中心
                    center_x = np.mean(cluster_points[:, 0])
                    center_y = np.mean(cluster_points[:, 1])
                    plt.scatter(center_x, center_y, marker='D', s=100, 
                              color=colors[i], edgecolors='black', linewidths=1.5, zorder=10)
                    plt.text(center_x, center_y, f'{i}', fontsize=fontsize, fontweight='bold',
                           ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, 
                                                             edgecolor='none', boxstyle='round,pad=0.2'), zorder=11)
                    
                    # 為每種官能基添加形狀標記
                    for group in np.unique(cluster_groups):
                        group_mask = cluster_groups == group
                        if np.any(group_mask):
                            marker = group_markers.get(group, default_marker)
                            plt.scatter(cluster_points[group_mask][:, 0], 
                                      cluster_points[group_mask][:, 1],
                                      marker=marker, s=80, facecolors='none', 
                                      edgecolors='black', linewidths=1.0, alpha=0.9, zorder=11)
                
                # 添加官能基圖例項目（只顯示實際存在的官能基）
                existing_groups = set(feature_groups)
                for group, marker in group_markers.items():
                    if group in existing_groups:
                        plt.scatter([], [], marker=marker, s=80, facecolors='none', 
                                  edgecolors='black', linewidths=1.0, 
                                  label=f'{group}', alpha=0.9)
                
                # 美化圖表
                plt.xlabel("t-SNE Dimension 1", fontsize=fontsize)
                plt.ylabel("t-SNE Dimension 2", fontsize=fontsize)
                plt.title(f"Functional Group t-SNE Distribution and K-means Clustering Results (k={n_clusters_kmeans})",
                        fontsize=fontsize, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tick_params(axis='x', labelsize=fontsize)
                plt.tick_params(axis='y', labelsize=fontsize)
                
                # 分開顯示兩種圖例
                handles, labels = plt.gca().get_legend_handles_labels()
                
                # 找出群集與官能基圖例的分界
                cluster_handles = []
                cluster_labels = []
                group_handles = []
                group_labels = []
                
                for h, label in zip(handles, labels):
                    if label.startswith('Cluster'):
                        cluster_handles.append(h)
                        cluster_labels.append(label)
                    else:
                        group_handles.append(h)
                        group_labels.append(label)
                
                # 調整布局，為圖例預留更多空間
                plt.subplots_adjust(right=0.65)  # 將右邊界進一步縮小，為圖例預留更多空間
                
                # 添加群集圖例，調整位置避免溢出
                cluster_legend = plt.legend(cluster_handles, cluster_labels,
                                          title="Clusters", bbox_to_anchor=(1.1, 1), 
                                          loc='upper left', fontsize=fontsize, frameon=True, 
                                          title_fontsize=12)
                plt.gca().add_artist(cluster_legend)
                
                # 添加官能基圖例，調整位置避免溢出
                if group_handles:
                    # 將官能基圖例放在群集圖例下方
                    group_legend = plt.legend(group_handles, group_labels,
                                            title="Functional Groups", bbox_to_anchor=(1.1, 0.5), 
                                            loc='center left', fontsize=fontsize, frameon=True, 
                                            title_fontsize=12)
                
                # 保存t-SNE圖，增加邊界以確保圖例完整顯示
                tsne_plot_filename = output_plot_filename.replace('.png', '_tsne.png')
                plt.savefig(tsne_plot_filename, dpi=300, bbox_inches='tight', pad_inches=1.5)
                print(f"t-SNE分群結果圖已保存至 {tsne_plot_filename}")
                plt.close()
            except Exception as e:
                print(f"t-SNE降維失敗：{e}")
        
    else:
        pass

    # 保存主圖，使用高分辨率
    # plt.tight_layout(rect=[0, 0, 1, 0.96])  # 這會覆蓋之前的布局調整
    plt.savefig(output_plot_filename, dpi=300, bbox_inches='tight')
    print(f"分群結果圖已保存至 {output_plot_filename}")
    plt.close()




