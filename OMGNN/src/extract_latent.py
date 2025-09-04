# src/extract_latent.py
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def collect_latent(model, loader, device="cuda"):
    model.eval().to(device)
    latents, labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)

            # 取得 (logits, pka, losses, latent_steps)
            *_, latent_steps = model(batch, return_latent=True)

            for step_idx, h in enumerate(latent_steps):
                latents.append(h.numpy())
                labels.append(step_idx)     # dissociation step = label

    return np.vstack(latents), np.array(labels)

def collect_latent_with_pka(model, loader, device="cuda"):
    """
    收集模型的潛在表示並返回相應的 pKa 值
    
    參數:
        model: 模型實例
        loader: 數據加載器
        device: 計算設備
    
    返回:
        latents: 形狀為 (n_samples, n_features) 的潛在表示矩陣
        pka_values: 形狀為 (n_samples,) 的 pKa 值數組
    """
    model.eval().to(device)
    Z, pkas = [], []

    with torch.no_grad():
        for batch in loader:                      # 建議 loader.batch_size = 1
            batch = batch.to(device, non_blocking=True)

            *_, latent_steps, pka_steps = model(batch, return_latent=True)

            # 兩條 list 長度一定相等
            for h, pk in zip(latent_steps, pka_steps):
                Z.append(h.numpy())               # hidden vector
                pkas.append(pk.item())            # scalar pKa

    return np.vstack(Z), np.array(pkas)

def collect_latent_with_pka_ver2(model, loader, device="cuda"):
    """
    回傳 Z, pKa, smiles  (3 個 array / list)
    """
    model.eval().to(device)
    Z, pkas, smiles_all = [], [], []

    with torch.no_grad():
        for batch in loader:            # 建議 batch_size = 1
            batch = batch.to(device, non_blocking=True)

            # ---- 你需要把 SMILES 放在 batch 裡，例如 batch.smiles ----
            smiles_batch = batch.smiles      # list[str]，長度 = dissociation steps

            *_, latent_steps, pka_steps = model(batch, return_latent=True)

            for h, pk, smi in zip(latent_steps, pka_steps, smiles_batch):
                Z.append(h.cpu().numpy())
                pkas.append(pk.item())
                smiles_all.append(smi)

    return np.vstack(Z), np.array(pkas), np.array(smiles_all, dtype=object)

from sklearn.neighbors import NearestNeighbors

def tsne_and_plot_ver2(
    X, y, smiles,
    title="t-SNE of latent steps",
    save_png=None,
    perplexity=30,
    metric="cosine",
    colormap="tab10",
    colorbar_label=None,
    *,
    pka_threshold=5.0,
    n_neighbors=10,          # 鄰近點個數 (KNN)
    radius=None              # 或用半徑；兩者擇一生效
):
    """
    X          : (n, d) 特徵
    y          : (n,) 連續或離散標籤，用於著色
    smiles     : (n,) SMILES 字串
    pka_threshold : 若 pk > threshold 代表「高 pKa」
    n_neighbors / radius : 定義鄰近點。若兩者皆提供，優先用 radius
    """
    n_samples, n_features = X.shape
    # ---------- 1. 降維 ----------
    if n_features >= 2:
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, n_samples - 1),
            metric=metric,
            init="pca",
            random_state=42,
        )
        X2 = tsne.fit_transform(X)
    else:   # 低維 fallback（同你原本的處理）
        X2 = np.hstack([X, 0.01 * np.random.randn(n_samples, 1)])

    # ---------- 2. 畫散點 ----------
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X2[:, 0], X2[:, 1], c=y, s=18, cmap=colormap, alpha=0.8)
    plt.title(title)

    # ---------- 3. 找「鄰近且 pKa > 閾值」 ----------
    # 建立 KNN / radius tree
    if radius is None:
        # k-NN 模式：先 fit，再對每點取 k 鄰居
        knn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X2)
        idx_high = np.where(y > pka_threshold)[0]      # 先挑本身 pKa 高的點
        for idx in idx_high:
            dist, neigh = knn.kneighbors(X2[idx, :][None, :], n_neighbors=n_neighbors + 1)
            neigh = neigh[0][1:]                       # 去掉自己
            # 檢查鄰居中是否也有人 pKa 高
            if np.any(y[neigh] > pka_threshold):
                plt.text(
                    X2[idx, 0], X2[idx, 1],
                    smiles[idx],
                    fontsize=6,  # 視需求調整
                    ha="center",
                    va="center"
                )
    else:
        # radius 模式：取固定半徑內的鄰居
        tree = NearestNeighbors(radius=radius).fit(X2)
        idx_high = np.where(y > pka_threshold)[0]
        for idx in idx_high:
            neigh = tree.radius_neighbors(X2[idx, :][None, :], radius=radius, return_distance=False)[0]
            neigh = neigh[neigh != idx]
            if np.any(y[neigh] > pka_threshold):
                plt.text(X2[idx, 0], X2[idx, 1], smiles[idx], fontsize=6, ha="center", va="center")

    # ---------- 4. legend / colorbar ----------
    if colorbar_label is not None:
        cbar = plt.colorbar(sc)
        cbar.set_label(colorbar_label)
    else:
        handles, _ = sc.legend_elements()
        plt.legend(handles, [f"step {i}" for i in range(len(handles))],
                   title="Dissociation Steps", fontsize=9, framealpha=0.7)

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()

    if save_png:
        plt.savefig(save_png, dpi=300, bbox_inches="tight")
        print(f"Plot saved → {save_png}")
    else:
        plt.show()

def tsne_and_plot(X, y, title="t-SNE of latent steps", save_png=None, perplexity=30, metric="cosine", colormap="tab10", colorbar_label=None):
    """
    使用t-SNE將高維特徵降為2D並繪製散點圖
    
    參數:
        X: 形狀為(n_samples, n_features)的特徵矩陣
        y: 形狀為(n_samples,)的標籤向量
        title: 圖表標題
        save_png: 保存圖像的路徑，若為None則顯示圖像
        perplexity: t-SNE的perplexity參數，控制局部與全局結構的平衡
        metric: 距離度量方式，可選"cosine"、"euclidean"等
        colormap: 色彩映射，可以是"tab10"、"viridis"等
        colorbar_label: 顏色條標籤，若為None則不顯示顏色條
    """
    print(f"運行t-SNE，特徵矩陣形狀: {X.shape}，標籤數量: {len(np.unique(y))}")
    
    # 檢查特徵維度，如果維度太低，則進行處理
    n_samples, n_features = X.shape
    
    if n_features < 2:
        print(f"警告: 特徵維度 ({n_features}) 小於2。無法進行t-SNE降維，將直接使用原始特徵繪圖。")
        # 如果特徵維度為1，直接用該維度作為x軸，創建一個隨機噪聲作為y軸
        if n_features == 1:
            X2 = np.zeros((n_samples, 2))
            X2[:, 0] = X.flatten()  # 使用原始一維特徵作為x軸
            X2[:, 1] = 0.01 * np.random.randn(n_samples)  # 添加微小噪聲作為y軸
        else:
            X2 = X  # 這種情況不應該發生，但以防萬一
    else:
        # 正常使用t-SNE降維
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, n_samples - 1),  # 確保perplexity不會過大
            metric=metric,
            random_state=42,
            init="pca",
        )
        X2 = tsne.fit_transform(X)

    # 創建散點圖
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(X2[:,0], X2[:,1], c=y, s=15, cmap=colormap, alpha=0.8)
    plt.title(title)
    
    # 檢查是否為連續值 (如 pKa) 還是離散類別 (如步驟)
    if colorbar_label is not None:
        # 連續值著色 - 顯示顏色條
        cbar = plt.colorbar(sc)
        cbar.set_label(colorbar_label)
    else:
        # 離散類別著色 - 顯示圖例
        handles, labels = sc.legend_elements()
        step_labels = [f"step {i}" for i in range(len(handles))]
        plt.legend(handles, step_labels, title="Dissociation Steps", loc="best", 
                   fontsize=10, framealpha=0.7)
    
    # 添加軸標籤
    plt.xlabel("t-SNE 1" if n_features >= 2 else "Feature 1")
    plt.ylabel("t-SNE 2" if n_features >= 2 else "Noise")
    
    # 添加網格線
    plt.grid(alpha=0.3, linestyle="--")
    
    plt.tight_layout()
    
    # 保存或顯示圖像
    if save_png:
        plt.savefig(save_png, dpi=300, bbox_inches="tight")
        print(f"圖像已保存 → {save_png}")
    else:
        plt.show()

# 示例使用方法 (可以添加為註釋)
"""
X, y_pka = collect_latent_with_pka(model, test_loader, device)

tsne_and_plot(
    X              = X,
    y              = y_pka,               # 直接拿 pKa 當 color
    title          = f"t-SNE – {version} (colored by pKa)",
    save_png       = f"{version}_latent_tsne_pka.png",
    perplexity     = 30,
    metric         = "cosine",
    colormap       = "viridis",           # 選連續色譜
    colorbar_label = "experimental pKa"   # 只要非 None → 顯示 color-bar
)
""" 