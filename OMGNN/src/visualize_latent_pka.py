# src/visualize_latent_pka.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import argparse
from pathlib import Path
import sys

# 添加src目錄到路徑
sys.path.append("src/")

from extract_latent import collect_latent_with_pka, tsne_and_plot
from OMGNN.src.error_self_pka_models import pka_GNN
from self_pka_dataset import PKADataset
from self_pka_trainutils import load_config_by_version, pka_Dataloader
from self_pka_chemutils import tensorize_for_pka


def parse_args():
    parser = argparse.ArgumentParser(description='使用t-SNE可視化模型潛在空間與pKa關聯')
    parser.add_argument('--version', type=str, required=True,
                        help='模型版本，例如 pka_ver10')
    parser.add_argument('--config_csv', type=str, default='../data/pka_parameters.csv',
                        help='模型配置CSV文件路徑')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='t-SNE的perplexity參數')
    parser.add_argument('--metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'], 
                        help='距離度量方式')
    parser.add_argument('--colormap', type=str, default='viridis',
                        choices=['viridis', 'plasma', 'inferno', 'magma'],
                        help='色彩映射方案')
    parser.add_argument('--device', type=str, default='cuda',
                        help='計算設備，例如 cuda 或 cpu')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 載入配置
    print(f"\n載入模型配置: {args.version}")
    cfg = load_config_by_version(args.config_csv, args.version)
    
    # 設置路徑
    model_path = f"../results/{args.version}"
    output_dir = f"../results/{args.version}/viz"
    best_model_path = os.path.join(model_path, f"{args.version}_best.pkl")
    
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 確定設備
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用設備: {device}")
    
    # 載入數據
    print(f"載入數據: {cfg['input']}")
    cols = ["smiles", "pka_values", "pka_matrix"]
    train_loader, test_loader = pka_Dataloader.data_loader_pka(
        file_path=cfg["input"],
        columns=cols,
        tensorize_fn=tensorize_for_pka,
        batch_size=1,  # 固定為1，確保正確提取潛在表示
        test_size=cfg["test_size"],
    )
    
    # 創建模型
    print("初始化模型...")
    model = pka_GNN(
        node_dim=cfg["num_features"],
        bond_dim=11,
        hidden_dim=cfg["num_features"],
        output_dim=1,
        dropout=cfg["dropout"],
        depth=cfg["depth"]
    ).to(device)
    
    # 載入最佳檢查點
    if not os.path.exists(best_model_path):
        print(f"警告: 找不到最佳模型檢查點 {best_model_path}，嘗試載入最終模型...")
        best_model_path = os.path.join(model_path, f"{args.version}_final.pkl")
        
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"找不到模型檢查點: {best_model_path}")
    
    print(f"載入模型檢查點: {best_model_path}")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 獲取pKa標準化參數
    try:
        # 嘗試從訓練集計算pKa統計數據
        print("計算pKa均值和標準差...")
        vals = []
        for batch in train_loader:
            mask = batch.pka_labels > 0
            vals.append(batch.pka_labels[mask].float())
        vals = torch.cat(vals)
        pka_mean = vals.mean().item()
        pka_std = max(vals.std(unbiased=False).item(), 1e-6)
        print(f"pKa統計: 均值={pka_mean:.3f}, 標準差={pka_std:.3f}")
        
        # 更新模型的標準化參數
        model.set_pka_normalization(pka_mean, pka_std)
    except Exception as e:
        print(f"無法計算pKa統計數據: {e}")
        print("使用模型內置的默認標準化參數")
    
    # 處理訓練集
    print(f"\n從訓練集收集潛在向量進行t-SNE可視化...")
    X_train, y_pka_train = collect_latent_with_pka(model, train_loader, device=device)
    print(f"收集完成: {X_train.shape[0]}個特徵向量")
    
    # 統計pKa值
    print(f"訓練集pKa值統計: 最小={y_pka_train.min():.2f}, 最大={y_pka_train.max():.2f}, 平均={y_pka_train.mean():.2f}")
    
    # 創建訓練集t-SNE可視化
    train_output_path = os.path.join(output_dir, f"{args.version}_latent_tsne_train_pka.png")
    tsne_and_plot(
        X=X_train,
        y=y_pka_train,
        title=f"t-SNE - {args.version} (colored by pKa, Training Set)",
        save_png=train_output_path,
        perplexity=args.perplexity,
        metric=args.metric,
        colormap=args.colormap,
        colorbar_label="experimental pKa"
    )
    print(f"訓練集t-SNE圖已保存到: {train_output_path}")
    
    # 對數轉換的訓練集可視化
    if np.max(y_pka_train) > 0:
        y_pka_train_log = np.log10(np.maximum(y_pka_train, 1e-6))
        train_output_path_log = os.path.join(output_dir, f"{args.version}_latent_tsne_train_log_pka.png")
        tsne_and_plot(
            X=X_train,
            y=y_pka_train_log,
            title=f"t-SNE - {args.version} (colored by log10(pKa), Training Set)",
            save_png=train_output_path_log,
            perplexity=args.perplexity,
            metric=args.metric,
            colormap=args.colormap,
            colorbar_label="log10(pKa)"
        )
        print(f"訓練集對數轉換的t-SNE圖已保存到: {train_output_path_log}")
    
    # 處理測試集
    print(f"\n從測試集收集潛在向量進行t-SNE可視化...")
    X_test, y_pka_test = collect_latent_with_pka(model, test_loader, device=device)
    print(f"收集完成: {X_test.shape[0]}個特徵向量")
    
    # 統計pKa值
    print(f"測試集pKa值統計: 最小={y_pka_test.min():.2f}, 最大={y_pka_test.max():.2f}, 平均={y_pka_test.mean():.2f}")
    
    # 創建測試集t-SNE可視化
    test_output_path = os.path.join(output_dir, f"{args.version}_latent_tsne_test_pka.png")
    tsne_and_plot(
        X=X_test,
        y=y_pka_test,
        title=f"t-SNE - {args.version} (colored by pKa, Test Set)",
        save_png=test_output_path,
        perplexity=args.perplexity,
        metric=args.metric,
        colormap=args.colormap,
        colorbar_label="experimental pKa"
    )
    print(f"測試集t-SNE圖已保存到: {test_output_path}")
    
    # 對數轉換的測試集可視化
    if np.max(y_pka_test) > 0:
        y_pka_test_log = np.log10(np.maximum(y_pka_test, 1e-6))
        test_output_path_log = os.path.join(output_dir, f"{args.version}_latent_tsne_test_log_pka.png")
        tsne_and_plot(
            X=X_test,
            y=y_pka_test_log,
            title=f"t-SNE - {args.version} (colored by log10(pKa), Test Set)",
            save_png=test_output_path_log,
            perplexity=args.perplexity,
            metric=args.metric,
            colormap=args.colormap,
            colorbar_label="log10(pKa)"
        )
        print(f"測試集對數轉換的t-SNE圖已保存到: {test_output_path_log}")
    
    print(f"\n完成! 所有t-SNE圖像已保存到: {output_dir}")


if __name__ == "__main__":
    main() 