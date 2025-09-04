import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np

def plot_learning_curves(csv_path, output_dir, version_name, epoch_count=None):
    """
    繪製訓練和測試損失曲線
    
    Args:
        csv_path: 包含訓練日誌的CSV文件路徑
        output_dir: 輸出圖像的目錄
        version_name: 模型版本名稱
        epoch_count: 當前訓練的epoch數（如果是在訓練過程中調用）
    """
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    title_suffix = f" (Progress: Epoch {epoch_count})" if epoch_count is not None else ""
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), dpi=300)
    
    fontsize = 16
    # 1. 總損失曲線
    ax = axes[0]
    ax.plot(df['epoch'], df['train_loss'], 'b-', label='Training Loss')
    ax.plot(df['epoch'], df['test_loss'], 'r-', label='Validation Loss')
    ax.set_title(f'Total Loss Curves{title_suffix}', fontsize=fontsize)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_ylabel('Loss', fontsize=fontsize)
    ax.set_yscale('log')  # 使用對數刻度
    ax.legend(fontsize=fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 2. 分類損失曲線
    ax = axes[1]
    ax.plot(df['epoch'], df['train_cla_loss'], 'b-', label='Training Classification Loss')
    ax.plot(df['epoch'], df['test_cla_loss'], 'r-', label='Validation Classification Loss')
    ax.set_title(f'Classification Loss Curves{title_suffix}', fontsize=fontsize)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_ylabel('Classification Loss', fontsize=fontsize)
    ax.set_yscale('log')  # 使用對數刻度
    ax.legend(fontsize=fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 3. 回歸損失曲線
    ax = axes[2]
    ax.plot(df['epoch'], df['train_reg_loss'], 'b-', label='Training Regression Loss')
    ax.plot(df['epoch'], df['test_reg_loss'], 'r-', label='Validation Regression Loss')
    ax.set_title(f'Regression Loss Curves{title_suffix}', fontsize=fontsize)
    ax.set_xlabel('Epoch', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_ylabel('Regression Loss', fontsize=fontsize)
    ax.set_yscale('log')  # 使用對數刻度
    ax.legend(fontsize=fontsize)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{version_name}_loss_curves.png'))
    # print(f"損失曲線已保存到 {os.path.join(output_dir, f'{version_name}_loss_curves.png')}")
    plt.close(fig)
    
    # 繪製RMSE曲線
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=300)
    ax = axes[0]
    ax.plot(df['epoch'], df['rmse'], 'm-', label='RMSE')
    ax.set_title(f'RMSE Curves{title_suffix}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_yscale('log')  # 使用對數刻度
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # # 分類指標曲線 (Precision, Recall, F1)
    # ax = axes[1]
    # ax.plot(df['epoch'], df['accuracy'], 'g-', label='Accuracy')
    # ax.plot(df['epoch'], df['precision'], 'r-', label='Precision')
    # ax.plot(df['epoch'], df['recall'], 'b-', label='Recall')
    # ax.set_title(f'Recall Curve{title_suffix}')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('Score')
    # ax.set_ylim(0, 1)
    # ax.legend()
    # ax.grid(True, linestyle='--', alpha=0.7)
    
    # 學習率曲線
    ax = axes[1]
    ax.plot(df['epoch'], df['learning_rate'], 'b-', linewidth=2)
    ax.set_title(f'Learning Rate Schedule{title_suffix}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')  # 使用對數刻度更好地顯示學習率變化
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, f'{version_name}_graph.png'))
    plt.close(fig)
    
    if epoch_count is not None:
        print(f"學習曲線已更新，當前進度: Epoch {epoch_count}")
        print(f"RMSE曲線已保存到 {os.path.join(output_dir, f'{version_name}_graph.png')}")
        print(f"學習率曲線已保存到 {os.path.join(output_dir, f'{version_name}_loss_curves.png')}")
    else:
        print(f"學習曲線已保存到 {output_dir}")
    
    # # 如果測試準確率達到令人滿意的水平，則顯示祝賀消息
    # if 'f1' in df.columns and df['f1'].max() > 0.5:
    #     print(f"🎉 恭喜！模型達到了不錯的性能，最高F1分數為 {df['f1'].max():.4f}")
    
    return True

if __name__ == "__main__":
    # Set CSV file path and output directory
    csv_path = "../results/pka_ver10/pka_ver10_loss.csv"
    output_dir = "../results/pka_ver10/"
    version_name = "pka_ver10"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot learning curves
    plot_learning_curves(csv_path, output_dir, version_name)




