import os
import pandas as pd
import numpy as np
import sys
from typing import Dict, List, Tuple
sys.path.append("../model")
import matplotlib.pyplot as plt
from collections import Counter              
from pathlib import Path
import torch

def build_metal_vocab(df: pd.DataFrame) -> Dict[str, int]:
    metals = sorted(df["metal_ion"].unique().tolist())
    return {m: i for i, m in enumerate(metals)}

"""
此處函式有:
1. output_evaluation_results
2. plot_boxplot
3. create_weight_tables


"""


def output_evaluation_results(metrics_tr, metrics_te, VERSION, cfg):
    # 串接結果
    smiles = metrics_tr["smiles"] + metrics_te["smiles"]
    metal_ion = metrics_tr["metal_ion"] + metrics_te["metal_ion"]
    y_true  = np.concatenate([metrics_tr["y_true"],  metrics_te["y_true"]])
    y_pred  = np.concatenate([metrics_tr["y_pred"],  metrics_te["y_pred"]])
    is_train = np.concatenate([metrics_tr["is_train"], metrics_te["is_train"]])

    res = pd.DataFrame({
        'metal_ion': metal_ion,
        'smiles': smiles,
        'y_true': y_true,
        'y_pred': y_pred,
        'is_train': is_train,
    })


    df = pd.read_csv(cfg["metal_csv"])
    metal2idx = build_metal_vocab(df)
    # 創建反向映射：從索引到金屬離子名稱
    idx2metal = {v: k for k, v in metal2idx.items()}
    # 使用反向映射將索引轉換為金屬離子名稱
    res['metal_ion'] = res['metal_ion'].apply(lambda x: idx2metal[x])
    res['error'] = res['y_pred'] - res['y_true']
    res['abs_error'] = res["error"].abs()
    res.sort_values(by='abs_error', ascending=True, inplace=True)
    
    res['y_true'] = res['y_true'].apply(lambda x: round(x, 2))
    res['y_pred'] = res['y_pred'].apply(lambda x: round(x, 4))
    res.to_csv(os.path.join(cfg["output_dir"], f"{VERSION}_evaluation_results.csv"), index=False)
    output_path = os.path.join(cfg["output_dir"], f"{VERSION}_evaluation_results.csv")
    res.to_csv(output_path, index=False)
    print(f"evaluation_results已保存至: {output_path}")
    
    
    font_size = 18
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(y_true, bins=100, alpha=0.5, label="True binding constant", color="#303030")
    ax.hist(y_pred, bins=100, alpha=0.5, label="Predicted binding constant", color="#ff0050")
    ax.set_xlabel("Binding constant", fontsize=font_size)
    ax.set_ylabel("Frequency", fontsize=font_size)
    ax.set_title("Binding Constant Distribution", fontsize=font_size)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"../output/{VERSION}/pka_combined_{VERSION}.png")
    print(f"[PLOT]分布圖： ../output/{VERSION}/pka_combined_{VERSION}.png saved")
    
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), dpi=300)
    ax = axes[0]
    ax.hist(res['error'], bins=100)
    ax.set_title("Error Distribution of pKa", fontsize=font_size)
    ax.set_xlabel("Error", fontsize=font_size)
    ax.set_ylabel("Frequency", fontsize=font_size)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # output_path = os.path.join(cfg["output_dir"], f"{VERSION}_error_distribution.png")
    # plt.savefig(output_path)
    ax = axes[1]
    ax.hist(res['abs_error'], bins=100)
    ax.set_title("Absolute Error Distribution of pKa", fontsize=font_size)
    ax.set_xlabel("Absolute error", fontsize=font_size)
    ax.set_ylabel("Frequency", fontsize=font_size)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    output_path = os.path.join(cfg["output_dir"], f"{VERSION}_error_distribution.png")
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"[PLOT] Error Distribution: {output_path} saved")
    
    
    
    from scipy import stats
        
    true_np = np.asarray(y_true, dtype=float)
    pred_np = np.asarray(y_pred, dtype=float)
    residual = pred_np - true_np
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    stats.probplot(residual, dist="norm", plot=ax)
    ax.set_title(f"QQ-plot of residuals of {VERSION}")
    ax.set_xlabel("Theoretical Quantiles (Normal)")
    ax.set_ylabel("Ordered residuals")
    out_png = f"../output/{VERSION}/residual_qq_{VERSION}.png"
    fig.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print(f"[PLOT]殘差QQ-plot ../output/{VERSION}/residual_qq_{VERSION}.png saved")

def plot_boxplot(VERSION, output_dir):
    csv_path = os.path.join(output_dir, f"{VERSION}_evaluation_results.csv")
    df = pd.read_csv(csv_path)
    metal_counts = Counter(df['metal_ion'])
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    metrics = ['y_true', 'y_pred', 'error', 'abs_error']
    titles  = ['Experimental binding constant (y_true)',
            'Predicted binding constant (y_pred)',
            'Error (y_pred - y_true)',
            'Absolute Error |y_pred - y_true|']

    for ax, col, title in zip(axes.flat, metrics, titles):
        df.boxplot(column=col, by='metal_ion',
                ax=ax, grid=False, showfliers=False)
        if col == 'y_true' or col == 'y_pred':
            ax.set_ylim(-5, 30)
            
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('')
        ax.set_ylabel(col)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(False)                           # 先關掉原本格線
        ax.axhline(0, color='gray', linestyle='--',
                linewidth=1, alpha=0.7)       # y = 0 虛線

        # ── 在每個 box 上方標註 n ───────────────────────────
        ymin, ymax_vis = ax.get_ylim()
        y_text = ymax_vis - 0.05 * (ymax_vis - ymin)   # 距頂端 5 %
        for x, metal in zip(ax.get_xticks(),
                            [lbl.get_text() for lbl in ax.get_xticklabels()]):
            n = metal_counts[metal]
            ax.text(x, y_text, f"n={n}",
                    ha='center', va='bottom', fontsize=9)

    fig.suptitle('')
    plt.tight_layout()
    joint_path = os.path.join(output_dir, f"{VERSION}_boxplot_4metrics.png")
    fig.savefig(joint_path, bbox_inches='tight', dpi=300)
    print(f"四合一盒鬚圖已輸出： {joint_path}")


