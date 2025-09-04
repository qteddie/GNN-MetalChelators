#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parity-plot utility
───────────────────
• 主要對外函式：parity_plot(y_true, y_pred, is_train, save_path, ...)
    - y_true / y_pred : 一維 array-like（長度相同）
    - is_train        : Bool mask，標註每一筆資料是否屬於「訓練集」
• 亦保留 CLI：給一個含三欄 (y_true,y_pred,is_train) 的 CSV 也可單獨畫圖
"""

from __future__ import annotations
import argparse, os, numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ──────────────────── 小工具 ────────────────────
def _reg_metrics(y_t: np.ndarray, y_p: np.ndarray):
    """RMSE / MAE / R²"""
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae  = float(mean_absolute_error(y_t, y_p))
    r2   = float(r2_score(y_t, y_p))
    return rmse, mae, r2


def _ensure_dir(p: Path | str):
    p = Path(p)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ──────────────────── 主函式 ────────────────────
def parity_plot(
    y_true,
    y_pred,
    is_train: np.ndarray | list | None,
    save_path: str | Path,
    *,
    title: str = "pKa Prediction",
    dpi: int = 300,
    annotate: bool = True,
):
    """
    畫「Train / Test / 綜合」三組指標的 parity plot。

    Parameters
    ----------
    y_true / y_pred : array-like
    is_train        : Bool mask（與 y_true 同長度）; 若為 None 全視為 Test
    save_path       : 輸出 PNG
    title           : 圖表標題
    dpi             : 解析度
    annotate        : 是否顯示指標文字方塊
    """
    # ---------- 資料檢查 ----------
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("`y_true` 與 `y_pred` 需為相同 shape 且不可為空")

    if is_train is None:
        is_train = np.zeros_like(y_true, dtype=bool)
    else:
        is_train = np.asarray(is_train, dtype=bool).ravel()
        if is_train.shape != y_true.shape:
            raise ValueError("`is_train` 長度需與 `y_true` 一致")

    # ---------- 指標 ----------
    rmse,  mae,  r2  = _reg_metrics(y_true,            y_pred)
    tr_rmse, tr_mae, tr_r2 = _reg_metrics(y_true[is_train],   y_pred[is_train]) if is_train.any() else (np.nan,)*3
    te_rmse, te_mae, te_r2 = _reg_metrics(y_true[~is_train],  y_pred[~is_train])

    # ---------- 視覺化 ----------
    mn, mx = min(y_true.min(), y_pred.min()) - 1, max(y_true.max(), y_pred.max()) + 1
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    # y = x 基準線
    ax.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=.7)

    # 散點 – 訓練集
    if is_train.any():
        ax.scatter(
            y_true[is_train], y_pred[is_train],
            s=22, alpha=.75, edgecolors="k", label="Train"
        )
    # 散點 – 測試集
    ax.scatter(
        y_true[~is_train], y_pred[~is_train],
        s=22, alpha=.75, marker="s", edgecolors="k", label="Test"
    )

    # 格線 & 座標
    ax.set_xlim(mn, mx);  ax.set_ylim(mn, mx)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.grid(ls="--", alpha=.3)

    ax.set_xlabel("Experimental binding constant (LogK)")
    ax.set_ylabel("Predicted binding constant (LogK)")
    ax.set_title(title)

    # ---------- 指標文字塊 ----------
    if annotate:
        bbox_cfg = dict(boxstyle="round", facecolor="white", alpha=.8)

        txt_all = f"RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nR²   = {r2:.2f}"
        ax.text(0.04, 0.96, txt_all, transform=ax.transAxes,
                va="top", ha="left", bbox=bbox_cfg)

        if is_train.any():
            txt_tr  = f"Train RMSE = {tr_rmse:.2f}\nTrain MAE = {tr_mae:.2f}\nTrain R² = {tr_r2:.2f}"
            ax.text(0.04, 0.80, txt_tr, transform=ax.transAxes,
                    va="top", ha="left", bbox=bbox_cfg)

        txt_te  = f"Test RMSE = {te_rmse:.2f}\nTest MAE = {te_mae:.2f}\nTest R² = {te_r2:.2f}"
        ax.text(0.04, 0.64, txt_te, transform=ax.transAxes,
                va="top", ha="left", bbox=bbox_cfg)

    ax.legend(loc="lower right")
    plt.tight_layout()

    # ---------- 儲存 ----------
    save_path = _ensure_dir(save_path)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print(f"[PLOT] {save_path} ✔")
    return Path(save_path)


# ──────────────────── CLI ────────────────────
def _cli():
    """
    用法：
        python 5_parity_plot.py result.csv -o out.png -t "My Title"
    其中 CSV 至少要有 3 欄：y_true, y_pred, is_train (0/1)
    """
    p = argparse.ArgumentParser(description="Draw parity plot from CSV (y_true,y_pred,is_train).")
    p.add_argument("csv", help="包含 y_true, y_pred, is_train 的 CSV")
    p.add_argument("-o", "--output", default="parity_plot.png", help="輸出 PNG 路徑")
    p.add_argument("-t", "--title", default="pKa Prediction", help="圖表標題")
    args = p.parse_args()

    data = np.loadtxt(args.csv, delimiter=",", skiprows=1)
    if data.shape[1] < 3:
        raise ValueError("CSV 需至少三欄 (y_true,y_pred,is_train)")
    y_t, y_p, is_tr = data[:,0], data[:,1], data[:,2].astype(bool)
    parity_plot(y_t, y_p, is_tr, args.output, title=args.title)

if __name__ == "__main__":
    _cli()
