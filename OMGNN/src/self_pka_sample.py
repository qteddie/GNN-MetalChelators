"""pka_all_in_one.py  (超精簡版)
==================================================
*唯一* 支援的 CLI 子指令：``all-in-one``
--------------------------------------------------
流程：
1. 隨機分割 mapping CSV → train / test
2. 對兩組分子逐一預測 pKa
3. 將結果輸出成 train / test CSV 並自動合併
4. 產生 parity‑plot (RMSE / MAE / R² 標註)

其他功能一概移除，確保程式碼 *最小、易讀*。
"""
from __future__ import annotations

# ─── 標準函式庫 ─────────────────────────────────────────────
import ast
import os
import sys
import csv
from pathlib import Path
from typing import List, Tuple, Dict, Sequence, Optional

# ─── 第三方套件 ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# safe matplotlib import（無 X11 也能跑）
import matplotlib
if os.getenv("DISPLAY", "") == "":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ─── 專案內部 ───────────────────────────────────────────────
from self_pka_models import pka_GNN_ver3 # type: ignore
from self_pka_trainutils import load_config_by_version

# ─── 常數 ───────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent
DATA_DIR        = ROOT / "../output"
MODEL_DIR       = ROOT / "../results"
MAPPING_FILE    = DATA_DIR / "pka_mapping_results.csv"
DEFAULT_VERSION = "pka_ver21"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FIELDNAMES      = ["smiles", "true_pka", "pred_pka", "rmse", "acc", "prec", "rec", "f1", "true_cla", "pred_cla",  "is_train"]
MODEL_VERSION   = pka_GNN_ver3
# ────────────────────────────────────────────────────────────
# 小工具
# ────────────────────────────────────────────────────────────

def ensure_dir(path: Path | str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

def save_csv(rows: Sequence[dict], path: Path) -> None:
    ensure_dir(path)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader(); w.writerows(rows)
    print(f"[CSV] {path.relative_to(ROOT)}  ←  {len(rows)} 行")


def parse_pka_matrix(s: str) -> Dict[int, float]:
    try:
        return {int(i): float(v) for i, v in ast.literal_eval(s)}
    except Exception:
        return {}

# ────────────────────────────────────────────────────────────
# 模型 (只載一次)
# ────────────────────────────────────────────────────────────
_MODEL_CACHE: Dict[str, MODEL_VERSION] = {}

def load_model(version: str) -> MODEL_VERSION:
    if version in _MODEL_CACHE:
        return _MODEL_CACHE[version]
    ckpt = MODEL_DIR / version / f"{version}_best.pkl"
    if not ckpt.exists():
        sys.exit(f"[ERROR] 找不到 checkpoint: {ckpt}")
    print(f"[MODEL] 載入 {version} → {ckpt.relative_to(ROOT)}")
    config_csv = "../data/pka_parameters.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config_by_version(config_csv, version)
    bond_dim = 11 + cfg["max_dist"] + 1 # 11 + 6 = 17
    # print(bond_dim)
    model = pka_GNN_ver3(
            node_dim  = 153,
            bond_dim  = bond_dim,
            hidden_dim= cfg["hidden_size"],
            output_dim= 1,
            dropout   = cfg["dropout"],
            depth     = cfg["depth"],
            heads      = cfg["heads"],
            pe_dim     = cfg["pe_dim"],
            max_dist   = cfg["max_dist"],
         ).to(device)
    state = torch.load(ckpt, map_location=DEVICE, weights_only=True)
    # 修正舊權重命名
    fixed = {k.replace("rnn_gate", "gate") if k.startswith("rnn_gate") else k: v
             for k, v in state.items() if not k.startswith("GCN1")}
    model.load_state_dict(fixed, strict=False)
    model.to(DEVICE).eval()
    _MODEL_CACHE[version] = model
    return model


def predict_pka(smiles: str, model: MODEL_VERSION) -> Tuple[List[int], List[float]] | None:
    res = model.sample(smiles, MAPPING_FILE, DEVICE)
    return res

# ────────────────────────────────────────────────────────────
# parity‑plot
# ────────────────────────────────────────────────────────────

def parity_plot(true_v: np.ndarray, pred_v: np.ndarray, *, title: str, out_png: Path, is_train: np.ndarray) -> None:
    rmse = float(np.sqrt(mean_squared_error(true_v, pred_v)))
    mae  = float(mean_absolute_error(true_v, pred_v))
    r2   = float(r2_score(true_v, pred_v))
    mn, mx = min(true_v.min(), pred_v.min()) - 1, max(true_v.max(), pred_v.max()) + 1
    print(f"綜合性能指標:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    # 分開計算訓練集和測試集的性能指標
    train_rmse = float(np.sqrt(mean_squared_error(true_v[is_train], pred_v[is_train])))
    train_mae  = float(mean_absolute_error(true_v[is_train], pred_v[is_train]))
    train_r2   = float(r2_score(true_v[is_train], pred_v[is_train]))
    test_rmse  = float(np.sqrt(mean_squared_error(true_v[~is_train], pred_v[~is_train])))
    test_mae   = float(mean_absolute_error(true_v[~is_train], pred_v[~is_train]))
    test_r2    = float(r2_score(true_v[~is_train], pred_v[~is_train]))
    print(f"訓練集性能指標:")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  R²: {train_r2:.4f}")
    print(f"測試集性能指標:")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R²: {test_r2:.4f}")
    plt.figure(figsize=(6, 6), dpi=300)
    plt.plot([mn, mx], [mn, mx], "k--", lw=1, alpha=.7)
    plt.scatter(true_v[is_train], pred_v[is_train], s=20, label="Train", alpha=.7, edgecolors="k")
    plt.scatter(true_v[~is_train], pred_v[~is_train], s=20, label="Test",  alpha=.7, marker="s", edgecolors="k")
    plt.legend(loc='lower right')
    txt1 = f"RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nR²   = {r2:.2f}"
    txt2 = f"Train RMSE = {train_rmse:.2f}\nTrain MAE = {train_mae:.2f}\nTrain R² = {train_r2:.2f}"
    txt3 = f"Test RMSE = {test_rmse:.2f}\nTest MAE = {test_mae:.2f}\nTest R² = {test_r2:.2f}"
    ax = plt.gca(); ax.set_xlabel("Experimental pKa"); ax.set_ylabel("Predicted pKa"); ax.set_title(title)
    ax.xaxis.set_major_locator(MultipleLocator(2)); ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.grid(ls="--", alpha=.3)
    ax.text(0.05, 0.95, txt1, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=.75))
    ax.text(0.05, 0.82, txt2, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=.75))
    ax.text(0.05, 0.69, txt3, transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=.75))
    plt.tight_layout(); ensure_dir(out_png); plt.savefig(out_png, dpi=300); plt.close()
    print(f"[PLOT] {out_png}  ✔")

# ────────────────────────────────────────────────────────────
# 核心：all‑in‑one
# ────────────────────────────────────────────────────────────
from function import *
def all_in_one(*, test_size: float = 0.2, title: str = "pKa Prediction (Train vs Test)",
               out_png: Optional[Path] = None, model_version: str = DEFAULT_VERSION) -> None:
    model = load_model(model_version)
    df = pd.read_csv(MAPPING_FILE)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    print(f"[SPLIT] Train={len(train_df)}  Test={len(test_df)}")

    def process(df_part: pd.DataFrame, tag: str, is_train_flag: bool) -> List[dict]:
        rows: List[dict] = []
        for idx, row in tqdm(df_part.iterrows(), total=len(df_part), desc=tag):
            res = predict_pka(row.smiles, model)
            if res is None:
                continue
            rows.append({
                "smiles": row.smiles,
                "true_pka": res["true_pka"],
                "pred_pka": res["pred_pka"],
                "rmse": res["rmse"],
                "true_cla": res["true_cla"],
                "pred_cla": res["pred_cla"],
                "acc": res["acc"],
                "prec": res["prec"],
                "rec": res["rec"],
                "f1": res["f1"],
                "is_train": is_train_flag
            })
        return rows

    train_rows = process(train_df, "train", True)
    test_rows  = process(test_df,  "test",  False)

    res_dir = MODEL_DIR / model_version; ensure_dir(res_dir)
    train_csv = res_dir / f"pka_train_{model_version}.csv"; save_csv(train_rows, train_csv)
    test_csv  = res_dir / f"pka_test_{model_version}.csv";  save_csv(test_rows,  test_csv)

    combined_csv = res_dir / f"pka_combined_{model_version}.csv"
    pd.concat([pd.DataFrame(train_rows), pd.DataFrame(test_rows)]).to_csv(combined_csv, index=False)
    print(f"[CSV] {combined_csv.relative_to(ROOT)}  ←  combined")
    
    
    
    ############################################################
    # 測試用，要記得註解掉
    # res_dir = MODEL_DIR / model_version; ensure_dir(res_dir)
    # combined_csv = "../results/pka_ver19/pka_combined_pka_ver19.csv"
    
    
    if out_png is None:
        out_png = res_dir / f"pka_parity_{model_version}.png"
    from ast import literal_eval
    df_valid = pd.read_csv(combined_csv).dropna()
    true_lists = df_valid["true_pka"].apply(literal_eval)
    pred_lists = df_valid["pred_pka"].apply(literal_eval)
    true_pka = np.concatenate(true_lists)
    pred_pka = np.concatenate(pred_lists)
    counts    = true_lists.str.len()            # 每列有幾個 pKa
    is_train  = np.repeat(df_valid["is_train"].values.astype(bool), counts)

    parity_plot(true_pka, pred_pka, title=title, out_png=out_png, is_train=is_train)
    draw_pka_distribution(model_version)
    merge_mapping_results(model_version)
    draw_boxplot(model_version)
    calculate_pka_rmse(model_version)
    calculate_rmse_by_functional_group(model_version)
    draw_residual_histogram(model_version)
    draw_residual_qq(model_version)
    draw_html_outlier(model_version)
    print("[DONE] all‑in‑one completed!")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("pKa all‑in‑one utility")
    p.add_argument("--version", default=DEFAULT_VERSION)
    args = p.parse_args()
    all_in_one(model_version=args.version)
