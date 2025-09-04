#!/usr/bin/env python3
"""pka_prediction_refactored.py
=================================================
Versatile utility for running pKa predictions **and** generating parity‑plots
with a pretrained ``pka_GNN`` model.

新增功能 (v2)
--------------
* **plot 子指令** – 直接從先前 `export` / `split` 產出的 CSV 生成 parity‑plot
* 內建 `plot_parity()` 以及 `combine_csv_files()` helper，無須額外腳本
* 自動偵測缺少 matplotlib backend 並 fallback 到 'Agg'（CLI環境也安全）

其餘結構沿用 v1：Argparse 子命令、tqdm 進度列、Pathlib 路徑、dataclass 型別提示。
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Standard libs
# -----------------------------------------------------------------------------
import ast
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Sequence, Optional, Tuple

# -----------------------------------------------------------------------------
# Third‑party deps
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# matplotlib 安全導入（無 X11 亦可）
import matplotlib
if os.getenv("DISPLAY", "") == "":  # headless 環境
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# -----------------------------------------------------------------------------
# Project‑internal imports
# -----------------------------------------------------------------------------
import sys
sys.path.append("/work/s6300121/LiveTransForM-main/metal")
from src.metal_models import pka_GNN  # type: ignore

# 如果後續需要這些工具，可保留以避免循環導入問題
try:
    from metal_trainutils import load_config_by_version, pka_Dataloader  # noqa: F401
except ImportError:
    pass

# -----------------------------------------------------------------------------
# Constants & paths
# -----------------------------------------------------------------------------
DATA_DIR = Path("/work/s6300121/LiveTransForM-main/metal/output")
MAPPING_FILE = DATA_DIR / "Metal/Metal_pka_mapping_results.csv"
MODEL_BASE_DIR = Path("/work/s6300121/LiveTransForM-main/results/")
DEFAULT_MODEL_VERSION = "pka_ver5"
DEFAULT_CHECKPOINT_PATH = MODEL_BASE_DIR / DEFAULT_MODEL_VERSION / f"{DEFAULT_MODEL_VERSION}_best.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------
@dataclass
class PkaPrediction:
    """A single pKa prediction paired with ground truth (if available)."""
    smiles: str
    atom_position: int
    true_pka: Optional[float] = None
    predicted_pka: Optional[float] = None

    @property
    def diff(self) -> Optional[float]:
        if self.true_pka is None or self.predicted_pka is None:
            return None
        return abs(self.true_pka - self.predicted_pka)


@dataclass
class MoleculeStats:
    """Aggregated statistics for one molecule across all matching atoms."""
    molecule_name: str
    smiles: str
    true_pka_count: int = 0
    predicted_pka_count: int = 0
    matched_count: int = 0
    rmse: float = 0.0
    mae: float = 0.0

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def parse_pka_matrix(pka_str: str) -> Dict[int, float]:
    """Convert the stored pKa string to a dict {atom_idx: pKa}."""
    try:
        pairs = ast.literal_eval(pka_str)
        return {int(pos): float(val) for pos, val in pairs}
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] 無法解析 pKa 字串: {exc}; str = {pka_str}")
        return {}


def ensure_dir(path: Path | str) -> None:
    """確保目錄存在。接受Path物件或字串路徑。"""
    # 如果收到字串，先轉換為Path物件
    if isinstance(path, str):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def save_csv(rows: Sequence[dict], path: Path, fieldnames: Sequence[str]) -> None:
    """Save a list of dicts to CSV with given header."""
    ensure_dir(path)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] CSV saved → {path}")

# -----------------------------------------------------------------------------
# Plotting utilities
# -----------------------------------------------------------------------------

def plot_parity(true_v, pred_v, title="", save_path=None, is_train=None):
    """
    繪製分組 parity plot：左上顯示 train/test 指標，右下圖例顯示顏色/形狀
    """
    if len(true_v) == 0 or len(pred_v) == 0:
        print("警告: 沒有有效的預測數據可用於繪製 parity plot")
        return {
            "RMSE": float('nan'),
            "MAE": float('nan'),
            "R2": float('nan')
        }

    true_v = np.array(true_v)
    pred_v = np.array(pred_v)

    # 分組指標
    if is_train is not None:
        is_train = np.array(is_train).astype(bool)
        train_true = true_v[is_train]
        train_pred = pred_v[is_train]
        test_true = true_v[~is_train]
        test_pred = pred_v[~is_train]
        train_metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(train_true, train_pred))),
            "MAE": float(mean_absolute_error(train_true, train_pred)),
            "R2": float(r2_score(train_true, train_pred))
        }
        test_metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(test_true, test_pred))),
            "MAE": float(mean_absolute_error(test_true, test_pred)),
            "R2": float(r2_score(test_true, test_pred))
        }
    else:
        # 若無 is_train，全部視為 test
        is_train = np.zeros_like(true_v, dtype=bool)
        train_true = np.array([])
        train_pred = np.array([])
        test_true = true_v
        test_pred = pred_v
        train_metrics = {"RMSE": float('nan'), "MAE": float('nan'), "R2": float('nan')}
        test_metrics = {
            "RMSE": float(np.sqrt(mean_squared_error(test_true, test_pred))),
            "MAE": float(mean_absolute_error(test_true, test_pred)),
            "R2": float(r2_score(test_true, test_pred))
        }

    # 畫圖
    plt.figure(figsize=(8, 8))
    
    # 固定x與y軸範圍在-18到32之間
    min_val = -18
    max_val = 32
    
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1, alpha=.7)

    # 設定顏色
    train_color = "#1f77b4"  # 藍色
    test_color = "#ff7f0e"   # 橙色

    # 畫點
    plt.scatter(train_true, train_pred, s=40, alpha=.7, label="Train", edgecolors="black", color=train_color)
    plt.scatter(test_true, test_pred, s=40, alpha=.7, label="Test", marker="s", edgecolors="black", color=test_color)

    # 右下圖例
    plt.legend(loc="lower right")

    # 左上指標
    txt = (f"Train ({len(train_true)}):\n"
           f"RMSE = {train_metrics['RMSE']:.2f}\n"
           f"MAE = {train_metrics['MAE']:.2f}\n"
           f"R²  = {train_metrics['R2']:.2f}\n\n"
           f"Test ({len(test_true)}):\n"
           f"RMSE = {test_metrics['RMSE']:.2f}\n"
           f"MAE = {test_metrics['MAE']:.2f}\n"
           f"R²  = {test_metrics['R2']:.2f}")

    plt.gca().text(0.05, 0.95, txt, transform=plt.gca().transAxes,
                   va="top", ha="left", fontsize=13,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=.85))

    plt.xlabel("Experimental pKa")
    plt.ylabel("Predicted pKa")
    plt.title(title)
    
    # 設置x軸和y軸的範圍
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.gca().xaxis.set_major_locator(MultipleLocator(5))  # 每5個單位一個主刻度
    plt.gca().yaxis.set_major_locator(MultipleLocator(5))  # 每5個單位一個主刻度
    plt.grid(True, ls="--", alpha=.3)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Plot saved → {save_path}")
    plt.close()

    # 回傳所有指標
    return {
        "train": train_metrics,
        "test": test_metrics
    }


def combine_csv_files(train_file: Path, test_file: Path, out_file: Path) -> Path:
    train_df = pd.read_csv(train_file); train_df["is_train"] = True
    test_df = pd.read_csv(test_file);  test_df["is_train"] = False
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined.to_csv(out_file, index=False)
    print(f"[INFO] Combined CSV → {out_file}")
    return out_file

# -----------------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------------

def load_pretrained_model(ckpt_path: Path, *,
                          node_dim: int = 153,
                          bond_dim: int = 11,
                          hidden_dim: int = 153,
                          output_dim: int = 1,
                          dropout: float = 0.0,
                          depth: int = 4) -> pka_GNN:
    """Load model architecture and weights, fixing mismatched keys."""
    model = pka_GNN(
        node_dim=node_dim,
        bond_dim=bond_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=dropout,
        depth=depth,
    )

    if not ckpt_path.exists():
        sys.exit(f"[ERROR] 找不到 checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)

    new_state: dict = {}
    for k, v in checkpoint.items():
        if k.startswith("rnn_gate"):
            new_state[k.replace("rnn_gate", "gate")] = v
        elif k.startswith("GCN1"):
            # skip obsolete layer
            continue
        else:
            new_state[k] = v

    missing = [k for k in model.state_dict().keys() if k not in new_state]
    if missing:
        print(f"[WARN] 以下權重缺失, 將使用隨機初始化: {missing}")

    model.load_state_dict(new_state, strict=False)
    model.to(DEVICE).eval()
    return model


def get_model_checkpoint_path(model_version: str = DEFAULT_MODEL_VERSION) -> Path:
    """根據版本獲取模型checkpoint路徑"""
    ckpt_path = MODEL_BASE_DIR / model_version / f"{model_version}_best.pkl"
    if not ckpt_path.exists():
        sys.exit(f"[ERROR] 找不到模型checkpoint: {ckpt_path}")
    return ckpt_path


# 全局模型實例 - 當需要指定不同版本時可動態創建
MODEL = None


def get_model(model_version: str = DEFAULT_MODEL_VERSION) -> pka_GNN:
    """獲取或首次加載指定版本的模型"""
    global MODEL
    
    if MODEL is None:
        ckpt_path = get_model_checkpoint_path(model_version)
        print(f"[INFO] 加載模型 {model_version} 從 {ckpt_path}")
        MODEL = load_pretrained_model(ckpt_path)
    
    return MODEL

# -----------------------------------------------------------------------------
# Core prediction logic
# -----------------------------------------------------------------------------

def predict_pka(smiles: str, *, model: pka_GNN | None = None, model_version: str = DEFAULT_MODEL_VERSION) -> Tuple[List[int], List[float]] | None:
    """Return predicted positions and values (or ``None`` if failed)."""
    if model is None:
        model = get_model(model_version)
    try:
        result = model.sample(smiles, DEVICE)
        if not result:
            return None
        return result["pka_positions"], result["pka_values"]
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] 預測失敗 {smiles}: {exc}")
        return None


# -----------------------------------------------------------------------------
# CLI commands
# -----------------------------------------------------------------------------

def interactive_mode(model_version: str = DEFAULT_MODEL_VERSION) -> None:
    """互動模式 – 手動輸入 SMILES 並顯示預測結果"""
    # 預載入模型
    get_model(model_version)
    
    while True:
        smiles = input("請輸入分子 SMILES (輸入 q 離開): ")
        if smiles.strip().lower() == "q":
            break

        pred = predict_pka(smiles, model_version=model_version)
        if pred is None:
            continue
        positions, values = pred
        print("\n預測結果:")
        for pos, val in zip(positions, values):
            print(f"  原子 #{pos:>3}: pKa = {val:6.2f}")
        print()


def export_full_results(out_dir: Path, *, threshold: float = 0.5, model_version: str = DEFAULT_MODEL_VERSION) -> None:
    """Export parity / visualisation CSVs for **all** molecules in database."""
    # 預載入模型
    get_model(model_version)
    
    df = pd.read_csv(MAPPING_FILE)

    parity_rows: List[dict] = []
    viz_rows: List[dict] = []

    print(f"[INFO] Processing {len(df)} molecules ...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smiles = row.smiles
        mol_name = f"Molecule_{idx}"
        true_dict = parse_pka_matrix(row.pka_matrix)
        true_positions = list(true_dict.keys())
        true_values = list(true_dict.values())

        pred = predict_pka(smiles, model_version=model_version)
        if pred is None:
            continue
        pred_positions, pred_values = pred

        # parity rows (only matched atoms)
        for pos, pval in zip(pred_positions, pred_values):
            if pos in true_dict:
                parity_rows.append({
                    "molecule_name": mol_name,
                    "smiles": smiles,
                    "atom_position": pos,
                    "true_pka": true_dict[pos],
                    "predicted_pka": pval,
                    "difference": abs(pval - true_dict[pos]),
                })

        # viz rows (all predictions)
        viz_rows.append({
            "molecule_name": mol_name,
            "smiles": smiles,
            "pka_positions": json.dumps(pred_positions),
            "pka_values": json.dumps(pred_values),
            "true_pka_positions": json.dumps(true_positions),
            "true_pka_values": json.dumps(true_values),
        })

    save_csv(parity_rows, out_dir / f"pka_prediction_results_{model_version}.csv",
             ["molecule_name", "smiles", "atom_position", "true_pka", "predicted_pka", "difference"])
    save_csv(viz_rows, out_dir / f"pka_visualization_data_{model_version}.csv",
             ["molecule_name", "smiles", "pka_positions", "pka_values", "true_pka_positions", "true_pka_values"])

    # global stats
    if parity_rows:
        all_true = np.array([r["true_pka"] for r in parity_rows])
        all_pred = np.array([r["predicted_pka"] for r in parity_rows])
        rmse = float(np.sqrt(np.mean((all_true - all_pred) ** 2)))
        mae = float(np.mean(np.abs(all_true - all_pred)))
        print(f"\n[INFO] 全體 RMSE = {rmse:.4f} | MAE = {mae:.4f}")


def check_errors(threshold: float = 0.5, model_version: str = DEFAULT_MODEL_VERSION) -> None:
    """列出 |pred−true| > threshold 的分子."""
    # 預載入模型
    get_model(model_version)
    
    df = pd.read_csv(MAPPING_FILE)
    errors: List[dict] = []

    for row in tqdm(df.itertuples(), total=len(df)):
        true_dict = parse_pka_matrix(row.pka_matrix)
        pred = predict_pka(row.smiles, model_version=model_version)
        if pred is None:
            continue
        pos_pred, val_pred = pred
        for pos, pval in zip(pos_pred, val_pred):
            if pos in true_dict and abs(pval - true_dict[pos]) > threshold:
                errors.append({
                    "smiles": row.smiles,
                    "position": pos,
                    "pred_pKa": round(pval, 2),
                    "true_pKa": round(true_dict[pos], 2),
                    "diff": round(abs(pval - true_dict[pos]), 2),
                })

    if errors:
        out_file = DATA_DIR / f"pka_errors_t{threshold}_{model_version}.csv"
        save_csv(errors, out_file, ["smiles", "position", "pred_pKa", "true_pKa", "diff"])
        print(f"[INFO] 找到 {len(errors)} 筆誤差 > {threshold}")
    else:
        print("[INFO] 沒有誤差超過閾值")


def split_and_sample(test_size: float = 0.2, model_version: str = DEFAULT_MODEL_VERSION) -> None:
    """隨機分割資料集並個別輸出預測結果."""
    # 預載入模型
    get_model(model_version)
    
    df = pd.read_csv(MAPPING_FILE)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    print(f"Train = {len(train_df)}, Test = {len(test_df)}")

    _process_and_save(train_df, "train", model_version)
    _process_and_save(test_df, "test", model_version)


def _process_and_save(df: pd.DataFrame, tag: str, model_version: str = DEFAULT_MODEL_VERSION) -> None:
    parity_rows: List[dict] = []
    viz_rows: List[dict] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{tag}"):
        mol_name = f"{tag}_Molecule_{idx}"
        true_dict = parse_pka_matrix(row.pka_matrix)

        pred = predict_pka(row.smiles, model_version=model_version)
        if pred is None:
            continue
        pos_pred, val_pred = pred

        # parity rows
        for pos, pval in zip(pos_pred, val_pred):
            if pos in true_dict:
                parity_rows.append({
                    "metal_ion": row.metal_ion,
                    "smiles": row.smiles,
                    "atom_position": pos,
                    "true_pka": true_dict[pos],
                    "predicted_pka": pval,
                    "difference": abs(pval - true_dict[pos]),
                })

        # viz rows
        viz_rows.append({
            "metal_ion": row.metal_ion,
            "smiles": row.smiles,
            "pka_positions": json.dumps(pos_pred),
            "pka_values": json.dumps(val_pred),
            "true_pka_positions": json.dumps(list(true_dict.keys())),
            "true_pka_values": json.dumps(list(true_dict.values())),
        })

    # 添加版本號到檔案名稱
    save_csv(parity_rows, DATA_DIR / f"pka_{tag}_prediction_{model_version}.csv",
             ["metal_ion", "smiles", "atom_position", "true_pka", "predicted_pka", "difference"])
    save_csv(viz_rows, DATA_DIR / f"pka_{tag}_visualization_{model_version}.csv",
             ["metal_ion", "smiles", "pka_positions", "pka_values", "true_pka_positions", "true_pka_values"])

    _print_metrics(parity_rows, tag)


def _print_metrics(rows: Sequence[dict], name: str) -> None:
    if not rows:
        print(f"[WARN] {name} 沒有可用數據")
        return
    true = np.array([r["true_pka"] for r in rows])
    pred = np.array([r["predicted_pka"] for r in rows])
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mae = float(np.mean(np.abs(true - pred)))
    print(f"{name} → RMSE={rmse:.4f}, MAE={mae:.4f}")


def sample_all(model_version: str = DEFAULT_MODEL_VERSION) -> None:
    """Predict pKa for every atom of every molecule and dump a single CSV."""
    # 預載入模型
    get_model(model_version)
    
    df = pd.read_csv(MAPPING_FILE)
    rows: List[dict] = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        pred = predict_pka(row.smiles, model_version=model_version)
        if pred is None:
            continue
        pos_pred, val_pred = pred
        for pos, pval in zip(pos_pred, val_pred):
            rows.append({
                "molecule_id": idx,
                "molecule_name": f"Molecule_{idx}",
                "smiles": row.smiles,
                "atom_position": pos,
                "predicted_pka": round(pval, 2),
            })
    save_csv(rows, DATA_DIR / f"all_pka_samples_{model_version}.csv",
             ["molecule_id", "molecule_name", "smiles", "atom_position", "predicted_pka"])
    print(f"總共寫入 {len(rows)} rows")


# -----------------------------------------------------------------------------
# 新增: plot 子命令
# -----------------------------------------------------------------------------

def plot_cmd(csv: Path | None, train_csv: Path | None, test_csv: Path | None, title: str, out_png: Path, model_version: str = DEFAULT_MODEL_VERSION) -> None:
    """從先前產生的 CSV 檔案生成 parity-plot 圖"""
    if csv is None:
        if train_csv is None or test_csv is None:
            sys.exit("[ERROR] 必須提供 --csv 或同時提供 --train-csv/--test-csv")
        csv = combine_csv_files(train_csv, test_csv, DATA_DIR / f"_tmp_combined_{model_version}.csv")
    
    df = pd.read_csv(csv)
    required = {"true_pka", "predicted_pka"}
    if not required.issubset(df.columns):
        sys.exit(f"[ERROR] CSV 缺少必要欄位: {required - set(df.columns)}")
    
    # 如果out_png沒有明確設置，則自動生成包含版本的文件名
    if out_png == DATA_DIR / "pka_parity.png":  # 默認值
        out_png = DATA_DIR / f"pka_parity_{model_version}.png"
    
    true_v = df["true_pka"].values
    pred_v = df["predicted_pka"].values
    is_train = df["is_train"].values if "is_train" in df.columns else None
    
    metrics = plot_parity(true_v, pred_v, title=title, save_path=out_png, is_train=is_train)
    print(f"[INFO] 整體指標 → RMSE {metrics['RMSE']:.4f} | MAE {metrics['MAE']:.4f} | R2 {metrics['R2']:.4f}")
    if 'train' in metrics and 'test' in metrics:
        print(f"[INFO] 訓練集指標 → RMSE {metrics['train']['RMSE']:.4f} | MAE {metrics['train']['MAE']:.4f} | R2 {metrics['train']['R2']:.4f}")
        print(f"[INFO] 測試集指標 → RMSE {metrics['test']['RMSE']:.4f} | MAE {metrics['test']['MAE']:.4f} | R2 {metrics['test']['R2']:.4f}")


# -----------------------------------------------------------------------------
# 新增: 一鍵完成功能 (分割→預測→繪圖)
# -----------------------------------------------------------------------------

def load_model(model_version: str = DEFAULT_MODEL_VERSION) -> Tuple[pka_GNN, torch.device]:
    """
    加載指定版本的模型
    
    Args:
        model_version: 模型版本
        
    Returns:
        Tuple[pka_GNN, torch.device]: 模型實例和設備
    """
    try:
        # 獲取模型checkpoint路徑
        ckpt_path = get_model_checkpoint_path(model_version)
        print(f"[INFO] 加載模型 {model_version} 從 {ckpt_path}")
        
        # 加載模型
        model = load_pretrained_model(ckpt_path)
        model.to(DEVICE).eval()
        
        return model, DEVICE
    except Exception as e:
        print(f"加載模型時出錯: {str(e)}")
        return None, DEVICE


def all_in_one(test_size: float = 0.2, title: str = "pKa Prediction (Train vs Test)", 
               out_png: Path | None = None, model_version: str = DEFAULT_MODEL_VERSION) -> None:
    """
    整合所有功能：預測、評估和繪圖
    
    Args:
        test_size: 測試集比例
        title: 圖表標題
        out_png: 輸出圖表路徑
        model_version: 模型版本
    """
    try:
        # 1. 加載模型和數據
        model, device = load_model(model_version)
        if model is None:
            print("無法加載模型，請檢查模型文件是否存在")
            return
            
        # 2. 分割數據集
        df = pd.read_csv(MAPPING_FILE)
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        print(f"Train = {len(train_df)}, Test = {len(test_df)}")
        
        # 3. 處理訓練集和測試集
        train_parity_rows = _process_df_for_parity(train_df, "train", model)
        test_parity_rows = _process_df_for_parity(test_df, "test", model)
        
        # 4. 保存結果
        train_csv = DATA_DIR / f"pka_train_prediction_{model_version}.csv"
        test_csv = DATA_DIR / f"pka_test_prediction_{model_version}.csv"
        combined_csv = DATA_DIR / f"pka_combined_prediction_{model_version}.csv"
        
        fieldnames = ["metal_ion", "molecule_name", "smiles", "atom_position", "true_pka", "predicted_pka", "difference"]
        save_csv(train_parity_rows, train_csv, fieldnames)
        save_csv(test_parity_rows, test_csv, fieldnames)
        combine_csv_files(train_csv, test_csv, combined_csv)
        
        # 5. 繪製 parity plot
        if out_png is None:
            out_png = DATA_DIR / f"pka_parity_{model_version}.png"
            
        df = pd.read_csv(combined_csv)
        true_v = df["true_pka"].values
        pred_v = df["predicted_pka"].values
        
        if len(true_v) > 0 and len(pred_v) > 0:
            is_train = df["is_train"].values
            metrics = plot_parity(true_v, pred_v, title=title, save_path=out_png, is_train=is_train)
            print(f"\n評估指標:")
            print(f"RMSE: {metrics['RMSE']:.2f}")
            print(f"MAE: {metrics['MAE']:.2f}")
            print(f"R²: {metrics['R2']:.2f}")
        else:
            print("沒有足夠的數據來生成 parity plot")
            
    except Exception as e:
        print(f"處理過程中出錯: {str(e)}")
        import traceback
        traceback.print_exc()


def _process_df_for_parity(df: pd.DataFrame, tag: str, model: pka_GNN = None) -> List[dict]:
    """處理DataFrame並生成用於parity plot的行數據"""
    parity_rows: List[dict] = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{tag}"):
        mol_name = f"{tag}_Molecule_{idx}"
        true_dict = parse_pka_matrix(row.pka_matrix)
        pred = predict_pka(row.smiles, model=model)
        if pred is None:
            continue
        pos_pred, val_pred = pred

        # debug print
        print(f"[{tag}] {row.smiles}")
        print(f"  true positions: {list(true_dict.keys())}")
        print(f"  pred positions: {list(pos_pred)}")

        for pos, pval in zip(pos_pred, val_pred):
            if pos in true_dict:
                parity_rows.append({
                    "metal_ion": row.metal_ion,
                    "molecule_name": mol_name,
                    "smiles": row.smiles,
                    "atom_position": pos,
                    "true_pka": true_dict[pos],
                    "predicted_pka": pval,
                    "difference": abs(pval - true_dict[pos]),
                })
    return parity_rows

# -----------------------------------------------------------------------------
# Main – argparse sub-command dispatcher
# -----------------------------------------------------------------------------

import argparse  # placed late to avoid slowing import as module

def plot_parity_by_metal_ion(df, title="", save_path=None, is_train=True):
    """
    Plot parity plot grouped by metal ion types
    
    Args:
        df: DataFrame containing true_pka, predicted_pka and metal_ion columns
        title: Plot title
        save_path: Save path for the plot
        is_train: Whether this is training set data
    
    Returns:
        dict: Dictionary containing overall metrics
    """
    if len(df) == 0:
        print("Warning: No valid prediction data available for parity plot")
        return {
            "RMSE": float('nan'),
            "MAE": float('nan'),
            "R2": float('nan')
        }
    
    # Prepare data
    true_v = df["true_pka"].values
    pred_v = df["predicted_pka"].values
    
    # Calculate overall metrics
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(true_v, pred_v))),
        "MAE": float(mean_absolute_error(true_v, pred_v)),
        "R2": float(r2_score(true_v, pred_v))
    }
    
    # Get unique metal ion types
    metal_ions = df["metal_ion"].fillna("No Metal").unique()
    
    # Count samples for each metal ion and sort by count (descending)
    metal_counts = {}
    for ion in metal_ions:
        count = sum(df["metal_ion"].fillna("No Metal") == ion)
        metal_counts[ion] = count
    
    # Sort metal ions by sample count (descending)
    sorted_metals = sorted(metal_counts.keys(), key=lambda x: metal_counts[x], reverse=True)
    
    # Create a more vibrant color map
    cmap = plt.cm.tab20b  # for more distinct colors
    colors = {ion: cmap(i % 11) for i, ion in enumerate(sorted_metals)}
    
    # Prepare plot
    plt.figure(figsize=(10, 8))
    
    # Fixed x and y axis range
    min_val = -18
    max_val = 32
    
    plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=1, alpha=.7)
    
    # Plot data points for each metal ion in order of decreasing sample count
    for ion in sorted_metals:
        mask = df["metal_ion"].fillna("No Metal") == ion
        if not mask.any():
            continue
            
        ion_df = df[mask]
        ion_true = ion_df["true_pka"].values
        ion_pred = ion_df["predicted_pka"].values
        
        # Calculate metrics for this metal ion
        ion_metrics = {}
        if len(ion_true) > 0:
            ion_metrics = {
                "RMSE": float(np.sqrt(mean_squared_error(ion_true, ion_pred))),
                "MAE": float(mean_absolute_error(ion_true, ion_pred)),
                "R2": float(r2_score(ion_true, ion_pred)) if len(ion_true) > 1 else float('nan')
            }
        
        # Plot data points
        label = f"{ion} (n={len(ion_true)})"
        plt.scatter(ion_true, ion_pred, s=40, alpha=.7, label=label, 
                    color=colors[ion], edgecolors="black")
    
    # Add legend
    plt.legend(loc="lower right", title="Metal Ion")
    
    # Show overall metrics at top-left
    dataset_type = "Training Set" if is_train else "Test Set"
    txt = (f"{dataset_type} (n={len(true_v)}):\n"
           f"RMSE = {metrics['RMSE']:.2f}\n"
           f"MAE = {metrics['MAE']:.2f}\n"
           f"R²  = {metrics['R2']:.2f}")
    
    plt.gca().text(0.05, 0.95, txt, transform=plt.gca().transAxes,
                   va="top", ha="left", fontsize=13,
                   bbox=dict(boxstyle="round", facecolor="white", alpha=.85))
    
    plt.xlabel("pKa$_{\mathrm{true}}$ (unit: LogK)")
    plt.ylabel("pKa$_{\mathrm{pred}}$ (unit: LogK)")
    plt.title(title)
    
    # Set x and y axis range
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.gca().xaxis.set_major_locator(MultipleLocator(5))  # Major tick every 5 units
    plt.gca().yaxis.set_major_locator(MultipleLocator(5))  # Major tick every 5 units
    plt.grid(True, ls="--", alpha=.3)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Metal ion grouped plot saved → {save_path}")
    
    plt.close()
    
    return metrics

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="pKa prediction utility script")
    sub = p.add_subparsers(dest="command", required=True)

    # 為所有命令添加通用參數
    def add_model_version_arg(parser):
        parser.add_argument("--model-version", type=str, default=DEFAULT_MODEL_VERSION,
                          help=f"模型版本 (預設: {DEFAULT_MODEL_VERSION})")

    # 各子命令
    interactive = sub.add_parser("interactive", help="手動輸入 SMILES 進行預測")
    add_model_version_arg(interactive)

    exp = sub.add_parser("export", help="匯出完整預測 (parity + viz)")
    exp.add_argument("--out-dir", type=Path, default=DATA_DIR, help="輸出資料夾")
    add_model_version_arg(exp)

    chk = sub.add_parser("check", help="列出誤差大於閾值的分子")
    chk.add_argument("--threshold", type=float, default=0.5)
    add_model_version_arg(chk)

    split = sub.add_parser("split", help="分割資料集並分別採樣")
    split.add_argument("--test-size", type=float, default=0.2)
    add_model_version_arg(split)

    sample = sub.add_parser("sample-all", help="將所有分子的樣本預測存單一檔案")
    add_model_version_arg(sample)
    
    # 新增 plot 子命令
    plt_cmd = sub.add_parser("plot", help="從 CSV 生成 parity-plot")
    plt_cmd.add_argument("--csv", type=Path, help="已含 is_train 欄位之 combined CSV")
    plt_cmd.add_argument("--train-csv", type=Path, help="train CSV (若未給 --csv)")
    plt_cmd.add_argument("--test-csv", type=Path, help="test CSV (若未給 --csv)")
    plt_cmd.add_argument("--title", type=str, default="pKa Prediction (Train vs Test)")
    plt_cmd.add_argument("--out", type=Path, default=DATA_DIR / "pka_parity.png")
    add_model_version_arg(plt_cmd)
    
    # 新增一鍵完成子命令
    aio = sub.add_parser("all-in-one", help="一鍵完成: 分割→預測→繪圖")
    aio.add_argument("--test-size", type=float, default=0.2, help="測試集比例")
    aio.add_argument("--title", type=str, default="pKa Prediction (Train vs Test)", help="圖表標題")
    aio.add_argument("--out", type=Path, help="輸出圖檔路徑 (默認: ../output/pka_parity_{model_version}.png)")
    add_model_version_arg(aio)

    return p


def main() -> None:  # noqa: D401
    """Command-line entry point."""
    args = _build_parser().parse_args()
    if args.command == "interactive":
        interactive_mode(args.model_version)
    elif args.command == "export":
        export_full_results(args.out_dir, model_version=args.model_version)
    elif args.command == "check":
        check_errors(args.threshold, args.model_version)
    elif args.command == "split":
        split_and_sample(args.test_size, args.model_version)
    elif args.command == "sample-all":
        sample_all(args.model_version)
    elif args.command == "plot":
        plot_cmd(args.csv, args.train_csv, args.test_csv, args.title, args.out, args.model_version)
    elif args.command == "all-in-one":
        all_in_one(args.test_size, args.title, args.out, args.model_version)
    else:  # pragma: no cover – argparse already prevents this
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
