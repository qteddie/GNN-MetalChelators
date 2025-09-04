# -*- coding: utf-8 -*-
"""Train pKa‐GNN with autoregressive GCN3"""
import os, time, sys, math, argparse, json, ast
import torch, torch.nn as nn
from torch_geometric.loader import DataLoader
import pandas as pd
from torch.optim.lr_scheduler import ExponentialLR
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- your own utils / models ---
import sys
sys.path.append("/work/s6300121/LiveTransForM-main/metal")
from src.metal_trainutils import load_config_by_version, pka_Dataloader
from src.metal_models   import pka_GNN, MetalFeatureExtractor
from src.metal_chemutils import tensorize_for_pka
from MetalGCN.src.learning_curve import plot_learning_curves
from src.metal_sample import all_in_one

# ---------------- argparse ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--config_csv", default="/work/s6300121/LiveTransForM-main/metal/pka_parameters.csv")
parser.add_argument("--version"   , default="pka_ver5")
args = parser.parse_args()

cfg = load_config_by_version(args.config_csv, args.version)
print("\n=== Config ===")
for k, v in cfg.items(): 
    print(f"{k:18}: {v}")
print("==============\n")

# ---------------- I/O dirs ----------------
os.makedirs(cfg["save_path"] , exist_ok=True)
os.makedirs(cfg["model_path"], exist_ok=True)

# ---------------- Data ----------------
cols = ["smiles", "pka_values", "pka_matrix", "metal_ion"]  # 添加 metal_ion 列
train_loader, test_loader = pka_Dataloader.data_loader_pka(
    file_path   = cfg["input"],
    columns     = cols,
    tensorize_fn= tensorize_for_pka,
    batch_size  = cfg["batch_size"],
    test_size   = cfg["test_size"],
)

# ------------- compute μ / σ -------------
print("⏳  Computing global pKa mean / std …")
vals = []
for batch in train_loader:               # 先 CPU 收集，省顯卡記憶體
    mask = batch.pka_labels > 0
    vals.append(batch.pka_labels[mask].float())

vals = torch.cat(vals)
pka_mean = vals.mean().item()
pka_std  = max(vals.std(unbiased=False).item(), 1e-6)
print(f"   → mean={pka_mean:.3f}, std={pka_std:.3f}")

# ---------------- Model ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = pka_GNN(
            node_dim  = cfg["num_features"],
            bond_dim  = 11,
            hidden_dim= cfg["num_features"],
            output_dim= 1,
            dropout   = cfg["dropout"],
            depth     = cfg["depth"],
            use_metal_features = True  # 啟用金屬特徵
         ).to(device)
model.set_pka_normalization(pka_mean, pka_std)

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

# 使用簡單的學習率衰減
scheduler = ExponentialLR(optimizer, gamma=cfg["anneal_rate"])

grad_clip = 1.0  # 適度放寬梯度裁剪
df_loss   = []

best_ckpt, best_loss = None, float("inf")
start = time.time()

for epoch in range(cfg["num_epochs"]):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    lr_now = optimizer.param_groups[0]["lr"]
    print(f"\nEpoch {epoch+1}/{cfg['num_epochs']}  lr={lr_now:.5e}")

    # -------- train loop --------
    tr_loss = tr_cla = tr_reg = n_batch = 0
    for batch in train_loader:
        batch = batch.to(device, non_blocking=True)
        
        # 修改金屬離子特徵處理方式 - 為每個批次創建金屬特徵列表
        if hasattr(batch, 'metal_ion'):
            # 為每個分子創建金屬特徵
            metal_features_list = []
            
            for ion in batch.metal_ion:
                if not ion or ion == "None" or ion == "":
                    metal_features_list.append(None)
                    continue
                    
                try:
                    # 獲取金屬特徵元組
                    metal_features = MetalFeatureExtractor.get_metal_features(ion)
                    metal_features_list.append(metal_features)
                except Exception as e:
                    print(f"處理金屬離子 {ion} 時出錯: {e}")
                    metal_features_list.append(None)
            
            # 將金屬特徵列表添加到batch
            batch.metal_features = metal_features_list
        
        ###### 前向傳播 #######
        logits, pka_pred, (loss, loss_cla, loss_reg) = model(batch)

        if torch.isnan(loss):  # very rare, just skip
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)

        tr_loss += loss.item(); tr_cla += loss_cla.item(); tr_reg += loss_reg.item()
        n_batch += 1

    if n_batch == 0:
        raise RuntimeError("all training batches skipped due to NaN")
    tr_loss /= n_batch; tr_cla /= n_batch; tr_reg /= n_batch

    # -------- evaluate --------
    tst_loss, tst_cla, tst_reg, tst_metrics = pka_Dataloader.evaluate_pka_model(
        model, test_loader, device,
        output_file=f"{cfg['version']}_epoch_{epoch}_test.csv",
        save_path = cfg["save_path"]
    )

    print(f"[train]  loss={tr_loss:.4f} (cla {tr_cla:.3f}  reg {tr_reg:.3f})")
    print(f"[test ]  loss={tst_loss:.4f} (cla {tst_cla:.3f}  reg {tst_reg:.3f})")

    df_loss.append({
        "epoch": epoch, "train_loss": tr_loss, "train_cla_loss": tr_cla,
        "train_reg_loss": tr_reg, "test_loss": tst_loss,
        "test_cla_loss": tst_cla, "test_reg_loss": tst_reg,
        "rmse_gt": tst_metrics["rmse_gt"], "rmse_hit": tst_metrics["rmse_hit"],
        "learning_rate": lr_now,
        "precision": tst_metrics["precision"],
        "recall": tst_metrics["recall"],
        "f1": tst_metrics["f1"],
                
        "test_classification_metrics": json.dumps(tst_metrics)
    })
    # -------- checkpoint --------
    if tst_loss < best_loss:
        best_loss, best_ckpt = tst_loss, model.state_dict().copy()
        torch.save(best_ckpt, os.path.join(cfg["model_path"], f"{cfg['version']}_best.pkl"))

    # -------- 每個epoch更新學習曲線 --------
    # 保存當前的損失數據
    temp_df = pd.DataFrame(df_loss)
    temp_df.to_csv(os.path.join(cfg["save_path"], f"{cfg['version']}_loss.csv"), index=False)
    
    # 更新學習曲線圖
    plot_learning_curves(
        csv_path=os.path.join(cfg["save_path"], f"{cfg['version']}_loss.csv"),
        output_dir=cfg["save_path"],
        version_name=cfg["version"],
        epoch_count=epoch+1
    )
    
    scheduler.step()

# ---------------- end of training ----------------
pd.DataFrame(df_loss).to_csv(
    os.path.join(cfg["save_path"], f"{cfg['version']}_loss.csv"), index=False)
print(f"Loss file saved to {cfg['save_path']}/{cfg['version']}_loss.csv")
torch.save(model.state_dict(), os.path.join(cfg["model_path"], f"{cfg['version']}_final.pkl"))

print(f"\nTraining finished in {(time.time()-start):.1f}s; "
      f"best test‑loss={best_loss:.4f}")

# -------- load best & final evaluation --------
if best_ckpt is not None:
    model.load_state_dict(best_ckpt)
    test_loss, test_cla, test_reg, tst_metrics = pka_Dataloader.evaluate_pka_model(
        model, test_loader, device,
        output_file=f"{cfg['version']}_best_final.csv",
        save_path=cfg["save_path"]
    )
    print(f"Best ckpt  loss={test_loss:.3f}  pka‑atom‑acc={tst_metrics['pka_atom_accuracy']:.3f}")

# -------- 生成parity plot --------
print("\nGenerating parity plot...")
plot_title = f"pKa Prediction - {cfg['version']}"
# Convert path to Path object
plot_path = Path(cfg["model_path"]) / f"{cfg['version']}_parity_plot.png"
# Use the same test set ratio as in training
all_in_one(test_size=cfg["test_size"], title=plot_title, out_png=plot_path, model_version=cfg['version'])

# -------- 新增：生成按金屬離子分組的parity plot --------
print("\nGenerating metal ion grouped parity plots...")
try:
    # 首先確保已經執行了分割和預測
    # 獲取映射文件中的數據
    mapping_df = pd.read_csv(cfg["input"])
    # 分割數據集
    train_df, test_df = train_test_split(mapping_df, test_size=cfg["test_size"], random_state=42)
    
    # 準備存儲預測結果的列表
    train_parity_rows = []
    test_parity_rows = []
    
    # 使用最佳模型進行預測
    model.load_state_dict(best_ckpt)
    model.eval()
    
    # 處理訓練集
    print("Processing training set data...")
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        mol_name = f"train_Molecule_{idx}"
        true_dict = {}
        try:
            # 解析pka_matrix
            if isinstance(row.pka_matrix, str):
                pairs = ast.literal_eval(row.pka_matrix)
                true_dict = {int(pos): float(val) for pos, val in pairs}
        except Exception as e:
            print(f"Error parsing pka_matrix: {e}")
            continue
            
        # 預測
        try:
            result = model.sample(row.smiles, device=device)
            if not result:
                continue
                
            # 將結果添加到parity_rows
            for pos, pval in zip(result["pka_positions"], result["pka_values"]):
                if pos in true_dict:
                    train_parity_rows.append({
                        "metal_ion": row.metal_ion if "metal_ion" in row else "No Metal",
                        "smiles": row.smiles,
                        "atom_position": pos,
                        "true_pka": true_dict[pos],
                        "predicted_pka": pval,
                        "difference": abs(pval - true_dict[pos]),
                    })
        except Exception as e:
            print(f"Prediction error: {e}")
            continue
    
    # 處理測試集
    print("Processing test set data...")
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        mol_name = f"test_Molecule_{idx}"
        true_dict = {}
        try:
            # 解析pka_matrix
            if isinstance(row.pka_matrix, str):
                pairs = ast.literal_eval(row.pka_matrix)
                true_dict = {int(pos): float(val) for pos, val in pairs}
        except Exception as e:
            print(f"Error parsing pka_matrix: {e}")
            continue
            
        # 預測
        try:
            result = model.sample(row.smiles, device=device)
            if not result:
                continue
                
            # 將結果添加到parity_rows
            for pos, pval in zip(result["pka_positions"], result["pka_values"]):
                if pos in true_dict:
                    test_parity_rows.append({
                        "metal_ion": row.metal_ion if "metal_ion" in row else "No Metal",
                        "smiles": row.smiles,
                        "atom_position": pos,
                        "true_pka": true_dict[pos],
                        "predicted_pka": pval,
                        "difference": abs(pval - true_dict[pos]),
                    })
        except Exception as e:
            print(f"Prediction error: {e}")
            continue
    
    # 將結果轉換為DataFrame
    train_df = pd.DataFrame(train_parity_rows)
    test_df = pd.DataFrame(test_parity_rows)
    
    # 保存到CSV
    train_csv_path = Path(cfg["save_path"]) / f"pka_train_prediction_{cfg['version']}.csv"
    test_csv_path = Path(cfg["save_path"]) / f"pka_test_prediction_{cfg['version']}.csv"
    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)
    
    # 生成訓練集的按金屬離子分組的parity plot
    from src.metal_sample import plot_parity_by_metal_ion
    
    if not train_df.empty:
        train_metal_plot_path = Path(cfg["model_path"]) / f"{cfg['version']}_train_metal_parity_plot.png"
        train_metal_plot_title = f"Training Set pKa Prediction (By Metal Ion) - {cfg['version']}"
        train_metrics = plot_parity_by_metal_ion(train_df, title=train_metal_plot_title, 
                                               save_path=train_metal_plot_path, is_train=True)
        print(f"Training set metal ion grouped parity plot generated: {train_metal_plot_path}")
    else:
        print("No valid prediction results for training set")
    
    # 生成測試集的按金屬離子分組的parity plot
    if not test_df.empty:
        test_metal_plot_path = Path(cfg["model_path"]) / f"{cfg['version']}_test_metal_parity_plot.png"
        test_metal_plot_title = f"Test Set pKa Prediction (By Metal Ion) - {cfg['version']}"
        test_metrics = plot_parity_by_metal_ion(test_df, title=test_metal_plot_title, 
                                              save_path=test_metal_plot_path, is_train=False)
        print(f"Test set metal ion grouped parity plot generated: {test_metal_plot_path}")
    else:
        print("No valid prediction results for test set")
    
except Exception as e:
    print(f"Error generating metal ion grouped parity plots: {str(e)}")
    import traceback
    traceback.print_exc()

