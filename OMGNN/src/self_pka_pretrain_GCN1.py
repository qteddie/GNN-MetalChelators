# -*- coding: utf-8 -*-
"""
Train pKa‐GNN with autoregressive GCN3
"""
import os, time, sys, math, argparse, json
import torch, torch.nn as nn
from torch_geometric.loader import DataLoader
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path

# --- your own utils / models ---
sys.path.append("src/")
from self_pka_trainutils import load_config_by_version, pka_Dataloader
from self_pka_models   import pka_GNN, pka_GNN_ver2, pka_GNN_ver3
from self_pka_chemutils import tensorize_for_pka
from pka_learning_curve import plot_learning_curves
from self_pka_sample import all_in_one
# ---------------- argparse ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--config_csv", default="../data/pka_parameters.csv")
parser.add_argument("--version"   , default="pka_ver1")
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
cols = ["smiles", "pka_values", "pka_matrix"]
train_loader, test_loader = pka_Dataloader.data_loader_pka(
    file_path   = cfg["input"],
    columns     = cols,
    tensorize_fn= tensorize_for_pka,
    batch_size  = cfg["batch_size"],
    test_size   = cfg["test_size"],
    k_pe        = cfg["pe_dim"],
    max_dist    = cfg["max_dist"],
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
bond_dim = 11 + cfg["max_dist"] + 1 # 11 + 6 = 17
model  = pka_GNN_ver3(
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
model.set_pka_normalization(pka_mean, pka_std)
optimizer = AdamW(
    model.parameters(),
    lr=cfg["lr"],
    weight_decay=cfg["weight_decay"],
)
scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=0)
# scheduler = ExponentialLR(optimizer, gamma=cfg["anneal_rate"])
grad_clip = 1.0 
df_loss   = []
best_loss = float("inf")
start = time.time()
for epoch in range(cfg["num_epochs"]):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    lr_now = optimizer.param_groups[0]["lr"]
    print(f"\nEpoch {epoch+1}/{cfg['num_epochs']}  lr={lr_now:.2e}")
    # -------- train loop --------
    tr_loss = tr_cla = tr_reg = n_batch = 0
    for batch in train_loader:
        batch = batch.to(device, non_blocking=True)
        logitss, pkas, target_final, pka_labels, (loss, loss_cla, loss_reg) = model(batch)
        loss = loss_cla + loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        tr_loss += loss.item(); tr_cla += loss_cla.item(); tr_reg += loss_reg.item()
        n_batch += 1
    if n_batch == 0:    
        raise RuntimeError("all training batches skipped due to NaN")
    tr_loss /= n_batch; tr_cla /= n_batch; tr_reg /= n_batch
    eval_model = model
    # -------- evaluate --------
    tst_loss, tst_cla, tst_reg, tst_metrics = pka_Dataloader.evaluate_pka_model(
        eval_model, test_loader, device
    )
    tst_loss = tst_cla + tst_reg
    print(f"[train]  loss={tr_loss:.4f} (cla {tr_cla:.3f}  reg {tr_reg:.3f})")
    print(f"[test ]  loss={tst_loss:.4f} (cla {tst_cla:.3f}  reg {tst_reg:.3f})")
    df_loss.append({
        "epoch": epoch, "train_loss": tr_loss, "train_cla_loss": tr_cla,
        "train_reg_loss": tr_reg, "test_loss": tst_loss,
        "test_cla_loss": tst_cla, "test_reg_loss": tst_reg,
        "rmse": tst_metrics["rmse"],
        "learning_rate": lr_now,
        "accuracy": tst_metrics["accuracy"],
        "precision": tst_metrics["precision"],
        "recall": tst_metrics["recall"],
    })
    # -------- checkpoint --------
    if tst_loss < best_loss:
        best_loss = tst_loss
        ckpt = eval_model.state_dict()
        torch.save(
            ckpt, 
            os.path.join(cfg["model_path"], f"{cfg['version']}_best.pkl")
        )

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
    # 更新學習率
    scheduler.step()
# ---------------- end of training ----------------
pd.DataFrame(df_loss).to_csv(
    os.path.join(cfg["save_path"], f"{cfg['version']}_loss.csv"), index=False)
print(f"Loss file saved to {cfg['save_path']}/{cfg['version']}_loss.csv")
torch.save(model.state_dict(), os.path.join(cfg["model_path"], f"{cfg['version']}_final.pkl"))

print(f"\nTraining finished in {(time.time()-start):.1f}s; "
      f"best test‑loss={best_loss:.4f}")

# -------- load best & final evaluation --------
model.load_state_dict(
    torch.load(
        os.path.join(cfg["model_path"], f"{cfg['version']}_best.pkl"), 
        map_location=device, 
        weights_only=True
    )
)
test_loss, test_cla, test_reg, tst_metrics = pka_Dataloader.evaluate_pka_model(
    model, test_loader, device
)
print(f"Best ckpt  test_total_loss={test_loss:.3f}, test_cla_loss={test_cla:.3f}, test_reg_loss={test_reg:.3f}")

# -------- 生成parity plot --------
print("\n正在生成parity plot...")
plot_title = f"pKa Prediction - {cfg['version']}"
# 將路徑轉換為Path物件
plot_path = Path(cfg["model_path"]) / f"{cfg['version']}_parity_plot.png"
all_in_one(test_size=cfg["test_size"], title=plot_title, out_png=plot_path, model_version=cfg['version'])
