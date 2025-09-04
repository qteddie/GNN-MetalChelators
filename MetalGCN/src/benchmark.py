import torch, random
from torch.utils.benchmark import Timer
import pickle
import sys
import os
import torch
import pandas as pd
import numpy as np
sys.path.append("../model")
VERSION = "metal_ver17"


from metal_pka_transfer import load_config_by_version, CustomMetalPKA_GNN, evaluate, load_preprocessed_metal_data, create_weight_tables
cfg = load_config_by_version("../data/parameters.csv", VERSION)
with open(cfg["dataloader_path"], "rb") as f:
    dl_train, dl_val, dl_test, mu, sigma, n_metals = pickle.load(f)
print(f"已加載預處理的資料: {cfg['dataloader_path']}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bin_edges, bin_w = create_weight_tables(cfg["version"])
model = CustomMetalPKA_GNN(n_metals, cfg["metal_emb_dim"],
                               cfg["num_features"],  # node_dim
                               9, # 先測試沒有spd的版本
                               cfg["hidden_size"],
                               cfg["output_size"],
                               cfg["dropout"],
                               cfg["depth"],
                               cfg["heads"],
                               spd_dim=cfg["spd_dim"],
                               max_dist=cfg["max_dist"],
                                bin_edges=bin_edges,
                                bin_weights=bin_w,
                                huber_beta=cfg["huber_beta"],
                                reg_weight=cfg["reg_weight"],
                               ).to(device)

model.set_pka_normalization(mu, sigma)

model.load_state_dict(torch.load(f"../output/{VERSION}/best_model.pt", weights_only=True))
model.eval()

# 1⃣ 先從 dataloader 抓一個 batch，搬到 GPU
batch = next(iter(dl_train))
batch = batch.to(device)

import torch.nn as nn
from torch_geometric.nn import TransformerConv

num_metals = n_metals
metal_emb_dim = cfg["metal_emb_dim"]
node_dim = cfg["num_features"]
hidden_dim = cfg["hidden_size"]
heads = cfg["heads"]
bond_dim = 9
dropout = cfg["dropout"]


device = torch.device("cuda", 0)          # 或沿用前面的 device 變數

metal_emb = nn.Embedding(num_metals, metal_emb_dim, device=device)
node_proj  = nn.Linear(node_dim + metal_emb_dim, hidden_dim, device=device)
gate       = nn.Sequential(
                 nn.Linear(hidden_dim, hidden_dim, device=device), nn.Tanh()
             )
transformer = TransformerConv(
                 in_channels = hidden_dim,
                 out_channels = hidden_dim // heads,
                 edge_dim = bond_dim,
                 heads = heads,
                 concat = True,
                 dropout = dropout,
             ).to(device) 


dim_reduction = nn.Identity()
# ── pKa 標準化常數 ───────────────────────
pka_mu    = torch.tensor(mu,    device=device)
pka_sigma = torch.tensor(sigma, device=device)

# ── ➊ pKa-bin 相關 ──────────────────────
bin_edges_np, bin_weights_np = create_weight_tables(cfg["version"])
bin_edges   = torch.tensor(bin_edges_np,   device=device)
bin_weights = torch.tensor(bin_weights_np, device=device)

huber_beta  = cfg["huber_beta"]
reg_weight  = cfg["reg_weight"]

# metal embedding broadcast to nodes
m_emb = metal_emb(batch.metal_id)  # [G, E]
m_emb_nodes = m_emb[batch.batch]        # [N, E]
x_cat = torch.cat([batch.x, m_emb_nodes], dim=1)
h = node_proj(x_cat)

h = transformer(h, batch.edge_index, edge_attr=batch.edge_attr)
h_cur = dim_reduction(h)

gt_mask = batch.pka_labels != 0
idx_gt = torch.nonzero(gt_mask).squeeze(1)
target = gt_mask.long()


idx_sorted = idx_gt[torch.argsort(batch.pka_labels[idx_gt])]
idx_sorted = torch.cat([idx_sorted, torch.tensor([-1], device=device)])

target_final = target.repeat(idx_sorted.shape[0] + 1, 1)
for i in range(len(idx_sorted) - 1):
    target_final[i + 1][idx_sorted[: i + 1]] = 0

logitss = []
pred_reg = []
loss_cla_steps = []
loss_reg_steps = []

for step_idx, idx in enumerate(idx_sorted):
    logits = model.atom_classifier(h_cur)
    logitss.append(logits)

    t = target_final[step_idx]
    neg_pos_ratio = float((t == 0).sum()) / (t.sum() + 1e-6)
    neg_pos_ratio = min(neg_pos_ratio, 50.0)   # ★ 上限 50
    loss_c = nn.functional.cross_entropy(
        logits,
        t,
        weight=torch.tensor([1.0, neg_pos_ratio], device=device),
        reduction="none",
    )
    loss_cla_steps.extend(loss_c)

    if step_idx != len(idx_sorted) - 1:
        pred_pka = model.atom_regressor(h_cur).view(-1)[idx]
        true_pka = batch.pka_labels[idx]
        
        # 標準化後計算 base loss（元素級，reduction='none' 以便乘權重）
        pred_norm = (pred_pka - pka_mu) / pka_sigma
        true_norm = (true_pka - pka_mu) / pka_sigma
        base_loss = nn.functional.smooth_l1_loss(
            pred_norm,
            true_norm,
            beta=huber_beta,
            reduction="none",
        )
        bin_idx = torch.bucketize(true_pka, bin_edges).to(device) - 1
        weight  = bin_weights[bin_idx]
        loss_r = (weight * base_loss).mean()
        
        loss_reg_steps.append(loss_r)
        pred_reg.append(pred_pka)

    if idx != -1:
        h_upd = h_cur.clone()
        h_upd[idx] = h[idx] + h_cur[idx] * gate(h_cur[idx])
        h = transformer(h_upd, batch.edge_index, edge_attr=batch.edge_attr)
        h_cur = dim_reduction(h)

loss_cla = torch.stack(loss_cla_steps).mean()
loss_reg = torch.stack(loss_reg_steps).mean() if loss_reg_steps else torch.tensor(0.0, device=device)


loss_reg = reg_weight * loss_reg
total = loss_cla + loss_reg 





