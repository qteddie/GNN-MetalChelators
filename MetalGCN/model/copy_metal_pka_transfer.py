#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Metal pKa Transfer Learning Model with Extensible Ligand Support

This module implements a Graph Neural Network for predicting metal-ligand binding 
constants (pKa values) with support for extensible vectors handling different numbers 
of ligands (ML, ML₂, ML₃ complexes).

Key Features:
- Extensible architecture supporting 1-3 ligands per metal complex
- Metal embedding with adaptive gating mechanisms
- Transfer learning from proton pKa prediction models
- Weighted loss functions for handling imbalanced pKa distributions
- Support for 10 different metal ions (Cu2+, Ag+, Zn2+, etc.)

Authors: Modified for extensible ligand support (2025-06-29)
"""

from __future__ import annotations

# 抑制常見警告
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 抑制 TensorFlow 警告
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import sys
import time
import traceback
from typing import Dict, List, Tuple

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from collections import defaultdict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms

# 抑制 RDKit 警告
RDLogger.DisableLog('rdApp.*')
# -----------------------------------------------------------------------------
# 本地專案模組
# -----------------------------------------------------------------------------

sys.path.append("../src/")
from metal_chemutils import tensorize_for_pka  # noqa: E402
from learning_curve import plot_learning_curves
from parity_plot import parity_plot 
from function import output_evaluation_results
################################################################################
# 工具函式
################################################################################


def load_config_by_version(csv_path: str, version: str) -> Dict:
    df = pd.read_csv(csv_path)
    row = df[df["version"] == version]
    if row.empty:
        raise ValueError(f"No configuration found for version {version}")
    cfg = row.iloc[0].to_dict()

    int_keys = [
        "seed",
        "output_size",
        "num_features",
        "hidden_size",
        "batch_size",
        "num_epochs",
        "depth",
        "heads",
        "pe_dim",
        "max_dist",
        "spd_dim",
        "max_ligands",
    ]
    float_keys = ["dropout", "lr", "anneal_rate", "weight_decay", "reg_weight", "huber_beta"]

    for k in int_keys:
        cfg[k] = int(cfg[k])
    for k in float_keys:
        cfg[k] = float(cfg[k])

    cfg.setdefault("metal_emb_dim", 8)
    return cfg


def parse_args():
    p = argparse.ArgumentParser("金屬 pKa 轉移學習 (fix‑up 版)")
    p.add_argument("--config_csv", dest='csv_path', default="../data/parameters.csv")
    p.add_argument("--version", default="metal_ver11")
    return load_config_by_version(**vars(p.parse_args()))


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

################################################################################
# Dataset
################################################################################


def build_metal_vocab(df: pd.DataFrame) -> Dict[str, int]:
    metals = sorted(df["metal_ion"].unique().tolist())
    return {m: i for i, m in enumerate(metals)}


def find_potential_acidic_sites(mol):  # 簡化版採用氧/氮常見酸位
    patt = Chem.MolFromSmarts("[O,N;H1]")
    return [idx for match in mol.GetSubstructMatches(patt) for idx in match]


def fallback_acidic_site_detection(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in {"O", "N"}]


def add_spd_edge_attr(edge_index, bond_attr, num_nodes, max_dist=5):
    """
    Args
    ----
    edge_index : LongTensor [2, E]      ─ 原始（雙向）邊
    bond_attr  : FloatTensor [E, d_bond]── 你的化學鍵特徵
    num_nodes  : int
    max_dist   : 最遠考慮幾 hop；>max_dist 與不連通統一歸一類
    Returns
    -------
    ei_full    : LongTensor [2, E_full]── 完整 k-hop 邊（含原本邊）
    edge_attr  : FloatTensor [E_full, d_bond + (max_dist+1)]
    """
    device = edge_index.device
    # BFS求shortest path distance
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    dist = torch.full((num_nodes, num_nodes), float('inf'), device=device)
    for v in range(num_nodes):
        dist[v, v] = 0
        frontier = [v]
        d = 1
        while frontier and d <= max_dist:
            nxt = []
            for u in frontier:
                neigh = adj[u].nonzero(as_tuple=False).view(-1)
                for w in neigh:
                    if dist[v, w] == float('inf'):
                        dist[v, w] = d
                        nxt.append(w.item())
            frontier = nxt
            d += 1
    dist = dist.clamp_max(max_dist)
    ei_full, d_flat = dense_to_sparse(dist)
    
    # 2) SPD one-hot
    spd_onehot = torch.nn.functional.one_hot(
        d_flat.long(), num_classes=max_dist+1
    ).float()
    
    # 3) combine
    E_full = ei_full.size(1)
    d_bond = bond_attr.size(1)
    bond_feat_full = torch.zeros(E_full, d_bond, device=device)
    # build (u, v) -> index map
    edge2idx = {(int(u), int(v)): i 
                for i, (u, v) in enumerate(edge_index.t().tolist())}
    for i in range(E_full):
        u, v = int(ei_full[0, i]), int(ei_full[1, i])
        if (u, v) in edge2idx:
            bond_feat_full[i] = bond_attr[edge2idx[(u, v)]]
    edge_attr = torch.cat([bond_feat_full, spd_onehot], dim=-1)
    return ei_full, edge_attr

def add_spd_feature(data: Data, max_dist: int = 5) -> Data:
    """
    為每條邊附上一個最短路徑距離 (Shortest-Path Distance, SPD)。
    若實際距離 > max_dist，統一設為 max_dist+1。
    結果放到 `data.spd` (LongTensor, shape = [E])
    """
    g = to_networkx(data, to_undirected=True)  # 轉 NetworkX
    spd_dict = dict(nx.all_pairs_shortest_path_length(g, cutoff=max_dist))

    dists = []
    for i, j in data.edge_index.t().tolist():
        dij = spd_dict[i].get(j, max_dist + 1)
        dists.append(dij)

    data.spd = torch.tensor(dists, dtype=torch.long)
    return data


def create_weight_tables(version: str,
                         pka_bin_edges=None,
                         bin_width: float = 2.0,
                         gamma: float = 0.4):

    csv_path = Path(f"../output/metal_ver11/metal_ver11_evaluation_results.csv")
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df      = pd.read_csv(csv_path)
    df_train = df[df["is_train"] == True]
    # ---------- pKa-bin -------------------------------------------
    y_train = df_train["y_true"].values.astype(np.float32)
    if pka_bin_edges is None:
        lo, hi   = np.floor(y_train.min()), np.ceil(y_train.max())
        pka_bin_edges = np.arange(lo, hi + bin_width, bin_width, dtype=np.float32)
        if pka_bin_edges[-1] <= hi:           # 確保能 cover max
            pka_bin_edges = np.append(pka_bin_edges, np.inf)
    hist, _ = np.histogram(y_train, bins=pka_bin_edges)
    freq    = hist / hist.sum()
    # pka_w   = (1.0 / (freq + 1e-6)) ** gamma
    # 強調低pKa的樣本
    pka_w = (1.0 / (freq + 1e-6)) ** gamma  # instead of 1 / freq
    pka_w   = pka_w / pka_w.mean()
    # ---------- save npz ------------------------------------------
    out_dir = Path("../data"); out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{version}_pka_weight_table.npz",
             bin_edges=pka_bin_edges, weights=pka_w.astype(np.float32))
    print(f"[NPZ] 權重表已存於 ../data/{version}_pka_weight_table.npz")
    return pka_bin_edges, pka_w


def make_data_obj(
    row: pd.Series,
    metal2idx: Dict[str, int],
    cfg: Dict,
) -> Tuple[Data, List[float], float]:
    """
    Convert a DataFrame row to PyTorch Geometric Data object with extensible ligand support.
    
    Args:
        row: DataFrame row containing SMILES, metal_ion, and pKa_value
        metal2idx: Dictionary mapping metal ion strings to indices
        cfg: Configuration dictionary
        
    Returns:
        tuple: (PyG Data object, list of pKa values, max pKa value)
        
    Supports multiple ligands (ML, ML₂, ML₃) by processing pKa_value lists.
    """

    from ast import literal_eval
    smiles     = row["SMILES"].tolist()[0]
    metal_ion  = row["metal_ion"].tolist()[0]
    # smiles     = row["SMILES"].tolist()[0]
    # metal_ion  = row["metal_ion"]
    # pka_values = eval(row["pKa_value"])
    pka_val_raw = row["pKa_value"].iloc[0]
    pka_values = literal_eval(pka_val_raw)
    print(smiles,metal_ion,pka_values)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    # ── 基本 node / edge 特徵 ───────────────────────────────
    fatoms, edge_index, edge_attr = tensorize_for_pka(smiles)
    # ── 加入 SPD 特徵 ─────────────────────────────────
    # 這邊先註解掉
    # ei_full, edge_features_spd = add_spd_edge_attr(
    #     edge_index, edge_attr, fatoms.size(0), max_dist=cfg['max_dist']
    # )
    num_atoms = fatoms.size(0)

    # ── pKa label（每個原子一個 scalar；沒資料=0）────────────
    pka_labels = torch.zeros(num_atoms)
    acidic_sites = (
        find_potential_acidic_sites(mol) or fallback_acidic_site_detection(mol)
    )
    # ★ 修改：支援多配體情況，確保有足夠的酸性位點
    n_ligands = len(pka_values)
    if len(acidic_sites) < n_ligands:
        # 如果酸性位點不足，補充更多可能的位點
        additional_sites = fallback_acidic_site_detection(mol)
        all_sites = list(set(acidic_sites + additional_sites))
        acidic_sites = all_sites[:n_ligands] if len(all_sites) >= n_ligands else all_sites
    
    # 使用向量化操作提高效率
    for i, pka in enumerate(pka_values):
        if i < len(acidic_sites) and acidic_sites[i] < num_atoms:
            pka_labels[acidic_sites[i]] = float(pka)  # 確保是float類型
    # ── 建立 Data 物件（暫時不含 SPD）──────────────────────
    # 如果要改回去就把edge_index, edge_attr改成edge_index, edge_attr
    data = Data(
        x=fatoms,
        edge_index=edge_index, # 從ei_full改回原來的edge_index
        edge_attr=edge_attr, # 從edge_features_spd改回原來的edge_attr
        pka_labels=pka_labels,
        smiles=smiles,
        metal_id=torch.tensor([metal2idx[metal_ion]], dtype=torch.long),
    )
    # ── ★ 加入 SPD 特徵 ★ ─────────────────────────────────
    # data = add_spd_feature(data, max_dist=cfg["max_dist"])
    max_pka = max(pka_values) if len(pka_values) else 0.0   # ★
    return data, pka_values, max_pka                        # ★



from sklearn.model_selection import StratifiedShuffleSplit

def stratified_split(data_list, train_ratio=0.7, val_ratio=0.15, seed=42):
    y = [d.metal_id.item() for d in data_list]
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(sss.split(np.zeros(len(y)), y))

    # 再把 temp 切成 val / test
    y_temp = [y[i] for i in temp_idx]
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        train_size=val_ratio / (1 - train_ratio),
        random_state=seed)
    val_idx, test_idx = next(sss2.split(np.zeros(len(y_temp)), y_temp))

    val_idx = [temp_idx[i] for i in val_idx]
    test_idx = [temp_idx[i] for i in test_idx]

    to_subset = lambda idx: [data_list[i] for i in idx]
    return to_subset(train_idx), to_subset(val_idx), to_subset(test_idx)


def load_preprocessed_metal_data(cfg: Dict, csv_path: str, batch_size: int, num_workers: int = 4):
    
    df = pd.read_csv(csv_path)
    metal2idx = build_metal_vocab(df)

    data_list, all_pkas, max_pka_list = [], [], []   # ★ 新增


    for _, row in tqdm(df.iterrows(), total=len(df), desc="CSV→Data"):
        try:
            d, pkas, max_pka = make_data_obj(row, metal2idx, cfg)
            data_list.append(d)
            all_pkas.extend(pkas)
            max_pka_list.append(max_pka)
        except Exception:
            traceback.print_exc()

    mu, sigma = float(np.mean(all_pkas)), float(np.std(all_pkas, ddof=0))
    print(f"pKa scaling  μ={mu:.2f}, σ={sigma:.2f}")

    train, val, test = stratified_split(data_list, train_ratio=0.7, val_ratio=0.15)
    from collections import Counter

    def print_metal_distribution(name, dataset):
        ids = [d.metal_id.item() for d in dataset]
        print(f"{name} metal 分佈:", Counter(ids))

    print_metal_distribution("Train", train)
    print_metal_distribution("Val", val)
    print_metal_distribution("Test", test)

    common_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test, shuffle=False, **common_kwargs)

    with open(cfg["dataloader_path"], "wb") as f:
        pickle.dump((train_loader, val_loader, test_loader, mu, sigma, len(metal2idx), all_pkas), f)

    return train_loader, val_loader, test_loader, mu, sigma, len(metal2idx), all_pkas


################################################################################
# Model
################################################################################



class CustomMetalPKA_GNN(nn.Module):
    def __init__(self, num_metals: int, metal_emb_dim: int,
                 node_dim, bond_dim, hidden_dim, output_dim,
                 dropout, depth, heads,
                 *,  # ← 星號表示之後隻能用關鍵字
                 bin_edges, bin_weights,
                 huber_beta: float,
                 reg_weight: float,
                 max_ligands: int = 3,  # ★ 新增：最大配體數量
                 ):      
        super().__init__()
        
        # ── 基本參數設置 ──
        self.node_dim = node_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.depth = depth
        self.heads = heads
        # ── 金屬嵌入和節點投影 ──
        self.metal_emb = nn.Embedding(num_metals, metal_emb_dim)
        self.node_proj = nn.Linear(node_dim + metal_emb_dim, hidden_dim)
        
        # ★ 新增：可擴展的門控機制，支援不同配體數量
        self.max_ligands = max_ligands
        self.gates = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), 
                nn.Tanh()
            ) for i in range(1, max_ligands + 1)
        })

        # ── TransformerConv 層 ──
        self.transformer = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            edge_dim=bond_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        
        # ── 分類器和回歸器 ──
        self.atom_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(256, 2)
        )
        self.atom_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )
        
        self.dim_reduction = nn.Identity()
        
        # ── pKa 標準化參數 ──
        self.register_buffer("pka_mu", torch.tensor(0.0))
        self.register_buffer("pka_sigma", torch.tensor(1.0))

        # ── ➊ pKa-bin 相關 ──────────────────────
        self.register_buffer("bin_edges", torch.tensor(bin_edges))
        self.register_buffer("bin_weights", torch.tensor(bin_weights))
        self.huber_beta = huber_beta
        self.reg_weight = reg_weight
        
        # ★ 溫和的輸出正則化參數
        self.soft_output_threshold = 25.0  # 軟閾值
        self.output_penalty_strength = 0.001  # 懲罰強度（很小）
        
        # ★ 分段平滑參數：針對分位數結果
        self.seg3_threshold = 8.8   # Seg3/Seg4 分界點
        self.seg4_smoothing = 0.002  # Seg4 額外平滑強度

    # ------------------------------------------------------------------
    def set_pka_normalization(self, mu: float, sigma: float):
        self.pka_mu.fill_(mu)
        self.pka_sigma.fill_(sigma if sigma > 0 else 1.0)

    # ------------------------------------------------------------------
    def forward(self, batch):  # noqa: C901, PLR0915
        device = batch.x.device
        
        # metal embedding broadcast to nodes
        m_emb = self.metal_emb(batch.metal_id)  # [G, E]
        m_emb_nodes = m_emb[batch.batch]        # [N, E]
        # 1. 取得金屬嵌入向量，去掉多餘的 1×1 維度 → [8]
        metal_vec = m_emb_nodes.squeeze(0).squeeze(0)      # 或 m_emb_nodes.view(-1)

        # 2. 複製到所有原子節點 → [11, 8]
        metal_mat = metal_vec.expand(batch.x.size(0), -1)  # 省記憶體；或用 repeat(11, 1)
        # 3. 與原子特徵串接 → [11, 66+8]
        x_cat = torch.cat([batch.x, metal_mat], dim=1)

        
        # x_cat = torch.cat([batch.x, m_emb_nodes], dim=1)
        h = self.node_proj(x_cat)
        # # metal embedding broadcast to nodes
        # m_emb = self.metal_emb(batch.metal_id)  # [G, E]
        # m_emb_nodes = m_emb[batch.batch]        # [N, E]
        # x_cat = torch.cat([batch.x, m_emb_nodes], dim=1)
        # h = self.node_proj(x_cat)

        h = self.transformer(h, batch.edge_index, edge_attr=batch.edge_attr)
        h_cur = self.dim_reduction(h)

        gt_mask = batch.pka_labels > 0
        idx_gt = torch.nonzero(gt_mask).squeeze(1)
        target = gt_mask.long()

        if idx_gt.numel() == 0:
            logits  = self.atom_classifier(h_cur)
            pka_raw = self.atom_regressor(h_cur).view(-1)
            loss_cla = nn.functional.cross_entropy(logits, target)
            loss_reg = torch.tensor(0.0, device=device)
            total    = loss_cla                     # 或 loss_cla + loss_reg

            dummy_tensor = torch.empty(0, device=device, dtype=pka_raw.dtype)
            target_final = target.unsqueeze(0)      # shape [1,N]，保持維度一致

            return logits, dummy_tensor, dummy_tensor, target_final, (total, loss_cla, loss_reg)


        idx_sorted = idx_gt[torch.argsort(batch.pka_labels[idx_gt])]
        idx_sorted = torch.cat([idx_sorted, torch.tensor([-1], device=device)])

        # ★ 新增：計算配體數量，用於選擇對應的門控機制
        num_ligands = len(idx_gt)
        num_ligands = min(num_ligands, self.max_ligands)  # 限制在最大配體數量內
        
        target_final = target.repeat(idx_sorted.shape[0] + 1, 1)
        for i in range(len(idx_sorted) - 1):
            target_final[i + 1][idx_sorted[: i + 1]] = 0

        logitss: List[torch.Tensor] = []
        pred_reg: List[torch.Tensor] = []
        loss_cla_steps: List[torch.Tensor] = []
        loss_reg_steps: List[torch.Tensor] = []

        for step_idx, idx in enumerate(idx_sorted):
            logits = self.atom_classifier(h_cur)
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
                # pred_pka = self.atom_regressor(h_cur).view(-1)[idx]
                # pred_norm = (pred_pka - self.pka_mu) / self.pka_sigma
                # true_norm = (batch.pka_labels[idx] - self.pka_mu) / self.pka_sigma
                # loss_r = nn.functional.smooth_l1_loss(pred_norm, true_norm, beta=self.huber_beta)
                
                pred_pka = self.atom_regressor(h_cur).view(-1)[idx]
                true_pka = batch.pka_labels[idx]
                
                # 標準化後計算 base loss（元素級，reduction='none' 以便乘權重）
                pred_norm = (pred_pka - self.pka_mu) / self.pka_sigma
                true_norm = (true_pka - self.pka_mu) / self.pka_sigma
                
                # ★ 分段自適應 Huber 損失：根據分位數結果優化
                if true_pka <= 2.9:        # Seg1: 表現好，保持
                    adaptive_beta = self.huber_beta * 1.0
                    range_weight = 1.0
                elif true_pka <= 5.2:      # Seg2: 表現好，保持
                    adaptive_beta = self.huber_beta * 1.0
                    range_weight = 1.0
                elif true_pka <= 8.8:      # Seg3: 中等，稍微放寬
                    adaptive_beta = self.huber_beta * 1.5
                    range_weight = 1.1  # 稍微增加權重
                else:                       # Seg4: 高值，大幅調整
                    adaptive_beta = self.huber_beta * 2.5
                    range_weight = 0.8  # 降低權重避免過度擬合
                
                diff = torch.abs(pred_norm - true_norm)
                base_loss = torch.where(
                    diff < adaptive_beta,
                    0.5 * diff.pow(2) / adaptive_beta,
                    diff - 0.5 * adaptive_beta
                )
                
                bin_idx = torch.bucketize(true_pka, self.bin_edges) - 1
                bin_weight = self.bin_weights[bin_idx]
                total_weight = bin_weight * range_weight
                
                loss_r = (total_weight * base_loss).mean()
                
                loss_reg_steps.append(loss_r)
                pred_reg.append(pred_pka)

            if idx != -1:
                h_upd = h_cur.clone()
                # ★ 使用基於配體數量的門控機制
                gate_key = str(min(step_idx + 1, self.max_ligands))
                current_gate = self.gates[gate_key]
                
                # 這邊降低gate的影響
                gate_coef = 1.0
                h_upd[idx] = h[idx] + gate_coef * h_cur[idx] * current_gate(h_cur[idx])
                h = self.transformer(h_upd, batch.edge_index, edge_attr=batch.edge_attr)
                h_cur = self.dim_reduction(h)

        loss_cla = torch.stack(loss_cla_steps).mean()
        loss_reg = torch.stack(loss_reg_steps).mean() if loss_reg_steps else torch.tensor(0.0, device=device)
        
        # ★ 分段輸出正則化：針對不同區域施加不同約束
        output_penalty = torch.tensor(0.0, device=device)
        if pred_reg:
            all_predictions = torch.stack(pred_reg)
            
            # 通用軟約束
            excess = torch.relu(torch.abs(all_predictions) - self.soft_output_threshold)
            general_penalty = self.output_penalty_strength * torch.mean(excess ** 2)
            
            # Seg4 額外平滑：對應高值目標的預測
            seg4_targets = []
            seg4_preds = []
            for pred_pka, idx in zip(pred_reg, idx_gt[:len(pred_reg)]):
                target_pka = batch.pka_labels[idx]
                if target_pka > self.seg3_threshold:  # Seg4 區域
                    seg4_targets.append(target_pka)
                    seg4_preds.append(pred_pka)
            
            seg4_penalty = torch.tensor(0.0, device=device)
            if seg4_preds and len(seg4_preds) > 1:
                seg4_preds_tensor = torch.stack(seg4_preds)
                # 對 Seg4 的預測施加額外的變異懲罰
                pred_variance = torch.var(seg4_preds_tensor)
                seg4_penalty = self.seg4_smoothing * pred_variance
            
            output_penalty = general_penalty + seg4_penalty
        
        loss_reg = self.reg_weight * loss_reg
        total = loss_cla + loss_reg + output_penalty

        return logitss, pred_reg, idx_sorted[:-1], target_final, (total, loss_cla, loss_reg)

    
    def sample(self, smiles: str, file_path: str, cfg: Dict, device=None):
        """
        對分子進行pKa預測的採樣方法
        
        Args:
            smiles: 分子SMILES字符串
            file_path: 包含數據的CSV文件路径
            cfg: 配置字典
            device: 計算設備
            
        Returns:
            dict: 包含預測結果的字典
        """
        if device is None:
            device = next(self.parameters()).device
        self.to(device).eval()
        
        df = pd.read_csv(file_path)
        row = df.loc[df['SMILES'] == smiles]
        
        metal2idx = {
            'Ag+': 0, 'Ca2+': 1, 'Cd2+': 2, 'Co2+': 3, 'Cu2+': 4,
            'Mg2+': 5, 'Mn2+': 6, 'Ni2+': 7, 'Pb2+': 8, 'Zn2+': 9
        }
        
        data, pka_values, _ = make_data_obj(row, metal2idx, cfg)
        batch = data.to(device)

        with torch.no_grad():
            _, pkas, pred_indices, _, _ = self(batch)
        
        all_preds, all_targets = [], []
        
        # 收集預測及對應 label
        for pred_pka, idx in zip(pkas, pred_indices):
            all_preds.append(pred_pka.item())
            all_targets.append(batch.pka_labels[idx].item())
        
        all_preds = np.array(all_preds, dtype=float)
        all_targets = np.array(all_targets, dtype=float)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        
        result = {
            "smiles": smiles,
            "true_pka": all_targets,
            "pred_pka": all_preds,
            "rmse": rmse,
            "num_ligands": len(pka_values),  # ★ 新增：配體數量信息
        }
        return result

################################################################################
# Train / Evaluate
################################################################################


class EarlyStopper:
    def __init__(self, patience=10):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, val):
        if val < self.best:
            self.best = val
            self.counter = 0
            return False  # not stop
        self.counter += 1
        return self.counter > self.patience


def evaluate(model, data_loader, epoch, device, *, is_train_data: bool = False):
    """
    評估模型效能。

    Parameters
    ----------
    model : torch.nn.Module
    data_loader : DataLoader  (train / val / test 都可)
    epoch : str | int         ⟹ 只用來顯示 tqdm 標題
    device : torch.device
    is_train_data : bool, default False
        - True  ➜  表示這批資料屬於「訓練集」
        - False ➜  視為驗證或測試集
    """
    model.eval()

    test_loss = test_cls_loss = test_reg_loss = 0.0
    all_preds, all_targets, all_is_train = [], [], []
    smiles_list, metal_ion_list = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'[Eval-{("Train" if is_train_data else "Test")} {epoch}]'):
            batch = batch.to(device, non_blocking=True)

            # 前向傳播
            _, pkas, pred_indices, _, (loss, cls_loss, reg_loss) = model(batch)

            # 累積損失
            test_loss     += loss.item()
            test_cls_loss += cls_loss.item()
            test_reg_loss += reg_loss.item()

            # 收集預測及對應 label
            for pred_pka, idx in zip(pkas, pred_indices):
                all_preds.append(pred_pka.item())
                all_targets.append(batch.pka_labels[idx].item())
                all_is_train.append(is_train_data)  
                
                mol_idx = batch.batch[idx]           # 將節點映射回分子 index
                smiles_list.append(batch.smiles[mol_idx])
                metal_ion_list.append(batch.metal_id[mol_idx].cpu().item())
            # smiles_list.extend(batch.smiles)
            # metal_ion_list.extend(batch.metal_id.cpu().numpy())
    # 平均化損失
    n_batch = len(data_loader)
    test_loss     /= n_batch
    test_cls_loss /= n_batch
    test_reg_loss /= n_batch

    # 統計指標
    all_preds   = np.array(all_preds,   dtype=float)
    all_targets = np.array(all_targets, dtype=float)
    mae  = mean_absolute_error(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

    return {
        "test_loss":  test_loss,
        "test_cls_loss": test_cls_loss,
        "test_reg_loss": test_reg_loss,
        "mae": mae,
        "rmse": rmse,
        "y_pred":   all_preds,
        "y_true":   all_targets,
        "is_train": np.array(all_is_train, dtype=bool),   # <─ 新增
        "smiles": smiles_list,
        "metal_ion": metal_ion_list
    }
    
def train(model, loaders, cfg, device, output_dir, df_loss, writer):
    train_loader, val_loader, test_loader = loaders
    scaler = torch.amp.GradScaler(init_scale=2**14)
    clip_val = 400.0           # 起手值
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], fused=True)
    
    
    # 2. Warm-up Scheduler
    # warmup_epochs = 3
    # from torch.optim.lr_scheduler import SequentialLR, LinearLR
    # main_sched = CosineAnnealingLR(opt, T_max=cfg['num_epochs']-warmup_epochs)
    # scheduler = SequentialLR(opt, schedulers=[LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs),
    #                                     main_sched],
    #                     milestones=[warmup_epochs])
    scheduler = CosineAnnealingLR(opt, T_max=cfg["num_epochs"])
    # scheduler = ExponentialLR(opt, gamma=cfg["anneal_rate"])
    stopper = EarlyStopper(patience=15)

    os.makedirs(output_dir, exist_ok=True)
    best_rmse = float("inf")
    # TensorBoard
    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()
        # ------------------- train -------------------
        model.train()
        tr_loss = cls_loss = reg_loss = 0.0
        grad_ratio_first = None  # 第 0 個 batch 的 grad norm 比（reg / cls）
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"[Train {epoch}]")):
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type):
                _, _, _, _, (loss, l_cls, l_reg) = model(batch)

            # --- 只在第一個 batch 估計梯度範圍，紀錄 grad_norm_reg / grad_norm_cls ---
            if batch_idx == 0:
                # 分別計算 classification / regression 部分的梯度 L2-norm
                grads_cls = torch.autograd.grad(
                    l_cls, model.parameters(), retain_graph=True, allow_unused=True)
                grads_reg = torch.autograd.grad(
                    l_reg, model.parameters(), retain_graph=True, allow_unused=True)

                def _norm(tensors):
                    return torch.sqrt(sum([g.detach().pow(2).sum() for g in tensors if g is not None]))

                norm_cls = _norm(grads_cls)
                norm_reg = _norm(grads_reg)
                if norm_cls.item() > 0:
                    grad_ratio_first = (norm_reg / (norm_cls + 1e-8)).item()

            scaler.scale(loss).backward()
            ## 先 unscale，再 clip
            scaler.unscale_(opt)
            if epoch <= 2:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_val)
            # ── 溢位防護 ───────────────────────────────────────────
            skip_batch = any(
                p.grad is not None and
                (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in model.parameters()
            )
            if skip_batch:
                # print("⚠️  skip batch: nan/inf in grads")
                opt.zero_grad()
                scaler.update()
                continue
            # ─────────────────────────────────────────────────────
            ## 更新參數與 scaler
            scaler.step(opt)
            scaler.update()
            tr_loss += loss.item(); cls_loss += l_cls.item(); reg_loss += l_reg.item()
            
            
        n_batches = len(train_loader)
        tr_loss /= n_batches; cls_loss /= n_batches; reg_loss /= n_batches
        # ------------------- val -------------------
        model.eval()
        v_loss = v_cls = v_reg = 0.0
        with torch.no_grad(), torch.amp.autocast(device_type=device.type):
            for batch in tqdm(val_loader, desc=f"[Val {epoch}]"):
                batch = batch.to(device, non_blocking=True)
                _, _, _, _, (loss, l_cls, l_reg) = model(batch)
                v_loss += loss.item(); v_cls += l_cls.item(); v_reg += l_reg.item()
        v_loss /= len(val_loader); v_cls /= len(val_loader); v_reg /= len(val_loader)
        metrics = evaluate(model, val_loader, epoch, device)
        metrics_te = evaluate(model, test_loader, epoch, device)
        rmse = metrics['rmse']
        rmse_te = metrics_te['rmse']
        scheduler.step()
        # --- 量化區段誤差 ---
        qs = [0.25, 0.5, 0.75]
        thr = np.quantile(metrics['y_true'], qs)
        bins = np.concatenate([[-np.inf], thr, [np.inf]])   # 五段
        digit = np.digitize(metrics['y_true'], bins) - 1    # 0~3 對應四段
        seg_rmses = []
        print(f"Val:")
        for i in range(4):
            idx = digit == i
            if idx.any():
                seg_rmse = np.sqrt(mean_squared_error(metrics['y_true'][idx],
                                                    metrics['y_pred'][idx]))
                seg_mae  = mean_absolute_error(metrics['y_true'][idx],
                                            metrics['y_pred'][idx])
                print(f"Seg{i+1} ({bins[i]:.1f}~{bins[i+1]:.1f}) | "
                    f"RMSE {seg_rmse:.3f}, MAE {seg_mae:.3f}, n={idx.sum()}")
                seg_rmses.append(seg_rmse)
        # 更新loss.csv
        df_loss.append({
            "epoch": epoch, "train_loss": tr_loss, "train_cla_loss": cls_loss,
            "train_reg_loss": reg_loss, "test_loss": v_loss,
            "test_cla_loss": v_cls, "test_reg_loss": v_reg,
            "rmse": rmse,
            "rmse_te": rmse_te,
            "learning_rate": scheduler.get_last_lr()[0],
            "ratio": reg_loss/cls_loss,
            "seg_rmses": seg_rmses
        })
        temp_df = pd.DataFrame(df_loss)
        output_path = os.path.join(output_dir, f"{cfg['version']}_loss.csv")
        temp_df.to_csv(output_path, index=False)
        print(f"loss已更新: {output_path}")
        # check early‑stop
        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            # print("  ↳ 保存新最佳模型")
        if stopper.step(v_loss):
            print("Early stopping triggered.")
            break
        # 更新學習曲線圖
        plot_learning_curves(
            csv_path=os.path.join(output_dir, f"{cfg['version']}_loss.csv"),
            output_dir=output_dir,
            version_name=cfg["version"],
            epoch_count=epoch
        )
        dt = time.time() - t0
        print(f"Epoch {epoch:02d} | train {tr_loss:.3f} (c{cls_loss:.2f}, r{reg_loss:.2f}) | "
              f"val {v_loss:.3f} (c{v_cls:.2f}, r{v_reg:.2f}), rmse: {rmse:.3f} | {dt:.1f}s")
        # # --- 量化區段誤差 ---
        # qs = [0.25, 0.5, 0.75]
        # thr = np.quantile(metrics['y_true'], qs)
        # bins = np.concatenate([[-np.inf], thr, [np.inf]])   # 五段
        # digit = np.digitize(metrics['y_true'], bins) - 1    # 0~3 對應四段
        # seg_rmses = []
        # print(f"Val:")
        # for i in range(4):
        #     idx = digit == i
        #     if idx.any():
        #         seg_rmse = np.sqrt(mean_squared_error(metrics['y_true'][idx],
        #                                             metrics['y_pred'][idx]))
        #         seg_mae  = mean_absolute_error(metrics['y_true'][idx],
        #                                     metrics['y_pred'][idx])
        #         print(f"Seg{i+1} ({bins[i]:.1f}~{bins[i+1]:.1f}) | "
        #             f"RMSE {seg_rmse:.3f}, MAE {seg_mae:.3f}, n={idx.sum()}")
        #         seg_rmses.append(seg_rmse)
        
        # # --- TensorBoard scalar logging ---
        writer.add_scalar('Loss/train', tr_loss, epoch)
        writer.add_scalar('Loss/val', v_loss, epoch)
        writer.add_scalar('Loss/train/cla', cls_loss, epoch)
        writer.add_scalar('Loss/train/reg', reg_loss, epoch)
        writer.add_scalar('Loss/val/cla', v_cls, epoch)
        writer.add_scalar('Loss/val/reg', v_reg, epoch)

        writer.add_scalar('RMSE/val', rmse, epoch)
        writer.add_scalar('RMSE/test', rmse_te, epoch)
        writer.add_scalar('RegRatio/train', reg_loss/cls_loss, epoch)
        if grad_ratio_first is not None:
            writer.add_scalar('GradNormRatio/train', grad_ratio_first, epoch)
        for i, srm in enumerate(seg_rmses):
            writer.add_scalar(f'RMSE/val_q{i+1}', srm, epoch)
        # for i, srm in enumerate(seg_rmses_te):
        writer.flush()
    print(f"Training finished, Best_rmse: {best_rmse:.3f}")
    # writer.close()
    return model



################################################################################
# Main
################################################################################

def main():
    cfg = parse_args()
    print("\n=== Config ===")
    for k, v in cfg.items():
        print(f"{k:18}: {v}")
    print("===============\n")
    # 讀取權重表
    bin_edges, bin_w = create_weight_tables(cfg["version"])
    
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(cfg["dataloader_path"]):
        with open(cfg["dataloader_path"], "rb") as f:
            loaded_data = pickle.load(f)
            if len(loaded_data) == 7:  # 新版本包含 all_pkas
                dl_train, dl_val, dl_test, mu, sigma, n_metals, _ = loaded_data
            else:  # 舊版本相容性
                dl_train, dl_val, dl_test, mu, sigma, n_metals = loaded_data
        print(f"已加載預處理的資料: {cfg['dataloader_path']}")
    else:
        print(f"未找到預處理資料，重新預處理資料: {cfg['dataloader_path']}")
        dl_train, dl_val, dl_test, mu, sigma, n_metals, _ = load_preprocessed_metal_data(
        cfg, cfg["metal_csv"], cfg["batch_size"])
    # 如果換資料集的話要把dataloader.pt刪掉，重新預處理資料

    # metal_weights_tensor = torch.ones(len(metal_w_dict), dtype=torch.float32)
    # cfg["num_features"] += 2
    model = CustomMetalPKA_GNN(n_metals, cfg["metal_emb_dim"],
                               cfg["num_features"],  # node_dim
                               9, # 先測試沒有spd的版本
                               cfg["hidden_size"],
                               cfg["output_size"],
                               cfg["dropout"],
                               cfg["depth"],
                               cfg["heads"],
                               bin_edges=bin_edges,
                               bin_weights=bin_w,
                               huber_beta=cfg["huber_beta"],
                               reg_weight=cfg["reg_weight"],
                               max_ligands=cfg.get("max_ligands", 3),  # ★ 新增可配置的最大配體數量
                               ).to(device)
    model.set_pka_normalization(mu, sigma)

    # 是否要載入以訓練模型繼續訓練
    # ckpt = "../output/metal_ver16/best_model.pt"
    # model.load_state_dict(torch.load(ckpt, weights_only=True))
    
    # model = torch.compile(model, mode="max-autotune")
    output_dir = cfg.get("output_dir", "./outputs")
    df_loss   = []
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs"))
    model = train(model, (dl_train, dl_val, dl_test), cfg, device, output_dir, df_loss, writer)
    
    # 1. 取得 train / test 兩批評估結果
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=True))
    model.eval()
    metrics_tr = evaluate(model, dl_train, "Final", device, is_train_data=True)
    metrics_te = evaluate(model, dl_test,  "Final", device, is_train_data=False)
    # metrics_val = evaluate(model, dl_val,  "Final", device, is_train_data=False)
    
    y_true_te  = np.concatenate([metrics_tr["y_true"],  metrics_te["y_true"]])
    y_pred_te  = np.concatenate([metrics_tr["y_pred"],  metrics_te["y_pred"]])
    is_train_te = np.concatenate([metrics_tr["is_train"], metrics_te["is_train"]])
    
    # y_true_val  = np.concatenate([metrics_tr["y_true"],  metrics_val["y_true"]])
    # y_pred_val  = np.concatenate([metrics_tr["y_pred"],  metrics_val["y_pred"]])
    # is_train_val = np.concatenate([metrics_tr["is_train"], metrics_val["is_train"]])
    
    # 輸出
    output_evaluation_results(metrics_tr, metrics_te, cfg["version"], cfg)
    
    parity_png_test = os.path.join(output_dir, f'{cfg["version"]}_parity_plot.png')
    parity_plot(y_true_te, y_pred_te, is_train_te, parity_png_test, title=f"Parity Plot - Test RMSE {metrics_te['rmse']:.3f}")
    
    # parity_png_val = os.path.join(output_dir, f'{cfg["version"]}_parity_plot_val.png')
    # parity_plot(y_true_val, y_pred_val, is_train_val, parity_png_val, title=f"Parity Plot - Val RMSE {metrics_val['rmse']:.3f}")
    
    # 上傳parity plot到tensorboard
    img_test = Image.open(parity_png_test).convert("RGB")
    # img_val = Image.open(parity_png_val).convert("RGB")
    img_tensor_test = transforms.ToTensor()(img_test)
    # img_tensor_val = transforms.ToTensor()(img_val)
    writer.add_image("ParityPlot/test", img_tensor_test, cfg["num_epochs"])  # step 記得給個 epoch 數
    # writer.add_image("ParityPlot/val", img_tensor_val, cfg["num_epochs"])  # step 記得給個 epoch 數
    writer.flush()
    writer.close()

if __name__ == "__main__":
    main()
