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

# 
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #  TensorFlow 
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')
warnings.filterwarnings('ignore', category=FutureWarning)

import argparse
import sys
import time
import traceback
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TransformerConv, GCNConv, GINEConv
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import torchvision.transforms as transforms

#  RDKit 
RDLogger.DisableLog('rdApp.*')
# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------

sys.path.append("../src/")
sys.path.append("./src/")
from metal_chemutils import tensorize_for_pka  # noqa: E402

from learning_curve import plot_learning_curves
from parity_plot import parity_plot 
from function import output_evaluation_results
from config import load_config_from_yaml, list_available_versions  # noqa: E402

################################################################################
# 
################################################################################

def parse_args():
    """
    
    
    Returns:
        dict: 
    """
    parser = argparse.ArgumentParser(
        description="Metal-ligand pKa transfer learning model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  python metal_pka_transfer.py --version metal_ver1
  python metal_pka_transfer.py --config custom_config.yaml --version metal_ver2
  python metal_pka_transfer.py --list-versions
        """
    )
    
    parser.add_argument(
        "--config", 
        dest='config_path', 
        default="../configs/config.yaml",
        help="Path to unified YAML configuration file"
    )
    
    parser.add_argument(
        "-v", "--version", 
        default="metal_ver1",
        help="Configuration version name"
    )
    
    parser.add_argument(
        "--list-versions", 
        action='store_true',
        help="List all available configuration versions"
    )
    
    args = parser.parse_args()
    
    # List available versions
    if args.list_versions:
        versions = list_available_versions(
            os.path.dirname(args.config_path), 
            os.path.basename(args.config_path)
        )
        print(f"Available versions: {', '.join(versions)}")
        exit(0)
    
    # Load configuration
    config_path = args.config_path
    version = args.version
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    print(f"Using unified YAML configuration: {config_path}")
    print(f"Version: {version}")
    
    # Load specified version configuration
    config_obj = load_config_from_yaml(config_path, version)
    return config_obj.to_dict()


def set_seed(seed: int):
    """
    Set random seed for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to: {seed}")

################################################################################
# 
################################################################################

def build_metal_vocab(df: pd.DataFrame) -> Dict[str, int]:
    """
    Build metal ion vocabulary mapping
    
    Args:
        df (pd.DataFrame): DataFrame containing metal ion information
        
    Returns:
        Dict[str, int]: Metal ion to index mapping dictionary
    """
    metals = sorted(df["metal_ion"].unique().tolist())
    metal_vocab = {metal: idx for idx, metal in enumerate(metals)}
    print(f"Metal ion types: {len(metals)} types")
    print(f"Metal ions: {', '.join(metals)}")
    return metal_vocab


def fallback_acidic_site_detection(mol):
    """
    Fallback method for acidic site detection
    
    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object
        
    Returns:
        List[int]: List of all oxygen and nitrogen atom indices
    """
    return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() in {"O", "N"}]


def validate_binding_sites(mol, pred_positions, min_neighbors=1):
    """
    改進結合位點預測：驗證預測位點的化學合理性
    
    Args:
        mol (rdkit.Chem.Mol): RDKit molecule object
        pred_positions (List[int]): Predicted binding atom positions
        min_neighbors (int): Minimum number of neighbors for valid binding site
        
    Returns:
        List[int]: Validated and potentially corrected binding positions
    """
    if mol is None:
        return pred_positions
    
    validated_positions = []
    
    for pos in pred_positions:
        if pos >= mol.GetNumAtoms():
            continue
            
        atom = mol.GetAtomWithIdx(pos)
        
        # 檢查原子類型 - 金屬配位常見的原子（O, N, S, P等）
        if atom.GetSymbol() not in {'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I'}:
            # 尋找附近的合適原子
            neighbors = [n.GetIdx() for n in atom.GetNeighbors() 
                        if n.GetSymbol() in {'O', 'N', 'S', 'P'}]
            if neighbors:
                validated_positions.append(neighbors[0])  # 取第一個鄰居
            else:
                validated_positions.append(pos)  # 保持原位置
        else:
            # 檢查鄰居數量 - 避免過度飽和的原子
            num_neighbors = len([n for n in atom.GetNeighbors()])
            if num_neighbors >= min_neighbors:
                validated_positions.append(pos)
            else:
                # 可能是孤立原子，尋找替代位置
                alternative_atoms = [i for i, a in enumerate(mol.GetAtoms()) 
                                   if a.GetSymbol() in {'O', 'N'} and 
                                   len([n for n in a.GetNeighbors()]) >= min_neighbors]
                if alternative_atoms:
                    validated_positions.append(alternative_atoms[0])
                else:
                    validated_positions.append(pos)  # 保持原位置
    
    return validated_positions if validated_positions else pred_positions

def create_weight_tables(version: str,
                         pka_bin_edges=None,
                         bin_width: float = 2.0,
                         gamma: float = 0.4):
    """
    Create pKa value weight tables to handle data imbalance
    
    Args:
        version (str): Model version name
        pka_bin_edges (np.ndarray, optional): pKa bin edges
        bin_width (float): Bin width for discretization
        gamma (float): Weight calculation parameter
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (bin edges, weight array)
    """
    csv_path = Path(f"../output/{version}/{version}_evaluation_results.csv")
    
    if not csv_path.exists():
        # Create default weight table if result file doesn't exist
        print(f"Result file not found {csv_path}, creating default weight table")
        if pka_bin_edges is None:
            # Use typical pKa range
            pka_bin_edges = np.arange(-2.0, 20.0, bin_width, dtype=np.float32)
            pka_bin_edges = np.append(pka_bin_edges, np.inf)
        
        # Create uniform weights
        pka_w = np.ones(len(pka_bin_edges) - 1, dtype=np.float32)
        
        # Save weight table
        out_dir = Path("../data")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / f"{version}_pka_weight_table.npz",
                 bin_edges=pka_bin_edges, weights=pka_w)
        print(f"Default weight table saved to: ../data/{version}_pka_weight_table.npz")
        return pka_bin_edges, pka_w
    
    # Calculate weights from existing results
    df = pd.read_csv(csv_path)
    df_train = df[df["is_train"] == True]
    
    if len(df_train) == 0:
        print("Training data is empty, using default weights")
        pka_bin_edges = np.arange(-2.0, 20.0, bin_width, dtype=np.float32)
        pka_bin_edges = np.append(pka_bin_edges, np.inf)
        pka_w = np.ones(len(pka_bin_edges) - 1, dtype=np.float32)
    else:
        # Calculate pKa distribution weights
        y_train = df_train["y_true"].values.astype(np.float32)
        if pka_bin_edges is None:
            lo, hi = np.floor(y_train.min()), np.ceil(y_train.max())
            pka_bin_edges = np.arange(lo, hi + bin_width, bin_width, dtype=np.float32)
            if pka_bin_edges[-1] <= hi:  # Ensure coverage of max value
                pka_bin_edges = np.append(pka_bin_edges, np.inf)
        
        hist, _ = np.histogram(y_train, bins=pka_bin_edges)
        freq = hist / hist.sum()
        
        # Calculate weights (emphasize low-frequency samples)
        pka_w = (1.0 / (freq + 1e-6)) ** gamma
        pka_w = pka_w / pka_w.mean()  # Normalize
        
        print(f"Weight calculation completed, bins: {len(pka_w)}")
    
    # Save weight table
    out_dir = Path("../data")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{version}_pka_weight_table.npz",
             bin_edges=pka_bin_edges, weights=pka_w.astype(np.float32))
    print(f"Weight table saved to: ../data/{version}_pka_weight_table.npz")
    
    return pka_bin_edges, pka_w


def make_data_obj(
    row: pd.Series,
    metal2idx: Dict[str, int],
) -> Tuple[Data, List[float], float]:
    """
    Convert a DataFrame row to PyTorch Geometric Data object with extensible ligand support.
    
    Args:
        row: DataFrame row containing SMILES, metal_ion, and pKa_value
        metal2idx: Dictionary mapping metal ion strings to indices
        
    Returns:
        tuple: (PyG Data object, list of pKa values, max pKa value)
        
    Supports multiple ligands (ML, ML₂, ML₃) by processing pKa_value lists.
    """

    from ast import literal_eval
    smiles     = row["SMILES"]
    metal_ion  = row["metal_ion"]
    pka_val_raw = row["pKa_value"]
    pred_pos = row["predicted_positions"]
    #  Series 
    if hasattr(smiles, 'iloc'):
        smiles = smiles.iloc[0] if len(smiles) > 0 else smiles
    if hasattr(metal_ion, 'iloc'):
        metal_ion = metal_ion.iloc[0] if len(metal_ion) > 0 else metal_ion
    if hasattr(pka_val_raw, 'iloc'):
        pka_val_raw = pka_val_raw.iloc[0] if len(pka_val_raw) > 0 else pka_val_raw
    
    #  pKa - 添加排序以解決置換問題
    if isinstance(pka_val_raw, str):
        pka_values = literal_eval(pka_val_raw)
    elif isinstance(pka_val_raw, (list, tuple)):
        pka_values = list(pka_val_raw)
    else:
        pka_values = [float(pka_val_raw)]
    
    # 對pKa值進行排序（從小到大）- 解決置換問題
    pka_values = sorted(pka_values)
    
    # pred_pos
    if isinstance(pred_pos, str):
        pred_pos = literal_eval(pred_pos)
    elif isinstance(pred_pos, (list, tuple)):
        pred_pos = list(pred_pos)
    else:
        pred_pos = [float(pred_pos)]
    
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    #   node / edge  
    fatoms, edge_index, edge_attr = tensorize_for_pka(smiles)
    num_atoms = fatoms.size(0)

    #  pKa label scalar=0
    
    pka_labels = torch.zeros(num_atoms)
    for i, pka in enumerate(pka_values):
        if i < len(pred_pos) and pred_pos[i] < num_atoms:
            pka_labels[pred_pos[i]] = float(pka)  # float
    
    # n_ligands
    n_ligands = len(pka_values)

    #   Data  
    data = Data(
        x=fatoms,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pred_pos=pred_pos,
        true_pka=pka_values,
        pka_labels=pka_labels,
        smiles=smiles,
        metal_id=torch.tensor([metal2idx[metal_ion]], dtype=torch.long),
        ligand_num=torch.tensor([n_ligands], dtype=torch.long),  # Add ligand count
    )
    
    max_pka = max(pka_values) if len(pka_values) else 0.0
    return data, pka_values, max_pka



from sklearn.model_selection import StratifiedShuffleSplit

def stratified_split(data_list, train_ratio=0.7, val_ratio=0.15, seed=42):
    y = [d.metal_id.item() for d in data_list]
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=train_ratio, random_state=seed)
    train_idx, temp_idx = next(sss.split(np.zeros(len(y)), y))

    #  temp  val / test
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

    data_list, all_pkas, max_pka_list = [], [], []   #  


    for _, row in tqdm(df.iterrows(), total=len(df), desc="CSV→Data"):
        try:
            d, pkas, max_pka = make_data_obj(row, metal2idx)
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
        print(f"{name} metal :", Counter(ids))

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
#   pseudo code
################################################################################

# class CustomMetalPKA_GNN(nn.Module):
#     def __init__(self, num_metals: int, metal_emb_dim: int,
#                  node_dim, bond_dim, hidden_dim, output_dim,
#                  dropout, depth, heads,
#                  *,  # ← 
#                  bin_edges, bin_weights,
#                  huber_beta: float,
#                  reg_weight: float,
#                  max_ligands: int = 3,  #  
#                  ):      
#         super().__init__()
        
        
#         num_metals = 10
#         metal_emb_dim = 8
#         node_dim = 66
#         bond_dim = 4
#         hidden_dim = 384
#         output_dim = 1
#         dropout = 0.0
#         depth = 4
#         heads = 8
#         max_ligands = 3
#         # #   
#         # self.node_dim = node_dim
#         # self.bond_dim = bond_dim
#         # self.hidden_dim = hidden_dim
#         # self.output_dim = output_dim
#         # self.dropout = dropout
#         # self.depth = depth
#         # self.heads = heads
#         # self.max_ligands = max_ligands
#         # # 
#         #   
#         metal_emb = nn.Embedding(num_metals, metal_emb_dim)
#         metal_proj = nn.Sequential(
#             nn.Linear(metal_emb_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         #   
#         ligand_proj = nn.Sequential(
#             nn.Linear(node_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         #   binding atoms  backbone atoms
#         gate_network = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.Tanh(),
#             nn.Linear(hidden_dim // 2, 1),
#             nn.Sigmoid()
#         )
        
#         #  TransformerConv -
#         transformer_conv = TransformerConv(
#             in_channels=hidden_dim,
#             out_channels=hidden_dim // heads,
#             edge_dim=bond_dim,
#             heads=heads,
#             concat=True,
#             dropout=dropout,
#         )
        
#         #   
#         predictor = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             # nn.ReLU(), 
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim // 2, output_dim)
#         )
        
#         #  pKa  
#         register_buffer("pka_mu", torch.tensor(0.0))
#         register_buffer("pka_sigma", torch.tensor(1.0))

#         #   pKa-bin  
#         register_buffer("bin_edges", torch.tensor(bin_edges))
#         register_buffer("bin_weights", torch.tensor(bin_weights))
#         huber_beta = huber_beta
#         reg_weight = reg_weight

#     def set_pka_normalization(self, mu: float, sigma: float):
#         self.pka_mu.fill_(mu)
#         self.pka_sigma.fill_(sigma if sigma > 0 else 1.0)

#     def forward(self, batch):
#         """
#         Forward pass with dynamic ligand count based on batch.ligand_num
#         """
#         batch 
#         device = batch.x.device
#         # Ensure we don't exceed model's max_ligands
#         max_ligands = batch.ligand_num
#         #  embedding metal and project to hidden_dim
#         metal_emb = metal_emb(batch.metal_id)  # [1, 8]
#         metal_emb.shape # [1, 8]
#         metal_proj_feature = metal_proj(metal_emb)  
#         metal_proj_feature.shape # [1, 384]
#         all_predictions = []
#         all_losses = []
        
#         #  - only predict up to actual ligand count
#         for eq_type_idx in range(max_ligands):
#             # 1.  h1_L 將fatoms 投影到 hidden_dim
#             h1_L = ligand_proj(batch.x)  # [num_atoms, hidden_dim]
#             h1_L.shape # [60, 384]
#             # 2. 
#             binding_atom_indices = batch.pred_pos
#             binding_atom_indices
#             # 3.  binding atoms (h_B)  backbone atoms (h_BB)
#             binding_mask = torch.zeros(h1_L.size(0), dtype=torch.bool, device='cpu')
#             binding_mask
#             binding_mask[binding_atom_indices] = True
#             binding_mask.shape # [60]
            
#             # 原子分組和特徵計算
#             h_B = h1_L[binding_mask]  # 
#             h_B
#             h_B.shape # [1, 384]
#             h_BB = h1_L[~binding_mask]  # 
#             h_BB
#             h_BB.shape # [59, 384]
#             if h_BB.size(0) == 0:  # 
#                 h_bar_BB = torch.zeros_like(h_B[0:1])
#             else:
#                 h_bar_BB = torch.mean(h_BB, dim=0, keepdim=True)  # [1, hidden_dim]
#             h_bar_BB.shape # [1, 384]
#             # 4. h2_L = h_B + gate(h̄_BB) · h_BB
#             if h_BB.size(0) > 0:
#                 gate_weights = gate_network(h_bar_BB)  # [1, 1]
#                 # 
#                 h2_L = h_B + gate_weights * h_bar_BB.expand(h_B.size(0), -1)
#             else:
#                 h2_L = h_B
#             h2_L.shape # [1, 384]
#             # 
#             if h2_L.size(0) > 0:
#                 h2_L_repr = torch.mean(h2_L, dim=0, keepdim=True)  # [1, hidden_dim]
#             else:
#                 continue
#             h2_L_repr.shape # [1, 384]
#             # 5. -h2_mL = cat([mpf, h2_L], dim=0)
#             h2_mL = torch.cat([metal_proj_feature, h2_L_repr], dim=0)  # [2, hidden_dim]
#             metal_proj_feature.shape # [1, 384]
#             h2_mL.shape # [2, 384]
#             # 6. -ei_mL = [[0,1],[1,0]]
#             ei_mL = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device='device').t()
#             ei_mL.shape # [2, 2]
#             # 7. 
#             eq_mL = torch.zeros(2, bond_dim, device=device)
#             eq_mL.shape # [2, 4]
#             # 8. TransformerConvh3_mL = transformer_conv(h2_mL, ei_mL, eq_mL)
#             h3_mL = transformer_conv(h2_mL.cpu(), ei_mL.cpu(), edge_attr=eq_mL.cpu())  # [2, hidden_dim]
#             h3_mL.shape # [2, 384]
#             # 9. V_mL = mean(h3_mL)
#             V_mL = torch.mean(h3_mL, dim=0, keepdim=True)  # [1, hidden_dim]
#             V_mL.shape # [1, 384]
#             # 10. pk = predictor(V_mL)
#             pka_pred = predictor(V_mL).squeeze()  # scalar
#             pka_pred.shape # []
#             all_predictions.append(pka_pred)
            
#             # 11. 
#             true_pka = batch.true_pka
#             true_pka.shape # [1]
#             # 
#             pred_norm = (pka_pred - pka_mu) / pka_sigma
#             true_norm = (true_pka - pka_mu) / pka_sigma
            
#             #  Huber 
#             loss = nn.functional.smooth_l1_loss(pred_norm, true_norm, beta=huber_beta)
#             all_losses.append(loss)
        
#         final_pred = torch.stack(all_predictions)
#         total_loss = torch.stack(all_losses).mean()
        
#         return final_pred, total_loss

class CustomMetalPKA_GNN(nn.Module):
    def __init__(self, num_metals: int, metal_emb_dim: int,
                 node_dim, bond_dim, hidden_dim, output_dim,
                 dropout, depth, heads,
                 *,  # ← 
                 bin_edges, bin_weights,
                 huber_beta: float,
                 reg_weight: float,
                 max_ligands: int = 3,  #  
                 ):      
        super().__init__()
        
        self.node_dim = node_dim
        self.bond_dim = bond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.depth = depth
        self.heads = heads
        self.max_ligands = max_ligands
        
        #   
        self.metal_emb = nn.Embedding(num_metals, metal_emb_dim)
        self.metal_proj = nn.Sequential(
            nn.Linear(metal_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 配體內部圖卷積層 - 先處理配體內部結構再進行特徵投射
        self.ligand_conv1 = GCNConv(node_dim, hidden_dim)
        self.ligand_conv2 = GCNConv(hidden_dim, hidden_dim)
        
        #   
        self.ligand_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        #   binding atoms  backbone atoms
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        #  TransformerConv -
        self.transformer_conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            edge_dim=bond_dim,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        
        #   
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            # nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        #  pKa  
        self.register_buffer("pka_mu", torch.tensor(0.0))
        self.register_buffer("pka_sigma", torch.tensor(1.0))

        #   pKa-bin  
        self.register_buffer("bin_edges", torch.tensor(bin_edges))
        self.register_buffer("bin_weights", torch.tensor(bin_weights))
        self.huber_beta = huber_beta
        self.reg_weight = reg_weight

    def set_pka_normalization(self, mu: float, sigma: float):
        self.pka_mu.fill_(mu)
        self.pka_sigma.fill_(sigma if sigma > 0 else 1.0)

    def forward(self, batch):
        """
        Forward pass with correct multi-ligand representation
        Implements explicit multi-ligand graph construction as per diagnosis
        """
        device = batch.x.device
        
        # Get actual ligand count from batch data
        if hasattr(batch, 'ligand_num') and batch.ligand_num.size(0) > 0:
            max_ligands = batch.ligand_num[0].item()
        else:
            # Fallback: use length of true_pka list
            if hasattr(batch, 'true_pka'):
                max_ligands = len(batch.true_pka) if isinstance(batch.true_pka, list) else 1
            else:
                max_ligands = (batch.pka_labels != 0).sum().item()
        
        # Ensure we don't exceed model's max_ligands
        max_ligands = min(max_ligands, self.max_ligands)
        
        # Metal embedding and projection
        metal_emb = self.metal_emb(batch.metal_id)  # [batch_size, metal_emb_dim]
        metal_proj_feature = self.metal_proj(metal_emb)  # [batch_size, hidden_dim]
        
        # 先對配體內部結構進行圖卷積，然後投射到隱層維度
        # Step 1: 配體內部圖卷積 - 讓每個原子融合鄰居原子資訊
        h_conv = F.relu(self.ligand_conv1(batch.x, batch.edge_index))
        h_conv = F.dropout(h_conv, p=0.1, training=self.training)
        h_conv = F.relu(self.ligand_conv2(h_conv, batch.edge_index))
        
        # Step 2: 特徵投射到隱層維度
        h1_L = self.ligand_proj(h_conv)  # [num_atoms, hidden_dim]
        
        # Use predefined binding positions from batch.pred_pos with validation
        if hasattr(batch, 'pred_pos') and batch.pred_pos.numel() > 0:
            raw_positions = batch.pred_pos.tolist()
            # Handle different tensor shapes
            if isinstance(raw_positions, int):
                raw_positions = [raw_positions]
            
            # 驗證結合位點的化學合理性
            if hasattr(batch, 'smiles') and len(batch.smiles) > 0:
                smiles = batch.smiles[0] if isinstance(batch.smiles, list) else batch.smiles
                mol = Chem.MolFromSmiles(smiles)
                binding_atom_indices = validate_binding_sites(mol, raw_positions)
            else:
                binding_atom_indices = raw_positions
        else:
            binding_atom_indices = []
        
        if not binding_atom_indices:
            # Fallback: use any non-zero pka_labels
            if hasattr(batch, 'pka_labels') and batch.pka_labels.numel() > 0:
                binding_atom_indices = torch.nonzero(batch.pka_labels, as_tuple=True)[0].tolist()
        
        if not binding_atom_indices:
            # Return dummy outputs if no binding atoms found
            print(f"Warning: No binding atoms found in batch")
            dummy_pred = self.predictor(torch.zeros(1, self.hidden_dim, device=device)).squeeze()
            dummy_loss = torch.zeros(1, device=device, requires_grad=True).squeeze()
            return dummy_pred, dummy_loss
        
        # 改進：為每個配體分別計算骨架特徵，而非全局平均
        # 假設每個配體包含相同數量的原子（對於同種配體情況）
        total_atoms = h1_L.size(0)
        max_possible_ligands = len(binding_atom_indices)
        
        # 如果有多個結合位點，嘗試推斷每個配體的原子範圍
        if max_possible_ligands > 1:
            atoms_per_ligand = total_atoms // max_possible_ligands
            ligand_backbone_features = []
            
            for i, binding_idx in enumerate(binding_atom_indices):
                # 計算第i個配體的原子範圍
                ligand_start = (binding_idx // atoms_per_ligand) * atoms_per_ligand
                ligand_end = ligand_start + atoms_per_ligand
                
                # 獲取該配體的所有原子，排除結合原子
                ligand_atoms = list(range(ligand_start, min(ligand_end, total_atoms)))
                if binding_idx in ligand_atoms:
                    ligand_atoms.remove(binding_idx)
                
                if ligand_atoms:
                    h_backbone_i = torch.mean(h1_L[ligand_atoms], dim=0, keepdim=True)
                else:
                    h_backbone_i = torch.zeros(1, self.hidden_dim, device=device)
                
                ligand_backbone_features.append(h_backbone_i)
        else:
            # 單配體情況：使用除結合原子外的所有原子
            non_binding_atoms = [i for i in range(total_atoms) if i not in binding_atom_indices]
            if non_binding_atoms:
                h_backbone = torch.mean(h1_L[non_binding_atoms], dim=0, keepdim=True)
            else:
                h_backbone = torch.zeros(1, self.hidden_dim, device=device)
            ligand_backbone_features = [h_backbone]
        
        all_predictions = []
        all_losses = []
        
        # Loop from full complex down to single ligand (correct order for stepwise dissociation)
        for n_ligands in range(max_ligands, 0, -1):
            # Create binding atom features for n ligands
            if len(binding_atom_indices) >= n_ligands:
                # Use first n binding atoms
                current_binding_indices = binding_atom_indices[:n_ligands]
                current_backbone_features = ligand_backbone_features[:n_ligands]
            else:
                # Duplicate binding atoms if we have fewer than needed (for identical ligands)
                current_binding_indices = []
                current_backbone_features = []
                for i in range(n_ligands):
                    idx = binding_atom_indices[i % len(binding_atom_indices)]
                    backbone_feat = ligand_backbone_features[i % len(ligand_backbone_features)]
                    current_binding_indices.append(idx)
                    current_backbone_features.append(backbone_feat)
            
            # Extract binding atom features
            h_B = h1_L[current_binding_indices]  # [n_ligands, hidden_dim]
            
            # 為每個配體分別應用gating機制
            h2_L_list = []
            for i, (h_binding, h_backbone) in enumerate(zip(h_B, current_backbone_features)):
                gate_weight = self.gate_network(h_backbone)  # [1, 1] 
                h2_ligand = h_binding + gate_weight.squeeze() * h_backbone.squeeze()  # [hidden_dim]
                h2_L_list.append(h2_ligand.unsqueeze(0))  # [1, hidden_dim]
            
            h2_L = torch.cat(h2_L_list, dim=0)  # [n_ligands, hidden_dim]
            
            # Create multi-ligand graph: metal + n ligand nodes
            # Metal node feature (use first batch element)
            if hasattr(batch, 'batch') and batch.batch is not None:
                metal_node = metal_proj_feature[batch.batch[0:1]]  # [1, hidden_dim]
            else:
                metal_node = metal_proj_feature[0:1]  # [1, hidden_dim]
            
            # Concatenate metal and ligand nodes
            h2_mL = torch.cat([metal_node, h2_L], dim=0)  # [1+n_ligands, hidden_dim]
            
            # Create edge index for star graph: metal (node 0) connected to all ligand nodes
            edge_list = []
            for ligand_idx in range(1, n_ligands + 1):
                # Metal to ligand and ligand to metal
                edge_list.extend([[0, ligand_idx], [ligand_idx, 0]])
            
            ei_mL = torch.tensor(edge_list, dtype=torch.long, device=device).t()  # [2, 2*n_ligands]
            
            # Create edge attributes (zeros as in original implementation)
            eq_mL = torch.zeros(ei_mL.size(1), self.bond_dim, device=device)  # [2*n_ligands, bond_dim]
            
            # Apply TransformerConv on multi-ligand graph
            h3_mL = self.transformer_conv(h2_mL, ei_mL, edge_attr=eq_mL)  # [1+n_ligands, hidden_dim]
            
            # Graph pooling: average all nodes (metal + all ligands)
            V_mL = torch.mean(h3_mL, dim=0, keepdim=True)  # [1, hidden_dim]
            
            # Predict pKa for this coordination state
            pka_pred = self.predictor(V_mL).squeeze()  # scalar
            all_predictions.append(pka_pred)
            
            # Calculate loss if true pKa is available
            target_pka = None
            
            # 改進target pKa匹配邏輯 - 根據優化建議正確配對
            if hasattr(batch, 'true_pka') and batch.true_pka is not None:
                if isinstance(batch.true_pka, list):
                    # 確保true_pka已排序（應該在make_data_obj中完成）
                    sorted_true_pka = sorted(batch.true_pka)
                    
                    # n_ligands從max降到1，對應排序後的pKa索引從0到max-1
                    target_idx = max_ligands - n_ligands
                    if target_idx < len(sorted_true_pka):
                        target_pka = sorted_true_pka[target_idx]
                    else:
                        # 如果索引超出範圍，跳過這個預測（不參與訓練）
                        target_pka = None
                else:
                    # 單值情況：只對第一個配體數量（最大配體數）計算loss
                    if n_ligands == max_ligands:
                        target_pka = batch.true_pka
                    else:
                        target_pka = None  # 其他配體數量不參與訓練
            
            # Fallback to pka_labels if true_pka not available
            if target_pka is None and hasattr(batch, 'pka_labels'):
                # Use first non-zero pka_label as target
                nonzero_labels = batch.pka_labels[batch.pka_labels != 0]
                if len(nonzero_labels) > 0:
                    target_pka = nonzero_labels[0].item()
            
            if target_pka is not None:
                # Normalize predictions and targets
                pred_norm = (pka_pred - self.pka_mu) / self.pka_sigma
                true_norm = (target_pka - self.pka_mu) / self.pka_sigma
                
                # Compute Huber loss
                if isinstance(true_norm, torch.Tensor):
                    true_norm_tensor = true_norm.clone().detach().to(device)
                else:
                    true_norm_tensor = torch.tensor(true_norm, device=device, dtype=torch.float32)
                
                loss = nn.functional.smooth_l1_loss(
                    pred_norm, 
                    true_norm_tensor, 
                    beta=self.huber_beta
                )
                all_losses.append(loss)
        
        # Return final predictions and loss - 確保所有預測都參與損失計算
        if all_predictions:
            # 預測值需要排序以對應排序後的true_pka
            final_pred = torch.stack(all_predictions)
            
            if all_losses:
                # 所有有效的預測都應該參與損失計算
                total_loss = torch.stack(all_losses).mean()
            else:
                # 如果沒有損失，創建一個小的正則化損失以防止梯度消失
                reg_loss = sum(p.pow(2.0).sum() for p in self.parameters()) * 1e-6
                total_loss = reg_loss
            return final_pred, total_loss
        else:
            # Fallback for edge cases
            dummy_pred = self.predictor(torch.zeros(1, self.hidden_dim, device=device)).squeeze()
            dummy_loss = torch.zeros(1, device=device, requires_grad=True).squeeze()
            return dummy_pred, dummy_loss
    
    def sample(self, smiles: str, metal_ion: str, pred_positions: List[int], 
               device=None, return_sorted=True):
        """
        改進的多配體pKa預測採樣方法
        
        Args:
            smiles: 分子SMILES字符串
            metal_ion: 金屬離子標識
            pred_positions: 預測的結合位點列表
            device: 計算設備
            return_sorted: 是否返回排序的pKa值
            
        Returns:
            dict: 包含預測結果的字典
        """
        if device is None:
            device = next(self.parameters()).device
        
        self.to(device).eval()
        
        # 金屬映射
        metal2idx = {
            'Ag+': 0, 'Ca2+': 1, 'Cd2+': 2, 'Co2+': 3, 'Cu2+': 4,
            'Mg2+': 5, 'Mn2+': 6, 'Ni2+': 7, 'Pb2+': 8, 'Zn2+': 9
        }
        
        if metal_ion not in metal2idx:
            raise ValueError(f"Unsupported metal ion: {metal_ion}")
        
        try:
            # 分子特徵化
            from metal_chemutils import tensorize_for_pka
            fatoms, edge_index, edge_attr = tensorize_for_pka(smiles)
            
            # 驗證結合位點
            mol = Chem.MolFromSmiles(smiles)
            validated_positions = validate_binding_sites(mol, pred_positions) if mol else pred_positions
            
            # 創建數據對象
            data = Data(
                x=fatoms,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pred_pos=torch.tensor(validated_positions, dtype=torch.long),
                metal_id=torch.tensor([metal2idx[metal_ion]], dtype=torch.long),
                ligand_num=torch.tensor([len(validated_positions)], dtype=torch.long),
                smiles=smiles
            )
            
            # 移動到設備並預測
            data = data.to(device)
            
            with torch.no_grad():
                predictions, _ = self(data)
            
            # 處理預測結果
            if torch.is_tensor(predictions):
                if predictions.dim() == 0:
                    pred_values = [predictions.cpu().item()]
                else:
                    pred_values = predictions.cpu().numpy().flatten().tolist()
            else:
                pred_values = [predictions] if not isinstance(predictions, list) else predictions
            
            # 根據需要排序預測值
            if return_sorted:
                pred_values = sorted(pred_values)
            
            # 生成配位狀態標籤
            coordination_states = []
            for i, pka_val in enumerate(pred_values):
                n_ligands = len(pred_values) - i
                if n_ligands == 1:
                    coordination_states.append("ML")
                else:
                    coordination_states.append(f"ML{n_ligands}")
            
            result = {
                "smiles": smiles,
                "metal_ion": metal_ion,
                "binding_sites": validated_positions,
                "num_ligands": len(validated_positions),
                "predictions": {
                    state: pka for state, pka in zip(coordination_states, pred_values)
                },
                "pka_values": pred_values,
                "sorted": return_sorted,
                "coordination_states": coordination_states
            }
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error during prediction: {e}")

################################################################################
# Train / Evaluate ()
################################################################################

class EarlyStopper:
    """ train_reproduce_tb.py"""
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_model_state = None
        
    def step(self, val_loss, model, epoch=None):
        """"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # 
            self.best_model_state = model.state_dict().copy()
            print(f"  New best validation loss: {val_loss:.6f}")
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
            
    def load_best_model(self, model):
        """"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            print(f"  Best model loaded (loss: {self.best_loss:.6f})")
            
    def get_status(self, epoch):
        """"""
        return {
            'counter': self.counter,
            'best_loss': self.best_loss,
            'patience': self.patience
        }

#  
def evaluate(model, data_loader, epoch, device, *, is_train_data: bool = False):
    """
     - 
    """
    model.eval()
    
    test_loss = 0.0
    all_preds, all_targets = [], []
    all_smiles, all_metal_ions = [], []
    successful_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f'[Eval-{("Train" if is_train_data else "Test")} {epoch}]'):
            batch = batch.to(device, non_blocking=True)
            
            try:
                preds, loss = model(batch)
                test_loss += loss.item()
                successful_batches += 1
                
                # Handle multi-ligand predictions correctly
                if torch.is_tensor(preds) and preds.numel() > 0:
                    # 根據優化建議改進評估時預測值與目標值的配對邏輯
                    if hasattr(batch, 'true_pka') and batch.true_pka is not None:
                        true_pka_values = batch.true_pka if isinstance(batch.true_pka, list) else [batch.true_pka]
                        
                        # 確保true_pka_values已排序
                        if isinstance(batch.true_pka, list):
                            true_pka_values = sorted(true_pka_values)
                        
                        # Convert predictions to list (model outputs multiple ligand state predictions)
                        if preds.dim() == 0:
                            pred_values = [preds.cpu().item()]
                        else:
                            pred_values = preds.cpu().numpy().flatten().tolist()
                        
                        # 預測值也需要排序以對應排序後的目標值
                        pred_values = sorted(pred_values)
                        
                        # 對於多配體預測，配對所有可用的預測和目標（都已排序）
                        if isinstance(batch.true_pka, list) and len(true_pka_values) > 1:
                            # 多階常數情況：配對所有可用的預測和目標
                            min_len = min(len(pred_values), len(true_pka_values))
                            for i in range(min_len):
                                all_preds.append(pred_values[i])
                                all_targets.append(true_pka_values[i])
                                all_smiles.append(batch.smiles[0] if hasattr(batch, 'smiles') and len(batch.smiles) > 0 else "")
                                all_metal_ions.append(batch.metal_id[0].item() if hasattr(batch, 'metal_id') and batch.metal_id.numel() > 0 else 0)
                        else:
                            # 單值情況：只取第一個預測值（對應最大配體數）
                            if len(pred_values) > 0:
                                all_preds.append(pred_values[0])  # 取第一個預測（max_ligands對應）
                                all_targets.append(true_pka_values[0] if isinstance(true_pka_values[0], (int, float)) else float(true_pka_values[0]))
                                all_smiles.append(batch.smiles[0] if hasattr(batch, 'smiles') and len(batch.smiles) > 0 else "")
                                all_metal_ions.append(batch.metal_id[0].item() if hasattr(batch, 'metal_id') and batch.metal_id.numel() > 0 else 0)
                    
                    else:
                        # Fallback to pka_labels method
                        valid_labels = batch.pka_labels[batch.pka_labels != 0]
                        
                        if len(valid_labels) > 0:
                            if preds.dim() == 0:
                                pred_value = preds.cpu().item()
                                all_preds.append(pred_value)
                                all_targets.append(valid_labels[0].cpu().item())
                            else:
                                pred_values = preds.cpu().numpy().flatten()
                                if len(pred_values) > 0:
                                    # Take first prediction for single target case
                                    all_preds.append(pred_values[0])
                                    all_targets.append(valid_labels[0].cpu().item())
                            
                            # Add SMILES and metal info for fallback case too
                            if hasattr(batch, 'smiles') and len(batch.smiles) > 0:
                                smiles_str = batch.smiles[0] if isinstance(batch.smiles[0], str) else str(batch.smiles[0])
                                all_smiles.append(smiles_str)
                            else:
                                all_smiles.append("")
                            
                            if hasattr(batch, 'metal_id') and batch.metal_id.numel() > 0:
                                all_metal_ions.append(batch.metal_id[0].item())
                            else:
                                all_metal_ions.append(0)
                    
            except Exception as e:
                print(f"Error during evaluation: {e}")
                continue
    
    # Calculate metrics
    min_len = min(len(all_preds), len(all_targets))
    if min_len > 0:
        all_preds = np.array(all_preds[:min_len])
        all_targets = np.array(all_targets[:min_len])
        
        
        mae = mean_absolute_error(all_targets, all_preds)
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    else:
        mae = rmse = float('inf')
    
    # 
    avg_loss = test_loss / successful_batches if successful_batches > 0 else float('inf')
    
    return {
        "test_loss": avg_loss,
        "test_cls_loss": 0.0,  # 
        "test_reg_loss": avg_loss,
        "mae": mae,
        "rmse": rmse,
        "y_pred": all_preds,
        "y_true": all_targets,
        "is_train": np.array([is_train_data] * len(all_preds)),
        "smiles": all_smiles,
        "metal_ion": all_metal_ions, 
        "successful_batches": successful_batches,
        "total_batches": len(data_loader)
    }

#   TensorBoard 
def train(model, loaders, cfg, device, output_dir, df_loss, writer):
    train_loader, val_loader, test_loader = loaders
    
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    # 修正學習率調度：避免過早衰減至零，使用更平緩的衰減策略
    scheduler = CosineAnnealingLR(opt, T_max=cfg["num_epochs"], eta_min=cfg["lr"] * 0.01)
    
    # 增加早停耐心度，讓模型有更充分時間收斂
    early_stopping = EarlyStopper(patience=25, min_delta=0.001)

    os.makedirs(output_dir, exist_ok=True)
    best_rmse = float("inf")
    
    # Record training start information
    print(f"Starting training - Goal: optimize metal pKa prediction")
    print(f"Training configuration:")
    print(f"  - Learning rate: {cfg['lr']}")
    print(f"  - Weight Decay: {cfg['weight_decay']}")
    print(f"  - Batch size: {cfg['batch_size']}")
    print(f"  - Max epochs: {cfg['num_epochs']}")
    print(f"  - Early stopping patience: {early_stopping.patience}")
    print(f"  - Device: {device}")
    print("-"*150)
    
    for epoch in range(1, cfg["num_epochs"] + 1):
        t0 = time.time()
        
        # ------------------- train -------------------
        model.train()
        tr_loss = 0.0
        successful_batches = 0
        total_norm = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"[Train {epoch}]")):
            batch = batch.to(device, non_blocking=True)
            opt.zero_grad()
            
            try:
                preds, loss = model(batch)
                
                # Debug: Check if loss and predictions are valid
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"   Batch {batch_idx}: Invalid loss {loss.item()}")
                    continue
                
                if loss.item() > 0:
                    loss.backward()
                    
                    # Gradient clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    total_norm += grad_norm.item()
                    
                    opt.step()
                    tr_loss += loss.item()
                    successful_batches += 1
                    
                    # TensorBoard logging with enhanced monitoring
                    global_step = (epoch - 1) * len(train_loader) + batch_idx
                    if batch_idx % 10 == 0:
                        writer.add_scalar('Loss/train_batch', loss.item(), global_step)
                        writer.add_scalar('Gradients/norm', grad_norm.item(), global_step)
                        
                        # 增加詳細的參數和梯度監控
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                # 參數統計
                                param_mean = param.data.mean().item()
                                param_std = param.data.std().item() if param.data.numel() > 1 else 0.0
                                param_max = param.data.max().item()
                                param_min = param.data.min().item()
                                
                                # 梯度統計
                                grad_mean = param.grad.data.mean().item()
                                grad_std = param.grad.data.std().item() if param.grad.data.numel() > 1 else 0.0
                                grad_max = param.grad.data.max().item()
                                grad_min = param.grad.data.min().item()
                                
                                # 記錄關鍵層的詳細統計
                                if any(key in name for key in ['metal_emb', 'ligand_conv', 'transformer_conv', 'predictor']):
                                    writer.add_scalar(f'Parameters_Detail/{name}_mean', param_mean, global_step)
                                    writer.add_scalar(f'Parameters_Detail/{name}_std', param_std, global_step)
                                    writer.add_scalar(f'Parameters_Detail/{name}_range', param_max - param_min, global_step)
                                    
                                    writer.add_scalar(f'Gradients_Detail/{name}_mean', grad_mean, global_step)
                                    writer.add_scalar(f'Gradients_Detail/{name}_std', grad_std, global_step)
                                    writer.add_scalar(f'Gradients_Detail/{name}_range', grad_max - grad_min, global_step)
                else:
                    if batch_idx % 20 == 0:  # Print occasional zero loss warnings
                        print(f"   Batch {batch_idx}: Zero loss {loss.item():.6f}")
                        print(f"     Ligand count: {batch.ligand_num.item() if hasattr(batch, 'ligand_num') else 'N/A'}")
                        print(f"     Predictions shape: {preds.shape if torch.is_tensor(preds) else type(preds)}")
                        if hasattr(batch, 'true_pka'):
                            print(f"     True pKa: {batch.true_pka}")
                
            except Exception as e:
                print(f"Batch {batch_idx} training error: {e}")
                print(f"   Batch info: SMILES count={len(batch.smiles) if hasattr(batch, 'smiles') else 'N/A'}, atoms={batch.x.size(0) if hasattr(batch, 'x') else 'N/A'}")
                if hasattr(batch, 'ligand_num'):
                    print(f"   Ligand count: {batch.ligand_num}")
                if hasattr(batch, 'pred_pos'):
                    print(f"   Pred positions: {batch.pred_pos}")
                import traceback
                traceback.print_exc()
                continue
        
        # 
        avg_train_loss = tr_loss / successful_batches if successful_batches > 0 else float('inf')
        avg_grad_norm = total_norm / successful_batches if successful_batches > 0 else 0.0
        
        # ------------------- validation -------------------
        val_metrics = evaluate(model, val_loader, epoch, device)
        
        val_loss = val_metrics['test_loss']
        val_rmse = val_metrics['rmse']
        
        # 
        scheduler.step()
        current_lr = opt.param_groups[0]['lr']
        
        # 
        early_stop = early_stopping.step(val_loss, model, epoch)
        status = early_stopping.get_status(epoch)
        
        #  RMSE
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
        
        #   TensorBoard 
        # 
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('RMSE/validation', val_rmse, epoch)
        writer.add_scalar('MAE/validation', val_metrics['mae'], epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # 
        writer.add_scalar('EarlyStopping/counter', status['counter'], epoch)
        writer.add_scalar('EarlyStopping/best_loss', status['best_loss'], epoch)
        
        
        # 
        df_loss.append({
            "epoch": epoch, 
            "train_loss": avg_train_loss, 
            "val_loss": val_loss,
            "rmse": val_rmse,
            "mae": val_metrics['mae'],
            "learning_rate": current_lr,
            "grad_norm": avg_grad_norm,
            "successful_batches": successful_batches,
            "early_stop_counter": status['counter']
        })
        
        #  CSV
        temp_df = pd.DataFrame(df_loss)
        output_path = os.path.join(output_dir, f"{cfg['version']}_loss.csv")
        temp_df.to_csv(output_path, index=False)
        
        # Training progress output
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{cfg['num_epochs']} | Training time: {dt:.1f}s")
        print(f"  Loss: Train {avg_train_loss:.4f} | Val {val_loss:.4f}")
        print(f"  RMSE: Val {val_rmse:.4f} {'↓' if val_rmse < best_rmse else '↑'}")
        print(f"  Learning rate: {current_lr:.8f} | Grad norm: {avg_grad_norm:.4f}")
        print(f"  Successful batches: {successful_batches}/{len(train_loader)} ({100*successful_batches/len(train_loader):.1f}%)")
        print(f"  Early stop counter: {status['counter']}/{status['patience']}")
        print("-"*150)
        
        #  TensorBoard
        writer.flush()
        # 
        if early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            print(f"   Best validation loss: {early_stopping.best_loss:.6f}")
            early_stopping.load_best_model(model)
            break
    
    print(f"Training finished!")
    print(f"   Best RMSE: {best_rmse:.4f}")
    print(f"   Best validation loss: {early_stopping.best_loss:.6f}")
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
    
    # 
    bin_edges, bin_w = create_weight_tables(cfg["version"])
    
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    if os.path.exists(cfg["dataloader_path"]):
        with open(cfg["dataloader_path"], "rb") as f:
            loaded_data = pickle.load(f)
            if len(loaded_data) == 7:
                dl_train, dl_val, dl_test, mu, sigma, n_metals, _ = loaded_data
            else:
                dl_train, dl_val, dl_test, mu, sigma, n_metals = loaded_data
        print(f"Loaded preprocessed data: {cfg['dataloader_path']}")
    else:
        print(f"Preprocessed data not found, reprocessing: {cfg['dataloader_path']}")
        dl_train, dl_val, dl_test, mu, sigma, n_metals, _ = load_preprocessed_metal_data(
            cfg, cfg["metal_csv"], cfg["batch_size"])
    
    # Print dataset information
    print(f"Dataset information:")
    print(f"  - Training set: {len(dl_train.dataset)} samples")
    print(f"  - Validation set: {len(dl_val.dataset)} samples") 
    print(f"  - Test set: {len(dl_test.dataset)} samples")
    print(f"  - Number of metal ions: {n_metals}")
    print(f"  - pKa normalization: μ={mu:.2f}, σ={sigma:.2f}")
    
    #  
    model = CustomMetalPKA_GNN(
        n_metals, cfg["metal_emb_dim"],
        cfg["num_features"],  # node_dim
        9,  # bond_dim
        cfg["hidden_size"],
        cfg["output_size"],
        cfg["dropout"],
        cfg["depth"],
        cfg["heads"],
        bin_edges=bin_edges,
        bin_weights=bin_w,
        huber_beta=cfg["huber_beta"],
        reg_weight=cfg["reg_weight"],
        max_ligands=cfg.get("max_ligands", 3),
    ).to(device)
    
    model.set_pka_normalization(mu, sigma)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model information:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Model size: {total_params * 4 / (1024**2):.2f} MB")
    
    #   TensorBoard 
    output_dir = cfg.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    runs_dir = os.path.join(output_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    log_dir = os.path.join(runs_dir, f"{cfg['version']}_{timestamp}")
    writer = SummaryWriter(log_dir=log_dir)
    
    print(f"TensorBoard setup:")
    print(f"  - Log directory: {log_dir}")
    print(f"  - Launch command: tensorboard --logdir={runs_dir}")
    
    #  TensorBoard
    config_text = "\n".join([f"{k}: {v}" for k, v in cfg.items()])
    writer.add_text("Config/Hyperparameters", config_text, 0)
    
    # 
    model_summary = f"""
    : CustomMetalPKA_GNN
    - : {n_metals}
    - : {cfg['metal_emb_dim']}
    - : {cfg['num_features']}
    - : {cfg['hidden_size']}
    - Transformer : {cfg['depth']}
    - : {cfg['heads']}
    - : {cfg.get('max_ligands', 3)}
    - : {total_params:,}
    """
    writer.add_text("Model/Architecture", model_summary, 0)
    
    # 
    dataset_info = f"""
    :
    - : {len(dl_train.dataset)} 
    - : {len(dl_val.dataset)}   
    - : {len(dl_test.dataset)} 
    - pKa : μ={mu:.4f}, σ={sigma:.4f}
    - : {cfg['batch_size']}
    """
    writer.add_text("Dataset/Info", dataset_info, 0)
    
    df_loss = []
    
    # Training
    print(f"\nStarting training...")
    model = train(model, (dl_train, dl_val, dl_test), cfg, device, output_dir, df_loss, writer)
    
    # Test best model
    print(f"\nLoading best model for final testing...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=True))
    model.eval()
    
    # 
    final_train_metrics = evaluate(model, dl_train, "Final", device, is_train_data=True)
    final_test_metrics = evaluate(model, dl_test, "Final", device, is_train_data=False)
    
    #  TensorBoard
    writer.add_scalar('Final/Train_RMSE', final_train_metrics['rmse'], 0)
    writer.add_scalar('Final/Test_RMSE', final_test_metrics['rmse'], 0)
    writer.add_scalar('Final/Train_MAE', final_train_metrics['mae'], 0)
    writer.add_scalar('Final/Test_MAE', final_test_metrics['mae'], 0)
    
    # 
    final_results = f"""
    :
    -  RMSE: {final_test_metrics['rmse']:.4f}
    -  MAE: {final_test_metrics['mae']:.4f}
    -  RMSE: {final_train_metrics['rmse']:.4f}
    -  MAE: {final_train_metrics['mae']:.4f}
    - : {final_test_metrics['successful_batches']}/{final_test_metrics['total_batches']}
    """
    writer.add_text("Results/Final", final_results, 0)
    
    print(f"Final test results:")
    print(f"   Test RMSE: {final_test_metrics['rmse']:.4f}")
    print(f"   Test MAE: {final_test_metrics['mae']:.4f}")
    print(f"   Train RMSE: {final_train_metrics['rmse']:.4f}")  
    print(f"   Train MAE: {final_train_metrics['mae']:.4f}")
    
    # 
    try:
        # 
        output_evaluation_results(final_train_metrics, final_test_metrics, cfg["version"], cfg)
        
        #  Parity Plot - 
        train_true = np.array(final_train_metrics["y_true"])
        train_pred = np.array(final_train_metrics["y_pred"])
        train_is_train = np.array(final_train_metrics["is_train"])
        
        test_true = np.array(final_test_metrics["y_true"])
        test_pred = np.array(final_test_metrics["y_pred"])
        test_is_train = np.array(final_test_metrics["is_train"])
        
        # 
        print(f"   Training data length: true={len(train_true)}, pred={len(train_pred)}, is_train={len(train_is_train)}")
        print(f"   Test data length: true={len(test_true)}, pred={len(test_pred)}, is_train={len(test_is_train)}")
        
        # 
        train_min_len = min(len(train_true), len(train_pred), len(train_is_train))
        test_min_len = min(len(test_true), len(test_pred), len(test_is_train))
        
        if train_min_len > 0 and test_min_len > 0:
            y_true_combined = np.concatenate([train_true[:train_min_len], test_true[:test_min_len]])
            y_pred_combined = np.concatenate([train_pred[:train_min_len], test_pred[:test_min_len]])
            is_train_combined = np.concatenate([train_is_train[:train_min_len], test_is_train[:test_min_len]])
        else:
            raise ValueError(f": train_min_len={train_min_len}, test_min_len={test_min_len}")
        
        parity_png_path = os.path.join(output_dir, f'{cfg["version"]}_parity_plot.png')
        parity_plot(y_true_combined, y_pred_combined, is_train_combined, 
                   parity_png_path, title=f"Parity Plot - Test RMSE {final_test_metrics['rmse']:.4f}")
        
        #  Parity Plot  TensorBoard
        if os.path.exists(parity_png_path):
            img = Image.open(parity_png_path).convert("RGB")
            img_tensor = transforms.ToTensor()(img)
            writer.add_image("Plots/Parity_Plot", img_tensor, 0)
            print(f"   Parity Plot saved: {parity_png_path}")
            
    except Exception as e:
        print(f"Error generating evaluation report: {e}")
    
    # 
    final_config = {
        'version': cfg['version'],
        'timestamp': timestamp,
        'final_test_rmse': float(final_test_metrics['rmse']),  # Python float
        'final_test_mae': float(final_test_metrics['mae']),    # Python float
        'model_params': int(total_params),                     # Python int
        'tensorboard_log': log_dir,
        'output_dir': output_dir
    }
    
    config_path = os.path.join(output_dir, f"{cfg['version']}_final_config.json")
    import json
    with open(config_path, 'w') as f:
        json.dump(final_config, f, indent=4)
    print(f"   Configuration saved: {config_path}")
    
    #  TensorBoard Writer
    writer.close()
    
    print(f"\nTraining and evaluation completed!")
    print(f"View TensorBoard: tensorboard --logdir={runs_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model weights: {os.path.join(output_dir, 'best_model.pt')}")
    print(f"Loss record: {os.path.join(output_dir, cfg['version'] + '_loss.csv')}")

if __name__ == "__main__":
    main()
