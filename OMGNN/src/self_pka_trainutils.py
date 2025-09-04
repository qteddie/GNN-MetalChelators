import os
import numpy as np
import pandas as pd

import torch
from torch_geometric.data     import Data
from torch_geometric.loader   import DataLoader
from sklearn.model_selection import train_test_split
import math
from self_pka_chemutils import *
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj, dense_to_sparse


def load_config_by_version(csv_path, version):
    """
    Loads configuration parameters from a CSV file for a specific version.
    
    The CSV should have a header row with parameter names, including a "version" column.
    """
    df = pd.read_csv(csv_path)
    # DataFrame for the given version.
    config_row = df[df['version'] == version]
    if config_row.empty:
        raise ValueError(f"No configuration found for version {version}")
    config = config_row.iloc[0].to_dict()
    
    # Convert parameters to appropriate types.
    # Adjust the casting based on your parameter types.
    config['test_size']    = float(config['test_size'])
    config['num_features'] = int(config['num_features'])
    config['hidden_size']  = int(config['hidden_size'])
    config['output_size']  = int(config['output_size'])
    config['batch_size']   = int(config['batch_size'])
    config['num_epochs']   = int(config['num_epochs'])
    config['dropout']      = float(config['dropout'])
    config['weight_decay'] = float(config['weight_decay'])
    config['depth']        = int(config['depth'])
    config['heads']        = int(config['heads'])
    config['pe_dim']       = int(config['pe_dim'])
    config['max_dist']     = int(config['max_dist'])
    
    return config
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
    

def build_data_item(row, tensorize_fn, max_dist=5):
    smiles = row["smiles"]
    pka_matrix = eval(row["pka_matrix"])
    fatoms, graphs, edge_features = tensorize_fn([smiles])
    
    # SPD one-hot
    ei_full, edge_features_spd = add_spd_edge_attr(
        graphs, edge_features, fatoms.size(0), max_dist=max_dist
    )
    
    num_atoms = fatoms.size(0)
    atom_pka_labels = torch.zeros(num_atoms)
    for atom_idx, pka_value in pka_matrix:
        atom_pka_labels[atom_idx] = pka_value
    
    data = Data(
        x=fatoms,
        edge_index=ei_full,
        edge_attr=edge_features_spd,
        pka_labels=atom_pka_labels,
        pka_count=len(pka_matrix),
        smiles=smiles,
    )
    return data
    
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import is_undirected as check_undir

class SafeAddLaplacianPE(T.BaseTransform):
    
    def __init__(self, k=10, attr_name="lap_pos", is_undirected=True):
        self.k = k
        self.attr_name = attr_name
        self.is_undirected = is_undirected
        
    def forward(self, data):
        N = data.num_nodes
        k_eff = max(1, min(self.k, N - 1))
        base = T.AddLaplacianEigenvectorPE(
            k=k_eff,
            attr_name=self.attr_name,
            is_undirected=self.is_undirected if check_undir(data.edge_index) else False,
        )
        
        data = base(data)
        pe = data[self.attr_name]
        if pe.size(1) < self.k:
            pad = torch.zeros(
                N, self.k - pe.size(1), device=pe.device, dtype=pe.dtype
            )
            data[self.attr_name] = torch.cat([pe, pad], dim=1)
        return data

class PkaDataset(InMemoryDataset):
    def __init__(self, 
                 df, 
                 tensorize_fn,
                 *,
                 k: int = 10, 
                 max_dist: int = 5, 
                 root: str = ".",
                 **kwargs,
    ):
        self.tensorize_fn = tensorize_fn
        self.k = k
        self.max_dist = max_dist
        transform = SafeAddLaplacianPE(
            k=k,
            attr_name="lap_pos",
            is_undirected=True,
        )
        super().__init__(root, transform=transform, pre_transform=None, **kwargs)
        self.data, self.slices = self.process_df(df)
        
    def process_df(self, df):
        data_list = [build_data_item(row, self.tensorize_fn, max_dist=self.max_dist) for _, row in df.iterrows()]
        return self.collate(data_list)
        

import os, shutil
from pathlib import Path
class pka_Dataloader():
    @staticmethod
    def data_loader_pka(
        file_path: str, 
        columns: list[str], 
        tensorize_fn, 
        batch_size: int,
        test_size: float = 0.2,
        k_pe: int = 10,
        max_dist: int = 5,
    ):
        df = pd.read_csv(file_path)[columns]
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # build InMemoryDataset
        train_dataset = PkaDataset(train_df, tensorize_fn, k=k_pe, max_dist=max_dist, root="dataset")
        test_dataset = PkaDataset(test_df, tensorize_fn, k=k_pe, max_dist=max_dist, root="dataset")
        
        # 創建資料加載器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    @staticmethod
    def evaluate_pka_model(model, loader, device):
        """
        回傳:  avg_loss, avg_cla_loss, avg_reg_loss, metrics(dict)
        metrics 內含
            accuracy / precision / recall / f1 / rmse
        """
                
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        import math
        model.eval()
        tot_loss = tot_c = tot_r = n_batch = 0
        acc_list, prec_list, rec_list, f1_list = [], [], [], []
        sqerr_total, n_labels = 0.0, 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                logitss, pkas, target_final, pka_labels, (total, loss_cla, loss_reg) = model(batch)
                pred_labels = torch.stack(logitss).argmax(dim=2)
                # ------ 累積損失 ------
                tot_loss += total.item(); tot_c += loss_cla.item(); tot_r += loss_reg.item()
                n_batch += 1
                # ------ 分類評估 ------
                gt_mask   = (batch.pka_labels > 0)
                target = gt_mask.to(torch.long)
                idx_gt  = torch.nonzero(gt_mask).squeeze(1)
                idx_sorted = idx_gt[torch.argsort(batch.pka_labels[idx_gt])]
                target_final = target.repeat(idx_sorted.shape[0]+1, 1)
                idx_sorted = torch.cat((idx_sorted, torch.tensor([-1], device=device)))
                for i in range(len(idx_sorted)-1):
                    target_final[i+1][idx_sorted[:i+1]] = 0
                for i in range(len(target_final)):
                    y_true = target_final[i].cpu().numpy()  
                    y_pred = pred_labels[i].cpu().numpy()  
                    acc     = accuracy_score(y_true, y_pred)
                    prec    = precision_score(y_true, y_pred, zero_division=1)
                    rec     = recall_score(y_true, y_pred, zero_division=1)
                    f1      = f1_score(y_true, y_pred, zero_division=1)
                    acc_list.append(acc)
                    prec_list.append(prec)
                    rec_list.append(rec)
                    f1_list.append(f1)
                    
                # ------ 回歸評估 ------
                true_pka = batch.pka_labels[idx_sorted[:-1]].to(device)
                pred_pka = torch.stack(pkas).to(device)
                true_pka, pred_pka = true_pka.to(device), pred_pka.to(device)
                
                sqerr_total += torch.sum((pred_pka - true_pka) ** 2).item()
                n_labels += true_pka.numel()

        # --------- 聚合 ---------
        avg_loss = tot_loss / n_batch
        avg_cla  = tot_c    / n_batch
        avg_reg  = tot_r    / n_batch
                
        global_rmse = math.sqrt(sqerr_total / n_labels)
        final_acc     = sum(acc_list) / len(acc_list)
        final_prec    = sum(prec_list) / len(prec_list)
        final_rec     = sum(rec_list) / len(rec_list)
        final_f1      = sum(f1_list) / len(f1_list)
        print(f"預測準確度: {final_acc:.3f}  |  precision={final_prec:.3f}  |  recall={final_rec:.3f}  |  RMSE={global_rmse:.3f}")
        metrics = dict(
            accuracy=final_acc, precision=final_prec, recall=final_rec, f1=final_f1, rmse=global_rmse,
        )
        return avg_loss, avg_cla, avg_reg, metrics
