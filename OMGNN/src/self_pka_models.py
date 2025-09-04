import torch 
import torch.nn as nn
import logging
import os
import numpy as np
from torch.autograd import set_detect_anomaly
import math
from torch_geometric.nn.conv   import TransformerConv
from self_pka_trainutils import SafeAddLaplacianPE, add_spd_edge_attr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from rdkit import Chem
from self_pka_chemutils import tensorize_for_pka
from torch_geometric.data import Data
import pandas as pd

# 配置logging，設置文件處理器
def setup_logger():
    """設置日誌記錄器"""
    # 確保logs目錄存在
    os.makedirs("logs", exist_ok=True)
    
    # 配置日誌記錄器
    logger = logging.getLogger('pka_model')
    logger.setLevel(logging.DEBUG)
    
    # 創建文件處理器
    file_handler = logging.FileHandler('logs/pka_model_debug.log')
    file_handler.setLevel(logging.DEBUG)
    
    # 創建控制台處理器，僅用於ERROR級別的消息
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    
    # 創建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加處理器到記錄器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 初始化logger
logger = setup_logger()

# 輔助函數：啟用autograd異常檢測
def enable_anomaly_detection():
    """啟用PyTorch梯度異常檢測，幫助定位就地操作問題"""
    set_detect_anomaly(True)
    logger.info("已啟用PyTorch梯度異常檢測，這會減慢訓練速度但能幫助定位問題")



class Squeeze1D(nn.Module):
    """自定義層，確保張量的最後一個維度為1時被壓縮"""
    def forward(self, x):
        # 如果最後一個維度為1，則移除該維度
        if x.size(-1) == 1:
            return x.squeeze(-1)
        return x

class pka_GNN_ver3(nn.Module):
    def __init__(self, 
                 node_dim: int, 
                 bond_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 *,
                 dropout: float = 0.0, 
                 depth: int = 4, 
                 heads: int = 1, 
                 max_dist: int = 5,
                 pe_dim: int = 10,
                 pka_weight_path: str = "../data/pka_weight_table.npz",
                 ):
        super().__init__()

        self.depth = depth
        self.dropout = dropout
        self.bond_dim = bond_dim
        self.max_dist = max_dist
        self.pe_dim = pe_dim
        self.x_proj = nn.Linear(node_dim, hidden_dim, bias=False)
        self.pe_proj = nn.Linear(pe_dim, hidden_dim, bias=False)
        self.gcn3 = TransformerConv(in_channels=hidden_dim, 
                                    out_channels=hidden_dim//heads,  
                                    edge_dim=bond_dim, 
                                    heads=heads,
                                    concat=True,
                                    dropout=dropout)
        table = np.load(pka_weight_path)
        self.register_buffer('bin_edges', torch.tensor(table['bin_edges']))
        self.register_buffer('bin_weights', torch.tensor(table['weights']))
        # --- (2) gate ---
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh())

        # --- (3) classifier / regressor ---
        self.atom_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        self.atom_regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, output_dim),  # scalar
            Squeeze1D()
        )
        # 測試SmoothL1loss
        self.criterion_reg = nn.SmoothL1Loss(beta=0.5)
        # self.criterion_reg = nn.MSELoss(reduction="mean")
        # pKa 標準化常數 (可由 set_pka_normalization 更新)
        self.register_buffer('pka_mean', torch.tensor([7.0]))
        self.register_buffer('pka_std',  torch.tensor([3.0]))
        

    def forward(self, batch, return_latent=False):
        if batch.edge_attr.size(-1) != self.bond_dim:
            raise ValueError(f"bond_dim 不匹配: {batch.edge_attr.size(-1)} != {self.bond_dim}")
        device = batch.x.device
        if not hasattr(batch, "lap_pos"):
            raise ValueError("batch 中缺少 lap_pos 欄位")
        
        x_orig = batch.x
        pe = batch.lap_pos.float()
        x = self.x_proj(x_orig) + self.pe_proj(pe)
        ei, ea = batch.edge_index, batch.edge_attr

        # --- 第 1 次 attention ------------------------------------------------
        h_cur = self.gcn3(x, ei, ea)

        # ---------- 取出 ground-truth pKa 原子順序 ----------
        gt_mask = (batch.pka_labels > 0)
        target = gt_mask.to(torch.long)
        idx_gt  = torch.nonzero(gt_mask).squeeze(1)           # [K]
        
        # 依真實 pKa 值排序
        idx_sorted = idx_gt[torch.argsort(batch.pka_labels[idx_gt])]
        target_final = target.repeat(idx_sorted.shape[0]+1, 1)
        idx_sorted = torch.cat((idx_sorted, torch.tensor([-1], device=device)))
        for i in range(len(idx_sorted)-1):
            target_final[i+1][idx_sorted[:i+1]] = 0
        # ---------- 逐-site loop ----------
        logitss, pkas = [], []
        loss_cla_steps, loss_reg_steps = [], []
        latent_steps, pka_steps = [], []   # <─ 新增
        
        for step_idx, idx in enumerate(idx_sorted):
            if return_latent:
                latent_steps.append(h_cur[idx].detach().cpu())
                pka_steps.append(batch.pka_labels[idx].detach().cpu())
                
            # 1) 分類 / 回歸
            logits = self.atom_classifier(h_cur)
            logitss.append(logits)
            target = target_final[step_idx]
            # print(f"target: {target}")
            # print(f"logits: {F.softmax(logits, dim=1).max(dim=1)[1]}")

            ratio  = float((target == 0).sum()) / (target.sum() + 1e-6)
            loss_c = nn.functional.cross_entropy(
                logits, target, weight=torch.tensor([1.0, ratio], device=device), reduction='none'
            )
            loss_cla_steps.extend(loss_c)
            
            if step_idx != len(idx_sorted)-1:
                
                pred_pka = self.atom_regressor(h_cur).view(-1)[idx]
                true_pka = batch.pka_labels[idx]
                
                # bucketize 會回傳 bin 索引 0…B-1
                bin_idx = torch.bucketize(true_pka, self.bin_edges) - 1
                w = self.bin_weights[bin_idx]
                
                pred_pka_norm  = (pred_pka - self.pka_mean) / self.pka_std
                true_pka_norm  = (batch.pka_labels[idx] - self.pka_mean) / self.pka_std
                loss_r = w * (pred_pka_norm - true_pka_norm).pow(2)
                loss_reg_steps.append(loss_r)
                pkas.append(pred_pka)

            # 2) gate 更新該原子特徵並重置其他原子 ---------------------- #
            # h_upd = h_static.clone()                           # (2) 其餘原子 → h_static
            h_upd = h_cur.clone()
            h_upd[idx] = x[idx] + h_cur[idx] * self.gate(h_cur[idx])

            # 3) 重新跑一次 gcn3，並取得新的 α
            h_cur = self.gcn3(
                h_upd, ei, ea
            )
            
        # ---------- 匯總損失 ----------
        loss_cla = torch.stack(loss_cla_steps).mean()
        loss_reg = torch.stack(loss_reg_steps).mean()
        
        # 在訓練腳本裡進行動態的調整
        lambda_reg = 1.0
        loss_reg = loss_reg * lambda_reg
        total     = loss_cla + loss_reg

        # 準備輸出
        outputs = (logitss, pkas, target_final, batch.pka_labels, 
                   (total, loss_cla, loss_reg))
        if return_latent:
            outputs += (latent_steps, pka_steps)

        return outputs

        
    def set_pka_normalization(self, mean, std):
        """
        設置 pKa 標準化參數，應使用全數據集的統計值
        
        Args:
            mean: 全數據集 pKa 值的均值
            std: 全數據集 pKa 值的標準差
        """
        # 確保這些值是浮點數
        mean_val = float(mean)
        std_val = max(float(std), 1e-6)  # 確保標準差不為零
        
        # 更新模型的標準化參數
        self.pka_mean[0] = mean_val
        self.pka_std[0] = std_val
        
        logger.info(f"設置 pKa 標準化參數: 均值={mean_val:.4f}, 標準差={std_val:.4f}")
        return mean_val, std_val

    def sample(self, smiles: str, file_path: str, device=None):

        if device is None:
            device = next(self.parameters()).device
        self.to(device).eval()
        df = pd.read_csv(file_path)
        row = df.loc[df['smiles'] == smiles]
        pka_matrix = row["pka_matrix"].tolist()[0]
        pka_matrix = eval(pka_matrix)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Can't parse SMILES: {smiles}")
        fatoms, ei_raw, ea_raw = tensorize_for_pka(smiles)
        
        ei_full, edge_attr = add_spd_edge_attr(
            ei_raw, ea_raw, fatoms.size(0), max_dist=self.max_dist
        )
        # print("max_dist =", self.max_dist)
        # print("edge_attr.shape =", edge_attr.shape)
        # print("ei_full.shape =", ei_full.shape)
        num_atoms = fatoms.size(0)
        atom_pka_labels = torch.zeros(num_atoms)
        for atom_idx, pka_value in pka_matrix:
            atom_pka_labels[atom_idx] = pka_value
        batch = Data(x=fatoms, 
                     edge_index=ei_full, 
                     edge_attr=edge_attr,
                     pka_labels=atom_pka_labels, 
                     smiles=smiles,
        ).to(device)
        # —— 加入 Laplacian PE ——
        transform = SafeAddLaplacianPE(k=self.pe_dim,
                                              attr_name='lap_pos',
                                              is_undirected=True)
        batch = transform(batch).to(device)

        with torch.no_grad():
            logitss, pkas, target_final, pka_labels, (loss, loss_cla, loss_reg) = self(batch) # Adjusted for new loss_attn
        pred_labels = torch.stack(logitss).argmax(dim=2)
        # ------ 分類評估 ------
        gt_mask   = (batch.pka_labels > 0)
        target = gt_mask.to(torch.long)
        idx_gt  = torch.nonzero(gt_mask).squeeze(1)
        idx_sorted = idx_gt[torch.argsort(batch.pka_labels[idx_gt])]
        target_final = target.repeat(idx_sorted.shape[0]+1, 1)
        idx_sorted = torch.cat((idx_sorted, torch.tensor([-1], device=device)))
        for i in range(len(idx_sorted)-1):
            target_final[i+1][idx_sorted[:i+1]] = 0
        # ------ 回歸評估 ------
        true_pka = batch.pka_labels[idx_sorted[:-1]].cpu().to(device)
        pred_pka = torch.stack(pkas)
        mse = torch.nn.functional.mse_loss(pred_pka, true_pka).item()
        rmse = math.sqrt(mse)
        acc_list, prec_list, rec_list, f1_list = [], [], [], []
    
        for i in range(len(target_final)):
            y_true = target_final[i].cpu().numpy()  
            y_pred = pred_labels[i].cpu().numpy()  
            # y_true_list.append(y_true)
            # y_pred_list.append(y_pred)
            acc     = accuracy_score(y_true, y_pred)
            prec    = precision_score(y_true, y_pred, zero_division=1)
            rec     = recall_score(y_true, y_pred, zero_division=1)
            f1      = f1_score(y_true, y_pred, zero_division=1)
            acc_list.append(acc)
            prec_list.append(prec)
            rec_list.append(rec)
            f1_list.append(f1)
        result = {
            "smiles": smiles,
            "true_pka": true_pka.tolist(),
            "pred_pka": pred_pka.tolist(),
            "rmse": rmse,
            "true_cla": target_final.tolist(),
            "pred_cla": pred_labels.tolist(),
            "acc": sum(acc_list) / len(acc_list),
            "prec": sum(prec_list) / len(prec_list),
            "rec": sum(rec_list) / len(rec_list),
            "f1": sum(f1_list) / len(f1_list),
        }
        return result
    
    
    # ===============================================================
    # 推論用：只給 SMILES，迭代產生分子級 pKa 預測
    # ===============================================================
    @torch.no_grad()
    def sample(
        self,
        smiles: str,
        *,
        confidence_threshold: float = 0.80,
        max_steps: int = 3,
        device: torch.device = None
    ) -> dict:
        """
        只輸入 SMILES，回傳分子的多段 pKa 預測。

        confidence_threshold : 二分類機率低於此值 -> 停止再找酸性位點
        max_steps            : 最多迭代預測多少個酸性位點
        """
        # ---------- 裝置 ----------
        if device is None:
            device = next(self.parameters()).device
        self.to(device).eval()

        # ---------- 1.  生成 PyG Data（pka_labels 全 0） ----------
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"無法解析 SMILES: {smiles}")

        fatoms, ei_raw, ea_raw = tensorize_for_pka(smiles)

        # 1-a. SPD edge + Laplacian PE
        ei_full, edge_attr = add_spd_edge_attr(
            ei_raw, ea_raw, fatoms.size(0), max_dist=self.max_dist
        )
        data = Data(
            x=fatoms,
            edge_index=ei_full,
            edge_attr=edge_attr,
            pka_labels=torch.zeros(fatoms.size(0)),  # 全 0
            smiles=smiles,
        )
        # Laplacian PE
        data = SafeAddLaplacianPE(
            k=self.pe_dim, attr_name="lap_pos", is_undirected=True
        )(data)
        batch = data.to(device)

        # ---------- 2. 初始節點嵌入 ----------
        x0 = self.x_proj(batch.x) + self.pe_proj(batch.lap_pos.float())
        h_cur = self.gcn3(x0, batch.edge_index, batch.edge_attr)  # 第一次 TransformerConv

        # ---------- 3.  迭代預測 ----------
        used = torch.zeros(batch.x.size(0), dtype=torch.bool, device=device)
        pred_pkas, pred_indices = [], []

        for _ in range(max_steps):
            # 3-a. 找最可能酸性位點
            logits = self.atom_classifier(h_cur)         # [N,2]
            probs  = logits.softmax(dim=1)[:, 1]         # 正類機率
            probs[used] = -1.0
            if probs[idx] < confidence_threshold:
                break
            idx = int(torch.argmax(probs).item())

            # 3-b. 取得該位點 pKa（反向標準化）
            pka_norm = self.atom_regressor(h_cur)[idx]        # tensor
            pka_val  = float(pka_norm * self.pka_std + self.pka_mean)
            pred_pkas.append(pka_norm)
            pred_indices.append(idx)
            used[idx] = True

            # 3-c. 更新圖狀態（gating + 再跑一層 gcn3）
            h_upd = h_cur.clone()
            h_upd[idx] = x0[idx] + self.gate(h_cur[idx]) * h_cur[idx]
            h_cur = self.gcn3(h_upd, batch.edge_index, batch.edge_attr)

        # ---------- 4.  回傳 ----------
        return {
            "smiles":       smiles,
            "pred_pka":     np.array(pred_pkas, dtype=float),
            "atom_indices": pred_indices,
            # 若想要單一「overall」數值，可再自行加:
            # "overall_pka_mean": float(np.mean(pred_pkas)) if pred_pkas else None,
            # "overall_pka_min":  float(np.min(pred_pkas))  if pred_pkas else None,
        }

    # def sample(self, smiles: str, file_path: str, device=None):

    #     if device is None:
    #         device = next(self.parameters()).device
    #     self.to(device).eval()
    #     df = pd.read_csv(file_path)
    #     row = df.loc[df['smiles'] == smiles]
    #     pka_matrix = row["pka_matrix"].tolist()[0]
    #     pka_matrix = eval(pka_matrix)
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is None:
    #         raise ValueError(f"Can't parse SMILES: {smiles}")
    #     fatoms, ei_raw, ea_raw = tensorize_for_pka(smiles)
        
    #     ei_full, edge_attr = add_spd_edge_attr(
    #         ei_raw, ea_raw, fatoms.size(0), max_dist=self.max_dist
    #     )
    #     # print("max_dist =", self.max_dist)
    #     # print("edge_attr.shape =", edge_attr.shape)
    #     # print("ei_full.shape =", ei_full.shape)
    #     num_atoms = fatoms.size(0)
    #     atom_pka_labels = torch.zeros(num_atoms)
    #     for atom_idx, pka_value in pka_matrix:
    #         atom_pka_labels[atom_idx] = pka_value
    #     batch = Data(x=fatoms, 
    #                  edge_index=ei_full, 
    #                  edge_attr=edge_attr,
    #                  pka_labels=atom_pka_labels, 
    #                  smiles=smiles,
    #     ).to(device)
    #     # —— 加入 Laplacian PE ——
    #     transform = SafeAddLaplacianPE(k=self.pe_dim,
    #                                           attr_name='lap_pos',
    #                                           is_undirected=True)
    #     batch = transform(batch).to(device)

    #     with torch.no_grad():
    #         logitss, pkas, target_final, pka_labels, (loss, loss_cla, loss_reg) = self(batch) # Adjusted for new loss_attn
    #     pred_labels = torch.stack(logitss).argmax(dim=2)
    #     # ------ 分類評估 ------
    #     gt_mask   = (batch.pka_labels > 0)
    #     target = gt_mask.to(torch.long)
    #     idx_gt  = torch.nonzero(gt_mask).squeeze(1)
    #     idx_sorted = idx_gt[torch.argsort(batch.pka_labels[idx_gt])]
    #     target_final = target.repeat(idx_sorted.shape[0]+1, 1)
    #     idx_sorted = torch.cat((idx_sorted, torch.tensor([-1], device=device)))
    #     for i in range(len(idx_sorted)-1):
    #         target_final[i+1][idx_sorted[:i+1]] = 0
    #     # ------ 回歸評估 ------
    #     true_pka = batch.pka_labels[idx_sorted[:-1]].cpu().to(device)
    #     pred_pka = torch.stack(pkas)
    #     mse = torch.nn.functional.mse_loss(pred_pka, true_pka).item()
    #     rmse = math.sqrt(mse)
    #     acc_list, prec_list, rec_list, f1_list = [], [], [], []
    
    #     for i in range(len(target_final)):
    #         y_true = target_final[i].cpu().numpy()  
    #         y_pred = pred_labels[i].cpu().numpy()  
    #         # y_true_list.append(y_true)
    #         # y_pred_list.append(y_pred)
    #         acc     = accuracy_score(y_true, y_pred)
    #         prec    = precision_score(y_true, y_pred, zero_division=1)
    #         rec     = recall_score(y_true, y_pred, zero_division=1)
    #         f1      = f1_score(y_true, y_pred, zero_division=1)
    #         acc_list.append(acc)
    #         prec_list.append(prec)
    #         rec_list.append(rec)
    #         f1_list.append(f1)
    #     result = {
    #         "smiles": smiles,
    #         "true_pka": true_pka.tolist(),
    #         "pred_pka": pred_pka.tolist(),
    #         "rmse": rmse,
    #         "true_cla": target_final.tolist(),
    #         "pred_cla": pred_labels.tolist(),
    #         "acc": sum(acc_list) / len(acc_list),
    #         "prec": sum(prec_list) / len(prec_list),
    #         "rec": sum(rec_list) / len(rec_list),
    #         "f1": sum(f1_list) / len(f1_list),
    #     }
    #     return result










    def sample_position(self, smiles: str, device=None, target_count=None, max_steps=5):
        if device is None:
            device = next(self.parameters()).device
        self.to(device).eval()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Can't parse SMILES: {smiles}")
        fatoms, ei_raw, ea_raw = tensorize_for_pka(smiles)
        
        ei_full, edge_attr = add_spd_edge_attr(
            ei_raw, ea_raw, fatoms.size(0), max_dist=self.max_dist
        )
        num_atoms = fatoms.size(0)
        
        batch = Data(x=fatoms, 
                     edge_index=ei_full, 
                     edge_attr=edge_attr,
                     pka_labels=torch.zeros(num_atoms),  # Dummy labels
                     smiles=smiles,
        ).to(device)
        
        # —— 加入 Laplacian PE ——
        transform = SafeAddLaplacianPE(k=self.pe_dim,
                                              attr_name='lap_pos',
                                              is_undirected=True)
        batch = transform(batch).to(device)
        
        with torch.no_grad():
            # Manual inference without ground truth dependency
            x_orig = batch.x
            pe = batch.lap_pos.float()
            x = self.x_proj(x_orig) + self.pe_proj(pe)
            ei, ea = batch.edge_index, batch.edge_attr
            
            # Initial attention
            h_cur = self.gcn3(x, ei, ea)
            
            predicted_positions = []
            selected_atoms = set()
            
            # Determine how many positions to predict
            if target_count is not None:
                steps_to_run = min(target_count, max_steps, num_atoms)
            else:
                steps_to_run = max_steps
            
            # Iterative prediction
            for step in range(steps_to_run):
                # Get classification logits
                logits = self.atom_classifier(h_cur)
                
                # Apply softmax and get probabilities for class 1 (pKa site)
                probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of being pKa site
                
                # Mask already selected atoms
                for selected_idx in selected_atoms:
                    probs[selected_idx] = 0.0
                
                # Get the most likely position
                most_likely_idx = torch.argmax(probs).item()
                confidence = probs[most_likely_idx].item()
                
                # If target_count is specified, continue even with low confidence
                # Otherwise, stop if confidence is too low
                if target_count is None and confidence < 0.1:
                    break
                    
                predicted_positions.append(most_likely_idx)
                selected_atoms.add(most_likely_idx)
                
                # Update hidden state with gate mechanism
                h_upd = h_cur.clone()
                h_upd[most_likely_idx] = x[most_likely_idx] + h_cur[most_likely_idx] * self.gate(h_cur[most_likely_idx])
                
                # Re-run attention
                h_cur = self.gcn3(h_upd, ei, ea)
            
        return predicted_positions






















