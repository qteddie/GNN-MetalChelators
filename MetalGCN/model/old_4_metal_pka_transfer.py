#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn.conv import TransformerConv
from tqdm import tqdm
import time
import torch.nn as nn
import math
import traceback
import pandas as pd
from rdkit import Chem
# 添加父目錄到路徑
import sys
sys.path.append("../src/")
from metal_chemutils import tensorize_for_pka
from metal_models import MetalPKA_GNN, MetalFeatureExtractor
from torch_geometric.data import Data
 

def load_config_by_version(csv_path, version):
    df = pd.read_csv(csv_path)
    config_row = df[df['version'] == version]
    if config_row.empty:
        raise ValueError(f"No configuration found for version {version}")
    config = config_row.iloc[0].to_dict()
    config['seed']         = int(config['seed'])
    config['output_size']  = int(config['output_size'])
    config['num_features'] = int(config['num_features'])
    config['hidden_size']  = int(config['hidden_size'])
    config['batch_size']   = int(config['batch_size'])
    config['num_epochs']   = int(config['num_epochs'])
    config['dropout']      = float(config['dropout'])
    config['lr']           = float(config['lr'])
    config['anneal_rate']  = float(config['anneal_rate'])
    config['weight_decay'] = float(config['weight_decay'])
    config['depth']        = int(config['depth'])
    config['heads']        = int(config['heads'])
    config['pe_dim']       = int(config['pe_dim'])
    config['max_dist']     = int(config['max_dist'])
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='金屬pKa轉移學習')
    parser.add_argument("--config_csv", default="../data/parameters.csv")
    parser.add_argument("--version"   , default="metal_ver1")
    args = parser.parse_args()
    cfg = load_config_by_version(args.config_csv, args.version)
    print("\n=== Config ===")
    for k, v in cfg.items(): 
        print(f"{k:18}: {v}")
    print("==============\n")
    return cfg
def set_seed(seed):
    """設置隨機種子以確保可重複性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def load_data(data_dir, batch_size, metal_csv=None):
    """加載金屬pKa數據集"""
    # 添加對預處理好的金屬資料集的支援
    if metal_csv and os.path.exists(metal_csv):
        print(f"檢測到預處理好的金屬資料集: {metal_csv}")
        return load_preprocessed_metal_data(metal_csv, batch_size)
    # 確保使用絕對路徑
    if data_dir is not None:
        data_dir = os.path.abspath(data_dir)
    else:
        raise ValueError("必須提供data_dir或metal_csv參數")
    # 原有的加載邏輯（假設數據集已經按照train/valid/test分割好）
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    # 創建自定義數據集類，直接加載 .pt 文件
    class PTDataset(torch.utils.data.Dataset):
        def __init__(self, directory):
            self.directory = directory
            self.file_list = [f for f in os.listdir(directory) if f.endswith('.pt')]
            print(f"找到 {len(self.file_list)} 個 .pt 文件在 {directory}")
            
        def __len__(self):
            return len(self.file_list)
            
        def __getitem__(self, idx):
            file_path = os.path.join(self.directory, self.file_list[idx])
            data = torch.load(file_path, weights_only=True)
            return data
    train_dataset = PTDataset(train_dir)
    valid_dataset = PTDataset(valid_dir)
    test_dataset = PTDataset(test_dir)
    # 計算pKa統計值用於標準化
    pka_values = []
    for dataset in [train_dataset]:
        for i in range(len(dataset)):
            data = dataset[i]
            if hasattr(data, 'pka_labels'):
                pka = data.pka_labels.numpy()
                pka = pka[pka > 0]  # 只考慮有效的pKa值
                pka_values.extend(pka)
    pka_mean = np.mean(pka_values) if pka_values else 7.0
    pka_std = np.std(pka_values) if pka_values else 3.0
    print(f"pKa 標準化參數 - 均值: {pka_mean:.2f}, 標準差: {pka_std:.2f}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x[0] if len(x) == 1 else x)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=lambda x: x[0] if len(x) == 1 else x)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=lambda x: x[0] if len(x) == 1 else x)
    return train_loader, valid_loader, test_loader, pka_mean, pka_std


def load_preprocessed_metal_data(csv_path, batch_size, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    """從預處理好的金屬CSV文件加載數據並切分為訓練/驗證/測試集"""

    # 讀取CSV文件
    df = pd.read_csv(csv_path)
    print(f"成功讀取CSV，含有 {len(df)} 條數據")
    data_list = []
    pka_values = []
    # 處理每一行
    for _, row in df.iterrows():
        try:
            metal_ion = row['metal_ion']
            smiles = row['SMILES']
            pka_value = eval(row['pKa_value'])
            pka_values.extend(pka_value)
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"無法解析SMILES: {smiles}，跳過")
                continue
            fatoms, edge_index, edge_attr = tensorize_for_pka(smiles)
            num_atoms = fatoms.size(0)
            pka_labels = torch.zeros(num_atoms)
            # 智能標記pKa位點
            acidic_atom_indices = find_potential_acidic_sites(mol)
            if len(acidic_atom_indices) == 0 and len(pka_value) > 0:
                acidic_atom_indices = fallback_acidic_site_detection(mol)
            if len(acidic_atom_indices) > 0 and len(pka_value) > 0:
                if len(acidic_atom_indices) >= len(pka_value):
                    for i, pka in enumerate(pka_value):
                        if i < len(acidic_atom_indices):
                            atom_idx = acidic_atom_indices[i]
                            if atom_idx < num_atoms:  # 確保索引有效
                                pka_labels[atom_idx] = pka
                else:
                    # pKa值多於酸性位點，取值平均
                    avg_pka = sum(pka_value) / len(pka_value)
                    for atom_idx in acidic_atom_indices:
                        if atom_idx < num_atoms:  # 確保索引有效
                            pka_labels[atom_idx] = avg_pka
            elif len(pka_value) > 0:
                pka_labels[0] = pka_value[0]
            metal_features = None
            if metal_ion:
                try:
                    metal_features = [MetalFeatureExtractor.get_metal_features(metal_ion)]
                except Exception as e:
                    print(f"處理金屬特徵時出錯 ({metal_ion}): {e}")
            data = Data(
                x=fatoms,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pka_labels=pka_labels,
                smiles=smiles,
                metal_ion=metal_ion,
                metal_features=metal_features
            )
            data_list.append(data)
        except Exception as e:
            traceback.print_exc()
    print(f"成功處理 {len(data_list)} / {len(df)} 條數據")
    # 計算pKa統計值用於標準化
    pka_mean = np.mean(pka_values)
    pka_std = np.std(pka_values)
    print(f"pKa 標準化參數 - 均值: {pka_mean:.2f}, 標準差: {pka_std:.2f}")
    from sklearn.model_selection import train_test_split
    train_data, temp_data = train_test_split(
        data_list, 
        train_size=train_ratio, 
        random_state=42
    )
    valid_ratio_adjusted = valid_ratio / (valid_ratio + test_ratio)  # 調整比例
    valid_data, test_data = train_test_split(
        temp_data, 
        train_size=valid_ratio_adjusted, 
        random_state=42
    )
    print(f"數據集分割: 訓練集 {len(train_data)}, 驗證集 {len(valid_data)}, 測試集 {len(test_data)}")
    from torch_geometric.loader import DataLoader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader, pka_mean, pka_std

def find_potential_acidic_sites(mol):
    if mol is None:
        return []
    acidic_indices = []
    acidic_patterns = [
        # 羧酸
        ('C(=O)[OH]', 2),  # OH中的O
        # 醇/酚
        ('[OH]', 0),  # OH中的O
        # 硫醇
        ('[SH]', 0),  # SH中的S
        # 磷酸
        ('P(=O)([OH])[OH]', 2),  # 第一個OH中的O
        ('P(=O)([OH])[OH]', 3),  # 第二個OH中的O
        # 胺
        ('[NH2]', 0),  # NH2中的N
        ('[NH]', 0),   # NH中的N
        # 咪唑
        ('c1ncnc1', 1),  # 咪唑環中的N
        # 硫酸
        ('S(=O)(=O)[OH]', 3),  # OH中的O
    ]
    # 尋找酸性官能團
    for pattern, atom_idx in acidic_patterns:
        patt = Chem.MolFromSmarts(pattern)
        if patt:
            matches = mol.GetSubstructMatches(patt)
            for match in matches:
                if atom_idx < len(match):
                    acidic_indices.append(match[atom_idx])
    return list(set(acidic_indices))

def fallback_acidic_site_detection(mol):
    
    if mol is None:
        return []
    candidates = []
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        formal_charge = atom.GetFormalCharge()
        if symbol in ['O', 'N', 'S']:
            if formal_charge < 0:
                candidates.append((atom_idx, 3))  # 高優先級
            elif symbol == 'O' and sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() != 'H') >= 1:
                candidates.append((atom_idx, 2))  # 中優先級
            elif symbol == 'N':
                candidates.append((atom_idx, 1))  # 低優先級
            else:
                candidates.append((atom_idx, 0))  # 最低優先級
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in candidates]


def create_model(cfg, device):
    """建立並初始化金屬pKa預測模型"""
    node_dim = cfg["num_features"]  
    bond_dim =  9  
    hidden_dim = cfg['hidden_size']  
    output_dim = cfg['output_size']  
    transformer_heads = cfg['heads']  
    transformer_out_dim = int(cfg['hidden_size'] / transformer_heads) 
    proton_model_path = cfg['proton_model']
    reset_output = True
    # 創建一個自定義類來修改MetalPKA_GNN
    class CustomMetalPKA_GNN(MetalPKA_GNN):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 重置Transformer層，使用自定義頭數和輸出維度
            self.transformer = TransformerConv(
                in_channels=hidden_dim, 
                out_channels=transformer_out_dim,  # 每個頭的輸出維度
                edge_dim=bond_dim, 
                heads=transformer_heads,  # 頭數
                dropout=kwargs.get('dropout', 0.0)
            )
            
            # 添加降維層，將Transformer輸出(512)降到hidden_dim(128)
            self.dim_reduction = nn.Sequential(
                nn.Linear(transformer_out_dim * transformer_heads, hidden_dim),
                nn.ReLU(),
                nn.Dropout(kwargs.get('dropout', 0.0))
            )
            
            print(f"自定義Transformer層: in_channels={hidden_dim}, out_channels={transformer_out_dim}, "
                  f"heads={transformer_heads}, total_out_dim={transformer_out_dim*transformer_heads}")
            print(f"添加降維層: {transformer_out_dim*transformer_heads} -> {hidden_dim}")
            
        # 重寫forward方法處理降維
        def forward(self, batch, return_latent=False):
            device = batch.x.device
            x, ei, ea = batch.x, batch.edge_index, batch.edge_attr
            
            # 處理金屬離子，添加為獨立節點
            metal_feature = None
            num_original_atoms = x.size(0)
            
            if hasattr(batch, 'metal_features') and batch.metal_features is not None and self.use_metal_features:
                try:
                    # 提取金屬原子特徵
                    metal_features = batch.metal_features[0]
                    
                    # 檢查金屬特徵的格式
                    if isinstance(metal_features, tuple) and len(metal_features) > 0:
                        metal_atom_feature = metal_features[0]  # 取出第一個元素，這應該是原子特徵
                        
                        # 確保 metal_atom_feature 是張量
                        if not isinstance(metal_atom_feature, torch.Tensor):
                            # 嘗試轉換為張量
                            if hasattr(metal_atom_feature, '__array__') or isinstance(metal_atom_feature, (list, np.ndarray)):
                                metal_atom_feature = torch.tensor(metal_atom_feature, device=device, dtype=torch.float)
                        
                        # 如果是標量張量或空張量，需要特別處理
                        if not hasattr(metal_atom_feature, 'shape') or len(metal_atom_feature.shape) == 0:
                            metal_atom_feature = torch.zeros(self.metal_feature_dim, device=device, dtype=torch.float)
                        
                        # 將金屬作為獨立節點添加，並建立配位邊
                        x, ei, ea = self._add_metal_node_and_edges(x, ei, ea, metal_atom_feature, device)
                except Exception as e:
                    print(f"處理金屬特徵時出錯: {e}")
                    # 繼續執行，不使用金屬特徵
                    
            rev_ei = self._rev_edge_index(ei)
            
            # 節點特徵投影
            h = self.node_proj(x)  # [N, hidden_dim]
            
            # 使用TransformerConv進行消息傳遞
            h_trans = self.transformer(h, ei, edge_attr=ea)  # [N, hidden_dim*heads]
            h_cur = self.dim_reduction(h_trans)  # [N, hidden_dim]
            
            # ---------- 取出 ground-truth pKa 原子順序 ----------
            # 只考慮原始原子（不含金屬節點）
            gt_mask = torch.zeros(x.size(0), dtype=torch.bool, device=device)
            if hasattr(batch, 'pka_labels') and batch.pka_labels is not None:
                # 確保 pka_labels 與原始原子數量匹配
                if batch.pka_labels.size(0) == num_original_atoms:
                    gt_mask[:num_original_atoms] = (batch.pka_labels > 0)
                else:
                    gt_mask[:batch.pka_labels.size(0)] = (batch.pka_labels > 0)
                    
            target = gt_mask.to(torch.long)
            idx_gt = torch.nonzero(gt_mask).squeeze(1)  # [K]
            
            if idx_gt.numel() == 0:  # 無可解離原子
                logits = self.atom_classifier(h_cur)
                pka_raw = self.atom_regressor(h_cur).view(-1)
                loss_cla = nn.functional.cross_entropy(
                    logits, torch.zeros_like(gt_mask, dtype=torch.long)
                )
                return logits, pka_raw, (0.5 * loss_cla, loss_cla, torch.tensor(0., device=device))
                
            # 依真實 pKa 值排序
            idx_sorted = idx_gt[torch.argsort(batch.pka_labels[idx_gt])]
            
            # 準備每一步的目標
            target_final = target.repeat(idx_sorted.shape[0]+1, 1)
            idx_sorted = torch.cat((idx_sorted, torch.tensor([-1], device=device)))
            for i in range(len(idx_sorted)-1):
                target_final[i+1][idx_sorted[:i+1]] = 0
                
            # ---------- 逐-site loop ----------
            logitss, pkas = [], []
            loss_cla_steps, loss_reg_steps = [], []
            latent_steps, pka_steps = [], []
            
            for step_idx, idx in enumerate(idx_sorted):
                # 存儲latent表示（如果需要）
                if return_latent and idx != -1:
                    latent_steps.append(h_cur[idx].detach().cpu())
                    pka_steps.append(batch.pka_labels[idx].detach().cpu())
                    
                # 分類
                logits = self.atom_classifier(h_cur)
                logitss.append(logits)
                target = target_final[step_idx]

                # 計算分類損失
                ratio = float((target == 0).sum()) / (target.sum() + 1e-6)
                loss_c = nn.functional.cross_entropy(
                    logits, target, weight=torch.tensor([1.0, ratio], device=device), reduction='none'
                )
                loss_cla_steps.extend(loss_c)
                
                # 回歸（對於最後一步EOS除外）
                if step_idx != len(idx_sorted)-1:
                    pred_pka = self.atom_regressor(h_cur).view(-1)[idx]
                    pred_pka_norm = (pred_pka - self.pka_mean) / self.pka_std
                    true_pka_norm = (batch.pka_labels[idx] - self.pka_mean) / self.pka_std
                    loss_r = self.criterion_reg(pred_pka_norm, true_pka_norm)
                    loss_reg_steps.append(loss_r)
                    pkas.append(pred_pka)

                # gate更新節點特徵
                if idx != -1:  # 最後一步不更新
                    h_upd = h_cur.clone()
                    h_upd[idx] = h[idx] + h_cur[idx] * self.gate(h_cur[idx])
                    
                    # 重新運行TransformerConv並應用降維
                    h_trans = self.transformer(h_upd, ei, edge_attr=ea)
                    h_cur = self.dim_reduction(h_trans)
                
            # ---------- 匯總損失 ----------
            loss_cla = torch.stack(loss_cla_steps).mean() if loss_cla_steps else torch.tensor(0., device=device)
            loss_reg = torch.stack(loss_reg_steps).mean() if loss_reg_steps else torch.tensor(0., device=device)
            total = loss_cla + loss_reg

            # 準備輸出
            outputs = (logitss, pkas, target_final, (total, loss_cla, loss_reg))
            if return_latent:
                outputs += (latent_steps, pka_steps)

            return outputs
            
        # 重寫sample方法以適配新的forward方法
        def sample(self, smiles: str, metal_ion=None, device=None, eval_mode="predicted"):
            """重寫sample方法以適配新的forward方法"""
            from rdkit import Chem
            from metal_chemutils import tensorize_for_pka
            from torch_geometric.data import Data

            if device is None:
                device = next(self.parameters()).device
            self.to(device).eval()

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"無法解析SMILES: {smiles}")

            fatoms, ei, ea = tensorize_for_pka(smiles)
            n = fatoms.size(0)
            data = Data(x=fatoms, edge_index=ei, edge_attr=ea,
                        pka_labels=torch.zeros(n, device=device), 
                        batch=torch.zeros(n, dtype=torch.long, device=device),
                        smiles=smiles).to(device)
            
            # 如果提供了金屬離子並且模型設置為使用金屬特徵
            if metal_ion and self.use_metal_features:
                try:
                    metal_features = MetalFeatureExtractor.get_metal_features(metal_ion)
                    data.metal_features = [metal_features]
                    print(f"為分子添加金屬離子: {metal_ion}")
                except Exception as e:
                    print(f"添加金屬離子特徵時出錯: {e}")

            with torch.no_grad():
                outputs = self(data)
                logitss, pkas = outputs[0], outputs[1]
                
            # 處理最終結果
            final_logits = logitss[-1]  # 最後一步的分類結果
            has_pka = (final_logits.argmax(1) == 1).cpu().numpy()
            
            # 收集所有預測的pKa值
            pred_pka_values = []
            pred_positions = []
            
            for i, logits in enumerate(logitss[:-1]):  # 排除最後一步（EOS）
                idx = logits[:, 1].argmax().item()
                if idx < n:  # 確保索引有效（不包括金屬節點）
                    pred_positions.append(idx)
                    pred_pka_values.append(pkas[i].item())
            
            # 根據評估模式決定返回的結果
            if eval_mode == "predicted":
                idx = pred_positions
                pka_values = pred_pka_values
            else:  # 'all'
                idx = list(range(n))
                pka_values = [pkas[i].item() if i in pred_positions else 0.0 for i in range(n)]

            result = {
                "smiles": smiles,
                "mol": mol,
                "metal_ion": metal_ion if metal_ion else None,
                "atom_has_pka": has_pka,
                "atom_pka_values": pka_values,
                "pka_positions": idx,
                "pka_values": [pka_values[i] for i in range(len(pka_values)) if i in idx],
                "eval_mode": eval_mode
            }
                
            return result
    
    # 創建自定義金屬pKa模型
    model = CustomMetalPKA_GNN(
        node_dim=node_dim,
        bond_dim=bond_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        dropout=cfg['dropout'],
        depth=cfg['depth'],
        use_metal_features=True
    ).to(device)
    
    # 加載預訓練模型
    if proton_model_path:
        try:
            model.load_proton_pretrained(proton_model_path, reset_output_layers=reset_output)
            print(f"成功加載預訓練模型: {proton_model_path}")
        except Exception as e:
            print(f"加載預訓練模型失敗: {e}")
    return model


def train(model, train_loader, valid_loader, optimizer, device, epochs, output_dir):
    """訓練模型"""
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    print(f"訓練結果將保存至: {output_dir}")
    
    # 初始化最佳損失
    best_loss = float('inf')
    
    # 初始化結果記錄
    results = {'train_loss': [], 'valid_loss': [], 'train_cls_loss': [],
               'valid_cls_loss': [], 'train_reg_loss': [], 'valid_reg_loss': []}
    
    # 開始訓練循環
    print(f"開始訓練，共{epochs}個輪次")
    try:
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # 訓練階段
            model.train()
            train_loss = 0
            train_cls_loss = 0
            train_reg_loss = 0
            
            train_iter = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch in train_iter:
                batch = batch.to(device)
                
                # 清除梯度
                optimizer.zero_grad()
                
                # 前向傳播
                _, _, _, (loss, cls_loss, reg_loss) = model(batch)
                
                # 反向傳播
                loss.backward()
                
                # 更新參數
                optimizer.step()
                
                # 更新損失
                train_loss += loss.item()
                train_cls_loss += cls_loss.item()
                train_reg_loss += reg_loss.item()
                
                # 更新進度條
                train_iter.set_postfix({'loss': loss.item()})
            
            # 計算平均損失
            train_loss /= len(train_loader)
            train_cls_loss /= len(train_loader)
            train_reg_loss /= len(train_loader)
            
            # 驗證階段
            model.eval()
            valid_loss = 0
            valid_cls_loss = 0
            valid_reg_loss = 0
            
            with torch.no_grad():
                valid_iter = tqdm(valid_loader, desc=f'Epoch {epoch+1}/{epochs} [Valid]')
                for batch in valid_iter:
                    batch = batch.to(device)
                    
                    # 前向傳播
                    _, _, _, (loss, cls_loss, reg_loss) = model(batch)
                    
                    # 更新損失
                    valid_loss += loss.item()
                    valid_cls_loss += cls_loss.item()
                    valid_reg_loss += reg_loss.item()
                    
                    # 更新進度條
                    valid_iter.set_postfix({'loss': loss.item()})
            
            # 計算平均損失
            valid_loss /= len(valid_loader)
            valid_cls_loss /= len(valid_loader)
            valid_reg_loss /= len(valid_loader)
            
            # 記錄結果
            results['train_loss'].append(train_loss)
            results['valid_loss'].append(valid_loss)
            results['train_cls_loss'].append(train_cls_loss)
            results['valid_cls_loss'].append(valid_cls_loss)
            results['train_reg_loss'].append(train_reg_loss)
            results['valid_reg_loss'].append(valid_reg_loss)
                
            # 計算本輪所需時間
            epoch_time = time.time() - epoch_start_time
            
            # 打印訓練信息
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'Train Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Reg: {train_reg_loss:.4f}), '
                  f'Valid Loss: {valid_loss:.4f} (Cls: {valid_cls_loss:.4f}, Reg: {valid_reg_loss:.4f}), '
                  f'時間: {epoch_time:.1f}秒')
            
            # 保存最佳模型
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_loss,
                    'pka_mean': model.pka_mean.item(),
                    'pka_std': model.pka_std.item()
                }, os.path.join(output_dir, 'best_model.pt'))
                print(f'保存最佳模型，驗證損失: {best_loss:.4f}')
    
    except KeyboardInterrupt:
        print("訓練被用戶中斷")
    
    # 保存最終模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': valid_loss,
        'pka_mean': model.pka_mean.item(),
        'pka_std': model.pka_std.item()
    }, os.path.join(output_dir, 'final_model.pt'))
    
    # 保存訓練結果
    np.save(os.path.join(output_dir, 'training_results.npy'), results)
    
    return model, results


def evaluate(model, test_loader, device):
    """評估模型"""
    model.eval()
    
    # 初始化評估指標
    test_loss = 0
    test_cls_loss = 0
    test_reg_loss = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        test_iter = tqdm(test_loader, desc='Testing')
        for batch in test_iter:
            batch = batch.to(device)
            
            # 前向傳播
            logitss, pkas, targets, (loss, cls_loss, reg_loss) = model(batch)
            
            # 更新損失
            test_loss += loss.item()
            test_cls_loss += cls_loss.item()
            test_reg_loss += reg_loss.item()
            
            # 收集預測結果
            for i, pka_list in enumerate(pkas):
                # 找出真實的pKa原子
                true_pka_mask = batch.pka_labels > 0
                true_pka_indices = torch.nonzero(true_pka_mask).squeeze(1)
                
                # 收集預測值
                if i < len(true_pka_indices):
                    idx = true_pka_indices[i]
                    pred = pka_list.item()
                    target = batch.pka_labels[idx].item()
                    
                    all_preds.append(pred)
                    all_targets.append(target)
    
    # 計算平均損失
    test_loss /= len(test_loader)
    test_cls_loss /= len(test_loader)
    test_reg_loss /= len(test_loader)
    
    # 計算回歸評估指標
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(np.mean(np.square(all_preds - all_targets)))
    
    # 打印評估結果
    print(f'Test Loss: {test_loss:.4f} (Cls: {test_cls_loss:.4f}, Reg: {test_reg_loss:.4f})')
    print(f'MAE: {mae:.4f}, RMSE: {rmse:.4f}')
    
    return {
        'test_loss': test_loss,
        'test_cls_loss': test_cls_loss,
        'test_reg_loss': test_reg_loss,
        'mae': mae,
        'rmse': rmse
    }


def demo_predict(model, smiles_list, metal_ion, device):
    """演示預測功能"""
    model.eval()
    
    results = []
    for smiles in smiles_list:
        try:
            # 簡化的預測方式
            print(f"\nSMILES: {smiles}")
            print(f"金屬離子: {metal_ion}")
            
            # 使用RDKit解析分子
            from rdkit import Chem
            from metal_chemutils import tensorize_for_pka
            from torch_geometric.data import Data

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"無法解析SMILES: {smiles}")
                continue

            # 生成分子特徵
            fatoms, ei, ea = tensorize_for_pka(smiles)
            n = fatoms.size(0)
            data = Data(x=fatoms, edge_index=ei, edge_attr=ea,
                        pka_labels=torch.zeros(n, device=device), 
                        batch=torch.zeros(n, dtype=torch.long, device=device),
                        smiles=smiles).to(device)
            
            # 添加金屬特徵
            if model.use_metal_features:
                try:
                    from src.metal_chemutils import atom_features
                    
                    # 簡化金屬特徵生成
                    metal_symbol = ''.join(c for c in metal_ion if c.isalpha())
                    oxidation_state = int(''.join(filter(str.isdigit, metal_ion)))
                    if '-' in metal_ion:
                        oxidation_state = -oxidation_state
                        
                    # 創建金屬原子
                    metal_mol = Chem.MolFromSmiles(f"[{metal_symbol}]")
                    metal_atom = metal_mol.GetAtomWithIdx(0)
                    
                    # 獲取特徵
                    metal_feat = atom_features(metal_atom, oxidation_state)
                    data.metal_features = [(metal_feat,)]
                    
                    print(f"成功添加金屬特徵: {metal_ion}")
                except Exception as e:
                    print(f"添加金屬特徵時出錯: {e}")
            
            # 直接使用模型前向傳播預測pKa，但不要依賴特定的輸出結構
            with torch.no_grad():
                try:
                    # 使用自定義模型的forward方法
                    outputs = model(data)
                    # 檢查輸出結構，適應不同格式
                    if isinstance(outputs, tuple) and len(outputs) >= 2:
                        logitss, pkas = outputs[0], outputs[1]
                    else:
                        # 如果輸出結構不符合預期，打印一條消息並繼續
                        print(f"模型輸出格式未預期: {type(outputs)}")
                        result = {
                            "smiles": smiles,
                            "metal_ion": metal_ion,
                            "pka_positions": [],
                            "pka_values": []
                        }
                        results.append(result)
                        continue
                        
                    # 檢查並打印一些輸出信息，幫助調試
                    print(f"模型輸出: logitss類型={type(logitss)}, 長度={len(logitss) if isinstance(logitss, list) else 'N/A'}")
                    if isinstance(logitss, list) and len(logitss) > 0:
                        print(f"第一個logits形狀: {logitss[0].shape}")
                    elif isinstance(logitss, torch.Tensor):
                        print(f"logits張量形狀: {logitss.shape}")
                    
                    # 根據輸出格式，找到可能的pKa位點和值
                    pka_positions = []
                    pka_values = []
                    
                    # 簡單方法：如果logitss和pkas都是有效的，選擇第一個pka值展示
                    if isinstance(logitss, list) and len(logitss) > 0 and isinstance(pkas, list) and len(pkas) > 0:
                        # 選擇第一個位點
                        if hasattr(logitss[0], 'shape') and len(logitss[0].shape) > 1 and logitss[0].shape[1] > 1:
                            max_idx = logitss[0][:, 1].argmax().item()
                            if max_idx < n:
                                pka_positions.append(max_idx)
                                pka_values.append(pkas[0].item() if hasattr(pkas[0], 'item') else float(pkas[0]))
                    elif isinstance(logitss, torch.Tensor) and isinstance(pkas, torch.Tensor):
                        # 如果logitss是單個張量，嘗試直接獲取預測
                        if len(logitss.shape) > 1 and logitss.shape[1] >= 2:
                            max_idx = logitss[:, 1].argmax().item()
                            if max_idx < n and pkas.numel() > 0:
                                pka_positions.append(max_idx)
                                pka_values.append(pkas[max_idx].item())
                                print(f"從張量中識別到pKa位點: 原子 {max_idx}, pKa = {pkas[max_idx].item():.2f}")
                    
                except Exception as e:
                    print(f"模型推理過程中出錯: {e}")
                    import traceback
                    traceback.print_exc()
                    result = {
                        "smiles": smiles,
                        "metal_ion": metal_ion,
                        "pka_positions": [],
                        "pka_values": []
                    }
                    results.append(result)
                    continue
            
            # 顯示預測結果
            if pka_positions:
                print("pKa預測結果:")
                for pos, pka in zip(pka_positions, pka_values):
                    print(f"  原子 {pos}: pKa = {pka:.2f}")
                result = {
                    "smiles": smiles,
                    "metal_ion": metal_ion,
                    "pka_positions": pka_positions,
                    "pka_values": pka_values
                }
                results.append(result)
            else:
                print("未檢測到pKa位點")
                result = {
                    "smiles": smiles,
                    "metal_ion": metal_ion,
                    "pka_positions": [],
                    "pka_values": []
                }
                results.append(result)
                
        except Exception as e:
            print(f"預測 {smiles} 時出錯: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "smiles": smiles,
                "metal_ion": metal_ion,
                "error": str(e),
                "pka_positions": [],
                "pka_values": []
            })
    
    return results


def analyze_predictions(model, test_loader, device, output_dir):
    """分析模型預測結果，並將結果保存為CSV文件"""
    print("\n分析預測結果:")
    
    model.eval()
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='分析預測'):
            batch = batch.to(device)
            
            # 提取分子信息
            smiles = batch.smiles if hasattr(batch, 'smiles') else "未知分子"
            metal_ion = batch.metal_ion if hasattr(batch, 'metal_ion') else "未知金屬"
            
            # 使用模型進行預測
            logitss, pkas, targets, _ = model(batch)
            
            # 找出真實的pKa原子
            true_pka_mask = batch.pka_labels > 0
            true_pka_indices = torch.nonzero(true_pka_mask).squeeze(1)
            true_pka_values = batch.pka_labels[true_pka_mask]
            
            # 預測的pKa值
            pred_pka_values = []
            pred_pka_indices = []
            
            # 處理預測結果
            for i, logits in enumerate(logitss[:-1]):  # 排除最後一步(EOS)
                max_idx = torch.argmax(logits[:, 1]).item()
                if max_idx < batch.x.size(0):  # 確保索引有效
                    pred_pka_indices.append(max_idx)
                    if i < len(pkas):
                        pred_pka_values.append(pkas[i].item())
            
            # 計算準確率，在金屬環境下pKa預測是比較困難的任務
            # 我們計算：
            # 1. 位點預測準確率: 預測位點與真實位點的交集/真實位點數
            # 2. 值預測誤差: 對於正確位點的pKa預測誤差
            
            # 轉換為集合便於計算交集
            true_indices_set = set(true_pka_indices.cpu().numpy())
            pred_indices_set = set(pred_pka_indices)
            
            # 計算位點預測準確率
            correct_sites = true_indices_set.intersection(pred_indices_set)
            site_accuracy = len(correct_sites) / len(true_indices_set) if len(true_indices_set) > 0 else 0
            
            # 計算值預測誤差
            value_errors = []
            for idx in correct_sites:
                true_idx = true_pka_indices.cpu().numpy().tolist().index(idx)
                pred_idx = pred_pka_indices.index(idx)
                
                true_val = true_pka_values[true_idx].item()
                pred_val = pred_pka_values[pred_idx]
                
                error = abs(true_val - pred_val)
                value_errors.append(error)
            
            avg_value_error = sum(value_errors) / len(value_errors) if value_errors else float('nan')
            
            # 保存結果
            result = {
                'smiles': smiles,
                'metal_ion': metal_ion,
                'true_pka_indices': true_pka_indices.cpu().numpy().tolist(),
                'true_pka_values': true_pka_values.cpu().numpy().tolist(),
                'pred_pka_indices': pred_pka_indices,
                'pred_pka_values': pred_pka_values,
                'site_accuracy': site_accuracy,
                'avg_value_error': avg_value_error
            }
            
            results.append(result)
    
    # 計算總體準確率和誤差
    site_accuracies = [r['site_accuracy'] for r in results]
    value_errors = [r['avg_value_error'] for r in results if not math.isnan(r['avg_value_error'])]
    
    avg_site_accuracy = sum(site_accuracies) / len(site_accuracies) if site_accuracies else 0
    avg_value_error = sum(value_errors) / len(value_errors) if value_errors else float('nan')
    
    print(f"位點預測平均準確率: {avg_site_accuracy:.3f}")
    print(f"pKa值預測平均絕對誤差: {avg_value_error:.3f}")
    
    # 保存為CSV
    csv_path = os.path.join(output_dir, 'prediction_analysis.csv')
    with open(csv_path, 'w') as f:
        f.write("SMILES,Metal_Ion,True_Sites,True_Values,Predicted_Sites,Predicted_Values,Site_Accuracy,Value_Error\n")
        for r in results:
            f.write(f"{r['smiles']},{r['metal_ion']},{r['true_pka_indices']},{r['true_pka_values']},{r['pred_pka_indices']},{r['pred_pka_values']},{r['site_accuracy']:.3f},{r['avg_value_error']}\n")
    
    print(f"分析結果已保存至: {csv_path}")
    
    return {
        'avg_site_accuracy': avg_site_accuracy,
        'avg_value_error': avg_value_error,
        'results': results
    }


def predict_metal_pka(model, smiles, metal_ion, device=None):
    """使用訓練好的模型預測金屬環境下的pKa值
    
    Args:
        model: 訓練好的金屬pKa模型
        smiles: 分子SMILES字符串
        metal_ion: 金屬離子 (例如 "Cu2+")
        device: 計算設備
        
    Returns:
        dict: 預測結果
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    try:
        # 解析分子
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {'error': f"無法解析SMILES: {smiles}"}
        
        # 使用模型的sample方法進行預測
        result = model.sample(smiles, metal_ion, device)
        
        # 格式化結果
        prediction = {
            'smiles': smiles,
            'metal_ion': metal_ion,
            'pka_sites': result['pka_positions'],
            'pka_values': result['pka_values']
        }
        
        # 如果有酸性位點，添加其解釋
        if prediction['pka_sites']:
            site_descriptions = []
            for idx in prediction['pka_sites']:
                atom = mol.GetAtomWithIdx(idx)
                symbol = atom.GetSymbol()
                hyb = str(atom.GetHybridization())
                neighbors = [n.GetSymbol() for n in atom.GetNeighbors()]
                desc = f"原子 {idx} ({symbol}, 雜化: {hyb}, 鄰居: {neighbors})"
                site_descriptions.append(desc)
            prediction['site_descriptions'] = site_descriptions
            
        return prediction
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'error': str(e)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = parse_args()
    data_source = cfg["metal_csv"]
    print(f"使用預處理好的金屬資料集: {data_source}")
    output_dir = cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    print(f"輸出將保存至: {output_dir}")
    # 設置隨機種子
    set_seed(cfg["seed"])
    train_loader, valid_loader, test_loader, pka_mean, pka_std = load_preprocessed_metal_data(
        data_source, cfg["batch_size"])
    print(f"創建模型並加載預訓練權重: {cfg['proton_model']}")
    model = create_model(cfg, device)
    model.set_pka_normalization(pka_mean, pka_std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    model, results = train(model, train_loader, valid_loader, optimizer,
                          device, cfg["num_epochs"], output_dir)
    metrics = evaluate(model, test_loader, device)
    np.save(os.path.join(output_dir, 'evaluation_results.npy'), metrics)
    analysis = analyze_predictions(model, test_loader, device, output_dir)
    np.save(os.path.join(output_dir, 'prediction_analysis.npy'), analysis)

if __name__ == '__main__':
    main() 