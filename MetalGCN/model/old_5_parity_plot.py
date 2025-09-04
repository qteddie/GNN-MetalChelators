#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metal pKa Prediction Parity Plot
===============================

這個腳本用於繪製金屬環境下pKa預測的parity plot，可以從訓練好的模型中
獨立出來運行，方便分析不同模型的預測結果。

用法:
    python 5_parity_plot.py

直接執行將使用預設的模型路徑和測試數據。

進階用法:
    python 5_parity_plot.py --model MODEL_PATH --test_data TEST_DATA_PATH --output_dir OUTPUT_DIR

可選參數:
    --model: 訓練好的模型路徑
    --test_data: 測試數據路徑 (可以是CSV文件或者PT文件目錄)
    --output_dir: 輸出目錄
    --batch_size: 批次大小
    --version: pKa模型版本號
"""

import os
import sys
import argparse
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 添加父目錄到路徑
import sys
sys.path.append("/work/s6300121/LiveTransForM-main/metal/src/")
from metal_models import MetalFeatureExtractor
from torch_geometric.data import Data

# 默認配置
# 請修改這些默認路徑以適應您的環境
DEFAULT_MODEL_PATH = 'output/metal_pka_results/best_model.pt'  # 默認模型路徑
DEFAULT_TEST_DATA_PATH = 'data/pre_metal_t15_T25_I0.1_E3_m11.csv'  # 默認測試數據路徑
DEFAULT_OUTPUT_DIR = 'output/parity_plots'  # 默認輸出目錄


def parse_args():
    parser = argparse.ArgumentParser(description='金屬pKa預測parity plot')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'訓練好的模型路徑 (默認: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--test_data', type=str, default=DEFAULT_TEST_DATA_PATH,
                        help=f'測試數據路徑 (CSV或PT文件目錄) (默認: {DEFAULT_TEST_DATA_PATH})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'輸出目錄 (默認: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小 (默認: 32)')
    parser.add_argument('--version', type=str, default=None,
                        help='pKa模型版本號，用於圖表標題 (默認: 自動從模型路徑推斷)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='計算設備 (默認: cuda 如果可用，否則 cpu)')
    return parser.parse_args()


def load_test_data(data_path, batch_size):
    """加載測試數據集，支持CSV和PT文件目錄"""
    # 檢查是否是CSV文件
    if data_path.endswith('.csv'):
        from src.metal_chemutils import tensorize_for_pka
        import pandas as pd
        from rdkit import Chem
        
        print(f"從CSV加載測試數據: {data_path}")
        
        # 讀取CSV文件
        df = pd.read_csv(data_path)
        print(f"成功讀取CSV，含有 {len(df)} 條數據")
        
        # 準備數據列表
        data_list = []
        
        # 處理每一行
        for idx, row in df.iterrows():
            try:
                # 提取數據
                metal_ion = row['metal_ion']
                smiles = row['SMILES']
                pka_num = row['pKa_num']
                pka_value = row['pKa_value']
                
                # 處理pKa值（可能是字符串表示的列表）
                if isinstance(pka_value, str):
                    if '[' in pka_value and ']' in pka_value:
                        # 解析列表字符串
                        pka_value = eval(pka_value)
                    else:
                        # 處理單個數值字符串
                        pka_value = [float(pka_value)]
                
                # 確保pka_value是列表
                if not isinstance(pka_value, list):
                    pka_value = [pka_value]
                
                # 解析分子
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    print(f"無法解析SMILES: {smiles}，跳過")
                    continue
                    
                # 生成分子圖特徵
                fatoms, edge_index, edge_attr = tensorize_for_pka(smiles)
                
                # 創建pKa標籤（初始化為0）
                num_atoms = fatoms.size(0)
                pka_labels = torch.zeros(num_atoms)
                
                # 智能標記pKa位點
                # 1. 首先識別分子中可能的酸性位點
                acidic_atom_indices = find_potential_acidic_sites(mol)
                
                # 2. 如果找不到酸性位點，但有pKa值，則使用啟發式規則進行分配
                if len(acidic_atom_indices) == 0 and len(pka_value) > 0:
                    print(f"警告: 在 '{smiles}' 中沒找到明確的酸性位點，使用啟發式規則")
                    acidic_atom_indices = fallback_acidic_site_detection(mol)
                
                # 3. 將pKa值分配給識別的酸性位點
                if len(acidic_atom_indices) > 0 and len(pka_value) > 0:
                    # 如果酸性位點數量與pKa值數量不匹配，進行最佳分配
                    if len(acidic_atom_indices) >= len(pka_value):
                        # 酸性位點多於或等於pKa值，直接分配
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
                    # 如果仍然找不到酸性位點，但有pKa值，標記第一個原子
                    pka_labels[0] = pka_value[0]
                    
                # 獲取金屬特徵
                metal_features = None
                if metal_ion:
                    try:
                        metal_features = [MetalFeatureExtractor.get_metal_features(metal_ion)]
                    except Exception as e:
                        print(f"處理金屬特徵時出錯 ({metal_ion}): {e}")
                
                # 創建PyTorch幾何數據對象
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
                print(f"處理第 {idx} 行時出錯: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"成功處理 {len(data_list)} / {len(df)} 條數據")
        
        # 創建數據加載器
        test_loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
        
        return test_loader
        
    # 檢查是否是目錄（假設包含PT文件）
    elif os.path.isdir(data_path):
        print(f"從目錄加載測試數據: {data_path}")
        
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
                data = torch.load(file_path)
                return data
        
        # 加載數據集
        test_dataset = PTDataset(data_path)
        
        # 確保數據集非空
        if len(test_dataset) == 0:
            raise ValueError(f"測試集為空! 目錄: {data_path}")
        
        # 創建數據加載器
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                                collate_fn=lambda x: x[0] if len(x) == 1 else x)
        
        return test_loader
    
    else:
        raise ValueError(f"不支持的數據路徑格式: {data_path}，請提供 .csv 文件或包含 .pt 文件的目錄")


def find_potential_acidic_sites(mol):
    """識別可能的酸性位點
    
    尋找分子中可能的酸性官能團，如羧酸、醇、硫醇、磷酸等
    
    Args:
        mol: RDKit分子對象
        
    Returns:
        list: 酸性原子的索引列表
    """
    from rdkit import Chem
    
    if mol is None:
        return []
    
    # 初始化酸性位點列表
    acidic_indices = []
    
    # 定義常見的酸性官能團SMARTS模式
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
    
    # 去重並返回
    return list(set(acidic_indices))


def fallback_acidic_site_detection(mol):
    """當無法通過SMARTS模式識別酸性位點時的備用方法
    
    使用簡單的啟發式規則識別可能的酸性位點
    
    Args:
        mol: RDKit分子對象
        
    Returns:
        list: 可能的酸性原子索引列表
    """
    from rdkit import Chem
    
    if mol is None:
        return []
    
    candidates = []
    
    # 遍歷分子中的所有原子
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        formal_charge = atom.GetFormalCharge()
        
        # 規則1: O, N, S原子是潛在的酸性位點
        if symbol in ['O', 'N', 'S']:
            # 規則2: 有負電荷的原子可能是酸性位點
            if formal_charge < 0:
                candidates.append((atom_idx, 3))  # 高優先級
            # 規則3: 與多個重原子相連的O可能是酸性位點(如羧酸)
            elif symbol == 'O' and sum(1 for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() != 'H') >= 1:
                candidates.append((atom_idx, 2))  # 中優先級
            # 規則4: 與非金屬元素相連的N可能是酸性位點
            elif symbol == 'N':
                candidates.append((atom_idx, 1))  # 低優先級
            # 規則5: 其他O, S原子
            else:
                candidates.append((atom_idx, 0))  # 最低優先級
    
    # 按優先級排序
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # 提取原子索引
    return [idx for idx, _ in candidates]


def load_model(model_path, device):
    """加載訓練好的模型"""
    print(f"加載模型: {model_path}")
    
    try:
        # 檢查模型文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        
        # 加載模型
        checkpoint = torch.load(model_path, map_location=device)
        
        # 確定模型類型
        if 'model_state_dict' in checkpoint:
            # 標準PyTorch保存的模型
            from src.metal_models import MetalPKA_GNN
            from torch.nn import Sequential, Linear, ReLU, Dropout
            from torch_geometric.nn.conv import TransformerConv
            import torch.nn as nn
            
            # 定義參數
            node_dim = 66  # 原子特徵維度
            bond_dim = 9   # 鍵特徵維度
            hidden_dim = 128  # 隱藏層維度
            output_dim = 1  # 輸出維度 (pKa值)
            transformer_heads = 4  # Transformer頭數
            transformer_out_dim = int(512 / transformer_heads)  # 每個頭的輸出維度
            
            # 創建自定義類以匹配4_metal_pka_transfer.py中的模型結構
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
                    from src.metal_chemutils import tensorize_for_pka
                    from torch_geometric.data import Data

                    if device is None:
                        device = next(self.parameters()).device
                    self.to(device).eval()
                    
                    # 確保smiles是字符串
                    if isinstance(smiles, list) and len(smiles) > 0:
                        smiles = smiles[0]
                    
                    # 確保metal_ion是字符串
                    if metal_ion is not None and isinstance(metal_ion, list) and len(metal_ion) > 0:
                        metal_ion = metal_ion[0]

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
            
            # 創建自定義模型實例
            model = CustomMetalPKA_GNN(
                node_dim=node_dim,
                bond_dim=bond_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                dropout=0.2,
                depth=4,
                use_metal_features=True  # 啟用金屬特徵處理
            ).to(device)
            
            # 加載模型參數
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 加載pKa標準化參數
            if 'pka_mean' in checkpoint and 'pka_std' in checkpoint:
                model.set_pka_normalization(checkpoint['pka_mean'], checkpoint['pka_std'])
                print(f"pKa 標準化參數 - 均值: {checkpoint['pka_mean']:.2f}, 標準差: {checkpoint['pka_std']:.2f}")
            
            print(f"成功加載模型，epoch: {checkpoint.get('epoch', 'unknown')}")
            
        else:
            # 假設是自定義模型格式
            model = checkpoint
            print("加載自定義格式模型")
        
        model.to(device)
        model.eval()
        return model
        
    except Exception as e:
        print(f"加載模型時出錯: {e}")
        import traceback
        traceback.print_exc()
        raise e


def plot_parity_by_metal_ion(model, test_loader, device, output_dir, version=None):
    """繪製各種金屬離子下的pKa預測parity plot
    
    Args:
        model: 訓練好的模型
        test_loader: 測試數據加載器
        device: 計算設備
        output_dir: 輸出目錄
        version: 版本號，可選
        
    Returns:
        dict: 包含評估指標和圖表路徑的字典
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    model.eval()
    
    # 收集預測結果和真實值
    all_preds = []
    all_targets = []
    all_metal_ions = []
    
    # 使用參考圖片中的顏色映射不同的金屬離子
    metal_colors = {
        'Cu2+': '#4c72b0',  # 藍色
        'Ni2+': '#55a868',  # 綠色
        'Zn2+': '#c44e52',  # 紅色
        'Co2+': '#8172b3',  # 紫色
        'Cd2+': '#ccb974',  # 淺黃色
        'Ca2+': '#64b5cd',  # 淺藍色
        'Mn2+': '#b0a474',  # 淺綠色
        'Mg2+': '#cf9f5f',  # 橙色
        'Ag+': '#5f9f6f',   # 深綠色
        'Pb2+': '#b07451',  # 棕色
        'Fe3+': '#d88b70'   # 橙紅色
    }
    
    try:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='收集預測數據'):
                try:
                    batch = batch.to(device)
                    
                    # 獲取金屬離子 (確保是字符串類型)
                    metal_ion = batch.metal_ion if hasattr(batch, 'metal_ion') else "未知金屬"
                    # 如果是列表或其他不可哈希類型，轉換為字符串
                    if not isinstance(metal_ion, str):
                        # 如果是列表，提取第一個元素（避免顯示整個列表）
                        if isinstance(metal_ion, list) and len(metal_ion) > 0:
                            metal_ion = metal_ion[0]
                        # 如果列表中元素是相同的（例如["Ag+", "Ag+", ...]），只取一個代表
                        elif metal_ion and str(metal_ion).startswith('[') and ']' in str(metal_ion):
                            try:
                                # 嘗試評估字符串為列表
                                if isinstance(eval(str(metal_ion)), list):
                                    metal_list = eval(str(metal_ion))
                                    if metal_list and all(x == metal_list[0] for x in metal_list):
                                        metal_ion = metal_list[0]
                                    else:
                                        metal_ion = metal_list[0] if metal_list else "未知金屬"
                            except:
                                # 如果無法評估，使用字符串表示
                                metal_ion = str(metal_ion)
                        else:
                            metal_ion = str(metal_ion)
                    
                    # 檢查是否為有效的批次
                    if not hasattr(batch, 'pka_labels') or not hasattr(batch, 'x'):
                        continue
                    
                    # 前向傳播
                    try:
                        outputs = model(batch)
                        # 檢查輸出結構，適應不同格式
                        if isinstance(outputs, tuple) and len(outputs) >= 4:
                            logitss, pkas, targets, _ = outputs
                        else:
                            print(f"模型輸出格式未預期: {type(outputs)}")
                            continue
                    except Exception as e:
                        print(f"前向傳播錯誤: {e}")
                        continue
                    
                    # 找出真實的pKa原子
                    true_pka_mask = batch.pka_labels > 0
                    true_pka_indices = torch.nonzero(true_pka_mask).squeeze(1)
                    
                    # 檢查是否有有效的pKa標籤
                    if true_pka_indices.numel() == 0:
                        continue
                        
                    true_pka_values = batch.pka_labels[true_pka_mask].cpu().numpy()
                    
                    # 處理預測結果
                    batch_preds = []
                    batch_targets = []
                    
                    # 方法1: 嘗試為每個真實位點找到匹配的預測
                    for idx in true_pka_indices:
                        true_val = batch.pka_labels[idx].item()
                        
                        # 找到對應的預測值
                        found = False
                        for i, logits in enumerate(logitss[:-1]):
                            if i < len(pkas):
                                max_idx = torch.argmax(logits[:, 1]).item()
                                if max_idx == idx.item():
                                    batch_preds.append(pkas[i].item())
                                    batch_targets.append(true_val)
                                    found = True
                                    break
                    
                    # 禁用sample方法，因為它會導致維度錯誤
                    # 如果無法找到匹配的預測，我們跳過這個例子
                    """
                    # 如果無法找到匹配的預測，可以使用model.sample方法嘗試獲取預測
                    if not found and hasattr(batch, 'smiles') and hasattr(model, 'sample'):
                        try:
                            # 確保smiles是字符串而非列表
                            smiles_str = batch.smiles
                            if isinstance(smiles_str, list) and len(smiles_str) > 0:
                                smiles_str = smiles_str[0]
                            
                            # 確保metal_ion是字符串
                            metal_ion_str = metal_ion
                            if isinstance(metal_ion_str, list) and len(metal_ion_str) > 0:
                                metal_ion_str = metal_ion_str[0]
                                
                            # 調用sample方法
                            result = model.sample(smiles_str, metal_ion_str, device)
                            for pos, val in zip(result['pka_positions'], result['pka_values']):
                                if pos == idx.item():
                                    batch_preds.append(val)
                                    batch_targets.append(true_val)
                                    found = True
                                    break
                        except Exception as e:
                            print(f"使用sample方法時出錯: {e}")
                    """
                    
                    # 檢查是否匹配到了預測
                    if batch_preds:
                        all_preds.extend(batch_preds)
                        all_targets.extend(batch_targets)
                        # 為每個預測添加相同的金屬離子標籤
                        all_metal_ions.extend([metal_ion] * len(batch_preds))
                except Exception as e:
                    print(f"處理批次時出錯: {e}")
                    continue
        
        # 確保我們有足夠的數據點
        if len(all_preds) < 10:
            print("收集到的數據點太少，無法繪製有意義的圖表")
            return {
                'error': '數據不足',
                'num_points': len(all_preds)
            }
        
        # 確保數據是numpy數組
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 計算整體評估指標
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        # 獲取唯一的金屬離子，確保元素是字符串類型
        all_metal_ions_str = [str(ion) for ion in all_metal_ions]
        unique_metal_ions = list(set(all_metal_ions_str))
        
        # 分類每種金屬離子下的數據
        metal_data = {}
        for ion in unique_metal_ions:
            indices = [i for i, metal in enumerate(all_metal_ions_str) if metal == ion]
            if indices:
                metal_data[ion] = {
                    'preds': [all_preds[i] for i in indices],
                    'targets': [all_targets[i] for i in indices],
                    'count': len(indices)
                }
        
        # 確定版本號
        if version is None:
            if hasattr(model, 'version'):
                version = model.version
            else:
                # 從輸出目錄名稱中提取版本號
                import re
                match = re.search(r'ver(\d+)', output_dir)
                version = match.group(1) if match else '5'
        
        # 繪製parity plot (參考圖片風格)
        plt.figure(figsize=(12, 10))
        
        # 設置圖表樣式
        plt.style.use('ggplot')
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.color'] = '#cccccc'
        
        # 繪製對角線
        min_val = min(np.min(all_targets), np.min(all_preds)) - 1
        max_val = max(np.max(all_targets), np.max(all_preds)) + 1
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        
        # 繪製各種金屬離子的數據點
        for ion, data in metal_data.items():
            color = metal_colors.get(ion, 'gray')
            plt.scatter(data['targets'], data['preds'], alpha=0.7, s=40, 
                        edgecolors='none', label=f"{ion} (n={data['count']})",
                        c=color)
        
        # 設置坐標軸範圍
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        
        # 添加網格線
        plt.grid(alpha=0.3, linestyle='-', color='#cccccc')
        
        # 添加標籤
        plt.xlabel('BindingConstant$_{true}$ (unit: LogK)', fontsize=12)
        plt.ylabel('BindingConstant$_{pred}$ (unit: LogK)', fontsize=12)
        plt.title(f'Metal BindingConstant Prediction (By Metal Ion) - Ver{version}', fontsize=14)
        
        # 添加評估指標文本框
        textstr = f'Test Set (n={len(all_targets)}):\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}\nR$^2$ = {r2:.2f}'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='top', bbox=props)
        
        # 添加圖例
        plt.legend(title='Metal Ion', loc='lower right', fontsize=10)
        
        # 調整佈局
        plt.tight_layout()
        
        # 保存圖片
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, f'bindingconstant_prediction_by_metal_ver{version}.png')
        plt.savefig(plot_path, dpi=300)
        print(f"已保存parity plot到: {plot_path}")
        
        # 關閉圖形
        plt.close()
        
        # 將數據保存為CSV，便於後續分析
        csv_path = os.path.join(output_dir, f'bindingconstant_prediction_data_ver{version}.csv')
        with open(csv_path, 'w') as f:
            f.write("metal_ion,true_bindingconstant,pred_bindingconstant\n")
            for i in range(len(all_targets)):
                # 確保金屬離子是單一值而不是列表
                metal_ion_str = all_metal_ions_str[i]
                
                # 處理如果字符串表示的列表
                if metal_ion_str.startswith('[') and ']' in metal_ion_str:
                    try:
                        # 嘗試評估為列表
                        metal_list = eval(metal_ion_str)
                        if isinstance(metal_list, list) and metal_list:
                            metal_ion_str = metal_list[0]
                    except:
                        # 如果無法評估，保持原樣
                        pass
                        
                f.write(f"{metal_ion_str},{all_targets[i]},{all_preds[i]}\n")
        print(f"已保存數據到: {csv_path}")
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'num_points': len(all_targets),
            'plot_path': plot_path,
            'data_path': csv_path
        }
        
    except Exception as e:
        print(f"繪製parity plot時出錯: {e}")
        import traceback
        traceback.print_exc()
        return {
            'error': str(e)
        }


def plot_parity_plots(model, test_loader, device, output_dir, version=None):
    """繪製多種不同的parity plot，包括按金屬離子和pKa範圍分組"""
    # 確保輸出目錄存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 繪製按金屬離子分類的parity plot
    print("繪製按金屬離子分類的parity plot...")
    metal_metrics = plot_parity_by_metal_ion(model, test_loader, device, output_dir, version)
    
    # 保存總體評估指標
    metrics_path = os.path.join(output_dir, f'bindingconstant_evaluation_metrics_ver{version}.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"測試集評估指標:\n")
        f.write(f"數據點數量: {metal_metrics.get('num_points', 'N/A')}\n")
        f.write(f"RMSE: {metal_metrics.get('rmse', 'N/A'):.4f}\n")
        f.write(f"MAE: {metal_metrics.get('mae', 'N/A'):.4f}\n")
        f.write(f"R^2: {metal_metrics.get('r2', 'N/A'):.4f}\n")
    
    print(f"評估指標已保存至: {metrics_path}")
    
    return {
        'metal_metrics': metal_metrics,
        'metrics_path': metrics_path
    }


def main():
    # 解析命令行參數
    args = parse_args()
    
    # 檢查默認模型路徑和數據路徑是否存在，如果不存在則嘗試獲取絕對路徑
    model_path = args.model
    test_data_path = args.test_data
    
    # 如果使用的是相對路徑，確保在當前目錄下查找
    if not os.path.isabs(model_path):
        # 首先檢查當前工作目錄
        if os.path.exists(model_path):
            model_path = os.path.abspath(model_path)
        else:
            # 然後檢查腳本所在目錄的相對路徑
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_model_path = os.path.join(script_dir, model_path)
            if os.path.exists(alt_model_path):
                model_path = alt_model_path
    
    if not os.path.isabs(test_data_path):
        if os.path.exists(test_data_path):
            test_data_path = os.path.abspath(test_data_path)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_test_data_path = os.path.join(script_dir, test_data_path)
            if os.path.exists(alt_test_data_path):
                test_data_path = alt_test_data_path
    
    
    # 創建輸出目錄（如果不存在）
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"輸出將保存至: {args.output_dir}")
    
    try:
        # 加載模型
        print(f"加載模型: {model_path}")
        model = load_model(model_path, args.device)
        
        # 加載測試數據
        print(f"加載測試數據: {test_data_path}")
        test_loader = load_test_data(test_data_path, args.batch_size)
        
        # 繪製parity plots
        results = plot_parity_plots(model, test_loader, args.device, args.output_dir, args.version)
        
        print(f"\n繪圖已完成，結果保存在 {args.output_dir}")
        
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == '__main__':
    main() 