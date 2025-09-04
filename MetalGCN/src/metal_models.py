import torch 
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing, GCNConv,  Linear, BatchNorm, GlobalAttention, GATConv
from torch.autograd import set_detect_anomaly
import math
from rdkit import Chem
import sys
sys.path.append("/work/s6300121/LiveTransForM-main/metal")
from src.metal_chemutils import atom_features
from torch_geometric.nn.conv import TransformerConv  # 添加 TransformerConv 導入

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

class BondGATMessagePassing(nn.Module):
    """
    使用 GATConv 取代手寫 message-passing：
    - 支援邊特徵：edge_dim=bond_features
    - 多頭注意力，可自行調整 heads
    - 仍保留 residual、dropout 與輸出線性層，方便銜接後續模型
    """
    def __init__(self,
                 node_features: int,
                 bond_features: int,
                 hidden_size: int,
                 depth: int = 5,
                 heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()

        self.depth   = depth
        self.dropout = dropout

        # (1) 第一層把 node+bond 映射到 hidden_size
        #     之後的層直接吃 hidden_size*heads
        gat_layers = []
        in_dim = node_features
        for i in range(depth):
            gat_layers.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=hidden_size,
                    heads=heads,
                    edge_dim=bond_features,
                    dropout=dropout,
                    add_self_loops=True,   # 保留自迴圈
                )
            )
            in_dim = hidden_size * heads  # 下一層輸入維度
        self.gats = nn.ModuleList(gat_layers)

        # (2) 最終線性層：concat(x, h_final) → hidden_size
        self.fc_out = nn.Linear(node_features + hidden_size * heads,
                                hidden_size)

        # (3) 初始化
        self.reset_parameters()

    def reset_parameters(self):
        for gat in self.gats:
            gat.reset_parameters()
        nn.init.kaiming_normal_(self.fc_out.weight,
                                mode='fan_out',
                                nonlinearity='relu')
        nn.init.zeros_(self.fc_out.bias)

    @staticmethod
    def _sanitize(t: torch.Tensor) -> torch.Tensor:
        """把 NaN / Inf 轉成有限值以避免梯度爆炸"""
        if torch.isfinite(t).all():
            return t
        return torch.nan_to_num(t, nan=0.0, posinf=1e3, neginf=-1e3)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor,
                rev_edge_index: torch.Tensor = None  # GAT 不需要，用不到
                ) -> torch.Tensor:
        """
        x            : [N, node_features]
        edge_index   : [2, E]
        edge_attr    : [E, bond_features]
        rev_edge_index : 保留參數位置以相容既有呼叫；GAT 不會用到
        """
        device = x.device
        edge_index = edge_index.to(device)
        edge_attr  = edge_attr.to(device)

        h = self._sanitize(x)

        # ----- (1) 多層 GAT -----
        for gat in self.gats:
            h = gat(h, edge_index, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self._sanitize(h)

        # ----- (2) Residual  + 輸出層 -----
        h_cat = torch.cat([x, h], dim=1)          # [N, node+hidden*heads]
        out   = F.relu(self.fc_out(h_cat))        # [N, hidden_size]
        out   = F.dropout(out, p=self.dropout, training=self.training)
        out   = self._sanitize(out)
        return out

class BondMessagePassing(nn.Module):
    def __init__(self, node_features, bond_features, hidden_size, depth=5, dropout=0.3):
        super(BondMessagePassing, self).__init__()
        # 先不初始化線性層，因為我們還不知道實際的輸入維度
        self.node_features = node_features
        self.bond_features = bond_features
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout = dropout
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        
        # 延遲初始化線性層
        self.W_i = None
        self.W_h = None
        self.W_o = None
        
    def _init_weights(self):
        """使用適當的初始化方法來防止梯度消失/爆炸"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Kaiming初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _ensure_initialized(self, x, edge_attr):
        """確保線性層已經正確初始化"""
        if self.W_i is None:
            # 計算實際的輸入維度
            actual_node_features = x.size(1)
            actual_bond_features = edge_attr.size(1)
            
            # 初始化線性層
            self.W_i = nn.Linear(actual_node_features + actual_bond_features, self.hidden_size)
            self.W_h = nn.Linear(self.hidden_size, self.hidden_size)
            self.W_o = nn.Linear(actual_node_features + self.hidden_size, self.hidden_size)
            
            # 將線性層移到正確的設備上
            device = x.device
            self.W_i = self.W_i.to(device)
            self.W_h = self.W_h.to(device)
            self.W_o = self.W_o.to(device)
            
            # 初始化權重
            self._init_weights()
            
            logger.info(f"BondMessagePassing 初始化完成: 輸入維度={actual_node_features + actual_bond_features}, 隱藏層維度={self.hidden_size}")

    def update(self, M_t, H_0):
        # 檢查並處理NaN和Inf值
        if torch.isnan(M_t).any() or torch.isinf(M_t).any():
            M_t = torch.nan_to_num(M_t, nan=0.0, posinf=1e3, neginf=-1e3)
            
        H_t = self.W_h(M_t)
        # 使用新變數而非就地修改
        H_t = self.relu(H_0 + H_t)  # 使用剩餘連接
        H_t = self.dropout_layer(H_t)

        return H_t

    def message(self, H, edge_index, rev_edge_index):
        """消息傳遞函數，現在支持直接接收邊索引和反向邊索引"""
        # 確保所有張量在同一設備上
        device = H.device
        edge_index = edge_index.to(device)
        rev_edge_index = rev_edge_index.to(device)
        
        # 檢查並處理NaN和Inf值
        if torch.isnan(H).any() or torch.isinf(H).any():
            H = torch.nan_to_num(H, nan=0.0, posinf=1e3, neginf=-1e3)
        
        # 確保index_torch在正確的設備上
        index_torch = edge_index[1].unsqueeze(1).repeat(1, H.shape[1]).to(device)
        
        # A: 使用更安全的方式計算消息
        try:
            # 1. 創建空的消息張量，並確保它在正確的設備上
            M_all = torch.zeros(H.shape[0], H.shape[1], dtype=H.dtype, device=device)
            
            # 2. 使用scatter_add_聚合消息，注意避免就地操作
            src_features = H[edge_index[0]]  # 源節點特徵
            # 創建新的張量而非使用就地操作
            M_all_accumulated = torch.zeros_like(M_all)
            M_all_accumulated.scatter_add_(0, index_torch, src_features)
            M_all = M_all_accumulated[edge_index[0]]  # 獲取特定位置的特徵
            
            # 3. 處理反向邊，首先創建一個與H相同形狀的零張量
            M_rev = torch.zeros_like(H[edge_index[0]], device=device)
            
            # 4. 只對有效的反向邊索引(非-1)進行查詢
            valid_rev_mask = rev_edge_index != -1
            if valid_rev_mask.any():
                valid_indices = rev_edge_index[valid_rev_mask]
                # 創建臨時張量來保存查詢結果
                temp_features = H[valid_indices]
                # 使用索引賦值而非就地修改
                new_M_rev = M_rev.clone()
                new_M_rev[valid_rev_mask] = temp_features
                M_rev = new_M_rev
            
            # 5. 返回差異作為最終消息
            return M_all - M_rev
            
        except Exception as e:
            print(f"消息傳遞出錯: {e}")
            # 返回一個形狀正確且在同一設備上的零張量
            return torch.zeros_like(H[edge_index[0]], device=device)

    def forward(self, x, edge_index, edge_attr, rev_edge_index):
        """修改接口以支持更靈活的輸入方式"""
        try:
            # 確保所有輸入張量在同一設備上
            device = x.device
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            rev_edge_index = rev_edge_index.to(device)
            
            # 確保線性層已經正確初始化
            self._ensure_initialized(x, edge_attr)
            
            # 檢查並處理NaN和Inf值
            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
            if torch.isnan(edge_attr).any() or torch.isinf(edge_attr).any():
                edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1e3, neginf=-1e3)
            
            # 結合節點和邊特徵
            combined = torch.cat([x[edge_index[0]], edge_attr], dim=1)
            H_0 = self.W_i(combined)
            H   = self.relu(H_0)
            
            # 多層消息傳遞
            current_H = H
            for _ in range(1, self.depth):
                M = self.message(current_H, edge_index, rev_edge_index)
                current_H = self.update(M, H_0)
            
            # 使用最終的H來計算
            H = current_H
            
            # 使用更安全的方式聚合最終結果
            index_torch = edge_index[1].unsqueeze(1).repeat(1, H.shape[1]).to(device)
            M_temp = torch.zeros(x.shape[0], H.shape[1], dtype=H.dtype, device=device)
            
            # 使用scatter_add_聚合消息，避免就地操作
            M_accumulated = torch.zeros_like(M_temp)
            M_accumulated.scatter_add_(0, index_torch, H)
            M = M_accumulated
            
            # 處理孤立節點，避免就地操作
            isolated_mask = M.sum(dim=1, keepdim=True) == 0
            if isolated_mask.any():
                M_new = torch.where(isolated_mask, x, M)
                M = M_new
            
            # 結合原始節點特徵和聚合消息
            final_features = torch.cat([x, M], dim=1)
            
            # 最終變換
            H_out = self.W_o(final_features)
            H_out = self.relu(H_out)    
            H_out = self.dropout_layer(H_out)
            
            # 確保輸出不包含NaN或Inf
            if torch.isnan(H_out).any() or torch.isinf(H_out).any():
                print("警告: BondMessagePassing輸出包含NaN或Inf值，已替換為有效值")
                H_out = torch.nan_to_num(H_out, nan=0.0, posinf=1e3, neginf=-1e3)
                
            return H_out
        except Exception as e:
            print(f"BondMessagePassing前向傳播出錯: {e}")
            # 返回一個安全的回退值，確保在正確的設備上
            return torch.zeros(x.shape[0], self.hidden_size, device=device)

class Squeeze1D(nn.Module):
    """自定義層，確保張量的最後一個維度為1時被壓縮"""
    def forward(self, x):
        # 如果最後一個維度為1，則移除該維度
        if x.size(-1) == 1:
            return x.squeeze(-1)
        return x

class MetalFeatureExtractor:
    @staticmethod
    def get_metal_features(metal_ion):
        """
        從金屬離子字串中提取特徵
        
        Args:
            metal_ion: 金屬離子字串，例如 'Ag+', 'Cu2+', 'Ag⁺'
            
        Returns:
            tuple: (原子特徵, 邊索引, 邊屬性)
        """
        try:
            # 處理上標符號
            metal_ion = metal_ion.replace('⁺', '+').replace('⁻', '-')
            
            # 提取氧化態
            digits = ''.join(filter(str.isdigit, metal_ion))
            if not digits:  # 如果沒有數字，根據正負號判斷
                if '+' in metal_ion:
                    oxidation_state = 1
                elif '-' in metal_ion:
                    oxidation_state = -1
                else:
                    oxidation_state = 0
            else:
                oxidation_state = int(digits)
                # 根據正負號調整
                if '-' in metal_ion:
                    oxidation_state = -oxidation_state
            
            # 提取金屬符號（移除所有數字和特殊符號）
            metal_symbol = ''.join(c for c in metal_ion if c.isalpha())
            
            if not metal_symbol:
                raise ValueError(f"無法從 {metal_ion} 中提取金屬符號")
            
            # 創建金屬原子
            mol = Chem.MolFromSmiles(f"[{metal_symbol}]")
            if mol is None:
                raise ValueError(f"無法創建金屬原子 {metal_symbol}")
                
            atom = mol.GetAtomWithIdx(0)
            
            # 創建邊索引和邊屬性
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            batch = torch.tensor([0], dtype=torch.long)
            edge_attr = torch.zeros((1, 11), dtype=torch.float)  # 11個邊特徵
            
            # 獲取原子特徵
            try:
                atom_features_tensor = atom_features(atom, oxidation_state)
            except Exception as e:
                logger.error(f"獲取原子特徵時出錯: {e}")
                # 使用默認特徵
                atom_features_tensor = torch.zeros(78, dtype=torch.float)  # 假設特徵維度為78
            
            logger.info(f"成功提取金屬特徵: {metal_ion} -> {metal_symbol}{oxidation_state:+d}")
            return atom_features_tensor, (edge_index, batch), edge_attr
            
        except Exception as e:
            logger.error(f"處理金屬離子特徵時出錯: {e}")
            # 獲取正確的特徵維度
            try:
                # 創建一個臨時原子來獲取特徵維度
                temp_mol = Chem.MolFromSmiles("[H]")
                temp_atom = temp_mol.GetAtomWithIdx(0)
                feature_dim = atom_features(temp_atom, 0).shape[0]
            except:
                feature_dim = 78  # 如果無法獲取，使用默認值
                
            # 返回默認值，使用正確的特徵維度
            default_features = torch.zeros(feature_dim, dtype=torch.float)
            default_edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            default_batch = torch.tensor([0], dtype=torch.long)
            default_edge_attr = torch.zeros((1, 11), dtype=torch.float)
            return default_features, (default_edge_index, default_batch), default_edge_attr

    @staticmethod
    def get_metal_feature_dim():
        """獲取金屬特徵的維度"""
        try:
            # 使用一個簡單的金屬離子來獲取特徵維度
            metal_feature, _, _ = MetalFeatureExtractor.get_metal_features("Ag+")
            return metal_feature.shape[0]
        except Exception as e:
            logger.error(f"獲取金屬特徵維度時出錯: {e}")
            return 78  # 默認值

    @staticmethod
    def get_metal_features_for_training(ion, device=None):
        """
        為訓練工具提供易於使用的金屬特徵格式
        
        Args:
            ion: 金屬離子字串
            device: 張量設備
            
        Returns:
            metal_feat: 金屬原子特徵張量
        """
        try:
            # 獲取金屬特徵
            features, _, _ = MetalFeatureExtractor.get_metal_features(ion)
            
            # 確保features是張量
            if not isinstance(features, torch.Tensor):
                logger.warning(f"金屬特徵不是張量，進行轉換: {type(features)}")
                features = torch.tensor(features, dtype=torch.float)
                
            # 移動到指定設備
            if device is not None:
                features = features.to(device)
                
            return features
        except Exception as e:
            logger.error(f"獲取訓練用金屬特徵時出錯: {e}")
            # 返回一個安全的默認值
            feature_dim = MetalFeatureExtractor.get_metal_feature_dim()
            return torch.zeros(feature_dim, dtype=torch.float, device=device)

class MetalPKA_GNN(nn.Module):
    """
    基於pka_GNN_ver3設計的金屬pKa預測模型
    結合TransformerConv與原有pka_GNN模型的金屬特徵處理功能
    """
    def __init__(self, node_dim, bond_dim, hidden_dim, output_dim, dropout=0.0, depth=4, use_metal_features=False):
        super().__init__()
        
        # 金屬特徵相關設置
        self.use_metal_features = use_metal_features
        self.metal_feature_dim = 0
        
        if use_metal_features:
            self.metal_feature_dim = MetalFeatureExtractor.get_metal_feature_dim()
            self.original_node_dim = node_dim
            logger.info(f"使用金屬特徵作為獨立節點，原子特徵維度保持不變: {node_dim}")
        
        # --- (1) TransformerConv 層 ---
        self.depth = depth
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        
        # 節點初始特徵投影層
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        
        # TransformerConv 層
        self.transformer = TransformerConv(
            in_channels=hidden_dim, 
            out_channels=hidden_dim,  
            edge_dim=bond_dim, 
            heads=4,
            dropout=dropout
        )
        
        # --- (2) gate 機制 ---
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

        # 損失函數
        self.criterion_reg = nn.MSELoss(reduction="mean")
        # pKa 標準化常數
        self.register_buffer('pka_mean', torch.tensor([7.0]))
        self.register_buffer('pka_std',  torch.tensor([3.0]))
        
        # 配位鍵特徵生成器
        self.coordination_edge_generator = nn.Linear(node_dim, bond_dim)
        
        # 權重初始化
        self._init_weights()
        
    def _init_weights(self):
        """初始化模型權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    @staticmethod
    def _rev_edge_index(edge_index):
        """計算反向邊索引"""
        device = edge_index.device
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i 
                         for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long, device=device)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        return rev_edge_index

    def _add_metal_node_and_edges(self, x, edge_index, edge_attr, metal_features, device):
        """
        將金屬作為獨立節點添加到分子圖中，並添加配位邊
        
        Args:
            x: 原子特徵 [N, node_dim]
            edge_index: 邊索引 [2, E]
            edge_attr: 邊特徵 [E, bond_dim]
            metal_features: 金屬特徵 [metal_feature_dim]
            device: 計算設備
            
        Returns:
            tuple: (新的x, 新的edge_index, 新的edge_attr)
        """
        if metal_features is None:
            return x, edge_index, edge_attr
        
        # 確保metal_features是正確的形狀和在正確的設備上
        if not isinstance(metal_features, torch.Tensor):
            metal_features = torch.tensor(metal_features, dtype=torch.float)
        
        if metal_features.dim() == 1:
            metal_features = metal_features.unsqueeze(0)
        
        metal_features = metal_features.to(device)
        
        # 根據原子特徵的維度，調整金屬特徵的維度
        if metal_features.size(1) != x.size(1):
            # 如果金屬特徵維度與原子特徵不同，使用線性層調整
            metal_features_adjusted = nn.Linear(metal_features.size(1), x.size(1)).to(device)(metal_features)
        else:
            metal_features_adjusted = metal_features
        
        # 添加金屬節點
        num_atoms = x.size(0)
        new_x = torch.cat([x, metal_features_adjusted], dim=0)  # [N+1, node_dim]
        
        # 計算離子配位傾向性（簡單實現：計算每個原子與金屬的相似度作為配位概率）
        metal_atom_sim = torch.matmul(x, metal_features_adjusted.t()).squeeze(-1)  # [N]
        
        # 選擇配位能力最強的k個原子（這裡假設k=3，可以根據實際需求調整）
        k = min(3, num_atoms)
        _, coordinating_atoms = torch.topk(metal_atom_sim, k)
        
        # 添加配位邊：從金屬離子到選定的原子，以及從這些原子到金屬離子
        metal_idx = num_atoms  # 金屬節點的索引
        
        # 創建新的邊索引
        # 從金屬到原子的邊
        metal_to_atoms = torch.stack([
            torch.full((k,), metal_idx, device=device),
            coordinating_atoms
        ])
        
        # 從原子到金屬的邊
        atoms_to_metal = torch.stack([
            coordinating_atoms,
            torch.full((k,), metal_idx, device=device)
        ])
        
        # 合併所有邊
        new_edge_index = torch.cat([edge_index, metal_to_atoms, atoms_to_metal], dim=1)
        
        # 創建配位邊的特徵
        coord_edge_features = self.coordination_edge_generator(x[coordinating_atoms])
        # 確保特徵維度正確
        if coord_edge_features.size(1) != edge_attr.size(1):
            coord_edge_features = nn.Linear(coord_edge_features.size(1), edge_attr.size(1)).to(device)(coord_edge_features)
        
        # 配位邊特徵（雙向）
        coord_edge_attr = torch.cat([coord_edge_features, coord_edge_features], dim=0)
        
        # 合併所有邊特徵
        new_edge_attr = torch.cat([edge_attr, coord_edge_attr], dim=0)
        
        return new_x, new_edge_index, new_edge_attr
    
    def forward(self, batch, return_latent=False):
        """
        模型前向傳播
        
        Args:
            batch: 輸入數據
            return_latent: 是否返回潛在表示
            
        Returns:
            Tuple: 包含模型的輸出和損失
        """
        device = batch.x.device
        x, ei, ea = batch.x, batch.edge_index, batch.edge_attr
        
        # ====== 處理金屬離子，添加為獨立節點 ======
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
                        logger.warning(f"金屬原子特徵不是張量: {type(metal_atom_feature)}")
                        # 嘗試轉換為張量
                        if hasattr(metal_atom_feature, '__array__') or isinstance(metal_atom_feature, (list, np.ndarray)):
                            metal_atom_feature = torch.tensor(metal_atom_feature, device=device, dtype=torch.float)
                    
                    # 如果是標量張量或空張量，需要特別處理
                    if not hasattr(metal_atom_feature, 'shape') or len(metal_atom_feature.shape) == 0:
                        logger.warning("金屬原子特徵沒有形狀或是標量，使用默認特徵")
                        metal_atom_feature = torch.zeros(self.metal_feature_dim, device=device, dtype=torch.float)
                    
                    # 將金屬作為獨立節點添加，並建立配位邊
                    x, ei, ea = self._add_metal_node_and_edges(x, ei, ea, metal_atom_feature, device)
                    
                    logger.info(f"成功添加金屬節點及配位邊，節點總數: {x.size(0)}")
                else:
                    logger.warning(f"金屬特徵格式不正確: {type(metal_features)}")
            except Exception as e:
                logger.error(f"處理金屬特徵時出錯: {e}")
                # 繼續執行，不使用金屬特徵
        
        rev_ei = self._rev_edge_index(ei)

        # 節點特徵投影
        h = self.node_proj(x)  # [N, hidden_dim]
        
        # 使用TransformerConv進行消息傳遞
        h_cur = self.transformer(h, ei, edge_attr=ea)
        h_cur = F.relu(h_cur)
        h_cur = F.dropout(h_cur, p=self.dropout, training=self.training)

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
            
                # 重新運行TransformerConv
                h_cur = self.transformer(h_upd, ei, edge_attr=ea)
                h_cur = F.relu(h_cur)
                h_cur = F.dropout(h_cur, p=self.dropout, training=self.training)
            
        # ---------- 匯總損失 ----------
        loss_cla = torch.stack(loss_cla_steps).mean() if loss_cla_steps else torch.tensor(0., device=device)
        loss_reg = torch.stack(loss_reg_steps).mean() if loss_reg_steps else torch.tensor(0., device=device)
        total = loss_cla + loss_reg

        # 準備輸出
        outputs = (logitss, pkas, target_final, (total, loss_cla, loss_reg))
        if return_latent:
            outputs += (latent_steps, pka_steps)

        return outputs
    
    def load_proton_pretrained(self, proton_model_path, reset_output_layers=True):
        """
        從預訓練的質子 pKa 模型加載權重
        
        Args:
            proton_model_path: 預訓練模型路徑
            reset_output_layers: 是否重置輸出層權重
            
        Returns:
            bool: 加載成功返回True
        """
        try:
            # 加載預訓練模型
            pretrained_dict = torch.load(proton_model_path, map_location='cpu')
            
            # 如果加載的是完整模型而非state_dict
            if not isinstance(pretrained_dict, dict) or 'state_dict' in pretrained_dict:
                pretrained_dict = pretrained_dict.get('state_dict', pretrained_dict)
                
            # 獲取當前模型的state_dict
            model_dict = self.state_dict()
            
            # 過濾不匹配的鍵
            filtered_dict = {}
            for k, v in pretrained_dict.items():
                # 去除可能的前綴如'module.'
                if k.startswith('module.'):
                    k = k[7:]
                    
                # 檢查鍵是否存在於當前模型中
                if k in model_dict:
                    # 檢查形狀是否匹配
                    if v.shape == model_dict[k].shape:
                        filtered_dict[k] = v
                    else:
                        logger.warning(f"忽略參數形狀不匹配的層: {k}, 預訓練: {v.shape}, 當前: {model_dict[k].shape}")
                else:
                    logger.warning(f"忽略當前模型中不存在的層: {k}")
            
            # 如果需要重置輸出層
            if reset_output_layers:
                output_layers = ['atom_classifier', 'atom_regressor']
                filtered_dict = {k: v for k, v in filtered_dict.items() 
                                if not any(k.startswith(layer) for layer in output_layers)}
                logger.info("已重置輸出層權重，這些層將從頭訓練")
            
            # 更新模型
            model_dict.update(filtered_dict)
            self.load_state_dict(model_dict)
            
            logger.info(f"成功從{proton_model_path}加載預訓練權重")
            return True
            
        except Exception as e:
            logger.error(f"加載預訓練模型失敗: {e}")
            return False
            
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
    
    def sample(self, smiles: str, metal_ion=None, device=None, eval_mode="predicted"):
        """
        對分子進行pKa預測
                   
        Args:
            smiles: 分子SMILES
            metal_ion: 金屬離子（可選）
            device: 計算設備
            eval_mode: 評估模式
            
        Returns:
            dict: 包含預測結果的字典
        """
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
                logger.info(f"為分子添加金屬離子: {metal_ion}")
            except Exception as e:
                logger.error(f"添加金屬離子特徵時出錯: {e}")
         
        with torch.no_grad():
            logitss, pkas, _, _ = self(data)
            
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
