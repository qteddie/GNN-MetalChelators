import torch 
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import MessagePassing, GCNConv, Linear, BatchNorm, GlobalAttention
import time  # 添加時間模組
import sys   # 添加系統模組
import numpy as np
from sklearn.preprocessing import StandardScaler

# 添加全局調試開關
DEBUG = True  # 設為True以查看詳細調試信息
MAX_RETRIES = 3  # 最大重試次數，避免無限循環

def debug_log(message):
    """調試日誌函數"""
    if DEBUG:
        timestamp = time.strftime('%H:%M:%S', time.localtime())
        print(f"[DEBUG {timestamp}] {message}", flush=True)
        sys.stdout.flush()  # 強制立即輸出

class BondMessagePassing(nn.Module):
    def __init__(self, node_features, bond_features, hidden_size, depth=5, dropout=0.3):
        super(BondMessagePassing, self).__init__()
        self.expected_node_features = node_features
        self.expected_bond_features = bond_features
        self.hidden_size = hidden_size
        self.depth = depth
        self.dropout_rate = dropout
        self.initialized = False
        self.device = None
        self.retry_count = 0  # 添加重試計數器
        
        debug_log(f"BondMessagePassing初始化: 預期節點特徵={node_features}, 邊特徵={bond_features}, 隱藏大小={hidden_size}")
    
    def _initialize_layers(self, node_dim, edge_dim):
        """根據實際數據維度動態初始化所有層"""
        debug_log(f"正在初始化層：節點特徵={node_dim}, 邊特徵={edge_dim}, 隱藏層={self.hidden_size}")
        
        # 核心網絡層
        self.W_i = nn.Linear(node_dim + edge_dim, self.hidden_size).to(self.device)
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size).to(self.device)
        self.W_o = nn.Linear(node_dim + self.hidden_size, self.hidden_size).to(self.device)
        self.node_transform = nn.Linear(node_dim, self.hidden_size).to(self.device)
        
        # 激活函數和正則化
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # 標記初始化完成
        self.initialized = True
        debug_log(f"層初始化完成: W_i.shape={self.W_i.weight.shape}, W_o.shape={self.W_o.weight.shape}")

    def update(self, M_t, H_0):
        # debug_log(f"執行update: M_t.shape={M_t.shape}, H_0.shape={H_0.shape}")
        H_t = self.W_h(M_t)
        H_t = self.relu(H_0 + H_t)
        H_t = self.dropout(H_t)
        return H_t

    def message(self, H, batch):
        # debug_log(f"執行message: H.shape={H.shape}, edge_index.shape={batch.edge_index.shape}")
        index_torch = batch.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M_all = torch.zeros(len(batch.x), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)[batch.edge_index[0]]
        M_rev = H[batch.rev_edge_index]
        return M_all - M_rev
        
    def forward(self, batch):
        # 添加進入forward的調試訊息
        debug_log(f"進入BondMessagePassing.forward: batch.x.shape={batch.x.shape}, batch.edge_attr.shape={batch.edge_attr.shape}")
        
        # 保存設備信息
        self.device = batch.x.device
        
        # 獲取實際數據維度
        actual_node_dim = batch.x.shape[1]
        actual_edge_dim = batch.edge_attr.shape[1]
        
        # 檢查是否需要初始化或重新初始化
        if not self.initialized:
            debug_log(f"首次執行: 節點特徵={actual_node_dim}, 邊特徵={actual_edge_dim}, 連接特徵總維度={actual_node_dim + actual_edge_dim}")
            self._initialize_layers(actual_node_dim, actual_edge_dim)
        
        # 核心前向傳播邏輯
        try:
            debug_log(f"開始前向傳播計算...")
            combined_features = torch.cat([batch.x[batch.edge_index[0]], batch.edge_attr], dim=1)
            debug_log(f"連接特徵完成: shape={combined_features.shape}")
            
            H_0 = self.W_i(combined_features)
            debug_log(f"W_i計算完成: H_0.shape={H_0.shape}")
            H = self.relu(H_0)
            
            # 報告循環進度
            for i in range(1, self.depth):
                debug_log(f"執行message-passing迭代 {i}/{self.depth-1}...")
                M = self.message(H, batch)
                H = self.update(M, H_0)
            
            debug_log("開始計算最終嵌入...")    
            index_torch = batch.edge_index[1].unsqueeze(1).repeat(1, H.shape[1]) 
            M = torch.zeros(len(batch.x), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)
            
            debug_log(f"計算節點轉換: batch.x.shape={batch.x.shape}, node_transform.weight.shape={self.node_transform.weight.shape}")
            transformed_x = self.node_transform(batch.x)
            M = torch.where(M.sum(dim=1, keepdim=True) == 0, transformed_x, M)
            
            debug_log(f"組合最終嵌入...")
            final_features = torch.cat([batch.x, M], dim=1)
            debug_log(f"最終特徵: shape={final_features.shape}, W_o.weight.shape={self.W_o.weight.shape}")
            H = self.W_o(final_features)
            H = self.relu(H)    
            H = self.dropout(H)
            
            debug_log(f"BondMessagePassing.forward完成: 輸出H.shape={H.shape}")
            # 重置重試計數器
            self.retry_count = 0
            return H
            
        except RuntimeError as e:
            # 詳細的錯誤訊息和診斷
            debug_log(f"錯誤發生在前向傳播中：{str(e)}")
            
            try:
                # 嘗試取得更多診斷信息
                cat_shape = torch.cat([batch.x[batch.edge_index[0]], batch.edge_attr], dim=1).shape
                debug_log(f"診斷信息：")
                debug_log(f"  - 連接張量形狀: {cat_shape}")
                debug_log(f"  - W_i權重形狀: {self.W_i.weight.shape}")
                debug_log(f"  - 節點特徵: 實際={actual_node_dim}, 預期={self.W_o.weight.shape[1] - self.hidden_size}")
            except Exception as diag_e:
                debug_log(f"無法獲取完整診斷信息: {diag_e}")
            
            # 防止無限重試
            self.retry_count += 1
            if self.retry_count > MAX_RETRIES:
                debug_log(f"達到最大重試次數({MAX_RETRIES})，終止執行")
                raise RuntimeError(f"達到最大重試次數({MAX_RETRIES})，原始錯誤: {str(e)}")
            
            # 重新初始化並重試
            if "shapes cannot be multiplied" in str(e):
                debug_log(f"重試 #{self.retry_count}: 重新初始化層以適應新的特徵維度...")
                self.initialized = False
                self._initialize_layers(actual_node_dim, actual_edge_dim)
                # 重新嘗試前向傳播
                debug_log(f"重新嘗試前向傳播...")
                return self.forward(batch)
            else:
                raise e

class SequentialBondMessagePassing(nn.Module):
    """
    按解離順序進行多層消息傳遞的模型
    """
    def __init__(self, node_dim, bond_dim, hidden_dim, num_layers=2, dropout=0.3):
        super(SequentialBondMessagePassing, self).__init__()
        self.num_layers = num_layers
        
        # 創建多個消息傳遞層，每層對應一個解離階段
        self.mp_layers = nn.ModuleList([
            BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=3, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # 層間轉換，用於將一層的輸出轉換為下一層的輸入
        self.layer_transitions = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) 
            for _ in range(num_layers-1)
        ])
        
        debug_log(f"初始化SequentialBondMessagePassing: num_layers={num_layers}")
    
    def forward(self, batch):
        debug_log(f"進入SequentialBondMessagePassing.forward")
        
        # 檢查是否有解離順序信息
        has_dissociation_order = hasattr(batch, 'dissociation_order')
        dissociation_order = batch.dissociation_order if has_dissociation_order else None
        
        # 獲取批次中最大的解離順序
        max_order = self.num_layers - 1
        if has_dissociation_order:
            max_order = max(0, dissociation_order.max().item())
            max_order = min(max_order, self.num_layers - 1)  # 確保不超過層數
        
        debug_log(f"檢測到最大解離順序: {max_order}")
        
        # 第一層處理所有節點
        H = self.mp_layers[0](batch)
        
        # 後續層根據解離順序處理
        for i in range(1, min(self.num_layers, max_order + 2)):
            debug_log(f"處理第{i}層消息傳遞...")
            
            # 應用層間轉換
            H_next = self.layer_transitions[i-1](H)
            
            # 第i層處理解離順序 >= i 的節點
            if has_dissociation_order:
                # 創建掩碼，標記當前層需要處理的節點
                mask = dissociation_order >= i
                
                # 如果沒有需要處理的節點，可以提前退出
                if not mask.any():
                    debug_log(f"第{i}層沒有需要處理的節點，提前退出")
                    break
                
                # 處理這些節點
                H_updated = self.mp_layers[i](batch)
                
                # 只更新需要處理的節點
                H = torch.where(mask.unsqueeze(1), H_updated, H_next)
            else:
                # 如果沒有解離順序信息，處理所有節點
                H = self.mp_layers[i](batch)
        
        debug_log(f"SequentialBondMessagePassing.forward完成")
        return H

class PKAScaler:
    def __init__(self):
        self.fitted = False
        self.mean = 0.0
        self.std = 1.0
        # 無需使用 sklearn 的 StandardScaler，直接計算統計量
    
    def fit(self, pka_values):
        """使用真實pKa值擬合標準化器"""
        valid_values = [x for x in pka_values if np.isfinite(x)]
        if len(valid_values) > 0:
            # 直接計算均值和標準差
            self.mean = np.mean(valid_values)
            self.std = np.std(valid_values) if np.std(valid_values) > 0 else 1.0
            self.fitted = True
            debug_log(f"PKAScaler已擬合: 平均值={self.mean:.4f}, 標準差={self.std:.4f}")
            
            # 在標準化器擬合後添加
            debug_log(f"標準化前的pKa統計: 最小={min(valid_values):.2f}, 最大={max(valid_values):.2f}")
            
            # 在標準化後檢查標準化值的分布
            std_values = [(x - self.mean) / self.std for x in valid_values]
            debug_log(f"標準化後的統計: 平均={np.mean(std_values):.2f}, 標準差={np.std(std_values):.2f}")
    
    def transform(self, pka_value):
        """將單個pKa值標準化"""
        if not self.fitted:
            return pka_value
        if not np.isfinite(pka_value):
            return 0.0
        return (pka_value - self.mean) / self.std
    
    def inverse_transform(self, scaled_value):
        """將標準化的值轉換回原始尺度"""
        if not self.fitted:
            return scaled_value
        return scaled_value * self.std + self.mean

class PKA_GNN(nn.Module):
    def __init__(self, node_dim, bond_dim, hidden_dim, max_dissociation_steps=2, dropout=0.1):
        super(PKA_GNN, self).__init__()
        debug_log(f"初始化PKA_GNN: node_dim={node_dim}, bond_dim={bond_dim}, hidden_dim={hidden_dim}, max_dissociation_steps={max_dissociation_steps}")
        
        # 使用順序性消息傳遞層
        self.sequential_mp = SequentialBondMessagePassing(
            node_dim, bond_dim, hidden_dim, num_layers=max_dissociation_steps, dropout=dropout)
        
        self.pool = global_mean_pool
        
        # 分類任務 - 判斷原子是否可解離
        self.dissociable_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2))  # [0, 1] - 不可解離/可解離
        
        # 回歸任務 - 預測pKa值，添加解離順序的考慮
        self.pka_regressor = nn.Sequential(
            nn.Linear(hidden_dim + 1, 256),  # 增加1個特徵用於解離順序
            nn.ReLU(),
            nn.Dropout(dropout * 1.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 解離順序預測器 - 新增
        self.order_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, max_dissociation_steps)  # 預測最多max_dissociation_steps階段的解離
        )
        
        # 損失函數
        self.criterion_cls = nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss()
        self.criterion_order = nn.CrossEntropyLoss()  # 新增解離順序損失
        
        # 門控RNN
        self.rnn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid())
        
        # 記錄最大解離階段
        self.max_dissociation_steps = max_dissociation_steps
        
        # 添加pKa值標準化器
        self.pka_scaler = PKAScaler()
        
        # 加大初始化權重範圍，使模型開始時就能產生多樣化預測
        self._init_weights()
        
        debug_log("PKA_GNN初始化完成")
    
    def _init_weights(self):
        """初始化權重，使模型開始時有更大的輸出範圍"""
        for m in self.pka_regressor.modules():
            if isinstance(m, nn.Linear):
                # 使用更大的標準差初始化
                nn.init.normal_(m.weight, mean=0.0, std=0.2)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    # 獲取反向邊緣索引
    @staticmethod
    def _rev_edge_index(edge_index):
        debug_log(f"計算反向邊緣索引: edge_index.shape={edge_index.shape}")
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i 
                         for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        debug_log(f"反向邊緣索引計算完成")
        return rev_edge_index

    # 圖的前向傳播
    def forward_graph(self, x, edge_index, batch, edge_attr, dissociation_order=None):
        debug_log(f"進入forward_graph: x.shape={x.shape}, edge_index.shape={edge_index.shape}, edge_attr.shape={edge_attr.shape}")
        # 獲取反向邊緣索引
        rev_edge_index = self._rev_edge_index(edge_index)
        
        # 構建圖
        graph_batch = Data(x=x,
                          edge_index=edge_index,
                          rev_edge_index=rev_edge_index,
                          edge_attr=edge_attr)
        
        # 如果有解離順序信息，添加到圖中
        if dissociation_order is not None:
            graph_batch.dissociation_order = dissociation_order
        
        debug_log("開始調用SequentialBondMessagePassing...")
        # 進行前向傳播，考慮解離順序
        node_embeddings = self.sequential_mp(graph_batch)
        debug_log(f"forward_graph完成: node_embeddings.shape={node_embeddings.shape}")
        
        # 返回節點嵌入
        return node_embeddings

    # 主函數
    def forward(self, batch):
        debug_log("進入PKA_GNN.forward")
        # 從批次中提取數據
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        dissociable_masks = batch.dissociable_masks if hasattr(batch, 'dissociable_masks') else None
        pka_values = batch.pka_values if hasattr(batch, 'pka_values') else None
        batch_idx = batch.batch if hasattr(batch, 'batch') else None
        
        # 解離順序信息
        dissociation_order = batch.dissociation_order if hasattr(batch, 'dissociation_order') else None
        current_dissociation = batch.current_dissociation if hasattr(batch, 'current_dissociation') else None
        
        # 統計和擬合pKa標準化器（僅在訓練階段第一次執行）
        if pka_values is not None and not self.pka_scaler.fitted:
            valid_pka_values = [pka.item() for pka in pka_values if not torch.isnan(pka)]
            self.pka_scaler.fit(valid_pka_values)
            debug_log(f"已擬合pKa標準化器: 平均值={self.pka_scaler.mean:.4f}, 標準差={self.pka_scaler.std:.4f}")
        
        # pKa值分析
        if pka_values is not None:
            valid_pka = pka_values[~torch.isnan(pka_values)]
            debug_log(f"真實pKa值統計: 最小={valid_pka.min().item():.2f}, 最大={valid_pka.max().item():.2f}, 平均={valid_pka.mean().item():.2f}, 數量={len(valid_pka)}")
        
        # 處理圖，獲取節點嵌入，傳遞解離順序
        debug_log("開始處理圖...")
        node_embeddings = self.forward_graph(x, edge_index, batch_idx, edge_attr, dissociation_order)
        
        # 初始化損失
        total_cls_loss = torch.tensor(0.0, device=node_embeddings.device)
        total_reg_loss = torch.tensor(0.0, device=node_embeddings.device)
        total_order_loss = torch.tensor(0.0, device=node_embeddings.device)
        correct_predictions = 0
        total_dissociable_atoms = 0
        
        # 記錄預測結果
        all_cls_predictions = []
        all_reg_predictions = []
        all_order_predictions = []
        
        # 收集pKa預測和真實值用於診斷
        all_pred_pkas = []
        all_true_pkas = []
        all_scaled_pred_pkas = []
        all_scaled_true_pkas = []
        all_reg_losses = []
        
        # 遍歷每個節點
        debug_log(f"開始遍歷節點進行預測，共{len(x)}個節點")
        progress_step = max(1, len(x) // 10)  # 每10%報告一次進度
        
        for i in range(len(x)):
            # 定期報告進度
            if i % progress_step == 0:
                debug_log(f"處理節點 {i}/{len(x)} ({i/len(x)*100:.1f}%)")
            
            # 進行分類任務 - 判斷是否可解離
            cls_pred = self.dissociable_classifier(node_embeddings[i].unsqueeze(0))
            cls_pred_label = torch.argmax(cls_pred, dim=1).item()
            all_cls_predictions.append(cls_pred_label)
            
            # 如果有標籤信息
            if dissociable_masks is not None:
                target_cls = 1 if dissociable_masks[i].item() > 0 else 0
                cls_loss = self.criterion_cls(cls_pred, torch.tensor([target_cls], device=cls_pred.device))
                total_cls_loss += cls_loss
                
                # 添加分類損失調試
                if i % 500 == 0:
                    prob_0 = torch.softmax(cls_pred, dim=1)[0, 0].item()
                    prob_1 = torch.softmax(cls_pred, dim=1)[0, 1].item()
                    debug_log(f"節點 {i} 分類: 真實={target_cls}, 預測={cls_pred_label}, 概率=[{prob_0:.4f}, {prob_1:.4f}], 損失={cls_loss.item():.4f}")
                
                # 計算分類準確率
                if cls_pred_label == target_cls:
                    correct_predictions += 1
                total_dissociable_atoms += 1
            
            # 如果預測為可解離，則預測pKa值和解離順序
            if cls_pred_label == 1 or (dissociable_masks is not None and dissociable_masks[i].item() > 0):
                # 預測解離順序
                if dissociation_order is not None:
                    order_pred = self.order_predictor(node_embeddings[i].unsqueeze(0))
                    pred_order = torch.argmax(order_pred, dim=1).item()
                    all_order_predictions.append((i, pred_order))
                    
                    # 如果有真實的解離順序
                    true_order = dissociation_order[i].item()
                    if true_order >= 0 and true_order < self.max_dissociation_steps:
                        order_loss = self.criterion_order(order_pred, torch.tensor([true_order], device=order_pred.device))
                        total_order_loss += order_loss
                else:
                    # 如果沒有解離順序信息，默認為0
                    pred_order = 0
                
                # 將解離順序作為回歸輸入的一部分
                order_feature = torch.tensor([[pred_order]], dtype=torch.float32, device=node_embeddings.device)
                regression_input = torch.cat([node_embeddings[i].unsqueeze(0), order_feature / self.max_dissociation_steps], dim=1)
                
                # 預測pKa值（標準化尺度）
                pka_pred = self.pka_regressor(regression_input)
                
                # 轉換回原始尺度
                scaled_pred_pka = pka_pred.item()  # 標準化後的預測
                pred_pka = self.pka_scaler.inverse_transform(scaled_pred_pka)  # 轉換回原始尺度
                
                all_reg_predictions.append((i, pred_pka))
                
                # 如果有真實pKa值
                if pka_values is not None and i < len(pka_values) and not torch.isnan(pka_values[i]):
                    true_pka = pka_values[i].item()
                    
                    # 標準化真實pKa值
                    scaled_true_pka = self.pka_scaler.transform(true_pka)
                    
                    # 收集用於診斷的值
                    all_pred_pkas.append(pred_pka)  # 原始尺度的預測
                    all_true_pkas.append(true_pka)  # 原始尺度的真實值
                    all_scaled_pred_pkas.append(scaled_pred_pka)  # 標準化後的預測
                    all_scaled_true_pkas.append(scaled_true_pka)  # 標準化後的真實值
                    
                    # 在標準化尺度上計算損失
                    reg_loss = self.criterion_reg(pka_pred, torch.tensor([[scaled_true_pka]], device=pka_pred.device))
                    all_reg_losses.append(reg_loss.item())
                    total_reg_loss += reg_loss
                    
                    # 添加回歸損失調試
                    if i % 100 == 0 or reg_loss.item() > 10:
                        debug_log(f"節點 {i} 回歸: 真實pKa={true_pka:.2f}(標準化:{scaled_true_pka:.2f}), " + 
                                 f"預測pKa={pred_pka:.2f}(標準化:{scaled_pred_pka:.2f}), 損失={reg_loss.item():.4f}")
                    
                    # 更新節點嵌入，使用門控機制
                    node_embeddings = node_embeddings * (1 - self.rnn_gate(node_embeddings))
        
        # 統計pKa預測結果
        if all_pred_pkas and all_true_pkas:
            # 原始尺度的統計
            pred_mean = sum(all_pred_pkas) / len(all_pred_pkas)
            true_mean = sum(all_true_pkas) / len(all_true_pkas)
            pred_min, pred_max = min(all_pred_pkas), max(all_pred_pkas)
            true_min, true_max = min(all_true_pkas), max(all_true_pkas)
            
            # 標準化尺度的統計
            scaled_pred_mean = sum(all_scaled_pred_pkas) / len(all_scaled_pred_pkas)
            scaled_true_mean = sum(all_scaled_true_pkas) / len(all_scaled_true_pkas)
            
            # 計算損失分布情況
            if all_reg_losses:
                loss_mean = sum(all_reg_losses) / len(all_reg_losses)
                loss_min, loss_max = min(all_reg_losses), max(all_reg_losses)
                high_losses = sum(1 for l in all_reg_losses if l > 10)
                debug_log(f"回歸損失統計: 平均={loss_mean:.2f}, 最小={loss_min:.2f}, 最大={loss_max:.2f}, 高損失比例={high_losses}/{len(all_reg_losses)}")
            
            debug_log(f"pKa預測統計: 樣本數={len(all_pred_pkas)}")
            debug_log(f"  原始預測值: 平均={pred_mean:.2f}, 範圍=[{pred_min:.2f}, {pred_max:.2f}]")
            debug_log(f"  原始真實值: 平均={true_mean:.2f}, 範圍=[{true_min:.2f}, {true_max:.2f}]")
            debug_log(f"  標準化預測值: 平均={scaled_pred_mean:.2f}")
            debug_log(f"  標準化真實值: 平均={scaled_true_mean:.2f}")
        
        # 計算平均損失
        if total_dissociable_atoms > 0:
            total_cls_loss = total_cls_loss / total_dissociable_atoms
        
        if len(all_reg_losses) > 0:
            # 使用 PyTorch 的 mean 函數而不是手動計算平均值
            total_reg_loss = torch.tensor(all_reg_losses, device=node_embeddings.device).mean()
        
        # 計算總損失
        total_loss = total_cls_loss + total_reg_loss
        if total_order_loss > 0:
            total_loss += 0.5 * total_order_loss
        
        # 計算分類準確率
        accuracy = correct_predictions / total_dissociable_atoms if total_dissociable_atoms > 0 else 0
        
        # 計算回歸準確率（使用相對誤差）
        reg_accuracy = 0
        if all_pred_pkas and all_true_pkas:
            # 計算相對誤差在10%以內的預測比例作為回歸準確率
            correct_predictions = sum(1 for pred, true in zip(all_pred_pkas, all_true_pkas)
                                    if abs(pred - true) / (abs(true) + 1e-6) <= 0.1)  # 允許10%的相對誤差
            reg_accuracy = correct_predictions / len(all_pred_pkas)
            debug_log(f"回歸準確率（相對誤差<=10%）: {reg_accuracy:.4f}")
        
        # 修改調試日誌，處理 float 對象
        debug_log(f"PKA_GNN.forward完成: 分類準確率={accuracy:.4f}, 分類損失={total_cls_loss:.4f}, 回歸損失={total_reg_loss:.4f}, 總損失={total_loss:.4f}")
        
        return all_cls_predictions, all_reg_predictions, (total_loss, total_cls_loss, total_reg_loss, accuracy, reg_accuracy)

    def predict(self, batch):
        debug_log("進入PKA_GNN.predict")
        # 預測模式，與forward類似但不計算損失
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        batch_idx = batch.batch if hasattr(batch, 'batch') else None
        
        # 解離順序信息
        dissociation_order = batch.dissociation_order if hasattr(batch, 'dissociation_order') else None
        
        # 處理圖，獲取節點嵌入
        debug_log("開始圖嵌入計算...")
        node_embeddings = self.forward_graph(x, edge_index, batch_idx, edge_attr, dissociation_order)
        
        # 記錄預測結果
        dissociable_nodes = []
        pka_predictions = []
        order_predictions = []
        
        # 遍歷每個節點
        debug_log(f"開始遍歷節點進行預測，共{len(x)}個節點")
        for i in range(len(x)):
            if i % 100 == 0:
                debug_log(f"處理節點 {i}/{len(x)} ({i/len(x)*100:.1f}%)")
                
            # 進行分類任務 - 判斷是否可解離
            cls_pred = self.dissociable_classifier(node_embeddings[i].unsqueeze(0))
            cls_pred_label = torch.argmax(cls_pred, dim=1).item()
            
            # 如果預測為可解離，則預測pKa值和解離順序
            if cls_pred_label == 1:
                dissociable_nodes.append(i)
                
                # 預測解離順序
                pred_order = 0
                if dissociation_order is not None:
                    order_pred = self.order_predictor(node_embeddings[i].unsqueeze(0))
                    pred_order = torch.argmax(order_pred, dim=1).item()
                order_predictions.append(pred_order)
                
                # 將解離順序作為回歸輸入的一部分
                order_feature = torch.tensor([[pred_order]], dtype=torch.float32, device=node_embeddings.device)
                regression_input = torch.cat([node_embeddings[i].unsqueeze(0), order_feature / self.max_dissociation_steps], dim=1)
                
                # 預測pKa值（標準化尺度），然後轉換回原始尺度
                scaled_pka_pred = self.pka_regressor(regression_input).item()
                pka_pred = self.pka_scaler.inverse_transform(scaled_pka_pred)
                
                pka_predictions.append(pka_pred)
                
                # 更新節點嵌入，使用門控機制
                node_embeddings = node_embeddings * (1 - self.rnn_gate(node_embeddings))
        
        debug_log(f"預測完成: 識別出{len(dissociable_nodes)}個可解離節點")
        return dissociable_nodes, pka_predictions, order_predictions 