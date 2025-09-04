import os
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch    import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch_geometric.data     import Data
from torch_geometric.loader   import DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from optparse  import OptionParser

import sys
sys.path.append("/work/s6300121/LiveTransForM-main/metal")        
from src.metal_chemutils import *

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
    config['output_size']  = int(config['output_size'])
    config['batch_size']   = int(config['batch_size'])
    config['num_epochs']   = int(config['num_epochs'])
    config['dropout']      = float(config['dropout'])
    config['weight_decay'] = float(config['weight_decay'])
    config['depth']        = int(config['depth'])
    
    return config




class pka_Dataloader():
    def __init__(self):
        pass

    def data_loader_pka(file_path, columns, tensorize_fn, batch_size, test_size=0.2):
        """
        載入已標記的pKa數據用於GCN模型訓練
        
        Args:
            file_path: CSV檔案路徑，包含SMILES和pKa標記資訊
            columns: 需要的CSV欄位清單
            tensorize_fn: 將SMILES轉換為圖形的函數
            batch_size: 批次大小
            test_size: 測試集大小比例
            
        Returns:
            訓練資料集和測試資料載入器
        """
        df = pd.read_csv(file_path)
        df = df[columns]
        
        # 分割訓練集和測試集
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        
        def tensorize_dataset(data):
            dataset = []
            for _, row in data.iterrows():
                try:
                    # 取得分子SMILES和pKa矩陣信息
                    smiles = row["smiles"]
                    # smiles = "Nc1ccncc1[N+]([O-])=O"
                    # pka_matrix_str = "[(4, 5.05)]"
                    # mol_tensor = tensorize_for_pka(smiles)
                    # fatoms, graphs, edge_features = mol_tensor[0], mol_tensor[1], mol_tensor[2]
                    
                    # 將pKa矩陣從字符串轉換回列表格式
                    # 格式如：[(0, 9.16), (4, 10.82)]
                    pka_matrix_str = row["pka_matrix"]
                    pka_matrix = eval(pka_matrix_str)
                    
                    # 使用tensorize_fn將SMILES轉換為圖形表示
                    mol_tensor = tensorize_fn([smiles])
                    fatoms, graphs, edge_features = mol_tensor[0], mol_tensor[1], mol_tensor[2]
                    
                    # 創建原子級別的pKa標籤
                    # 初始化所有原子的pKa標籤為0
                    num_atoms = fatoms.size(0)
                    atom_pka_labels = torch.zeros(num_atoms)
                    
                    # 對有pKa值的原子進行標記
                    for atom_idx, pka_value in pka_matrix:
                        atom_pka_labels[atom_idx] = pka_value
                    
                    # 計算pKa值的數量
                    pka_count = len(pka_matrix)
                    
                    metal_ion = row['metal_ion']
                    # 創建Data對象
                    data_item = Data(
                        x=fatoms,
                        edge_index=graphs,
                        edge_attr=edge_features, 
                        pka_labels=atom_pka_labels,  # 每個原子的pKa標籤
                        pka_count=pka_count,         # pKa值的數量
                        metal_ion=metal_ion,         # 金屬離子
                        smiles=smiles,
                    )
                    
                    dataset.append(data_item)
                except Exception as e:
                    print(f"處理資料時出錯: {e} (SMILES: {smiles})")
                    continue
                    
            return dataset
        
        # 創建訓練集和測試集
        train_dataset = tensorize_dataset(train_data)
        test_dataset = tensorize_dataset(test_data)
        
        # 創建資料加載器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    @staticmethod
    def evaluate_pka_model(model, loader, device, output_file="", save_path=""):
        """
        回傳:  avg_loss, avg_cla_loss, avg_reg_loss, metrics(dict)
        metrics 內含
            accuracy / precision / recall / f1 / pka_atom_accuracy
            rmse_gt   – 所有真實 pKa 原子
            rmse_hit  – 真實且模型也預測為正的原子
        """
        model.eval()
        tot_loss = tot_c = tot_r = n_batch = 0

        y_true, y_pred = [], []            # 分類
        tgt_all, pred_all = [], []         # 所有 gt pKa
        tgt_hit, pred_hit = [], []         # 命中才算

        results_data = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                
                # 處理金屬離子特徵，與訓練時保持一致
                if hasattr(batch, 'metal_ion'):
                    # 為每個分子創建金屬特徵張量
                    batch_size = len(batch.metal_ion)
                    # 從metal_models引入
                    from src.metal_models import MetalFeatureExtractor
                    
                    # 創建金屬特徵列表，用於批次處理
                    metal_features_list = []
                    
                    # 使用新的方法獲取金屬特徵
                    for ion in batch.metal_ion:
                        try:
                            # 使用新的特徵提取方法，直接獲取處理好的特徵
                            metal_feat = MetalFeatureExtractor.get_metal_features_for_training(ion, device)
                            metal_features_list.append([metal_feat, None, None])  # 保持與返回元組相同的結構
                        except Exception as e:
                            print(f"獲取金屬特徵時出錯: {e}")
                            # 使用默認特徵
                            metal_feat_dim = MetalFeatureExtractor.get_metal_feature_dim()
                            metal_feat = torch.zeros(metal_feat_dim, device=device)
                            metal_features_list.append([metal_feat, None, None])
                    
                    # 將金屬特徵列表添加到batch
                    batch.metal_features = metal_features_list
                
                logits, pka_raw, (loss, loss_c, loss_r) = model(batch)

                # ------ 累積損失 ------
                tot_loss += loss.item(); tot_c += loss_c.item(); tot_r += loss_r.item(); n_batch += 1

                # ------ 分類評估 ------
                gt_mask   = (batch.pka_labels > 0)
                pred_mask = logits.argmax(1) == 1

                y_true.append(gt_mask.cpu())
                y_pred.append(pred_mask.cpu())

                # ------ 回歸評估 ------
                # gt pKa 全部
                tgt_all.append(batch.pka_labels[gt_mask].cpu())
                pred_all.append(pka_raw[gt_mask].cpu())

                # 🎯 "命中（hit）" = 同時被正確分類為有 pKa 並且該原子的真實 label 是有 pKa 的
                hit_mask = gt_mask & pred_mask
                if hit_mask.any():
                    tgt_hit.append(batch.pka_labels[hit_mask].cpu())
                    pred_hit.append(pka_raw[hit_mask].cpu())

                # ------ (可選) 保存 per-molecule 結果 ------
                if output_file:
                    smiles = batch.smiles
                    idx_gt = torch.nonzero(gt_mask).squeeze(1).cpu().tolist()
                    true_v = batch.pka_labels[gt_mask].cpu().tolist()
                    pred_v = pka_raw[gt_mask].cpu().tolist()
                    results_data.append({
                        "smiles": smiles,
                        "gt_idx": idx_gt,
                        "gt_pka": true_v,
                        "pred_pka": pred_v
                    })

        # --------- 聚合 ---------
        avg_loss = tot_loss / n_batch
        avg_cla  = tot_c    / n_batch
        avg_reg  = tot_r    / n_batch

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        acc     = accuracy_score(y_true, y_pred)
        prec    = precision_score(y_true, y_pred, zero_division=0)
        rec     = recall_score(y_true, y_pred, zero_division=0)
        f1      = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        pka_atom_acc = rec   # ＝ TP / (TP+FN)

        # 回歸 RMSE
        import numpy as np, math
        rmse_gt  = math.sqrt(np.mean((torch.cat(pred_all).numpy() -
                                    torch.cat(tgt_all ).numpy())**2)) if tgt_all else 0.0
        rmse_hit = (math.sqrt(np.mean((torch.cat(pred_hit).numpy() -
                                    torch.cat(tgt_hit).numpy())**2))
                    if tgt_hit else 0.0)

        print(f"預測準確度: {rec:.3f}  |  "
            f"RMSE(gt)={rmse_gt:.3f}  RMSE(hit)={rmse_hit:.3f}")
        metrics = dict(
            accuracy=acc, precision=prec, recall=rec, f1=f1,
            pka_atom_accuracy=pka_atom_acc,
            rmse_gt=rmse_gt, rmse_hit=rmse_hit,
            confusion_matrix=dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
        )

        # --------- 儲存 CSV ---------
        if output_file and results_data:
            os.makedirs(save_path, exist_ok=True)
            df = pd.DataFrame(results_data)
            df.to_csv(os.path.join(save_path, f"pka-{output_file}"), index=False)
            print(f"結果已存 {os.path.join(save_path, f'pka-{output_file}')}")

        return avg_loss, avg_cla, avg_reg, metrics

