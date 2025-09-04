#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
數據分割腳本
=============

將金屬pKa數據集分割為訓練、驗證和測試集，並轉換為模型所需的格式。

用法:
    python 3_datasplita.py
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
import shutil
import random
import traceback
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# 添加父目錄到路徑
sys.path.append("../src/")
try:
    from metal_chemutils import tensorize_for_pka
    print("成功導入 metal_chemutils 模塊")
except ImportError as e:
    print(f"導入 metal_chemutils 模塊時出錯: {e}")
    print("請確保 ../src/ 目錄下存在 metal_chemutils.py 文件")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='數據分割')
    parser.add_argument('--csv_path', type=str, default='../data/pre_metal_t15_T25_I0.1_E3_m11.csv',
                        help='CSV數據文件路徑')
    parser.add_argument('--output_dir', type=str, default='../data/metal_pka_example',
                        help='輸出目錄')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='訓練集比例')
    parser.add_argument('--valid_ratio', type=float, default=0.15,
                        help='驗證集比例')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='測試集比例')
    parser.add_argument('--seed', type=int, default=33,
                        help='隨機種子')
    parser.add_argument('--debug', action='store_true',
                        help='啟用調試模式')
    return parser.parse_args()

def set_seed(seed):
    """設置隨機種子以確保可重複性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"隨機種子已設置為: {seed}")

def create_directories(output_dir):
    """創建必要的目錄結構"""
    # 創建主輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    print(f"創建主輸出目錄: {output_dir}")
    
    # 創建訓練/驗證/測試子目錄
    train_dir = os.path.join(output_dir, 'train')
    valid_dir = os.path.join(output_dir, 'valid')
    test_dir = os.path.join(output_dir, 'test')
    
    # 清空目錄（如果已存在）
    for dir_path in [train_dir, valid_dir, test_dir]:
        if os.path.exists(dir_path):
            print(f"清空現有目錄: {dir_path}")
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)
        print(f"創建目錄: {dir_path}")
    
    return train_dir, valid_dir, test_dir

def load_data(csv_path):
    """載入CSV數據"""
    try:
        print(f"嘗試載入數據: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"成功載入數據: {csv_path}")
        print(f"數據形狀: {df.shape}")
        
        # 顯示列名
        print("數據列名:", df.columns.tolist())
        
        # 顯示前幾行數據
        print("\n數據預覽:")
        print(df.head())
        
        # 檢查是否有必要的列
        required_cols = ['SMILES', 'pKa_value', 'metal_ion']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"警告: 缺少以下列: {missing_cols}")
            
        return df
    except Exception as e:
        print(f"載入數據時出錯: {e}")
        traceback.print_exc()
        return None

def process_molecule(smiles, pka_values=None, metal_ion=None, debug=False):
    """處理分子並創建PyG數據對象"""
    try:
        if debug:
            print(f"處理分子: {smiles}, pKa值: {pka_values}, 金屬離子: {metal_ion}")
            
        # 使用RDKit解析SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"無法解析SMILES: {smiles}")
            return None
            
        # 用tensorize_for_pka處理分子
        try:
            fatoms, edge_index, edge_attr = tensorize_for_pka(smiles)
            if debug:
                print(f"分子特徵: 原子數量={fatoms.size(0)}, 邊數量={edge_index.size(1)}")
        except Exception as e:
            print(f"張量化分子時出錯: {e}")
            traceback.print_exc()
            return None
            
        num_atoms = fatoms.size(0)
        
        # 創建pKa標簽
        pka_labels = torch.zeros(num_atoms)
        
        # 由於CSV中沒有原子索引信息，我們將使用第一個原子作為pKa位點
        # 實際應用中，這應該根據化學結構或其他規則來確定
        if pka_values and len(pka_values) > 0:
            try:
                # pKa_value是字符串形式的列表，例如 "[20.45]"
                if isinstance(pka_values, str):
                    if debug:
                        print(f"解析pKa值: {pka_values}")
                    pka_values = eval(pka_values)  # 將字符串轉換為Python列表
                
                # 確保我們有有效的值
                if isinstance(pka_values, list) and len(pka_values) > 0:
                    # 為簡單起見，我們只使用第一個pKa值，並分配給第一個合適的原子
                    pka_value = float(pka_values[0])
                    
                    # 尋找合適的原子作為pKa中心（這裡簡單地選擇第一個非碳原子，如果沒有則選擇第一個原子）
                    pka_atom_idx = 0
                    
                    # 在實際應用中，這應該是基於化學知識選擇的原子
                    for atom_idx in range(mol.GetNumAtoms()):
                        atom = mol.GetAtomWithIdx(atom_idx)
                        if atom.GetSymbol() in ['N', 'O', 'S', 'P']:  # 優先選擇這些原子作為pKa位點
                            pka_atom_idx = atom_idx
                            break
                    
                    # 設置pKa標簽
                    if pka_atom_idx < num_atoms:
                        pka_labels[pka_atom_idx] = pka_value
                        if debug:
                            print(f"設置pKa: 原子索引={pka_atom_idx}, 值={pka_value}")
                    else:
                        print(f"警告: 原子索引 {pka_atom_idx} 超出範圍 (總原子數: {num_atoms})")
            except Exception as e:
                print(f"處理pKa值時出錯: {e}, pka_values={pka_values}")
                traceback.print_exc()
        
        # 創建PyG數據對象
        data = Data(
            x=fatoms,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pka_labels=pka_labels,
            smiles=smiles
        )
        
        # 添加金屬離子信息（如果提供）
        if metal_ion is not None:
            data.metal_ion = metal_ion
            
        if debug:
            print(f"創建PyG數據對象: {data}")
            
        return data
    except Exception as e:
        print(f"處理分子時出錯: {e}")
        traceback.print_exc()
        return None

def split_and_save_data(df, train_dir, valid_dir, test_dir, train_ratio, valid_ratio, test_ratio, seed, debug=False):
    """分割數據並保存到相應目錄"""
    # 確保比例總和為1
    total_ratio = train_ratio + valid_ratio + test_ratio
    train_ratio = train_ratio / total_ratio
    valid_ratio = valid_ratio / total_ratio
    test_ratio = test_ratio / total_ratio
    
    # 分割數據
    train_df, temp_df = train_test_split(df, train_size=train_ratio, random_state=seed)
    # 調整驗證集比例
    valid_ratio_adjusted = valid_ratio / (valid_ratio + test_ratio)
    valid_df, test_df = train_test_split(temp_df, train_size=valid_ratio_adjusted, random_state=seed)
    
    print(f"數據分割完成: 訓練集 {len(train_df)}筆, 驗證集 {len(valid_df)}筆, 測試集 {len(test_df)}筆")
    
    # 定義數據處理與保存函數
    def process_and_save(subset_df, output_dir, subset_name):
        print(f"處理{subset_name}集...")
        successful_count = 0
        for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc=f"{subset_name}集進度"):
            try:
                # 從行中提取信息
                smiles = row.get('SMILES')
                pka_values = row.get('pKa_value')
                metal_ion = row.get('metal_ion')
                
                if debug:
                    print(f"處理數據 {idx}: SMILES={smiles}, pKa={pka_values}, 金屬={metal_ion}")
                
                if smiles and pka_values:
                    # 處理分子並創建數據對象
                    data = process_molecule(smiles, pka_values, metal_ion, debug)
                    if data is not None:
                        # 生成文件名
                        output_path = os.path.join(output_dir, f"data_{idx}.pt")
                        # 保存數據對象
                        torch.save(data, output_path)
                        successful_count += 1
                        if debug:
                            print(f"保存數據到: {output_path}")
                else:
                    if debug:
                        print(f"跳過數據 {idx}: 缺少SMILES或pKa值")
            except Exception as e:
                print(f"處理數據 {idx} 時出錯: {e}")
                if debug:
                    traceback.print_exc()
                continue
        
        print(f"{subset_name}集處理完成: {successful_count}/{len(subset_df)} 筆數據成功處理")
        return successful_count
    
    # 處理並保存每個子集
    train_success = process_and_save(train_df, train_dir, "訓練")
    valid_success = process_and_save(valid_df, valid_dir, "驗證")
    test_success = process_and_save(test_df, test_dir, "測試")
    
    # 檢查生成的數據集大小
    train_files = len([f for f in os.listdir(train_dir) if f.endswith('.pt')])
    valid_files = len([f for f in os.listdir(valid_dir) if f.endswith('.pt')])
    test_files = len([f for f in os.listdir(test_dir) if f.endswith('.pt')])
    
    # 檢查是否數量匹配
    if train_files != train_success or valid_files != valid_success or test_files != test_success:
        print("警告: 檔案數量與成功處理數量不匹配")
        print(f"訓練集: 成功處理={train_success}，檔案數量={train_files}")
        print(f"驗證集: 成功處理={valid_success}，檔案數量={valid_files}")
        print(f"測試集: 成功處理={test_success}，檔案數量={test_files}")
    
    print(f"數據處理完成: 訓練集 {train_files}個文件, 驗證集 {valid_files}個文件, 測試集 {test_files}個文件")
    return train_files, valid_files, test_files

def main():
    # 解析命令行參數
    args = parse_args()
    debug = args.debug
    
    print("=" * 80)
    print("金屬pKa數據分割工具")
    print("=" * 80)
    
    # 設置隨機種子
    set_seed(args.seed)
    
    # 創建輸出目錄
    train_dir, valid_dir, test_dir = create_directories(args.output_dir)
    
    # 載入數據
    print("開始載入數據...")
    df = load_data(args.csv_path)
    if df is None:
        print("載入數據失敗，退出程序")
        return
    
    # 確認數據集有足夠的列
    if 'SMILES' not in df.columns and 'smiles' not in df.columns:
        print("錯誤: 數據集缺少SMILES列")
        return
    
    # 分割並保存數據
    print("開始分割並處理數據...")
    train_files, valid_files, test_files = split_and_save_data(
        df, train_dir, valid_dir, test_dir,
        args.train_ratio, args.valid_ratio, args.test_ratio,
        args.seed, debug
    )
    
    # 打印結果摘要
    print("\n" + "=" * 80)
    print("數據處理摘要:")
    print("=" * 80)
    print(f"輸入CSV: {args.csv_path}")
    print(f"輸出目錄: {args.output_dir}")
    print(f"訓練集: {train_files}個文件 ({args.train_ratio*100:.1f}%)")
    print(f"驗證集: {valid_files}個文件 ({args.valid_ratio*100:.1f}%)")
    print(f"測試集: {test_files}個文件 ({args.test_ratio*100:.1f}%)")
    print("=" * 80)
    print(f"數據分割完成！現在可以使用 4_metal_pka_transfer.py 進行模型訓練。")
    print("=" * 80)

if __name__ == '__main__':
    main() 