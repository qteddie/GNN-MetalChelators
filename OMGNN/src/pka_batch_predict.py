#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pka_batch_predict.py

此腳本用於批量預測多個分子的pKa值。可以從CSV文件讀取SMILES字符串，預測pKa值並將結果保存到CSV文件。
也可以選擇性地生成分子的可視化圖像。
"""

import os
import sys
import argparse
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from multiprocessing import Pool, cpu_count

# 添加路徑以導入本地模塊
sys.path.append('src/')

# 導入本地模塊
from OMGNN.src.error_self_pka_models import pka_GNN
from pka_predict_sample import load_pka_model, visualize_pka_result

def ensure_dir(directory):
    """確保目錄存在，如果不存在則創建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_smiles(args):
    """
    處理單個SMILES，在多進程環境中使用
    
    Args:
        args: 包含(model, smiles, device, viz_dir, idx)的元組
        
    Returns:
        預測結果字典
    """
    model, smiles, device, viz_dir, idx = args
    
    try:
        # 進行預測
        result = model.sample(smiles, device)
        
        # 如果需要可視化並且預測成功
        if viz_dir and result:
            # 創建可視化文件名，使用索引確保唯一性
            output_path = os.path.join(viz_dir, f"mol_{idx}.png")
            # 生成可視化圖像，但不顯示
            visualize_pka_result(result, output_path=output_path, show=False)
        
        return result
    except Exception as e:
        print(f"處理SMILES時出錯 ({smiles}): {e}")
        return None

def batch_predict_pka(model, smiles_list, device=None, viz_dir=None, parallel=True, n_workers=None):
    """
    批量預測多個SMILES字符串的pKa值
    
    Args:
        model: 預訓練的pKa_GNN模型
        smiles_list: SMILES字符串列表
        device: 計算設備
        viz_dir: 可選，保存可視化結果的目錄
        parallel: 是否使用並行處理
        n_workers: 並行處理的工作者數量
        
    Returns:
        結果列表，每個元素是預測結果字典
    """
    if viz_dir:
        ensure_dir(viz_dir)
    
    if parallel and len(smiles_list) > 1:
        # 使用多進程加速批量處理
        if n_workers is None:
            n_workers = min(cpu_count(), len(smiles_list))
        
        # 準備參數
        args_list = [(model, smiles, device, viz_dir, i) 
                     for i, smiles in enumerate(smiles_list)]
        
        # 使用進程池並行處理
        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(process_smiles, args_list),
                total=len(args_list),
                desc="批量處理SMILES"
            ))
        
    else:
        # 順序處理
        results = []
        for i, smiles in enumerate(tqdm(smiles_list, desc="預測pKa")):
            try:
                # 進行預測
                result = model.sample(smiles, device)
                
                # 如果需要可視化並且預測成功
                if viz_dir and result:
                    output_path = os.path.join(viz_dir, f"mol_{i}.png")
                    visualize_pka_result(result, output_path=output_path, show=False)
                
                results.append(result)
            except Exception as e:
                print(f"處理SMILES時出錯 ({smiles}): {e}")
                results.append(None)
    
    return results

def format_pka_results_for_csv(results, smiles_list):
    """
    將pKa預測結果格式化為可保存到CSV的DataFrame
    
    Args:
        results: 預測結果列表
        smiles_list: 對應的SMILES字符串列表
        
    Returns:
        pandas DataFrame
    """
    data = []
    
    for i, (result, smiles) in enumerate(zip(results, smiles_list)):
        if result is None:
            # 處理失敗的情況
            data.append({
                'smiles': smiles,
                'success': False,
                'num_pka_sites': 0,
                'pka_positions': '',
                'pka_values': ''
            })
        else:
            # 處理成功的情況
            pka_positions = result['pka_positions']
            pka_values = result['pka_values']
            
            # 格式化pKa位置和值為字符串
            positions_str = ','.join(map(str, pka_positions))
            values_str = ','.join([f"{val:.2f}" for val in pka_values])
            
            data.append({
                'smiles': smiles,
                'success': True,
                'num_pka_sites': len(pka_positions),
                'pka_positions': positions_str,
                'pka_values': values_str
            })
    
    return pd.DataFrame(data)

def main():
    """主函數，處理命令行參數並執行批量pKa預測"""
    parser = argparse.ArgumentParser(description='批量預測SMILES的pKa值')
    parser.add_argument('--model', type=str, required=True, help='預訓練模型權重的路徑')
    parser.add_argument('--input', type=str, required=True, help='包含SMILES的CSV文件路徑')
    parser.add_argument('--smiles-col', type=str, default='smiles', help='SMILES列的名稱')
    parser.add_argument('--output', type=str, required=True, help='保存結果的CSV文件路徑')
    parser.add_argument('--viz-dir', type=str, default=None, help='保存可視化結果的目錄路徑')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='計算設備 (cuda 或 cpu)')
    parser.add_argument('--no-parallel', action='store_true', help='禁用並行處理')
    parser.add_argument('--workers', type=int, default=None, help='並行處理的工作者數量')
    
    args = parser.parse_args()
    
    # 檢查輸入文件
    if not os.path.exists(args.input):
        print(f"錯誤：輸入文件不存在: {args.input}")
        return
    
    # 檢查模型文件
    if not os.path.exists(args.model):
        print(f"錯誤：模型文件不存在: {args.model}")
        return
    
    # 載入SMILES
    try:
        df = pd.read_csv(args.input)
        if args.smiles_col not in df.columns:
            print(f"錯誤：在CSV文件中找不到SMILES列 '{args.smiles_col}'")
            return
        
        smiles_list = df[args.smiles_col].tolist()
        print(f"已載入 {len(smiles_list)} 個SMILES")
    except Exception as e:
        print(f"載入SMILES時出錯: {e}")
        return
    
    # 載入模型
    try:
        model = load_pka_model(args.model, args.device)
    except Exception as e:
        print(f"載入模型時出錯: {e}")
        return
    
    # 進行批量預測
    results = batch_predict_pka(
        model, 
        smiles_list, 
        args.device, 
        args.viz_dir, 
        not args.no_parallel,
        args.workers
    )
    
    # 處理結果並保存為CSV
    result_df = format_pka_results_for_csv(results, smiles_list)
    result_df.to_csv(args.output, index=False)
    print(f"已將pKa預測結果保存至: {args.output}")
    
    # 統計成功率
    success_count = result_df['success'].sum()
    success_rate = (success_count / len(smiles_list)) * 100
    print(f"預測成功率: {success_rate:.2f}% ({success_count}/{len(smiles_list)})")
    
    # 統計pKa位點
    pka_sites = result_df[result_df['success']]['num_pka_sites'].sum()
    print(f"共發現 {pka_sites} 個pKa位點")
    
    # 其他統計
    pka_dist = result_df[result_df['success']]['num_pka_sites'].value_counts()
    print("\n每個分子pKa位點數量分布:")
    for n_sites, count in pka_dist.items():
        print(f"  {n_sites} 個位點: {count} 個分子")

if __name__ == "__main__":
    main() 