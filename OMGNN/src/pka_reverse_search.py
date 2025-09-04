#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pka_reverse_search.py

此腳本用於根據指定的官能基類型和pKa區間進行反向搜尋，
並使用FunctionalGroupLabeler類視覺化結果。
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import argparse
from collections import defaultdict
import pickle
import time
import matplotlib.pyplot as plt
from rdkit import Chem

# 添加正確的路徑
sys.path.append('src/')
import importlib
import pka_smiles_query
importlib.reload(pka_smiles_query)
from pka_smiles_query import map_pka_to_functional_groups, visualize_pka_mapping
from functional_group_labeler import FunctionalGroupLabeler

# 忽略RDKit的警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def preprocess_database(csv_file, output_pickle_file, output_dir):
    """
    預處理整個數據庫，提取所有分子的官能基信息，並保存到pickle檔
    
    Args:
        csv_file: 輸入的CSV文件路徑
        output_pickle_file: 輸出的pickle檔案路徑
        
    Returns:
        包含所有分子資訊的字典
    """
    print(f"開始預處理資料庫: {csv_file}")
    
    # 讀取CSV文件
    df = pd.read_csv(csv_file)
    print(f"從CSV檔案載入了 {len(df)} 筆數據")
    
    # 用於儲存所有分子的資訊
    all_molecules_data = {}
    
    # 初始化error_list用於記錄官能基數量小於pKa數量的情況
    error_list = []
    
    # 處理每個分子
    print("正在提取所有分子的官能基資訊...")
    
    # 初始化FunctionalGroupLabeler
    labeler = FunctionalGroupLabeler()
    
    # 使用tqdm創建進度條
    for index, row in tqdm(df.iterrows(), total=len(df), desc="處理進度"):
        smiles = row['SMILES']
        
        try:
            # 標記分子官能基
            mol, atom_to_group = labeler.label_molecule(smiles)
            
            if mol and atom_to_group:
                # 將atom_to_group轉換為pka_prediction_kmeans需要的格式
                functional_groups = []
                for atom_idx, group_type in atom_to_group.items():
                    atom = mol.GetAtomWithIdx(atom_idx)
                    functional_groups.append({
                        'group_type': group_type,
                        'match_indices': [atom_idx],
                        'key_atom_index': atom_idx
                    })
                
                # 獲取pKa值
                pka_value_str = row['pKa_value']
                try:
                    # 處理可能的多個pKa值
                    if isinstance(pka_value_str, str) and "," in pka_value_str:
                        pka_values = [float(val.strip()) for val in pka_value_str.strip('"').split(',')]
                    else:
                        pka_values = [float(pka_value_str)]
                except (ValueError, TypeError):
                    print(f"\n無法解析pKa值: {pka_value_str} 對於SMILES: {smiles}")
                    continue
                
                # 使用已有的官能基信息映射pKa
                results = map_pka_to_functional_groups(smiles, csv_file, functional_groups)
                
                # 檢查找到的官能基數量是否小於pKa數量
                if len(results['functional_groups']) < len(results['pka_values']):
                    error_list.append(smiles)
                    continue
                
                if results and 'mapping' in results:
                    all_molecules_data[smiles] = {
                        'results': results,
                        'pka_values': pka_values,
                        'mol': mol,
                        'atom_to_group': atom_to_group
                    }
            else:
                print(f"\n警告: 無法標記SMILES中的官能基: {smiles}")
                
        except Exception as e:
            print(f"\n處理SMILES時出錯 {smiles}: {str(e)}")
            continue
    
    
    if error_list:
        print(f"找到的官能基數量小於pKa數量(error): {len(error_list)}")
        with open(os.path.join(output_dir, "functional_group_count_smaller_than_pka_count_list.txt"), "w") as f:
            for smiles in error_list:
                f.write(f"{smiles}\n")
        print(f"已儲存到 {os.path.join(output_dir, 'functional_group_count_smaller_than_pka_count_list.txt')}")
    else:
        print("沒有找到官能基數量小於pKa數量的情況")
        
    
        
    # 保存處理後的數據
    print(f"\n儲存預處理結果至 {output_pickle_file}")
    with open(output_pickle_file, 'wb') as f:
        pickle.dump(all_molecules_data, f)
    
    print(f"預處理完成! 共處理了 {len(all_molecules_data)} 個分子")
    return all_molecules_data

def reverse_search_by_functional_group_and_pka(csv_file, output_dir, 
                                             functional_group=None, 
                                             pka_min=None, pka_max=None,
                                             max_results=10,
                                             use_cached_data=True):
    """
    根據官能基類型和pKa區間進行反向搜尋
    
    Args:
        csv_file: 輸入的CSV文件路徑
        output_dir: 輸出圖像的目錄路徑
        functional_group: 要搜尋的官能基類型，若為None則搜尋所有官能基
        pka_min: pKa的最小值，若為None則不設下限
        pka_max: pKa的最大值，若為None則不設上限
        max_results: 最多顯示的結果數量
        use_cached_data: 是否使用預處理的緩存數據
    """
    # 創建輸出目錄（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 確定預處理數據的文件路徑
    pickle_file = csv_file.replace('.csv', '_processed.pkl')
    
    # 檢查是否使用緩存數據
    if use_cached_data and os.path.exists(pickle_file):
        print(f"從緩存檔案載入數據: {pickle_file}")
        start_time = time.time()
        with open(pickle_file, 'rb') as f:
            molecules_data = pickle.load(f)
        print(f"載入完成，耗時 {time.time() - start_time:.2f} 秒")
    else:
        print(f"未找到緩存檔案或指定不使用緩存，將預處理資料庫...")
        molecules_data = preprocess_database(csv_file, pickle_file, output_dir)
    
    # 設定搜尋條件的提示信息
    search_conditions = []
    if functional_group:
        search_conditions.append(f"官能基類型: {functional_group}")
    if pka_min is not None:
        search_conditions.append(f"pKa最小值: {pka_min}")
    if pka_max is not None:
        search_conditions.append(f"pKa最大值: {pka_max}")
    
    if search_conditions:
        print(f"搜尋條件: {', '.join(search_conditions)}")
    else:
        print("未指定搜尋條件，將顯示所有結果")
    
    # 用於儲存符合條件的分子
    matching_molecules = []
    
    # 處理每個分子
    print("正在搜尋符合條件的分子...")
    
    # 搜尋符合條件的分子
    for smiles, molecule_data in tqdm(molecules_data.items(), total=len(molecules_data), desc="搜尋進度"):
        results = molecule_data['results']
        
        # 用於判斷是否符合條件
        match_found = False
        matched_pkas = []
        
        # 檢查每個pKa與對應的官能基是否符合條件
        for pka, group_info in results['mapping'].items():
            group_type = group_info['type']
            # 移除可能的猜測標記
            if " (guess)" in group_type:
                group_type = group_type.replace(" (guess)", "")
            
            # 檢查官能基類型是否符合條件
            if functional_group and group_type != functional_group:
                continue
            
            # 檢查pKa是否在範圍內
            pka_in_range = True
            if pka_min is not None and float(pka) < pka_min:
                pka_in_range = False
            if pka_max is not None and float(pka) > pka_max:
                pka_in_range = False
            
            if pka_in_range:
                match_found = True
                matched_pkas.append(float(pka))
        
        # 如果找到符合條件的情況，保存結果
        if match_found:
            matching_molecules.append({
                'smiles': smiles,
                'results': results,
                'matched_pkas': matched_pkas,
                'mol': molecule_data.get('mol', None),
                'atom_to_group': molecule_data.get('atom_to_group', {})
            })
            
            # 如果已經找到足夠多的結果，停止搜尋
            if len(matching_molecules) >= max_results:
                print(f"\n已找到 {max_results} 個符合條件的分子，停止搜尋")
                break
    
    # 顯示搜尋結果
    if not matching_molecules:
        print("未找到符合條件的分子")
        return
    
    print(f"\n總共找到 {len(matching_molecules)} 個符合條件的分子")
    
    # 初始化FunctionalGroupLabeler
    labeler = FunctionalGroupLabeler()
    
    # 為每個符合條件的分子生成可視化結果
    for i, molecule in enumerate(matching_molecules):
        smiles = molecule['smiles']
        results = molecule['results']
        matched_pkas = molecule['matched_pkas']
        
        print(f"\n分子 {i+1}: {smiles}")
        print(f"匹配的pKa值: {', '.join([f'{pka:.2f}' for pka in matched_pkas])}")
        
        # 確保 results 包含正確的 smiles
        if results['smiles'] != smiles:
            print(f"警告: 修正 results 中的 SMILES 不匹配問題")
            results['smiles'] = smiles
        
        # 創建該分子的專屬輸出目錄
        molecule_output_dir = os.path.join(output_dir, f"molecule_{i+1}")
        os.makedirs(molecule_output_dir, exist_ok=True)
        
        # 獲取分子對象
        mol = molecule.get('mol', None)
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"無法從SMILES創建分子對象: {smiles}")
                continue
        
        # 從預處理數據中獲取atom_to_group
        atom_to_group = molecule.get('atom_to_group', None)
        if atom_to_group is None or not atom_to_group:
            # 如果不存在，重新標記分子
            mol, atom_to_group = labeler.label_molecule(smiles)
            if not atom_to_group:
                print(f"無法標記分子: {smiles}")
                continue
        
        # 生成基本分子圖像
        output_path = os.path.join(molecule_output_dir, f"molecule_{i+1}.png")
        pil_img = labeler.visualize_labeled_molecule(mol, atom_to_group, output_path=output_path)
        
        # 生成包含pKa值的圖像
        try:
            # 使用visualize_pka_mapping生成帶有pKa值的圖像
            # 設置matplotlib後端為Agg以避免顯示圖形
            plt.switch_backend('Agg')
            output_path_with_pka = os.path.join(molecule_output_dir, f"molecule_{i+1}_with_pka.png")
            visualize_pka_mapping(results, output_path_with_pka)
        except Exception as e:
            print(f"生成pKa圖像時出錯: {e}")
        
        # 將匹配信息保存到文本文件
        info_file = os.path.join(molecule_output_dir, "info.txt")
        with open(info_file, "w") as f:
            f.write(f"SMILES: {smiles}\n")
            f.write(f"搜尋條件:\n")
            for condition in search_conditions:
                f.write(f"  {condition}\n")
            f.write(f"\n匹配的pKa值:\n")
            for pka in matched_pkas:
                group_info = results['mapping'][pka]
                group_type = group_info['type']
                f.write(f"  pKa = {pka:.2f} => {group_type}\n")
            f.write(f"results: \n")
            for key, value in results.items():
                f.write(f"  {key}: {value}\n")
        print(f"分子 {i+1} 的資訊已保存到 {info_file}")
    
    # 生成摘要文件，包含所有分子的資訊和對應的pKa
    summary_file = os.path.join(output_dir, "search_summary.txt")
    with open(summary_file, "w") as f:
        # 寫入標題和搜尋條件
        f.write("=" * 50 + "\n")
        f.write("pKa 反向搜尋結果摘要\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("搜尋條件:\n")
        if search_conditions:
            for condition in search_conditions:
                f.write(f"  {condition}\n")
        else:
            f.write("  未指定搜尋條件\n")
        
        f.write(f"\n找到 {len(matching_molecules)} 個符合條件的分子\n\n")
        
        # 為每個分子寫入摘要資訊
        for i, molecule in enumerate(matching_molecules):
            smiles = molecule['smiles']
            results = molecule['results']
            matched_pkas = molecule['matched_pkas']
            
            f.write("-" * 50 + "\n")
            f.write(f"分子 {i+1}:\n")
            f.write(f"SMILES: {smiles}\n")
            f.write(f"匹配的pKa值及對應官能基:\n")
            
            # 按照pKa值排序
            sorted_pkas = sorted(matched_pkas)
            for pka in sorted_pkas:
                group_info = results['mapping'][pka]
                group_type = group_info['type']
                f.write(f"  pKa = {pka:.2f} => {group_type}\n")
            
            # 添加該分子圖像的路徑資訊
            molecule_dir = f"molecule_{i+1}"
            f.write(f"\n圖像檔案:\n")
            f.write(f"  分子結構: {molecule_dir}/molecule_{i+1}.png\n")
            f.write(f"  含pKa標記: {molecule_dir}/molecule_{i+1}_with_pka.png\n")
            f.write(f"  詳細資訊: {molecule_dir}/info.txt\n\n")
    
    print(f"\n已生成搜尋摘要: {summary_file}")
    print(f"\n搜尋完成！所有結果已保存到 {output_dir}")

def IM():
    """互動模式，讓用戶輸入搜尋條件"""
    print("=" * 50)
    print("pKa 反向搜尋工具 - 互動模式")
    print("=" * 50)
    
    # 預設值
    default_csv_file = '../data/processed_pka_data.csv'
    default_output_dir = '../output/reverse_search_results'
    
    # 可搜尋的官能基類型
    functional_groups = [
        'COOH', 'SulfonicAcid', 'PhosphonicAcid', 'PrimaryAmine', 
        'SecondaryAmine', 'TertiaryAmine', 'Phenol', 'Alcohol', 
        'AliphaticRingAlcohol', 'Imidazole', 'ImidazoleLike', 'Pyridine', 
        'Thiol', 'NHydroxylamine', 'PyrazoleNitrogen', 'Indole', 'Phosphate',
        'Aniline'
    ]
    
    # 進入無窮迴圈
    while True:
        print("\n" + "=" * 50)
        print("請選擇操作:")
        print("1. 開始新的搜尋")
        print("2. 預處理數據庫")
        print("3. 離開程式")
        choice = input("請輸入選項(1/2/3): ").strip()
        
        if choice == "3":
            print("謝謝使用，程式已結束。")
            break
        
        if choice not in ["1", "2"]:
            print("無效的選項，請重新選擇。")
            continue
        
        # 輸入CSV檔案路徑
        csv_file = input(f"\n請輸入CSV文件路徑 [{default_csv_file}](空白則使用預設值): ").strip()
        if not csv_file:
            csv_file = default_csv_file
        
        # 預處理數據庫
        if choice == "2":
            pickle_file = csv_file.replace('.csv', '_processed.pkl')
            preprocess_database(csv_file, pickle_file, default_output_dir)
            input("\n預處理完成，按Enter鍵繼續...")
            continue
        
        # 開始新的搜尋
        output_dir = input(f"請輸入輸出資料夾名稱 [{default_output_dir}/(資料夾名稱)](空白則使用預設值): ").strip()
        if not output_dir:
            output_dir = default_output_dir
        else:
            output_dir = os.path.join(default_output_dir, output_dir)
        
        # 詢問是否使用緩存數據
        use_cached = True
        cache_choice = input("\n是否使用預處理緩存數據? (y/n)[y]: ").strip().lower()
        if cache_choice == 'n' or cache_choice == 'no':
            use_cached = False
        
        # 列出可搜尋的官能基類型
        print("\n可選的官能基類型:")
        for i, group in enumerate(functional_groups):
            print(f"  {i+1}. {group}")
        
        functional_group = None
        group_choice = input("\n請選擇官能基類型 (輸入編號或名稱，留空則不限制): ").strip()
        if group_choice:
            try:
                # 檢查是否輸入了編號
                choice_idx = int(group_choice) - 1
                if 0 <= choice_idx < len(functional_groups):
                    functional_group = functional_groups[choice_idx]
                else:
                    # 檢查是否直接輸入了官能基名稱
                    if group_choice in functional_groups:
                        functional_group = group_choice
                    else:
                        print(f"警告: 未找到官能基 '{group_choice}'，將不限制官能基類型")
            except ValueError:
                # 不是數字，檢查是否直接輸入了官能基名稱
                if group_choice in functional_groups:
                    functional_group = group_choice
                else:
                    print(f"警告: 未找到官能基 '{group_choice}'，將不限制官能基類型")
        
        pka_min = input("\n請輸入pKa最小值 (留空則不設下限): ").strip()
        if pka_min:
            try:
                pka_min = float(pka_min)
            except ValueError:
                print("警告: pKa最小值必須是數字，將不設下限")
                pka_min = None
        else:
            pka_min = None
        
        pka_max = input("請輸入pKa最大值 (留空則不設上限): ").strip()
        if pka_max:
            try:
                pka_max = float(pka_max)
            except ValueError:
                print("警告: pKa最大值必須是數字，將不設上限")
                pka_max = None
        else:
            pka_max = None
        
        max_results = input("請輸入最多顯示的結果數量 [10]: ").strip()
        if max_results:
            try:
                max_results = int(max_results)
            except ValueError:
                print("警告: 結果數量必須是整數，將使用默認值 10")
                max_results = 10
        else:
            max_results = 10
        
        print("\n搜尋條件:")
        print(f"CSV文件: {csv_file}")
        print(f"輸出目錄: {output_dir}")
        print(f"使用緩存數據: {'是' if use_cached else '否'}")
        if functional_group:
            print(f"官能基類型: {functional_group}")
        else:
            print("官能基類型: 不限")
        if pka_min is not None:
            print(f"pKa最小值: {pka_min}")
        else:
            print("pKa最小值: 不限")
        if pka_max is not None:
            print(f"pKa最大值: {pka_max}")
        else:
            print("pKa最大值: 不限")
        print(f"最多結果數量: {max_results}")
        
        confirm = input("\n確認搜尋? (y/n): ").strip().lower()
        if confirm == 'y' or confirm == 'yes':
            print("\n開始搜尋...")
            reverse_search_by_functional_group_and_pka(
                csv_file, output_dir, functional_group, pka_min, pka_max, max_results, use_cached
            )
            
            input("\n按Enter鍵繼續...")
        else:
            print("搜尋已取消")

if __name__ == "__main__":
    # 使用互動模式
    IM() 