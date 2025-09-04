#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pka_smiles_query.py

該腳本用於查詢SMILES結構，顯示其官能基與對應的pKa值關係。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
import warnings
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import ast  # 添加ast模塊導入

# 添加正確的路徑
sys.path.append('src/')

# 忽略RDKit的警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')
warnings.filterwarnings("ignore", category=FutureWarning)






def map_pka_to_functional_groups(smiles_str, data_path="../data/processed_pka_data.csv", external_functional_groups=None):
    """
    將SMILES結構中的官能基映射到其對應的pKa值
    
    Args:
        smiles_str: 要查詢的SMILES字符串
        data_path: 原始數據CSV文件路徑
        external_functional_groups: 可選，外部提供的官能基列表，如果提供則不再調用extract_molecule_feature
        
    Returns:
        包含官能基與pKa對應關係的字典
    """
    print(f"查詢SMILES: {smiles_str}")
    
    # ===== STEP 1: 載入SMILES的pKa值 =====
    df = pd.read_csv(data_path)
    matched_row = df[df['SMILES'] == smiles_str]
    
    if matched_row.empty:
        print(f"警告: 在數據集中找不到SMILES: {smiles_str}")
        return None
    
    # 獲取pKa值
    pka_value_str = matched_row['pKa_value'].values[0]
    try:
        # 處理可能的多個pKa值
        if isinstance(pka_value_str, str):
            # 使用ast.literal_eval安全地將字符串轉換為Python對象
            try:
                pka_values = ast.literal_eval(pka_value_str)
                if not isinstance(pka_values, list):
                    pka_values = [float(pka_value_str)]
            except (ValueError, SyntaxError):
                # 如果轉換失敗，嘗試其他格式
                if "," in pka_value_str:
                    pka_values = [float(val.strip()) for val in pka_value_str.strip('"').split(',')]
                else:
                    pka_values = [float(pka_value_str)]
        else:
            pka_values = [float(pka_value_str)]
        print(f"找到pKa值: {pka_values}")
    except (ValueError, TypeError):
        print(f"無法解析pKa值: {pka_value_str}")
        return None
    
    # ===== STEP 2: 識別分子中的官能基 =====
    if external_functional_groups is not None:
        # 使用外部提供的官能基信息
        functional_groups = external_functional_groups
    else:
        # 使用原來的提取方法
        from functional_group_labeler import extract_functional_group_features
        functional_groups = extract_functional_group_features(smiles_str)
    
    if not functional_groups:
        print(f"在SMILES中未發現官能基: {smiles_str}")
        return None
    
    print(f"發現 {len(functional_groups)} 個官能基:")
    for i, fg in enumerate(functional_groups):
        # 確保match_indices是列表
        if isinstance(fg['match_indices'], tuple):
            fg['match_indices'] = list(fg['match_indices'])
        print(f"  {i+1}. {fg['group_type']} 於位置: {fg['match_indices']}")
    
    # ===== STEP 3: 將官能基分類為酸性和鹼性 =====
    pka_to_group_mapping = {}
    acid_groups = []  # 酸性官能基
    base_groups = []  # 鹼性官能基
    
    for i, fg in enumerate(functional_groups):
        group_type = fg['group_type']
        match_indices = list(fg['match_indices']) if isinstance(fg['match_indices'], tuple) else fg['match_indices']
        
        if group_type in ['COOH', 'SulfonicAcid', 'PhosphonicAcid', 'Thiol', 'Phosphate', 'Phenol']:
            acid_groups.append({
                'type': group_type,
                'indices': match_indices,
                'original_index': i,
                'used': False
            })
            print(f"  分類 {group_type} 為酸性官能基")
        elif group_type in ['PrimaryAmine', 'SecondaryAmine', 'TertiaryAmine', 'Imidazole', 'ImidazoleLike', 'Pyridine', 'PyrazoleNitrogen', 'NHydroxylamine', 'Alcohol', 'AliphaticRingAlcohol', 'Indole', 'Aniline']:
            base_groups.append({
                'type': group_type,
                'indices': match_indices,
                'original_index': i,
                'used': False
            })
            print(f"  分類 {group_type} 為鹼性官能基")
    
    # ===== STEP 4: 定義官能基的pKa範圍 =====
    special_pka_ranges = {
        'COOH': [(1.5, 5.5)],            # 羧酸通常在3.5-5.5之間
        'SulfonicAcid': [(-3.0, 4.0)],   # 硫酸通常在-3到4的範圍內
        'PhosphonicAcid': [(1.0, 7.0)],  # 磷酸通常在1.0-7.0的範圍內
        'Phosphate': [(1.0, 7.0), (6.5,7.5), (9.5, 11.0)],  # 磷酸酯的多個pKa範圍
        'Phenol': [(8.0, 10.5)],         # 酚的pKa通常在8-10.5之間
        'Thiol': [(6.0, 11.0)],          # 硫醇通常在6-11之間
        'Alcohol': [(15.0, 17.0)],       # 醇的pKa通常在15-17之間
        'AliphaticRingAlcohol': [(12.0, 14.5)],  # 脂肪環醇的pKa通常低於普通醇
        'SecondaryAmine': [(2.0, 4.0), (5.0, 7.0)],  # 二級胺的多個pKa範圍
        'PrimaryAmine': [(2.0, 4.5)],    # 一級胺特殊範圍
        'Aniline': [(3.0, 5.0)],         # 苯胺的pKa通常在3-5之間
        'TertiaryAmine': [(3.0, 5.0)],   # 三級胺特殊範圍
        'Pyridine': [(2.8, 6.5)],        # 吡啶的pKa範圍
        'Imidazole': [(5.0, 9.0)],       # 咪唑的pKa範圍
        'ImidazoleLike': [(1.0, 2.0), (5.0, 7.0)],  # 類咪唑的多個pKa範圍
        'PyrazoleNitrogen': [(2.0, 3.5)],  # 吡唑氮的pKa範圍
        'NHydroxylamine': [(8.5, 10.0)],  # 羥胺的pKa範圍
        'Indole': [(16.0, 17.0)],         # 吲哚的pKa範圍
    }
    
    # ===== STEP 5: 按pKa值大小排序 =====
    sorted_pkas = sorted(pka_values)
    
    # 檢查是否同時存在硫酸和酚
    has_sulfonic = any(group['type'] == 'SulfonicAcid' for group in acid_groups)
    has_phenol = any(group['type'] == 'Phenol' for group in acid_groups)
    # 檢查是否存在苯胺
    has_aniline = any(group['type'] == 'Aniline' for group in base_groups)
    # 檢查是否存在吡啶
    has_pyridine = any(group['type'] == 'Pyridine' for group in base_groups)
    # 檢查是否存在硫醇
    has_thiol = any(group['type'] == 'Thiol' for group in acid_groups)
    
    # ===== STEP 6: 特殊處理特定官能基共存的情況 =====
    # 特殊處理Phenol和SulfonicAcid共存的情況
    if has_phenol and has_sulfonic:
        print("  檢測到Phenol和SulfonicAcid共存，進行特殊處理")
        # 將pKa值分為兩組：低於4的（可能分配給SulfonicAcid）和高於4的（一定不分配給SulfonicAcid）
        low_pkas = [pka for pka in sorted_pkas if pka <= 4.0]
        high_pkas = [pka for pka in sorted_pkas if pka > 4.0]
        
        # 確保低pKa值不超過SulfonicAcid的數量
        sulfonic_count = sum(1 for group in acid_groups if group['type'] == 'SulfonicAcid')
        if len(low_pkas) > sulfonic_count:
            excess = len(low_pkas) - sulfonic_count
            high_pkas = low_pkas[-excess:] + high_pkas
            low_pkas = low_pkas[:-excess]
        
        # 重新排序處理後的pKa值
        sorted_pkas = low_pkas + high_pkas
        print(f"  重新排序後的pKa值: {sorted_pkas}")
    
    # 特殊處理Aniline和Phenol共存的情況
    if has_aniline and has_phenol:
        print("  檢測到Aniline和Phenol共存，進行特殊處理")
        # 將pKa值分為兩組：3-6範圍內的（優先分配給Aniline）和其他的
        aniline_pkas = [pka for pka in sorted_pkas if 3.0 <= pka <= 6.0]
        other_pkas = [pka for pka in sorted_pkas if pka < 3.0 or pka > 6.0]
        
        # 保存原始排序，以便在映射時參考
        original_pka_order = {pka: i for i, pka in enumerate(sorted_pkas)}
        
        # 如果有多個pKa值在3-6範圍，確保它們不會全部分配給Phenol
        aniline_count = sum(1 for group in base_groups if group['type'] == 'Aniline')
        if len(aniline_pkas) > aniline_count:
            # 確保至少有一個pKa值分配給Aniline
            if aniline_count > 0:
                # 先保留aniline_count個pKa值給Aniline
                reserved_for_aniline = aniline_pkas[:aniline_count]
                # 剩餘的pKa值可用於其他官能基
                remaining_aniline_pkas = aniline_pkas[aniline_count:]
                # 按原始順序將剩餘的pKa值插入other_pkas
                for pka in remaining_aniline_pkas:
                    idx = next((i for i, p in enumerate(other_pkas) if original_pka_order[p] > original_pka_order[pka]), len(other_pkas))
                    other_pkas.insert(idx, pka)
                # 最終排序：reserved_for_aniline放在前面，其他按原始順序
                sorted_pkas = reserved_for_aniline + other_pkas
                print(f"  為Aniline保留的pKa值: {reserved_for_aniline}")
                print(f"  重新排序後的pKa值: {sorted_pkas}")
    
    # 特殊處理Pyridine和Phenol共存的情況
    if has_pyridine and has_phenol:
        print("  檢測到Pyridine和Phenol共存，進行特殊處理")
        # 將pKa值分為兩組：2.8-6.5範圍內的（優先分配給Pyridine）和其他的
        pyridine_pkas = [pka for pka in sorted_pkas if 2.8 <= pka <= 6.5]
        other_pkas = [pka for pka in sorted_pkas if pka < 2.8 or pka > 6.5]
        
        # 保存原始排序，以便在映射時參考
        original_pka_order = {pka: i for i, pka in enumerate(sorted_pkas)}
        
        # 如果有多個pKa值在2.8-6.5範圍，確保它們不會全部分配給Phenol
        pyridine_count = sum(1 for group in base_groups if group['type'] == 'Pyridine')
        if len(pyridine_pkas) > pyridine_count:
            # 確保至少有一個pKa值分配給Pyridine
            if pyridine_count > 0:
                # 先保留pyridine_count個pKa值給Pyridine
                reserved_for_pyridine = pyridine_pkas[:pyridine_count]
                # 剩餘的pKa值可用於其他官能基
                remaining_pyridine_pkas = pyridine_pkas[pyridine_count:]
                # 按原始順序將剩餘的pKa值插入other_pkas
                for pka in remaining_pyridine_pkas:
                    idx = next((i for i, p in enumerate(other_pkas) if original_pka_order[p] > original_pka_order[pka]), len(other_pkas))
                    other_pkas.insert(idx, pka)
                # 最終排序：reserved_for_pyridine放在前面，其他按原始順序
                sorted_pkas = reserved_for_pyridine + other_pkas
                print(f"  為Pyridine保留的pKa值: {reserved_for_pyridine}")
                print(f"  重新排序後的pKa值: {sorted_pkas}")
    
    # 特殊處理Thiol和Pyridine共存的情況
    if has_thiol and has_pyridine:
        print("  檢測到Thiol和Pyridine共存，進行特殊處理")
        
        # 計算Pyridine和Thiol的數量
        pyridine_count = sum(1 for group in base_groups if group['type'] == 'Pyridine')
        thiol_count = sum(1 for group in acid_groups if group['type'] == 'Thiol')
        
        # 檢查pKa值的分佈情況
        lower_pkas = [pka for pka in sorted_pkas if pka < 6.0]  # Pyridine的優先範圍
        higher_pkas = [pka for pka in sorted_pkas if pka >= 6.0]  # Thiol的優先範圍
        
        # 如果pKa值分佈不合理，進行調整
        if len(lower_pkas) < pyridine_count or len(higher_pkas) < thiol_count:
            print(f"  pKa值分佈不合理，進行調整（低pKa: {len(lower_pkas)}，高pKa: {len(higher_pkas)}）")
            
            # 按從小到大排序
            all_pkas = sorted(sorted_pkas)
            
            # 如果有足夠的pKa值，按官能基數量分配
            if len(all_pkas) >= pyridine_count + thiol_count:
                # 前pyridine_count個分配給Pyridine
                reserved_for_pyridine = all_pkas[:pyridine_count]
                # 後thiol_count個分配給Thiol
                reserved_for_thiol = all_pkas[-thiol_count:]
                
                # 如果有中間值未分配，放入其他pKa中
                middle_pkas = []
                if pyridine_count + thiol_count < len(all_pkas):
                    middle_pkas = all_pkas[pyridine_count:-thiol_count]
                
                # 合併結果，優先保留Pyridine和Thiol的分配
                sorted_pkas = reserved_for_pyridine + middle_pkas + reserved_for_thiol
                print(f"  為Pyridine保留的pKa值: {reserved_for_pyridine}")
                print(f"  為Thiol保留的pKa值: {reserved_for_thiol}")
                print(f"  重新排序後的pKa值: {sorted_pkas}")
    
    # ===== STEP 7: 為每個pKa值分配官能基 =====
    print("\n開始映射pKa值到官能基:")
    
    for pka in sorted_pkas:
        print(f"\n處理pKa = {pka}:")
        mapped = False
        
        # ===== STEP 7.1: 首先嘗試按特定pKa範圍進行映射 =====
        print(f"  嘗試使用特定pKa範圍進行映射...")
        
        # 特別處理：當pKa在3-6範圍時，優先檢查是否有Aniline
        if 3.0 <= pka <= 6.0 and has_aniline:
            print(f"  pKa = {pka} 在Aniline的優先範圍(3.0-6.0)內")
            for base_group in base_groups:
                if base_group['type'] == 'Aniline' and not base_group['used']:
                    print(f"  將pKa = {pka} 優先映射到 Aniline")
                    pka_to_group_mapping[pka] = {
                        'type': 'Aniline',
                        'indices': base_group['indices']
                    }
                    base_group['used'] = True
                    mapped = True
                    break
        
        # 特別處理：當pKa在2.8-6.5範圍時，優先檢查是否有Pyridine
        if not mapped and 2.8 <= pka <= 6.5 and has_pyridine:
            print(f"  pKa = {pka} 在Pyridine的優先範圍(2.8-6.5)內")
            for base_group in base_groups:
                if base_group['type'] == 'Pyridine' and not base_group['used']:
                    print(f"  將pKa = {pka} 優先映射到 Pyridine")
                    pka_to_group_mapping[pka] = {
                        'type': 'Pyridine',
                        'indices': base_group['indices']
                    }
                    base_group['used'] = True
                    mapped = True
                    break
        
        # 如果不是優先處理Aniline或Pyridine的情況，則繼續正常的範圍映射
        if not mapped:
            for group_type in ['COOH', 'SulfonicAcid', 'PhosphonicAcid', 'Phosphate', 'Phenol', 'Thiol']:
                if mapped:
                    break
                    
                # 特別處理：如果pKa在3-6範圍，跳過對Phenol的映射（這範圍應優先考慮Aniline）
                if group_type == 'Phenol' and 3.0 <= pka <= 6.0 and has_aniline:
                    print(f"  跳過 {group_type}，因為pKa在Aniline的優先範圍(3.0-6.0)內")
                    continue
                
                # 特別處理：如果pKa在2.8-6.5範圍，跳過對Phenol的映射（這範圍應優先考慮Pyridine）
                if group_type == 'Phenol' and 2.8 <= pka <= 6.5 and has_pyridine:
                    print(f"  跳過 {group_type}，因為pKa在Pyridine的優先範圍(2.8-6.5)內")
                    continue
                
                # 特別處理：如果pKa < 6.0，跳過對Thiol的映射
                if group_type == 'Thiol' and pka < 6.0:
                    print(f"  跳過 {group_type}，因為pKa < 6.0，低於Thiol的正常範圍(6.0-11.0)")
                    continue
                    
                if group_type in special_pka_ranges:
                    for pka_range in special_pka_ranges[group_type]:
                        min_pka, max_pka = pka_range
                        if min_pka <= pka <= max_pka:
                            print(f"  pKa = {pka} 在 {group_type} 的範圍 {min_pka}-{max_pka} 內")
                            
                            # 特別處理SulfonicAcid
                            if group_type == 'SulfonicAcid' and pka > 4.0:
                                print(f"  跳過 {group_type}，因為pKa > 4.0")
                                continue
                            
                            # 檢查該類型的官能基是否有未使用的
                            for acid_group in acid_groups:
                                if acid_group['type'] == group_type and not acid_group['used']:
                                    print(f"  將pKa = {pka} 映射到 {group_type}（特定範圍匹配）")
                                    pka_to_group_mapping[pka] = {
                                        'type': group_type,
                                        'indices': acid_group['indices']
                                    }
                                    acid_group['used'] = True
                                    mapped = True
                                    break
                            
                            if mapped:
                                break
        
        # ===== STEP 7.2: 如果特定範圍沒有匹配，使用一般的酸鹼性判斷 =====
        if not mapped:
            if pka < 7.0:  # 酸性pKa
                print(f"  pKa = {pka} < 7.0，嘗試映射到酸性官能基...")
                best_match = None
                
                # 尋找最合適的酸性官能基
                for acid_group in acid_groups:
                    if acid_group['used']:
                        continue
                        
                    # 特別處理SulfonicAcid
                    if acid_group['type'] == 'SulfonicAcid' and pka > 4.0:
                        print(f"  跳過 {acid_group['type']}，因為pKa > 4.0")
                        continue
                        
                    # 如果pKa在8-10.5範圍，則跳過硫酸(優先考慮酚類)
                    if pka >= 8.0 and acid_group['type'] == 'SulfonicAcid':
                        print(f"  跳過 {acid_group['type']}，因為pKa >= 8.0")
                        continue
                    
                    # 特別處理Phenol - 當pKa較低且存在Aniline或Pyridine時，應優先考慮Aniline/Pyridine
                    if acid_group['type'] == 'Phenol':
                        if 3.0 <= pka <= 6.0 and has_aniline:
                            print(f"  跳過 {acid_group['type']}，因為pKa在Aniline的優先範圍(3.0-6.0)內")
                            continue
                        if 2.8 <= pka <= 6.5 and has_pyridine:
                            print(f"  跳過 {acid_group['type']}，因為pKa在Pyridine的優先範圍(2.8-6.5)內")
                            continue
                        if pka < 7.0:
                            print(f"  注意: 嘗試將pKa = {pka} < 7.0 映射到Phenol，但這超出了Phenol的正常範圍(8.0-10.5)")
                    
                    # 特別處理Thiol - 當pKa < 6.0時跳過
                    if acid_group['type'] == 'Thiol' and pka < 6.0:
                        print(f"  跳過 {acid_group['type']}，因為pKa < 6.0，低於Thiol的正常範圍(6.0-11.0)")
                        continue
                    
                    # 優先順序
                    if acid_group['type'] == 'COOH' and 1.5 <= pka <= 5.5:
                        print(f"  找到COOH與pKa = {pka}匹配")
                        best_match = acid_group
                        break
                    elif acid_group['type'] == 'SulfonicAcid' and -3.0 <= pka <= 4.0:
                        print(f"  找到SulfonicAcid與pKa = {pka}匹配")
                        best_match = acid_group
                        break
                    elif acid_group['type'] == 'PhosphonicAcid' and 1.0 <= pka <= 7.0:
                        print(f"  找到PhosphonicAcid與pKa = {pka}匹配")
                        best_match = acid_group
                        break
                    elif acid_group['type'] == 'Phosphate' and 1.0 <= pka <= 7.0:
                        print(f"  找到Phosphate與pKa = {pka}匹配")
                        best_match = acid_group
                        break
                    elif acid_group['type'] == 'Phenol' and 8.0 <= pka <= 10.5:
                        print(f"  找到Phenol與pKa = {pka}匹配")
                        best_match = acid_group
                        break
                    elif acid_group['type'] == 'Thiol' and 6.0 <= pka <= 11.0:
                        print(f"  找到Thiol與pKa = {pka}匹配")
                        best_match = acid_group
                    elif best_match is None:
                        print(f"  沒有最佳匹配，保存 {acid_group['type']} 作為後備選項")
                        best_match = acid_group
                
                # 使用最佳匹配
                if best_match:
                    # 警告：如果是將Phenol映射到pKa < 8.0的情況
                    if best_match['type'] == 'Phenol' and pka < 8.0:
                        print(f"  警告：將pKa = {pka} < 8.0 映射到 Phenol，這超出了Phenol的正常範圍(8.0-10.5)")
                    
                    print(f"  將pKa = {pka} 映射到 {best_match['type']}（一般酸性判斷）")
                    pka_to_group_mapping[pka] = {
                        'type': best_match['type'],
                        'indices': best_match['indices']
                    }
                    best_match['used'] = True
                    mapped = True
            
            else:  # 鹼性pKa
                print(f"  pKa = {pka} >= 7.0，嘗試映射到鹼性官能基...")
                # 按照優先順序尋找未使用的鹼性官能基
                priority_order = ['Aniline', 'PrimaryAmine', 'SecondaryAmine', 'TertiaryAmine', 
                                'Imidazole', 'ImidazoleLike', 'Pyridine', 'PyrazoleNitrogen',
                                'NHydroxylamine', 'Indole', 'AliphaticRingAlcohol', 'Alcohol']
                
                # 基於pKa範圍的優先選擇
                if 3.0 <= pka <= 5.0:  # 優先考慮苯胺和胺類
                    filtered_groups = [g for g in base_groups if g['type'] in 
                                    ['Aniline', 'PrimaryAmine', 'SecondaryAmine', 'TertiaryAmine']]
                    print(f"  pKa = {pka} 在3.0-5.0範圍，優先考慮苯胺和胺類")
                elif 5.0 <= pka <= 7.0:  # 優先考慮咪唑類
                    filtered_groups = [g for g in base_groups if g['type'] in 
                                    ['Imidazole', 'ImidazoleLike', 'Pyridine']]
                    print(f"  pKa = {pka} 在5.0-7.0範圍，優先考慮咪唑類")
                elif 10.0 <= pka <= 15.0:  # 優先考慮環狀醇
                    filtered_groups = [g for g in base_groups if g['type'] in 
                                    ['AliphaticRingAlcohol']]
                    print(f"  pKa = {pka} 在10.0-15.0範圍，優先考慮環狀醇")
                elif pka >= 15.0:  # 優先考慮普通醇
                    filtered_groups = [g for g in base_groups if g['type'] in 
                                    ['Alcohol', 'Indole']]
                    print(f"  pKa = {pka} >= 15.0，優先考慮普通醇和吲哚")
                else:
                    filtered_groups = base_groups
                    print(f"  pKa = {pka} 沒有特定優先級，考慮所有鹼性官能基")
                
                # 如果根據pKa範圍過濾後沒有找到合適的官能基，使用所有鹼性官能基
                if not filtered_groups:
                    filtered_groups = base_groups
                    print(f"  沒有找到過濾後的鹼性官能基，使用所有鹼性官能基")
                
                # 按照優先順序尋找未使用的鹼性官能基
                for group_type in priority_order:
                    if mapped:
                        break
                        
                    # 特殊處理
                    if group_type == 'Imidazole' and pka >= 9.0:
                        print(f"  跳過 Imidazole，因為pKa >= 9.0")
                        continue
                    
                    if group_type == 'AliphaticRingAlcohol' and (pka < 12.0 or pka > 14.5):
                        print(f"  跳過 AliphaticRingAlcohol，因為pKa不在12.0-14.5範圍內")
                        continue
                    
                    if group_type == 'Indole' and pka < 16.0:
                        print(f"  跳過 Indole，因為pKa < 16.0")
                        continue
                    
                    # 特殊處理Pyridine - 只在其正常pKa範圍(2.8-6.5)內進行匹配
                    if group_type == 'Pyridine' and pka > 7.0:
                        print(f"  跳過 Pyridine，因為pKa > 7.0，超出了Pyridine的正常範圍(2.8-6.5)")
                        continue
                    
                    for base_group in filtered_groups:
                        if base_group['type'] == group_type and not base_group['used']:
                            print(f"  將pKa = {pka} 映射到 {base_group['type']}（鹼性優先順序）")
                            pka_to_group_mapping[pka] = {
                                'type': base_group['type'],
                                'indices': base_group['indices']
                            }
                            base_group['used'] = True
                            mapped = True
                            break
        
        # ===== STEP 7.3: 如果仍然沒有找到匹配，使用猜測 =====
        if not mapped:
            print(f"  pKa = {pka} 未找到精確匹配，進入猜測階段")
            
            # 先嘗試尋找任何未使用的官能基
            all_groups = acid_groups + base_groups
            
            # 如果pKa在3-6範圍，先檢查是否有未使用的Aniline
            if 3.0 <= pka <= 6.0 and has_aniline:
                for group in all_groups:
                    if group['type'] == 'Aniline' and not group['used']:
                        print(f"  將pKa = {pka} 優先映射到未使用的 Aniline（猜測階段）")
                        pka_to_group_mapping[pka] = {
                            'type': 'Aniline',
                            'indices': group['indices']
                        }
                        group['used'] = True
                        mapped = True
                        break
            
            # 如果pKa在2.8-6.5範圍，先檢查是否有未使用的Pyridine
            if not mapped and 2.8 <= pka <= 6.5 and has_pyridine:
                for group in all_groups:
                    if group['type'] == 'Pyridine' and not group['used']:
                        print(f"  將pKa = {pka} 優先映射到未使用的 Pyridine（猜測階段）")
                        pka_to_group_mapping[pka] = {
                            'type': 'Pyridine',
                            'indices': group['indices']
                        }
                        group['used'] = True
                        mapped = True
                        break
            
            # 如果仍未映射，尋找其他未使用的官能基
            if not mapped:
                suitable_groups = []
                for group in all_groups:
                    if not group['used']:
                        # 特別處理各種官能基的pKa範圍
                        
                        # 特別處理SulfonicAcid
                        if group['type'] == 'SulfonicAcid' and pka > 4.0:
                            print(f"  跳過 {group['type']} 做猜測，因為pKa > 4.0")
                            continue
                        
                        # 特別處理Phenol
                        if group['type'] == 'Phenol':
                            if 3.0 <= pka <= 6.0 and has_aniline:
                                print(f"  跳過猜測 {group['type']}，因為pKa在Aniline的優先範圍(3.0-6.0)內")
                                continue
                            if 2.8 <= pka <= 6.5 and has_pyridine:
                                print(f"  跳過猜測 {group['type']}，因為pKa在Pyridine的優先範圍(2.8-6.5)內")
                                continue
                            if pka < 8.0:
                                print(f"  警告：考慮將pKa = {pka} < 8.0 映射到 Phenol，這超出了Phenol的正常範圍(8.0-10.5)")
                        
                        # 特別處理Thiol
                        if group['type'] == 'Thiol' and pka < 6.0:
                            print(f"  跳過猜測 {group['type']}，因為pKa < 6.0，低於Thiol的正常範圍(6.0-11.0)")
                            continue
                        
                        # 特別處理Pyridine
                        if group['type'] == 'Pyridine' and pka > 7.0:
                            print(f"  跳過猜測 {group['type']}，因為pKa > 7.0，超出了Pyridine的正常範圍(2.8-6.5)")
                            continue
                        
                        suitable_groups.append(group)
            
            # 如果找不到或pKa不在特定範圍，考慮其他官能基
            if not suitable_groups:
                suitable_groups = []  # 清空並重新初始化
                for fg in functional_groups:
                    # 特別處理各種官能基的pKa範圍
                    
                    # 特別處理SulfonicAcid
                    if fg['group_type'] == 'SulfonicAcid' and pka > 4.0:
                        print(f"  跳過 {fg['group_type']} 做猜測，因為pKa > 4.0")
                        continue
                    
                    # 特別處理Phenol
                    if fg['group_type'] == 'Phenol':
                        if 3.0 <= pka <= 6.0 and has_aniline:
                            print(f"  跳過猜測 {fg['group_type']}，因為pKa在Aniline的優先範圍(3.0-6.0)內")
                            continue
                        if 2.8 <= pka <= 6.5 and has_pyridine:
                            print(f"  跳過猜測 {fg['group_type']}，因為pKa在Pyridine的優先範圍(2.8-6.5)內")
                            continue
                        if pka < 8.0:
                            print(f"  警告：考慮將pKa = {pka} < 8.0 映射到 Phenol，這超出了Phenol的正常範圍(8.0-10.5)")
                    
                    # 特別處理Thiol
                    if fg['group_type'] == 'Thiol' and pka < 6.0:
                        print(f"  跳過猜測 {fg['group_type']}，因為pKa < 6.0，低於Thiol的正常範圍(6.0-11.0)")
                        continue
                    
                    # 特別處理Pyridine
                    if fg['group_type'] == 'Pyridine' and pka > 7.0:
                        print(f"  跳過猜測 {fg['group_type']}，因為pKa > 7.0，超出了Pyridine的正常範圍(2.8-6.5)")
                        continue
                    
                    suitable_groups.append(fg)
            
            # 如果找到合適的官能基，使用第一個
            if suitable_groups:
                first_group = suitable_groups[0]
                print(f"  使用第一個合適的官能基 {first_group['group_type']} 做猜測")
            else:
                # 如果實在沒有其他官能基，只好使用第一個官能基（無論是什麼類型）
                first_group = functional_groups[0]
                print(f"  沒有合適的官能基，只能使用第一個官能基 {first_group['group_type']} 做猜測")
            
            pka_to_group_mapping[pka] = {
                'type': first_group['group_type'] + " (guess)",
                'indices': list(first_group['match_indices']) if isinstance(first_group['match_indices'], tuple) else first_group['match_indices']
            }
    
    # ===== STEP 8: 整理結果 =====
    print("\n映射完成，結果摘要:")
    all_functional_groups = []
    for fg in functional_groups:
        # 確保match_indices是列表
        match_indices = list(fg['match_indices']) if isinstance(fg['match_indices'], tuple) else fg['match_indices']
        all_functional_groups.append({
            'type': fg['group_type'],
            'indices': match_indices
        })
    
    # 打印最終映射結果
    for pka, group_info in pka_to_group_mapping.items():
        print(f"  pKa = {pka:.2f} => {group_info['type']} 於位置: {group_info['indices']}")
    
    return {
        'smiles': smiles_str,
        'pka_values': pka_values,
        'functional_groups': all_functional_groups,
        'mapping': pka_to_group_mapping
    }


















def display_pka_mapping_results(results):
    """
    顯示pKa映射結果
    
    Args:
        results: map_pka_to_functional_groups函數的返回結果
    """
    if not results:
        print("無結果可顯示")
        return
    
    print("\n===== pKa值與官能基對應關係 =====")
    print(f"SMILES: {results['smiles']}")
    print(f"發現的官能基:")
    for i, fg in enumerate(results['functional_groups']):
        print(f"  {i+1}. {fg['type']} 於位置: {fg['indices']}")
    
    print("\npKa值對應:")
    
    # 按pKa值從小到大排序
    sorted_pkas = sorted(results['mapping'].keys())
    
    for pka in sorted_pkas:
        group_info = results['mapping'][pka]
        print(f"  pKa = {pka:.2f} => {group_info['type']} 於位置: {group_info['indices']}")
    
    print("===============================\n")

def visualize_pka_mapping(results, output_path="../src/pka_mapping_result.png"):
    """
    可視化pKa映射結果，並高亮顯示匹配到的官能基
    
    Args:
        results: map_pka_to_functional_groups函數的返回結果
        output_path: 輸出圖像的路徑，默認為../src/pka_mapping_result.png
    """
    if not results:
        print("無結果可可視化")
        return
    
    # 創建圖表，使用子圖佈局以便添加分子結構
    fig = plt.figure(figsize=(15, 8))
    
    # 創建網格佈局，2行1列，左側放pKa圖，右側放分子結構
    gs = fig.add_gridspec(1, 2, width_ratios=[2, 1])
    ax = fig.add_subplot(gs[0, 0])  # pKa圖
    mol_ax = fig.add_subplot(gs[0, 1])  # 分子結構圖
    
    # 設置pKa刻度
    pka_min = min(results['pka_values']) - 1
    pka_max = max(results['pka_values']) + 1
    ax.set_xlim(pka_min, pka_max)
    ax.set_ylim(0, 1)
    
    # 繪製pKa刻度線
    for pka in range(int(pka_min), int(pka_max)+1):
        ax.axvline(x=pka, color='lightgray', linestyle='-', alpha=0.7)
    
    # 色彩設置
    colors = {
        'COOH': '#1E90FF',             # 道奇藍
        'PrimaryAmine': '#32CD32',     # 萊姆綠
        'SecondaryAmine': '#9932CC',   # 深蘭紫色
        'TertiaryAmine': '#8B0000',    # 深紅色
        'Aniline': '#FF4500',          # 橙紅色
        'Phenol': '#FFD700',           # 金色
        'Alcohol': '#00CED1',          # 深青色
        'AliphaticRingAlcohol': '#DB7093',  # 淡紫紅色
        'SulfonicAcid': '#FF1493',     # 深粉色
        'PhosphonicAcid': '#006400',   # 深綠色
        'Pyridine': '#4682B4',         # 鋼藍色
        'Imidazole': '#2E8B57',        # 海洋綠
        'ImidazoleLike': '#556B2F',    # 深橄欖綠
        'Thiol': '#8B4513',            # 棕色
        'Indole': '#3D3D3D',           # 深灰色
        'Phosphate': '#728C00',        # 橄欖綠
        'NHydroxylamine': '#8B8B00',   # 黃褐色
        'PyrazoleNitrogen': '#4F94CD'  # 鋼藍色
    }
    
    # 對接近的pKa點進行處理
    sorted_pkas = sorted(results['pka_values'])
    
    # 設置點的垂直位置：接近的pKa點將有不同的垂直位置
    y_positions = {}
    y_labels_top = {}
    y_labels_bottom = {}
    
    # 判斷哪些pKa點需要錯開
    MIN_PKA_DISTANCE = 0.5  # 兩個點間的最小pKa距離，小於此值將錯開顯示
    
    for i, pka in enumerate(sorted_pkas):
        if i > 0 and abs(pka - sorted_pkas[i-1]) < MIN_PKA_DISTANCE:
            # 當前點與前一個點太近，需要錯開
            if i % 2 == 0:  # 偶數索引的點向上移動
                y_positions[pka] = 0.6
                y_labels_top[pka] = 0.85
                y_labels_bottom[pka] = 0.35
            else:  # 奇數索引的點向下移動
                y_positions[pka] = 0.4
                y_labels_top[pka] = 0.65
                y_labels_bottom[pka] = 0.15
        else:
            # 點間距足夠，居中顯示
            y_positions[pka] = 0.5
            y_labels_top[pka] = 0.75
            y_labels_bottom[pka] = 0.25
    
    # 繪製pKa點
    for pka in sorted_pkas:
        group_info = results['mapping'].get(pka, {"type": "未知", "indices": []})
        group_type = group_info['type']
        
        # 處理guess的情況
        if " (guess)" in group_type:
            group_type = group_type.replace(" (guess)", "")
            is_guess = True
        else:
            is_guess = False
        
        # 獲取顏色
        color = colors.get(group_type, '#808080')  # 默認為灰色
        
        # 繪製pKa點，使用計算好的y位置
        y_pos = y_positions[pka]
        if is_guess:
            ax.scatter(pka, y_pos, color=color, s=200, alpha=0.7, edgecolors='black', linewidths=2, marker='o', zorder=10)
            # 添加問號
            ax.text(pka, y_pos, '?', color='white', ha='center', va='center', fontweight='bold', zorder=11)
        else:
            ax.scatter(pka, y_pos, color=color, s=200, alpha=0.7, edgecolors='black', linewidths=2, marker='o', zorder=10)
        
        # 使用計算好的標籤位置
        # 添加pKa標籤
        ax.annotate(f'pKa = {pka:.2f}',
                  xy=(pka, y_pos),
                  xytext=(pka, y_labels_top[pka]),
                  fontsize=10, fontweight='bold',
                  ha='center', va='center',
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'),
                  zorder=12)
        
        # 添加官能基標籤
        ax.annotate(f'{group_type}',
                  xy=(pka, y_pos),
                  xytext=(pka, y_labels_bottom[pka]),
                  fontsize=10, fontweight='bold',
                  ha='center', va='center',
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round,pad=0.2'),
                  zorder=12)
    
    # 美化pKa圖表
    ax.set_title(f"pKa value and functional group mapping", fontsize=14, fontweight='bold')
    ax.set_xlabel("pKa value", fontsize=12)
    ax.set_yticks([])  # 隱藏y軸刻度
    ax.spines['left'].set_visible(False)  # 隱藏左邊框
    ax.spines['right'].set_visible(False)  # 隱藏右邊框
    ax.spines['top'].set_visible(False)  # 隱藏上邊框
    
    # 繪製分子結構，高亮顯示官能基
    try:
        smiles = results['smiles']
        mol = Chem.MolFromSmiles(smiles)
        
        if mol:
            # 添加2D坐標
            AllChem.Compute2DCoords(mol)
            
            try:
                # 創建帶有不同官能基的不同版本的分子
                smiles_variants = []
                labels = []
                group_types = []
                
                # 為每個pKa值創建一個帶有高亮官能基的版本
                pka_values = sorted(results['mapping'].keys())
                
                if not pka_values:  # 如果沒有映射，至少顯示原始分子
                    img = Draw.MolToImage(mol, size=(350, 350))
                else:
                    # 創建一組帶有高亮的分子
                    mols = [mol] * len(pka_values)
                    
                    # 為每個分子添加標題
                    legends = []
                    for pka in pka_values:
                        group_info = results['mapping'][pka]
                        group_type = group_info['type']
                        if " (guess)" in group_type:
                            group_type = group_type.replace(" (guess)", "")
                            legends.append(f"pKa = {pka:.2f} ({group_type}, guess)")
                        else:
                            legends.append(f"pKa = {pka:.2f} ({group_type})")
                    
                    # 獲取每個分子需要高亮的原子
                    highlights = []
                    highlight_colors = []
                    for pka in pka_values:
                        group_info = results['mapping'][pka]
                        group_type = group_info['type']
                        if " (guess)" in group_type:
                            group_type = group_type.replace(" (guess)", "")
                        
                        indices = group_info['indices']
                        # 確保索引是列表
                        if isinstance(indices, tuple):
                            indices = list(indices)
                        
                        # 獲取顏色
                        color = colors.get(group_type, '#808080')  # 默認為灰色
                        # 轉換為RGB元組
                        from matplotlib.colors import to_rgb
                        rgb_color = to_rgb(color)
                        
                        highlights.append(indices)
                        highlight_colors.append(rgb_color)
                    
                    # 使用Draw.MolsToGridImage繪製網格圖像
                    try:
                        # 嘗試使用較新的API
                        # 創建高亮原子顏色的字典列表
                        atom_colors_list = []
                        for i, (indices, color) in enumerate(zip(highlights, highlight_colors)):
                            atom_colors = {}
                            for idx in indices:
                                atom_colors[idx] = color
                            atom_colors_list.append(atom_colors)
                            
                        img = Draw.MolsToGridImage(
                            mols, 
                            molsPerRow=2,
                            subImgSize=(350, 350),
                            legends=legends,
                            highlightAtomLists=highlights,
                            highlightBondLists=[[] for _ in range(len(mols))],
                            highlightAtomColors=atom_colors_list,
                            useSVG=False
                        )
                    except Exception as e:
                        print(f"網格繪圖錯誤: {e}")
                        # 如果無法使用網格圖像或顏色高亮，嘗試使用其他方法
                        try:
                            # 檢查版本
                            from rdkit import rdBase
                            rdkit_version = rdBase.rdkitVersion.split('.')
                            version_supports_colors = False
                            if int(rdkit_version[0]) > 2020:
                                version_supports_colors = True
                            
                            # 確認是否支援顏色高亮
                            if version_supports_colors:
                                img = Draw.MolsToGridImage(
                                    mols, 
                                    molsPerRow=2,
                                    subImgSize=(350, 350),
                                    legends=legends,
                                    highlightAtomLists=highlights,
                                    highlightBondLists=[[] for _ in range(len(mols))],
                                    highlightAtomColors=atom_colors_list,
                                    useSVG=False
                                )
                            else:
                                # 使用一系列單獨的分子繪圖，然後拼接成一個網格
                                import numpy as np
                                from PIL import Image
                                
                                # 計算網格佈局
                                n_mols = len(mols)
                                if n_mols <= 2:
                                    n_cols = n_mols
                                    n_rows = 1
                                else:
                                    n_cols = 2
                                    n_rows = (n_mols + 1) // 2
                                
                                # 創建每個分子的圖像
                                mol_imgs = []
                                for i, (mol, indices, legend, color) in enumerate(zip(mols, highlights, legends, highlight_colors)):
                                    # 使用RDKit的Draw.MolToImage繪製單個分子
                                    mol_img = Draw.MolToImage(
                                        mol,
                                        size=(350, 350),
                                        highlightAtoms=indices,
                                        highlightColor=color,  # 使用顏色
                                    )
                                    
                                    # 添加標題
                                    from PIL import ImageDraw, ImageFont
                                    draw = ImageDraw.Draw(mol_img)
                                    try:
                                        font = ImageFont.truetype("arial.ttf", 12)
                                    except:
                                        font = ImageFont.load_default()
                                    
                                    # 在底部添加文本
                                    draw.text((10, 330), legend, fill=(0, 0, 0), font=font)
                                    
                                    mol_imgs.append(mol_img)
                                
                                # 創建網格佈局的新圖像
                                grid_width = n_cols * 350
                                grid_height = n_rows * 350
                                grid_img = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
                                
                                # 將分子圖像粘貼到網格中
                                for i, mol_img in enumerate(mol_imgs):
                                    row = i // n_cols
                                    col = i % n_cols
                                    x = col * 350
                                    y = row * 350
                                    grid_img.paste(mol_img, (x, y))
                                
                                img = grid_img
                        except Exception as e:
                            print(f"高級繪圖錯誤: {e}")
                            # 如果高級方法失敗，使用基本繪圖
                            try:
                                # 嘗試使用沒有顏色的網格圖像
                                img = Draw.MolsToGridImage(
                                    mols, 
                                    molsPerRow=2,
                                    subImgSize=(350, 350),
                                    legends=legends,
                                    highlightAtomLists=highlights,
                                    useSVG=False
                                )
                            except:
                                # 如果所有方法都失敗，使用最基本的繪圖
                                img = Draw.MolToImage(mol, size=(350, 350))
                
                # 分析官能基映射添加圖例
                used_colors = {}
                for pka, group_info in results['mapping'].items():
                    group_type = group_info['type']
                    if " (guess)" in group_type:
                        group_type = group_type.replace(" (guess)", "")
                    color = colors.get(group_type, '#808080')
                    used_colors[group_type] = color
                
                # 添加圖例到整個圖的上方而非右下角
                if used_colors:
                    legend_elements = []
                    from matplotlib.patches import Patch
                    for group_type, color in used_colors.items():
                        legend_elements.append(Patch(facecolor=color, edgecolor='black', label=group_type))
                    
                    # 使用fig.legend讓圖例位於整個圖的上方
                    fig.legend(handles=legend_elements, 
                            loc='upper center', 
                            title="Functional Groups", 
                            fontsize=9,
                            title_fontsize=10,
                            framealpha=0.8,
                            bbox_to_anchor=(0.8, 0.98),
                            ncol=min(len(used_colors), 4))  # 根據官能基數量設置列數
                
                # 在matplotlib中顯示這個圖像
                mol_ax.imshow(img)
                mol_ax.set_title(f"Molecular Structure: {results['smiles']}", fontsize=14, fontweight='bold')
                mol_ax.axis('off')  # 隱藏坐標軸
                
            except Exception as e:
                print(f"高亮繪圖錯誤: {e}")
                # 如果高亮失敗，改用基本繪圖
                img = Draw.MolToImage(mol, size=(400, 400), kekulize=True, fitImage=True)
                mol_ax.imshow(img)
                mol_ax.set_title(f"分子結構 (無高亮): {results['smiles']}", fontsize=14, fontweight='bold')
                mol_ax.axis('off')
                
        else:
            mol_ax.text(0.5, 0.5, "cannot draw molecule structure", 
                      ha='center', va='center', fontsize=14, color='red')
            mol_ax.axis('off')
    except Exception as e:
        print(f"繪製分子結構時出錯: {e}")
        mol_ax.text(0.5, 0.5, f"error when drawing molecule structure: {str(e)}", 
                  ha='center', va='center', fontsize=12, color='red', wrap=True)
        mol_ax.axis('off')
    
    # 保存圖像
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"可視化結果已保存至: {output_path}")
    
    
if __name__ == "__main__":

    output_dir = "../src"
    # 允許用戶輸入SMILES進行查詢
    while True:
        user_input = input("\n請輸入要查詢的SMILES (輸入'e'退出): ")
        if user_input.lower() == 'e':
            break
            
        results = map_pka_to_functional_groups(user_input)
        if results:
            display_pka_mapping_results(results)
            visualize_pka_mapping(results, output_dir + "/pka_mapping_result.png") 
    
            
            
