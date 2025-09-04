#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Metal pKa Dataset
===============

處理金屬離子環境下的分子pKa數據集
"""

import os
import torch
import numpy as np
import pandas as pd
from rdkit import Chem
from torch_geometric.data import Dataset, Data
import glob
import json
from src.metal_chemutils import tensorize_for_pka, atom_features
from src.metal_models import MetalFeatureExtractor


class MetalPKADataset(Dataset):
    """
    金屬pKa數據集類
    
    處理包含金屬離子環境下的分子及其pKa值的數據集
    """
    
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        """
        初始化金屬pKa數據集
        
        Args:
            root: 數據集根目錄
            transform: 轉換函數
            pre_transform: 預處理轉換函數
            pre_filter: 預處理過濾函數
        """
        self.root = root
        super(MetalPKADataset, self).__init__(root, transform, pre_transform, pre_filter)
        
        # 加載數據
        self._load_data()
    
    @property
    def raw_file_names(self):
        """獲取原始文件名列表"""
        # 支持CSV或JSON格式
        csv_files = glob.glob(os.path.join(self.root, "*.csv"))
        json_files = glob.glob(os.path.join(self.root, "*.json"))
        return csv_files + json_files
    
    @property
    def processed_file_names(self):
        """獲取處理後的文件名列表"""
        # 如果已經處理過，返回處理後的文件名
        processed_files = glob.glob(os.path.join(self.processed_dir, "data_*.pt"))
        if processed_files:
            return [os.path.basename(f) for f in processed_files]
        return []
    
    def _load_data(self):
        """載入數據集"""
        self.data_list = []
        
        # 如果存在處理過的文件，直接載入
        if self.processed_file_names:
            for i, file_name in enumerate(self.processed_file_names):
                data = torch.load(os.path.join(self.processed_dir, file_name))
                self.data_list.append(data)
            return
        
        # 否則從原始文件載入
        for file_path in self.raw_file_names:
            if file_path.endswith('.csv'):
                self._load_from_csv(file_path)
            elif file_path.endswith('.json'):
                self._load_from_json(file_path)
        
        # 保存處理後的數據
        self._save_processed_data()
    
    def _load_from_csv(self, file_path):
        """從CSV文件載入數據"""
        try:
            df = pd.read_csv(file_path)
            
            # 檢查必要的列
            required_columns = ['smiles', 'metal_ion', 'pka_matrix']
            if not all(col in df.columns for col in required_columns):
                print(f"警告: {file_path} 缺少必要的列: {required_columns}")
                return
            
            # 處理每一行數據
            for _, row in df.iterrows():
                try:
                    # 獲取SMILES和金屬離子
                    smiles = row['smiles']
                    metal_ion = row['metal_ion']
                    
                    # 解析pKa矩陣 (格式: [[atom_idx, pKa_value], ...])
                    if isinstance(row['pka_matrix'], str):
                        pka_matrix = eval(row['pka_matrix'])
                    else:
                        pka_matrix = row['pka_matrix']
                    
                    # 處理成PyTorch Geometric數據
                    data = self._create_data_object(smiles, metal_ion, pka_matrix)
                    if data is not None:
                        self.data_list.append(data)
                except Exception as e:
                    print(f"處理數據行時發生錯誤: {e}")
        except Exception as e:
            print(f"載入CSV文件 {file_path} 時發生錯誤: {e}")
    
    def _load_from_json(self, file_path):
        """從JSON文件載入數據"""
        try:
            with open(file_path, 'r') as f:
                data_list = json.load(f)
            
            for item in data_list:
                try:
                    # 檢查必要的鍵
                    required_keys = ['smiles', 'metal_ion', 'pka_matrix']
                    if not all(key in item for key in required_keys):
                        print(f"警告: 數據項缺少必要的鍵: {required_keys}")
                        continue
                    
                    # 獲取SMILES和金屬離子
                    smiles = item['smiles']
                    metal_ion = item['metal_ion']
                    pka_matrix = item['pka_matrix']
                    
                    # 處理成PyTorch Geometric數據
                    data = self._create_data_object(smiles, metal_ion, pka_matrix)
                    if data is not None:
                        self.data_list.append(data)
                except Exception as e:
                    print(f"處理JSON數據項時發生錯誤: {e}")
        except Exception as e:
            print(f"載入JSON文件 {file_path} 時發生錯誤: {e}")
    
    def _create_data_object(self, smiles, metal_ion, pka_matrix):
        """
        創建PyTorch Geometric數據對象
        
        Args:
            smiles: 分子SMILES字符串
            metal_ion: 金屬離子字符串 (例如: "Cu2+")
            pka_matrix: pKa矩陣，格式為 [[atom_idx, pKa_value], ...]
            
        Returns:
            Data: PyTorch Geometric數據對象
        """
        try:
            # 解析分子
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"無法解析SMILES: {smiles}")
                return None
            
            # 獲取原子特徵、邊索引和邊特徵
            fatoms, edge_index, edge_attr = tensorize_for_pka(smiles)
            
            # 創建pKa標籤
            num_atoms = fatoms.size(0)
            pka_labels = torch.zeros(num_atoms)
            
            for atom_idx, pka_value in pka_matrix:
                if 0 <= atom_idx < num_atoms:
                    pka_labels[atom_idx] = pka_value
            
            # 獲取金屬特徵
            metal_features = None
            if metal_ion:
                try:
                    metal_features = [MetalFeatureExtractor.get_metal_features(metal_ion)]
                except Exception as e:
                    print(f"處理金屬特徵時出錯: {e}")
            
            # 創建數據對象
            data = Data(
                x=fatoms,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pka_labels=pka_labels,
                smiles=smiles,
                metal_ion=metal_ion,
                metal_features=metal_features
            )
            
            return data
        except Exception as e:
            print(f"創建數據對象時發生錯誤: {e}")
            return None
    
    def _save_processed_data(self):
        """保存處理後的數據"""
        # 創建處理目錄
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # 保存每個數據對象
        for i, data in enumerate(self.data_list):
            torch.save(data, os.path.join(self.processed_dir, f'data_{i}.pt'))
    
    def len(self):
        """獲取數據集長度"""
        return len(self.data_list)
    
    def get(self, idx):
        """獲取指定索引的數據"""
        return self.data_list[idx]
    
    @property
    def processed_dir(self):
        """獲取處理目錄"""
        return os.path.join(self.root, 'processed')


# 生成示例數據集的工具函數
def create_example_dataset(output_dir, num_samples=10):
    """
    創建示例金屬pKa數據集
    
    Args:
        output_dir: 輸出目錄
        num_samples: 示例數量
        
    Returns:
        bool: 創建成功返回True
    """
    try:
        # 示例SMILES
        smiles_list = [
            'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',  # 咖啡因
            'OC1=C(O)C=CC=C1',  # 間苯二酚
            'CC(=O)OC1=CC=CC=C1C(=O)O',  # 阿司匹林
            'NC1=CC=C(C=C1)C(=O)O',  # 對氨基苯甲酸
            'CC1=CC=C(C=C1)O',  # 對甲基苯酚
            'CC(C)(C)C1=CC=C(C=C1)O',  # 對叔丁基苯酚
            'OC1=CC=CC=C1C(=O)O',  # 水楊酸
            'CCOC(=O)C1=CC=CC=C1C(=O)O',  # 阿司匹林酯
            'CC(=O)OC1=CC=CC=C1O',  # 乙酰基水楊酸
            'OC1=CC=C(C=C1)C(=O)O'   # 對羥基苯甲酸
        ]
        
        # 金屬離子
        metal_ions = ['Cu2+', 'Zn2+', 'Fe3+', 'Mg2+', 'Ca2+']
        
        # 創建輸出目錄
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成數據
        data_list = []
        for _ in range(num_samples):
            # 隨機選擇SMILES和金屬離子
            smiles = np.random.choice(smiles_list)
            metal_ion = np.random.choice(metal_ions)
            
            # 解析分子
            mol = Chem.MolFromSmiles(smiles)
            num_atoms = mol.GetNumAtoms()
            
            # 隨機生成pKa矩陣
            num_pka_sites = np.random.randint(1, 3)  # 1-2個pKa位點
            pka_matrix = []
            
            for _ in range(num_pka_sites):
                atom_idx = np.random.randint(0, num_atoms)
                pka_value = np.random.uniform(3.0, 10.0)  # 合理的pKa範圍
                pka_matrix.append([atom_idx, pka_value])
            
            # 添加到數據列表
            data_list.append({
                'smiles': smiles,
                'metal_ion': metal_ion,
                'pka_matrix': pka_matrix
            })
        
        # 保存為CSV文件
        df = pd.DataFrame(data_list)
        df.to_csv(os.path.join(output_dir, 'example_data.csv'), index=False)
        
        # 保存為JSON文件
        with open(os.path.join(output_dir, 'example_data.json'), 'w') as f:
            json.dump(data_list, f, indent=2)
        
        print(f"成功創建示例數據集，保存在 {output_dir}")
        return True
    except Exception as e:
        print(f"創建示例數據集時發生錯誤: {e}")
        return False


# 測試函數
def test_dataset(dataset_dir):
    """測試金屬pKa數據集"""
    try:
        dataset = MetalPKADataset(dataset_dir)
        print(f"數據集大小: {len(dataset)}")
        
        if len(dataset) > 0:
            # 檢查第一個數據
            data = dataset[0]
            print(f"SMILES: {data.smiles}")
            print(f"金屬離子: {data.metal_ion}")
            print(f"原子特徵形狀: {data.x.shape}")
            print(f"邊索引形狀: {data.edge_index.shape}")
            print(f"邊特徵形狀: {data.edge_attr.shape}")
            print(f"pKa標籤: {data.pka_labels}")
            
            # 檢查是否有金屬特徵
            if hasattr(data, 'metal_features') and data.metal_features is not None:
                print("金屬特徵存在")
            else:
                print("無金屬特徵")
        
        return True
    except Exception as e:
        print(f"測試數據集時發生錯誤: {e}")
        return False


if __name__ == '__main__':
    # 創建示例數據集
    example_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'metal_pka_example')
    create_example_dataset(example_dir)
    
    # 測試數據集
    test_dataset(example_dir) 