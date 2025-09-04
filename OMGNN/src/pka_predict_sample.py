#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
pka_predict_sample.py

此腳本演示如何加載訓練好的pKa預測模型，並使用sample方法從SMILES字符串預測pKa值。
同時提供結果的可視化功能。
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

# 添加路徑以導入本地模塊
sys.path.append('src/')

# 導入本地模塊
from OMGNN.src.error_self_pka_models import pka_GNN
from self_pka_chemutils import tensorize_for_pka


def load_pka_model(model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    載入預訓練的pKa預測模型
    
    Args:
        model_path: 模型權重文件路徑
        device: 計算設備 ('cuda' 或 'cpu')
        
    Returns:
        加載權重後的pKa_GNN模型
    """
    # 設定模型參數，這些應與訓練時相同
    node_dim = 153    # 節點特徵維度
    bond_dim = 11     # 鍵特徵維度
    hidden_dim = 128  # 隱藏層維度
    output_dim = 1    # 輸出維度
    dropout = 0.2     # Dropout率
    
    # 初始化模型
    model = pka_GNN(node_dim, bond_dim, hidden_dim, output_dim, dropout)
    
    # 載入預訓練權重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        # 檢查是否是完整的模型或僅權重
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"成功載入模型權重從: {model_path}")
    except Exception as e:
        print(f"載入模型權重時出錯: {e}")
        raise
    
    # 將模型移至指定設備並設為評估模式
    model = model.to(device)
    model.eval()
    
    return model


def visualize_pka_result(result, output_path=None, show=True):
    """
    將pKa預測結果可視化，標記有pKa值的原子
    
    Args:
        result: sample方法返回的結果字典
        output_path: 可選，保存圖像的路徑
        show: 是否顯示圖像
        
    Returns:
        None
    """
    if result is None:
        print("無法可視化：結果為None")
        return
    
    # 獲取分子和預測結果
    mol = result['mol']
    smiles = result['smiles']
    pka_positions = result['pka_positions']
    pka_values = result['pka_values']
    
    # 確保分子有三維座標用於更好的可視化
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    
    # 設置圖像尺寸和字體大小
    drawer = rdMolDraw2D.MolDraw2DCairo(800, 500)
    drawer.SetFontSize(0.8)
    
    # 準備原子標註：顯示原子索引和pKa值
    atom_notes = {}
    atom_colors = {}
    
    for i, pos in enumerate(pka_positions):
        pka_val = pka_values[i]
        atom_notes[pos] = f"{pos}: pKa={pka_val:.2f}"
        # 設置pKa原子顏色
        atom_colors[pos] = (0.2, 0.8, 0.2)  # 綠色
    
    # 生成帶有高亮和標註的分子圖像
    drawer.DrawMolecule(
        mol,
        highlightAtoms=pka_positions,
        highlightAtomColors=atom_colors,
        highlightBonds=[],
        atomLabels=atom_notes
    )
    drawer.FinishDrawing()
    
    # 獲取PNG數據
    png_data = drawer.GetDrawingText()
    
    # 如果指定了輸出路徑，保存圖像
    if output_path:
        with open(output_path, 'wb') as f:
            f.write(png_data)
        print(f"分子圖像已保存至: {output_path}")
    
    # 如果需要顯示，則創建matplotlib圖形
    if show:
        import io
        from PIL import Image
        
        # 從PNG數據創建圖像
        img = Image.open(io.BytesIO(png_data))
        
        # 顯示圖像和結果摘要
        plt.figure(figsize=(12, 8))
        
        # 圖像
        plt.subplot(1, 1, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"SMILES: {smiles}\npKa預測結果", fontsize=12)
        
        # 添加結果摘要文本
        result_text = "預測pKa位置及數值:\n"
        for i, pos in enumerate(pka_positions):
            result_text += f"原子 {pos}: pKa = {pka_values[i]:.2f}\n"
        plt.figtext(0.02, 0.02, result_text, fontsize=10, wrap=True)
        
        plt.tight_layout()
        plt.show()


def predict_pka_from_smiles(model, smiles, device=None, visualize=True, output_path=None):
    """
    使用給定模型從SMILES字符串預測pKa值
    
    Args:
        model: 預訓練的pKa_GNN模型
        smiles: 分子的SMILES字符串
        device: 計算設備
        visualize: 是否可視化結果
        output_path: 可選，保存圖像的路徑
        
    Returns:
        預測結果字典
    """
    # 進行pKa預測
    result = model.sample(smiles, device)
    
    # 打印預測結果
    if result:
        print(f"SMILES: {smiles}")
        print("預測的pKa位置及數值:")
        for i, pos in enumerate(result['pka_positions']):
            print(f"  原子 {pos}: pKa = {result['pka_values'][i]:.2f}")
        
        # 可視化結果
        if visualize:
            visualize_pka_result(result, output_path)
    else:
        print(f"預測 '{smiles}' 時出錯")
    
    return result


def main():
    """主函數，處理命令行參數並執行pKa預測"""
    parser = argparse.ArgumentParser(description='從SMILES字符串預測pKa值')
    parser.add_argument('--model', type=str, required=True, help='預訓練模型權重的路徑')
    parser.add_argument('--smiles', type=str, required=True, help='要預測的分子SMILES字符串')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='計算設備 (cuda 或 cpu)')
    parser.add_argument('--output', type=str, default=None, help='保存可視化結果的文件路徑')
    parser.add_argument('--no-viz', action='store_true', help='禁用可視化')
    
    args = parser.parse_args()
    
    # 檢查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"錯誤：模型文件不存在: {args.model}")
        return
    
    # 載入模型
    model = load_pka_model(args.model, args.device)
    
    # 進行預測並可視化
    predict_pka_from_smiles(
        model,
        args.smiles,
        args.device,
        visualize=not args.no_viz,
        output_path=args.output
    )


if __name__ == "__main__":
    main() 