#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CustomMetalPKA_GNN 模型預測演示腳本

這個腳本展示如何使用 CustomMetalPKA_GNN 模型進行單個預測的完整流程。
包含模型結構說明、預測過程和結果解釋。

Usage:
    python demo_metal_prediction.py
"""

import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
import os
import sys

# 添加路徑以導入模型
sys.path.append("../src/")
sys.path.append("./src/")

def demonstrate_metal_pka_prediction():
    """演示金屬配體 pKa 預測的完整流程"""
    
    print("=" * 70)
    print("           CustomMetalPKA_GNN 模型預測演示")
    print("=" * 70)
    
    # 1. 模型架構說明
    print("\n🏗️  模型架構特點:")
    print("   • 支持多配體複合物預測 (ML, ML₂, ML₃)")
    print("   • 配體內部圖卷積 + Transformer注意力機制")
    print("   • 金屬嵌入向量 + 自適應門控融合")
    print("   • 逐階穩定常數預測（模擬配體逐步解離）")
    
    # 2. 測試分子案例
    print("\n🧪 測試案例: 乙二胺 (Ethylenediamine) 與 Cu2+ 配位")
    
    test_smiles = "NCCN"  # 乙二胺
    metal_ion = "Cu2+"
    binding_sites = [0, 3]  # 兩個氮原子
    
    print(f"   • SMILES: {test_smiles}")
    print(f"   • 金屬離子: {metal_ion}")
    print(f"   • 結合位點: 原子索引 {binding_sites} (兩個氮原子)")
    
    # 3. 分子結構分析
    mol = Chem.MolFromSmiles(test_smiles)
    if mol is not None:
        print(f"\n🔬 分子結構分析:")
        print(f"   • 原子數量: {mol.GetNumAtoms()}")
        print(f"   • 鍵數量: {mol.GetNumBonds()}")
        
        print(f"   • 原子詳情:")
        for i, atom in enumerate(mol.GetAtoms()):
            symbol = atom.GetSymbol()
            marker = " ⭐" if i in binding_sites else ""
            print(f"     [{i}] {symbol}{marker}")
    
    # 4. 模型預測流程說明
    print(f"\n⚙️  模型預測流程:")
    print(f"   1. 配體分子圖卷積 → 原子特徵增強")
    print(f"   2. 金屬嵌入 → 金屬特性編碼")
    print(f"   3. 結合位點選擇 → 配位原子確定")
    print(f"   4. 骨架特徵融合 → 全分子信息整合")
    print(f"   5. 金屬-配體星狀圖構建 → 配位結構建模")
    print(f"   6. Transformer注意力 → 複合物表示學習")
    print(f"   7. 多階預測輸出 → ML₂, ML₁ 常數")
    
    # 5. 模擬預測結果（展示格式）
    print(f"\n📊 預測結果示例:")
    
    # 模擬一些合理的預測值
    np.random.seed(42)  # 為了演示一致性
    
    # 乙二胺-銅配合物的典型 pKa 值範圍
    simulated_predictions = {
        "ML₂": 18.8,  # [Cu(en)₂]²⁺ 第二個配體解離
        "ML₁": 10.7   # [Cu(en)]²⁺ 第一個配體解離
    }
    
    print(f"   配位狀態 | 預測 pKa | 化學意義")
    print(f"   --------|----------|------------------")
    for state, pka in simulated_predictions.items():
        if state == "ML₂":
            meaning = "雙配體複合物穩定性"
        else:
            meaning = "單配體複合物穩定性"
        print(f"   {state:7} | {pka:8.1f} | {meaning}")
    
    # 6. 結果解釋
    print(f"\n📈 結果解釋:")
    print(f"   • pKa 值越大 → 配位結合越強")
    print(f"   • ML₂ > ML₁ → 第二個配體更難解離（協同效應）")
    print(f"   • 乙二胺是雙齒配體，形成穩定的螯合結構")
    
    # 7. 預測可靠性評估
    print(f"\n🎯 預測可靠性指標:")
    
    reliability_factors = {
        "結合位點準確性": "高 (基於化學規則驗證)",
        "分子特徵完整性": "高 (包含鍵接和電子資訊)",
        "金屬特異性": "中 (依賴訓練數據覆蓋度)",
        "多配體建模": "高 (顯式建模配位結構)"
    }
    
    for factor, level in reliability_factors.items():
        print(f"   • {factor}: {level}")
    
    # 8. 實際應用建議
    print(f"\n💡 實際應用建議:")
    print(f"   1. 驗證結合位點預測的化學合理性")
    print(f"   2. 與實驗數據或文獻值對比校驗") 
    print(f"   3. 考慮溶劑、溫度、離子強度等環境因素")
    print(f"   4. 對於新型配體，建議謹慎解釋結果")
    
    # 9. 模型限制說明
    print(f"\n⚠️  模型限制:")
    print(f"   • 僅支持同種配體的配位複合物")
    print(f"   • 最多支持3個配體的配位狀態")
    print(f"   • 預測精度依賴訓練數據質量")
    print(f"   • 不考慮立體化學效應")
    
    print(f"\n" + "=" * 70)
    print(f"           演示完成！")
    print(f"   使用 quick_test_metal_model.py 進行實際預測測試")
    print(f"   使用 test_custom_metal_sample.py 進行批量評估")
    print(f"=" * 70)

def create_simple_visualization():
    """創建簡單的預測結果可視化"""
    
    print(f"\n📊 創建預測結果可視化...")
    
    # 模擬不同金屬離子的預測結果
    metals = ['Cu2+', 'Zn2+', 'Ni2+', 'Co2+', 'Mg2+']
    ml2_pkas = [18.8, 16.5, 17.2, 15.9, 12.3]  # ML₂ 複合物
    ml1_pkas = [10.7, 9.2, 10.1, 8.8, 6.5]     # ML₁ 複合物
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 圖1: 不同金屬離子的pKa比較
    x = np.arange(len(metals))
    width = 0.35
    
    ax1.bar(x - width/2, ml2_pkas, width, label='ML₂', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, ml1_pkas, width, label='ML₁', alpha=0.8, color='lightcoral')
    
    ax1.set_xlabel('metal ion')
    ax1.set_ylabel('predict')
    ax1.set_title('Prediction binding stability of metal-ligand complexes')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metals)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 圖2: 配位狀態穩定性趨勢
    ligand_states = ['ML₃', 'ML₂', 'ML₁']
    stability = [20.5, 18.8, 10.7]  # 示例穩定性常數
    
    ax2.plot(ligand_states, stability, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('binding state')
    ax2.set_ylabel('stability constant')
    ax2.set_title('Stability of different binding states')
    ax2.grid(True, alpha=0.3)
    
    # 添加標註
    for i, (state, pka) in enumerate(zip(ligand_states, stability)):
        ax2.annotate(f'{pka:.1f}', (i, pka), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    
    # 保存圖片
    output_dir = "./demo_output"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'metal_pka_prediction_demo.png'), 
                dpi=300, bbox_inches='tight')
    
    print(f"   ✓ 可視化圖表已保存到: {output_dir}/metal_pka_prediction_demo.png")
    
    # 不顯示圖片，避免在服務器環境中出錯
    plt.close()

if __name__ == "__main__":
    # 執行演示
    demonstrate_metal_pka_prediction()
    
    # 創建可視化
    try:
        create_simple_visualization()
    except Exception as e:
        print(f"   ⚠️  可視化創建失敗: {e}")
        print(f"   (這通常是因為缺少圖形環境，不影響主要功能)")
    
    print(f"\n🚀 下一步操作建議:")
    print(f"   1. python quick_test_metal_model.py  # 快速測試模型")
    print(f"   2. python test_custom_metal_sample.py  # 詳細評估")
    print(f"   3. 準備你自己的 SMILES 和金屬離子進行預測！")