
# OMGNN: pKa 預測模型

## 專案概述

OMGNN (Organic Molecule Graph Neural Network) 是一個用於預測有機分子 pKa 值的深度學習系統。該系統基於圖形神經網路(GNN)，能夠從分子結構中學習特徵並準確預測解離常數的負對數(pKa)。本專案特別強化了對解離順序的預測和處理能力。

## Workflow-進行pka的mapping
1. pka_prediction_kmeans.py:
[](src/pka_prediction_kmeans.py)
    定義了官能基種類以及重點函式extract_molecule_feature
    同時也能繪出t-SNE圖
    - 執行方式: sbatch

    1.5. 可另外使用evaluate_tsne.py看smiles在裡面的位置
[](src/evaluate_tsne.py)

2. pka_smiles_query.py:
[](src/pka_smiles_query.py)
    進行繪圖及單個分子測試
    定義了繪圖重點函式
    - 執行方式: python kernel
3. pka_draw_multiple_smiles.py:
[](src/pka_draw_multiple_smiles.py)
    將所有的分子都畫出單一的圖
    - 執行方式: sbatch
    - 執行時間: 大約15分鐘
4. pka_functional_group_analysis.py:
[](src/pka_functional_group_analysis.py)
    畫出官能基的分佈
    - 執行方式: python kernel
5. pka_reverse_search.py:
[](src/pka_reverse_search.py)
    Given pKa, functional group
    回傳範圍內的smiles
    - 執行方式: python kernel


## Database
processed_pka_data.csv
[](data/processed_pka_data.csv)

    儲存了smiles, pka等重要資訊

# Self-automation

## 其他功能性腳本:
functional_group_labeler.py:
[](src/functional_group_labeler.py)

    調用FunctionalGroupLabeler()類


pka_learning_curve.py:
[](src/pka_learning_curve.py)
    
    可以調用並畫學習曲線

## 模型核心
1. self_pka_models.py
[](src/self_pka_models.py)
- class BondMessagePassing(nn.Module):
- class pka_GNN(nn.Module):
    - forward()
    最後能呼叫self_pka_sample.py的all_in_one畫出parity plot
2. self_pka_chemutils.py
[](src/self_pka_chemutils.py)
- tensorize_for_pka(): 將分子轉為向量
3. self_pka_trainutils.py
[](src/self_pka_trainutils.py)
- class pka_Dataloader():
    - data_loader_pka(): 建立dataloader
    - evaluate_pka_model(): 進行測試集評估
4. self_pka_pretrain_GCN1.py
[](src/self_pka_pretrain_GCN1.py)
- 執行訓練
5. self_pka_sample.py
    all_in_one能進行train, test dataset做sample並且畫出parity plot
6. self_pka_preprocess.py:
[](src/self_pka_preprocess.py)

    可以生成模型所需的檔案: pka_mapping_results.csv
    裡面有儲存相對應(id, pka)的格式
    - 新增檢查機制，找到的官能基數量應該大於實際的pka數量
## Database
pka_mapping_results.csv
[](output/pka_mapping_results.csv)

## Workflow
![](src/test.drawio.png)
