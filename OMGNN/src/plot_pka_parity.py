import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def plot_parity(true_values, predicted_values, title="pKa Prediction vs. Experimental", out_png=None, mol_names=None, is_train=None):
    """
    繪製pKa預測值與真實值的對比圖（parity plot）
    
    Args:
        true_values: 真實pKa值的列表或陣列
        predicted_values: 預測pKa值的列表或陣列
        title: 圖表標題
        save_path: 如果提供，圖表將保存到此路徑
        mol_names: 分子名稱列表，用於資料點標籤（可選）
        is_train: 布林值列表，標記每個數據點是否為訓練集（True）或測試集（False）
    """
    # 確保輸入是numpy陣列
    true_values = np.array(true_values)
    predicted_values = np.array(predicted_values)
    
    # 計算誤差指標
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    
    # 確定圖表範圍
    min_val = min(np.min(true_values), np.min(predicted_values)) - 1
    max_val = max(np.max(true_values), np.max(predicted_values)) + 1
    
    # 創建圖表
    plt.figure(figsize=(8, 8), dpi=300)
    
    # 繪製對角線（y=x線）
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1)
    
    # 繪製資料點，區分訓練集和測試集
    if is_train is not None:
        # 將數據分為訓練集和測試集
        is_train = np.array(is_train)
        train_indices = np.where(is_train)[0]
        test_indices = np.where(~is_train)[0]
        
        # 繪製訓練集數據點（藍色）
        plt.scatter(true_values[train_indices], predicted_values[train_indices], 
                   c='royalblue', edgecolor='navy', alpha=0.7, s=50, label='Training Set')
        
        # 繪製測試集數據點（紅色）
        plt.scatter(true_values[test_indices], predicted_values[test_indices], 
                   c='crimson', edgecolor='darkred', alpha=0.7, s=50, label='Testing Set')
        
        # 添加圖例
        plt.legend(loc='lower right', fontsize=12)
    else:
        # 如果沒有指定訓練/測試標記，則所有點使用同一顏色
        plt.scatter(true_values, predicted_values, c='royalblue', edgecolor='navy', 
                  alpha=0.7, s=50)
    
    # 添加標籤（如果提供）
    if mol_names is not None:
        for i, name in enumerate(mol_names):
            plt.annotate(name, (true_values[i], predicted_values[i]), 
                         fontsize=8, alpha=0.7, 
                         xytext=(5, 5), textcoords='offset points')
    
    # 設置軸標籤和標題
    plt.xlabel('Experimental pKa', fontsize=14)
    plt.ylabel('Predicted pKa', fontsize=14)
    plt.title(title, fontsize=16)
    
    # 添加性能指標文字
    plt.text(0.05, 0.95, f'RMSE = {rmse:.2f}\nMAE = {mae:.2f}\nR² = {r2:.2f}',
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    # 設置刻度
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))
    plt.gca().yaxis.set_major_locator(MultipleLocator(2))
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
    
    # 設置網格線
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 調整圖表邊界和佈局
    plt.tight_layout()
    
    # 保存圖表（如果提供了路徑）
    if out_png:
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=300, bbox_inches='tight')
        print(f"圖表已保存至: {out_png}")
    
    # 顯示圖表
    plt.show()
    
    # 返回性能指標
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def load_results_from_csv(csv_file):
    """從CSV文件中載入結果資料"""
    df = pd.read_csv(csv_file)
    
    # 檢查必要的列是否存在
    required_cols = ['true_pka', 'predicted_pka']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV文件中缺少必要的列: {col}")
    
    true_values = df['true_pka'].values
    predicted_values = df['predicted_pka'].values
    
    # 如果存在分子名稱列，則載入
    mol_names = None
    if 'molecule_name' in df.columns:
        mol_names = df['molecule_name'].values
    
    # 如果存在數據集標記列，則載入
    is_train = None
    if 'is_train' in df.columns:
        is_train = df['is_train'].values.astype(bool)
    
    return true_values, predicted_values, mol_names, is_train

def combine_csv_files(train_file, test_file, output_file=None):
    """合併訓練集和測試集的CSV文件"""
    # 載入訓練集和測試集數據
    train_df = pd.read_csv(train_file)
    train_df['is_train'] = True
    
    test_df = pd.read_csv(test_file)
    test_df['is_train'] = False
    
    # 合併數據
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 如果提供輸出路徑，保存結果
    if output_file:
        combined_df.to_csv(output_file, index=False)
        print(f"已將合併數據保存到: {output_file}")
    
    return combined_df

def plot_train_test_comparison():
    """繪製訓練集和測試集預測結果的對比圖"""
    # 定義輸入輸出文件路徑
    train_file = "../output/pka_train_prediction_results.csv"
    test_file = "../output/pka_test_prediction_results.csv"
    combined_output = "../output/pka_combined_predictions.csv"
    plot_output = "../output/pka_train_test_comparison.png"
    
    # 合併訓練集和測試集數據
    # combined_df = combine_csv_files(train_file, test_file, combined_output)
    
    # 從合併數據中載入結果
    true_values, predicted_values, mol_names, is_train = load_results_from_csv(combined_output)
    
    # 繪製parity plot
    metrics = plot_parity(
        true_values,
        predicted_values,
        title="pKa Prediction (Training vs Testing)",
        save_path=plot_output,
        mol_names=None,  # 不顯示分子名稱以避免圖表過於擁擠
        is_train=is_train
    )
    
    # 輸出性能指標
    print(f"性能指標:")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R²: {metrics['R2']:.4f}")
    
    # 分別計算訓練集和測試集的指標
    train_indices = np.where(is_train)[0]
    test_indices = np.where(~is_train)[0]
    
    train_true = true_values[train_indices]
    train_pred = predicted_values[train_indices]
    train_rmse = np.sqrt(mean_squared_error(train_true, train_pred))
    train_mae = mean_absolute_error(train_true, train_pred)
    train_r2 = r2_score(train_true, train_pred)
    
    test_true = true_values[test_indices]
    test_pred = predicted_values[test_indices]
    test_rmse = np.sqrt(mean_squared_error(test_true, test_pred))
    test_mae = mean_absolute_error(test_true, test_pred)
    test_r2 = r2_score(test_true, test_pred)
    
    print(f"\n訓練集 (樣本數: {len(train_true)}):")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  R²: {train_r2:.4f}")
    
    print(f"\n測試集 (樣本數: {len(test_true)}):")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  MAE: {test_mae:.4f}")
    print(f"  R²: {test_r2:.4f}")

def plot_single_dataset(csv_file, output_file, title):
    """繪製單個數據集的parity plot"""
    # 從CSV文件載入數據
    true_values, predicted_values, mol_names, is_train = load_results_from_csv(csv_file)
    
    # 繪製parity plot
    metrics = plot_parity(
        true_values,
        predicted_values,
        title=title,
        save_path=output_file,
        mol_names=None  # 不顯示分子名稱以避免圖表過於擁擠
    )
    
    # 輸出性能指標
    print(f"性能指標 ({title}):")
    print(f"  樣本數: {len(true_values)}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  MAE: {metrics['MAE']:.4f}")
    print(f"  R²: {metrics['R2']:.4f}")

def main():
    """主函數，用於生成parity plot"""
    # 定義要使用的功能模式
    mode = 1  # 1: 訓練集和測試集比較, 2: 僅訓練集, 3: 僅測試集
    
    if mode == 1:
        # 繪製訓練集和測試集的對比圖
        plot_train_test_comparison()
    
    elif mode == 2:
        # 僅繪製訓練集
        plot_single_dataset(
            csv_file="../output/pka_train_prediction_results.csv",
            output_file="../output/pka_train_parity.png",
            title="pKa Prediction (Training Set)"
        )
    
    elif mode == 3:
        # 僅繪製測試集
        plot_single_dataset(
            csv_file="../output/pka_test_prediction_results.csv",
            output_file="../output/pka_test_parity.png",
            title="pKa Prediction (Testing Set)"
        )
    
    else:
        print("無效的模式!")

if __name__ == "__main__":
    main()

# 使用範例:
# 1. 從分開的訓練集和測試集評估文件生成對比圖:
#    python plot_pka_parity.py --train_eval "../output/pka-pka_ver1_epoch_10_train.csv" --test_eval "../output/pka-pka_ver1_epoch_10_test.csv" --output "../output/pka_parity_combined.png" --title "pKa Prediction (Train vs Test)"
#
# 2. 從單個評估文件生成對比圖:
#    python plot_pka_parity.py --from_evaluation "../output/pka-pka_ver1_epoch_10_test.csv" --output "../output/pka_test_parity.png" --title "pKa Test Set Prediction"
#
# 3. 從預處理好的CSV文件直接生成對比圖:
#    python plot_pka_parity.py --input "../output/combined_pka_predictions.csv" --output "../output/pka_parity_from_csv.png" 