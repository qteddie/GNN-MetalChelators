import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import torch
from PIL import Image
from io import BytesIO

def highlight_atoms_with_pka(mol, pka_positions, pka_values, atom_colors=None, 
                           image_size=(600, 600), title=None, 
                           show_pka_values=True):
    """
    高亮顯示具有pKa值的原子，並標註pKa值
    
    Args:
        mol: RDKit分子對象
        pka_positions: 具有pKa值的原子位置列表
        pka_values: 對應的pKa值列表
        atom_colors: 原子顏色映射 (位置 -> (r,g,b))
        image_size: 圖像尺寸
        title: 圖像標題
        show_pka_values: 是否顯示pKa數值
        
    Returns:
        PIL.Image: 含有高亮原子和標註的分子圖像
    """
    if atom_colors is None:
        # 創建一個從藍色到紅色的漸變色彩映射
        # 從較低的pKa（酸性，藍色）到較高的pKa（鹼性，紅色）
        colormap = plt.cm.get_cmap('coolwarm')
        
        # 找出pKa值的範圍，用於歸一化
        if pka_values:
            min_pka = min(pka_values)
            max_pka = max(pka_values)
            pka_range = max_pka - min_pka
            
            # 避免除以零
            if pka_range == 0:
                pka_range = 1
            
            atom_colors = {}
            for pos, val in zip(pka_positions, pka_values):
                # 歸一化pKa值到[0,1]範圍
                norm_val = (val - min_pka) / pka_range
                # 獲取顏色
                color = colormap(norm_val)
                # 存儲RGB值（不含Alpha通道）
                atom_colors[pos] = (color[0], color[1], color[2])
    
    # 準備高亮原子
    highlight_atoms = pka_positions
    highlight_bonds = []
    
    # 創建原子高亮對象
    atom_highlights = {}
    for pos in highlight_atoms:
        if pos in atom_colors:
            atom_highlights[pos] = atom_colors[pos]
        else:
            atom_highlights[pos] = (1, 0, 0)  # 默認為紅色
    
    # 創建繪圖對象
    drawer = rdMolDraw2D.MolDraw2DCairo(image_size[0], image_size[1])
    drawer.SetFontSize(0.9)  # 設置字體大小
    
    # 設置繪圖選項
    draw_options = drawer.drawOptions()
    draw_options.continuousHighlight = True
    
    # 繪製分子
    drawer.DrawMolecule(
        mol, 
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_highlights
    )
    drawer.FinishDrawing()
    
    # 獲取PNG圖像數據
    png_data = drawer.GetDrawingText()
    
    # 轉換為PIL圖像
    img = Image.open(BytesIO(png_data))
    
    # 如果需要顯示pKa值，添加標註
    if show_pka_values and pka_values:
        # 獲取原子坐標
        conf = mol.GetConformer()
        
        # 創建一個可繪製的圖像副本
        img_with_text = img.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        # 獲取分子的邊界框
        bounds = drawer.GetDrawCoords()
        
        # 為每個標記的原子添加pKa標籤
        for pos, val in zip(pka_positions, pka_values):
            atom_pos = conf.GetAtomPosition(pos)
            x, y = drawer.GetDrawCoords(pos)
            
            # 格式化pKa值
            pka_text = f"{val:.1f}"
            
            # 添加文字標籤
            draw.text((x, y-10), pka_text, fill=(0, 0, 0))
        
        img = img_with_text
    
    # 添加標題（如果有）
    if title:
        img_with_title = Image.new('RGB', (img.width, img.height + 30), (255, 255, 255))
        img_with_title.paste(img, (0, 30))
        draw = ImageDraw.Draw(img_with_title)
        
        # 使用系統字體或默認字體
        try:
            from PIL import ImageFont
            font = ImageFont.truetype("Arial", 16)
            draw.text((10, 10), title, fill=(0, 0, 0), font=font)
        except:
            draw.text((10, 10), title, fill=(0, 0, 0))
        
        img = img_with_title
    
    return img

def create_pka_visualization_grid(results, title=None, mols_per_row=3, 
                                mol_size=(400, 400), output_file=None):
    """
    創建具有pKa預測結果的分子網格
    
    Args:
        results: 包含分子和pKa預測的字典列表
        title: 網格標題
        mols_per_row: 每行分子數量
        mol_size: 每個分子圖像的大小
        output_file: 輸出文件路徑
        
    Returns:
        PIL.Image: 具有多個分子及其pKa預測的網格圖像
    """
    if not results:
        raise ValueError("No results provided for visualization")
    
    # 確定網格尺寸
    num_mols = len(results)
    num_rows = (num_mols + mols_per_row - 1) // mols_per_row  # 向上取整
    
    # 創建彩色映射以區分不同pKa值的原子
    # 獲取所有pKa值以確定範圍
    all_pka_values = []
    for result in results:
        if 'pka_values' in result and result['pka_values']:
            all_pka_values.extend(result['pka_values'])
    
    if all_pka_values:
        min_pka = min(all_pka_values)
        max_pka = max(all_pka_values)
    else:
        min_pka, max_pka = 0, 14  # 默認pKa範圍
    
    # 創建每個分子的圖像
    mol_images = []
    for result in results:
        mol = result.get('mol')
        if mol is None and 'smiles' in result:
            # 從SMILES創建分子
            mol = Chem.MolFromSmiles(result['smiles'])
        
        if mol:
            # 確保分子有坐標
            if mol.GetNumConformers() == 0:
                AllChem.Compute2DCoords(mol)
            
            # 創建具有pKa標記的分子圖像
            pka_positions = result.get('pka_positions', [])
            pka_values = result.get('pka_values', [])
            
            mol_name = result.get('molecule_name', result.get('smiles', ''))
            short_name = mol_name[:20] + '...' if len(mol_name) > 20 else mol_name
            
            try:
                mol_img = highlight_atoms_with_pka(
                    mol, pka_positions, pka_values, 
                    image_size=mol_size, title=short_name
                )
                mol_images.append(mol_img)
            except Exception as e:
                print(f"Error creating image for {mol_name}: {e}")
                # 創建一個空白圖像作為替代
                blank_img = Image.new('RGB', mol_size, (240, 240, 240))
                mol_images.append(blank_img)
        else:
            # 創建一個空白圖像作為替代
            blank_img = Image.new('RGB', mol_size, (240, 240, 240))
            mol_images.append(blank_img)
    
    # 創建網格圖像
    grid_width = mols_per_row * mol_size[0]
    grid_height = num_rows * mol_size[1]
    
    # 添加標題區域
    title_height = 50 if title else 0
    grid_img = Image.new('RGB', (grid_width, grid_height + title_height), (255, 255, 255))
    
    # 添加標題
    if title:
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(grid_img)
            try:
                font = ImageFont.truetype("Arial", 24)
                draw.text((20, 15), title, fill=(0, 0, 0), font=font)
            except:
                draw.text((20, 15), title, fill=(0, 0, 0))
        except ImportError:
            print("PIL.ImageDraw not available for adding title")
    
    # 放置分子圖像
    for i, img in enumerate(mol_images):
        row = i // mols_per_row
        col = i % mols_per_row
        x = col * mol_size[0]
        y = row * mol_size[1] + title_height
        grid_img.paste(img, (x, y))
    
    # 保存圖像
    if output_file:
        # 確保輸出目錄存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        grid_img.save(output_file, dpi=(300, 300))
        print(f"Image saved to: {output_file}")
    
    return grid_img

def load_results_from_json(json_file):
    """從JSON文件中載入pKa結果"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # 確定結果格式 - 可能是字典的列表或以分子名稱為鍵的字典
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        # 轉換為列表格式，添加分子名稱
        results = []
        for mol_name, mol_data in data.items():
            mol_data['molecule_name'] = mol_name
            results.append(mol_data)
        return results
    else:
        raise ValueError("Unsupported JSON format")

def load_results_from_csv(csv_file):
    """從CSV文件中載入pKa結果"""
    df = pd.read_csv(csv_file)
    
    # 檢查CSV格式
    required_cols = ['smiles']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV file missing required column: {col}")
    
    # 從CSV轉換為結果字典列表
    results = []
    for _, row in df.iterrows():
        # 創建分子對象
        mol = Chem.MolFromSmiles(row['smiles'])
        
        # 生成基本結果數據
        result = {
            'smiles': row['smiles'],
            'mol': mol
        }
        
        # 添加pKa位置和值（如果存在）
        if 'pka_positions' in df.columns and 'pka_values' in df.columns:
            try:
                # 處理可能存儲為字符串的列表
                pka_positions = row['pka_positions']
                pka_values = row['pka_values']
                
                if isinstance(pka_positions, str):
                    # 嘗試解析JSON字符串列表
                    pka_positions = json.loads(pka_positions.replace("'", '"'))
                if isinstance(pka_values, str):
                    pka_values = json.loads(pka_values.replace("'", '"'))
                
                result['pka_positions'] = pka_positions
                result['pka_values'] = pka_values
            except:
                print(f"Warning: Could not parse pKa positions or values for row {_}")
        
        # 添加分子名稱（如果存在）
        if 'molecule_name' in df.columns:
            result['molecule_name'] = row['molecule_name']
        
        results.append(result)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Visualize pKa prediction results')
    parser.add_argument('--input', required=True, help='Input file path (CSV or JSON format)')
    parser.add_argument('--output', default='pka_visualization.png', help='Output image path')
    parser.add_argument('--title', default='pKa Prediction Visualization', help='Image title')
    parser.add_argument('--mols-per-row', type=int, default=3, help='Number of molecules per row')
    parser.add_argument('--mol-size', type=int, default=400, help='Molecule image size')
    
    args = parser.parse_args()
    
    # 根據文件類型載入資料
    file_ext = os.path.splitext(args.input)[1].lower()
    
    try:
        if file_ext == '.json':
            results = load_results_from_json(args.input)
        elif file_ext == '.csv':
            results = load_results_from_csv(args.input)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or JSON file.")
        
        # 創建視覺化網格
        mol_size = (args.mol_size, args.mol_size)
        grid_img = create_pka_visualization_grid(
            results, 
            title=args.title,
            mols_per_row=args.mols_per_row,
            mol_size=mol_size,
            output_file=args.output
        )
        
        # 顯示或保存圖像
        grid_img.show()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 嘗試導入PIL的ImageDraw和ImageFont，如果可用的話
    try:
        from PIL import ImageDraw, ImageFont
    except ImportError:
        print("Warning: PIL's ImageDraw or ImageFont unavailable, titles and labels may not display correctly")
    
    main() 