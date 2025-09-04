import warnings
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges
from typing import List, Dict, Tuple, Optional, Set, Union

# 忽略 RDKit 的警告
warnings.filterwarnings("ignore", category=UserWarning, module='rdkit')
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 元素和原子參數定義 ---
elem_list = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

# 有效電荷字典 - 從原始程式碼複製
atom_eff_charge_dict = {'H': 0.1, 'He': 0.16875, 'Li': 0.12792, 'Be': 0.1912, 'B': 0.24214000000000002, 'C': 0.31358, 'N': 0.3834, 'O': 0.44532, 'F': 0.51, 'Ne': 0.57584, 'Na': 0.2507400000000001, 'Mg': 0.3307500000000001, 'Al': 0.40656000000000003, 'Si': 0.42852, 'P': 0.48864, 'S': 0.54819, 'Cl': 0.61161, 'Ar': 0.6764100000000002, 'K': 0.34952000000000005, 'Ca': 0.43979999999999997, 'Sc': 0.4632400000000001, 'Ti': 0.4816800000000001, 'V': 0.4981200000000001, 'Cr': 0.5133200000000002, 'Mn': 0.5283200000000001, 'Fe': 0.5434000000000001, 'Co': 0.55764, 'Ni': 0.5710799999999999, 'Cu': 0.5842399999999998, 'Zn': 0.5965199999999999, 'Ga': 0.6221599999999999, 'Ge': 0.6780400000000001, 'As': 0.7449200000000001, 'Se': 0.8287199999999999, 'Br': 0.9027999999999999, 'Kr': 0.9769199999999998, 'Rb': 0.4984499999999997, 'Sr': 0.60705, 'Y': 0.6256, 'Zr': 0.6445499999999996, 'Nb': 0.5921, 'Mo': 0.6106000000000003, 'Tc': 0.7226500000000002, 'Ru': 0.6484499999999997, 'Rh': 0.6639499999999998, 'Pd': 1.3617599999999996, 'Ag': 0.6755499999999999, 'Cd': 0.8192, 'In': 0.847, 'Sn': 0.9102000000000005, 'Sb': 0.9994500000000003, 'Te': 1.0808500000000003, 'I': 1.16115, 'Xe': 1.2424500000000003, 'Cs': 0.6363, 'Ba': 0.7575000000000003, 'Pr': 0.7746600000000001, 'Nd': 0.9306600000000004, 'Pm': 0.9395400000000003, 'Sm': 0.8011800000000001, 'Eu': 0.8121600000000001, 'Tb': 0.8300399999999997, 'Dy': 0.8343600000000002, 'Ho': 0.8439000000000001, 'Er': 0.8476199999999999, 'Tm': 0.8584200000000003, 'Yb': 0.8593199999999996, 'Lu': 0.8804400000000001, 'Hf': 0.9164400000000001, 'Ta': 0.9524999999999999, 'W': 0.9854399999999999, 'Re': 1.0116, 'Os': 1.0323000000000009, 'Ir': 1.0566599999999995, 'Pt': 1.0751400000000004, 'Au': 1.0938000000000003, 'Hg': 1.1153400000000004, 'Tl': 1.22538, 'Pb': 1.2393, 'Bi': 1.3339799999999997, 'Po': 1.4220600000000005, 'At': 1.5163200000000003, 'Rn': 1.6075800000000002}

# --- 官能基 SMARTS 定義 ---
FUNCTIONAL_GROUPS_SMARTS = {
    "COOH": Chem.MolFromSmarts('[C;X3](=[O;X1])[O;X2H1]'),
    "SulfonicAcid": Chem.MolFromSmarts('S(=O)(=O)O'),
    "PhosphonicAcid": Chem.MolFromSmarts('[#15](-[#8])(=[#8])(-[#8])-[#8]'),
    "PrimaryAmine": Chem.MolFromSmarts('[NH2;!$(NC=O)]'),
    "SecondaryAmine": Chem.MolFromSmarts('[NH1;!$(NC=O)]'),
    "TertiaryAmine": Chem.MolFromSmarts('[NX3;H0;!$(NC=O);!$(N=O);!$([N+]-[O-])]'),
    "Phenol": Chem.MolFromSmarts('[OH1][c]'),
    "Alcohol": Chem.MolFromSmarts('[OH1][C;!c;!$(C1[a])]'),
    "AliphaticRingAlcohol": Chem.MolFromSmarts('[OH1][C;!c;!$([C]=[*]);R]'),
    "Imidazole": Chem.MolFromSmarts('c1cnc[nH]1'),
    "ImidazoleLike": Chem.MolFromSmarts('[n;R1]1c[n;H0;R1]cc1'),
    "Pyridine": Chem.MolFromSmarts('c1ccccn1'),
    "Thiol": Chem.MolFromSmarts('[SH]'),
    "Nitro": Chem.MolFromSmarts('[N+](=O)[O-]'),
    "NHydroxylamine": Chem.MolFromSmarts('[NH]([OH1])'),
    "PyrazoleNitrogen": Chem.MolFromSmarts('c1c[nH]nc1'),
    "Indole": Chem.MolFromSmarts('c1c[nH]c2ccccc12'),
    "Phosphate": Chem.MolFromSmarts('[P](=O)([OH1])([OH1])'),
    "Aniline": Chem.MolFromSmarts('[NH2][c;a]')
}

# --- 官能基優先級 ---
FUNCTIONAL_GROUP_PRIORITY = [
    "SulfonicAcid",
    "PhosphonicAcid",
    "Phosphate",
    "COOH",
    # "Nitro", # 移除Nitro從優先級列表，使其不被標記
    "Thiol",
    "Phenol",
    "NHydroxylamine",
    "Imidazole",
    "ImidazoleLike",
    "Indole",
    "PyrazoleNitrogen",
    "Pyridine",
    "Aniline",
    "SecondaryAmine",
    "TertiaryAmine",
    "PrimaryAmine",
    "AliphaticRingAlcohol",
    "Alcohol"
]

# --- 定義每個官能基的關鍵原子識別規則 ---
# 使用原子符號和原子類型識別關鍵原子
KEY_ATOM_IDENTIFIERS = {
    "COOH": {"symbols": ["O"], "properties": {"has_hydrogen": True}},
    "SulfonicAcid": {"symbols": ["O"], "properties": {"has_hydrogen": True}},
    "PhosphonicAcid": {"symbols": ["O"], "properties": {"has_hydrogen": True}},
    "Phosphate": {"symbols": ["O", "P"], "properties": {"has_hydrogen": True}},
    "PrimaryAmine": {"symbols": ["N"], "properties": {"has_hydrogen": True}},
    "SecondaryAmine": {"symbols": ["N"], "properties": {"has_hydrogen": True}},
    "TertiaryAmine": {"symbols": ["N"], "properties": {}},
    "Phenol": {"symbols": ["O"], "properties": {"has_hydrogen": True}},
    "Alcohol": {"symbols": ["O"], "properties": {"has_hydrogen": True}},
    "AliphaticRingAlcohol": {"symbols": ["O"], "properties": {"has_hydrogen": True}},
    "Imidazole": {"symbols": ["N"], "properties": {"aromatic": True}},
    "ImidazoleLike": {"symbols": ["N"], "properties": {"aromatic": True}},
    "Indole": {"symbols": ["N"], "properties": {"has_hydrogen": True, "aromatic": True}},
    "Pyridine": {"symbols": ["N"], "properties": {"aromatic": True}},
    "Thiol": {"symbols": ["S"], "properties": {"has_hydrogen": True}},
    "Nitro": {"symbols": ["N"], "properties": {}},
    "NHydroxylamine": {"symbols": ["O"], "properties": {"has_hydrogen": True}},
    "PyrazoleNitrogen": {"symbols": ["N"], "properties": {"has_hydrogen": True, "aromatic": True}},
    "Aniline": {"symbols": ["N"], "properties": {"has_hydrogen": True}}
}

def onek_encoding_unk(value, allowable_set):
    """
    對值進行 one-hot 或 unknown 編碼。
    如果值在可允許的集合中，返回該位置為1的 one-hot 向量。
    如果值不在集合中，返回全0向量。
    
    參數:
        value: 待編碼的值
        allowable_set: 可允許的值集合
        
    返回:
        List[int]: one-hot 或 unknown 編碼向量
    """
    if value in allowable_set:
        return [1 if v == value else 0 for v in allowable_set]
    else:
        return [0] * len(allowable_set)

#=======回傳153維度的tensor向量=======
def atom_features(atom, oxidation_state=None):
    """
    生成原子特徵向量 (與 chemutils_old.py 中的實現相容)。
    
    參數:
        atom: RDKit 原子對象
        oxidation_state: 可選，原子的氧化態
        
    返回:
        torch.Tensor: 原子特徵向量
    """
    atom_symbol_encoding = onek_encoding_unk(atom.GetSymbol(), elem_list)  # 118
    atom_sym = atom.GetSymbol()
    atom_eff_charge = [atom_eff_charge_dict.get(atom_sym, 0.1)]  # 預設 0.1
    
    atom_degree_encoding = onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])  # 6
    
    formal_charge = atom.GetFormalCharge() if oxidation_state is None else oxidation_state
    formal_charge_encoding = onek_encoding_unk(formal_charge, [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 0])  # 13
    
    chiral_tag_encoding = onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])  # 4
    
    num_h_encoding = onek_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 5
    
    hybridization_encoding = onek_encoding_unk(
        int(atom.GetHybridization()), 
        [0, 1, 2, 3, 4, 5]  # [S, SP, SP2, SP3, SP3D, SP3D2]
    )  # 6
    
    is_aromatic = [atom.GetIsAromatic()]  # 1
    
    return torch.Tensor(
        atom_symbol_encoding +
        atom_degree_encoding +
        formal_charge_encoding +
        chiral_tag_encoding +
        num_h_encoding +
        hybridization_encoding +
        is_aromatic
    )  # 118 + 6 + 13 + 4 + 5 + 6 + 1 = 153 維

def is_key_atom(atom, group_name: str) -> bool:
    """
    判斷原子是否為特定官能基的關鍵原子。
    
    參數:
        atom: RDKit 原子物件
        group_name: 官能基名稱
    
    返回:
        bool: 是否為關鍵原子
    """
    if group_name not in KEY_ATOM_IDENTIFIERS:
        return False
    
    criteria = KEY_ATOM_IDENTIFIERS[group_name]
    
    # 檢查原子符號
    if atom.GetSymbol() not in criteria["symbols"]:
        return False
    
    # 檢查其他屬性
    properties = criteria["properties"]
    
    # 檢查是否有氫
    if properties.get("has_hydrogen", False) and atom.GetTotalNumHs() == 0:
        return False
    
    # 檢查是否為芳香原子
    if properties.get("aromatic", False) and not atom.GetIsAromatic():
        return False
    
    # 特殊處理磷酸
    if group_name == "Phosphate":
        # 如果是磷原子，應該至少有一個羥基氧連接
        if atom.GetSymbol() == 'P':
            oh_count = 0
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'O' and neighbor.GetTotalNumHs() > 0:
                    oh_count += 1
            return oh_count >= 1
        # 如果是氧原子，確保它連接到磷原子並有氫
        elif atom.GetSymbol() == 'O':
            for neighbor in atom.GetNeighbors():
                if neighbor.GetSymbol() == 'P':
                    return atom.GetTotalNumHs() > 0
            return False
    
    # 特殊處理咪唑 - 調整為標記2號位置的氮
    if group_name == "Imidazole":
        # 檢查是否在五員環中
        if not atom.IsInRingSize(5):
            return False
        
        # 對於咪唑，我們要確認這是非氫連接的環氮（2號位置）
        if atom.GetSymbol() != 'N' or not atom.GetIsAromatic():
            return False
            
        # 確認這是不帶氫的氮（2號位置），帶氫的是3號位置
        if atom.GetTotalNumHs() > 0:
            return False
            
        # 檢查連接模式，2號位置的氮連接兩個碳
        carbon_neighbors = 0
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C':
                carbon_neighbors += 1
                
        return carbon_neighbors == 2
    
    # 特殊處理ImidazoleLike - 標記9號位置的N（相當於咪唑的2號位置）
    if group_name == "ImidazoleLike":
        # 檢查是否在五員環中
        if not atom.IsInRingSize(5):
            return False
        
        # 對於類咪唑，我們要確認這是雜環中非氫連接的氮（9號位置）
        if atom.GetSymbol() != 'N' or not atom.GetIsAromatic():
            return False
            
        # 確認這是不帶氫的氮（9號位置）
        if atom.GetTotalNumHs() > 0:
            return False
            
        # 檢查這個氮是否連接到至少一個碳和一個其他原子
        carbon_count = 0
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C':
                carbon_count += 1
                
        # 9號位置的氮通常連接兩個碳
        return carbon_count >= 1 and len(list(atom.GetNeighbors())) >= 2
    
    # 特殊處理脂肪環醇 - 確保OH連接的碳在環上但不是芳香環
    if group_name == "AliphaticRingAlcohol":
        # 確認OH連接的碳在環上
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C' and neighbor.IsInRing() and not neighbor.GetIsAromatic():
                return True
        return False
    
    # 特殊處理苯胺(Aniline) - 確保只標記氮原子，且該氮原子連接到芳香環碳原子
    if group_name == "Aniline":
        # 確認氮原子連接到芳香環碳原子
        if atom.GetSymbol() != 'N' or atom.GetTotalNumHs() != 2:
            return False
            
        # 檢查是否連接到芳香環碳
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'C' and neighbor.GetIsAromatic():
                return True
        return False
    
    # 特殊處理N-羥基胺 - 確保選擇的是羥基氧，而不是氮
    if group_name == "NHydroxylamine":
        # 檢查原子是否為氧且連接到氮原子
        if atom.GetSymbol() != 'O':
            return False
            
        # 確認這個氧連接著氮
        connected_to_nitrogen = False
        for neighbor in atom.GetNeighbors():
            if neighbor.GetSymbol() == 'N':
                connected_to_nitrogen = True
                break
                
        return connected_to_nitrogen and atom.GetTotalNumHs() > 0
    
    return True

def extract_functional_group_features(smiles: str) -> List[Dict]:
    """
    為單個分子的 SMILES 提取官能基特徵。
    只對每個官能基的關鍵原子提取特徵，而不是整個子結構。
    
    參數:
        smiles: 分子的 SMILES 字符串
    
    返回:
        List[Dict]: 包含官能基特徵信息的列表，每個項目為一個字典，
        包含 'feature', 'group_type', 'smiles', 'match_indices' 和 'key_atom_index'
    """
    print(f"提取特徵: {smiles}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []

    group_instance_features = []  # 儲存這個分子中所有官能基實例的特徵
    
    # 跟踪已匹配的原子，避免重複
    matched_atoms = set()
    
    # 跟踪NHydroxylamine相關的原子，特別處理相連的氮原子
    nh_related_atoms = set()
    
    # 跟踪Nitro基團中的原子，避免將它們標記為其他類型
    nitro_related_atoms = set()
    
    # 首先識別所有的Nitro基團，但不標記它們
    nitro_pattern = FUNCTIONAL_GROUPS_SMARTS.get("Nitro")
    if nitro_pattern:
        try:
            nitro_matches = mol.GetSubstructMatches(nitro_pattern)
            for match_indices in nitro_matches:
                for idx in match_indices:
                    nitro_related_atoms.add(idx)
                    atom = mol.GetAtomWithIdx(idx)
                    # 將所有與氮連接的原子也加入到nitro_related_atoms
                    if atom.GetSymbol() == 'N':
                        for neighbor in atom.GetNeighbors():
                            nitro_related_atoms.add(neighbor.GetIdx())
        except Exception as e:
            print(f"識別Nitro基團時出錯: {e}")
    
    # 按優先級順序匹配官能基
    for group_name in FUNCTIONAL_GROUP_PRIORITY:
        pattern = FUNCTIONAL_GROUPS_SMARTS.get(group_name)
        if pattern is None:
            continue

        try:
            matches = mol.GetSubstructMatches(pattern)
        except Exception as e:
            continue  # 跳過這個官能基

        if matches:
            for match_indices in matches:
                # 檢查匹配原子是否已被更高優先級的官能基使用
                overlap = False
                for idx in match_indices:
                    if idx in matched_atoms:
                        overlap = True
                        break
                
                if overlap:
                    # 如果有重疊，跳過這個匹配
                    continue
                
                # 檢查是否有任何原子屬於Nitro基團，如果是TertiaryAmine則跳過此匹配
                if group_name in ["TertiaryAmine", "SecondaryAmine", "PrimaryAmine"]:
                    nitro_overlap = False
                    for idx in match_indices:
                        if idx in nitro_related_atoms:
                            nitro_overlap = True
                            break
                    if nitro_overlap:
                        continue
                
                # 特殊處理NHydroxylamine，標記相關的氮原子
                if group_name == "NHydroxylamine":
                    for idx in match_indices:
                        atom = mol.GetAtomWithIdx(idx)
                        if atom.GetSymbol() == 'N':
                            nh_related_atoms.add(idx)
                            # 檢查所有連接到這個氮的原子
                            for neighbor in atom.GetNeighbors():
                                # 把所有連接到這個氮的原子也標記為已使用
                                nh_related_atoms.add(neighbor.GetIdx())
                
                # 找到官能基匹配
                # 識別關鍵原子
                key_atom_indices = []
                for idx in match_indices:
                    if idx >= mol.GetNumAtoms():
                        continue
                    
                    # 對於SecondaryAmine和其他基團，檢查它們是否已被NHydroxylamine相關或屬於Nitro基團
                    if group_name in ["SecondaryAmine", "PrimaryAmine", "TertiaryAmine"] and (idx in nh_related_atoms or idx in nitro_related_atoms):
                        continue
                    
                    atom = mol.GetAtomWithIdx(idx)
                    if is_key_atom(atom, group_name):
                        key_atom_indices.append(idx)
                
                # 如果沒有找到關鍵原子，跳過此匹配
                if not key_atom_indices:
                    continue
                
                # 為每個關鍵原子提取特徵
                for key_atom_idx in key_atom_indices:
                    atom = mol.GetAtomWithIdx(key_atom_idx)
                    try:
                        # 提取原子特徵
                        features = atom_features(atom).numpy()
                        
                        # 將這個原子標記為已匹配
                        matched_atoms.add(key_atom_idx)
                        
                        group_instance_features.append({
                            'feature': features,
                            'group_type': group_name,
                            'smiles': smiles,
                            'match_indices': match_indices,
                            'key_atom_index': key_atom_idx
                        })
                    except Exception as e:
                        print(f"提取特徵時出錯: {e}")
                        continue
    
    # 打印找到的官能基及其重要性
    if group_instance_features:
        print(f"    總共找到 {len(group_instance_features)} 個關鍵官能基原子")
        for i, info in enumerate(group_instance_features):
            print(f"    官能基 {i+1}: {info['group_type']} at 關鍵原子位置 {info['key_atom_index']}")
    else:
        print(f"    未找到任何官能基關鍵原子")
        
    return group_instance_features



class FunctionalGroupLabeler:
    """
    官能基標記類，提供更高級的功能和組織化的 API。
    """
    
    def __init__(self):
        """初始化標記器"""
        self.functional_groups = FUNCTIONAL_GROUPS_SMARTS
        self.priority_order = FUNCTIONAL_GROUP_PRIORITY
        self.key_atom_identifiers = KEY_ATOM_IDENTIFIERS
    
    def label_molecule(self, smiles: str) -> Tuple[Optional[Chem.Mol], Optional[Dict[int, str]]]:
        """
        標記分子中的官能基。
        
        參數:
            smiles: 分子的 SMILES 字符串
        
        返回:
            Tuple[Optional[Chem.Mol], Optional[Dict[int, str]]]: 
            分子對象和原子索引到官能基類型的映射
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None
        
        features = extract_functional_group_features(smiles)
        
        # 構建原子索引到官能基類型的映射
        atom_to_group = {}
        for feature in features:
            atom_to_group[feature['key_atom_index']] = feature['group_type']
        
        return mol, atom_to_group
    
    def visualize_labeled_molecule(self, mol, atom_to_group, output_path=None, include_atom_indices=True):
        """
        使用 RDKit 可視化標記後的分子。
        
        參數:
            mol: RDKit 分子對象
            atom_to_group: 原子索引到官能基類型的映射
            output_path: 可選，輸出圖像的路徑
            include_atom_indices: 是否包含原子索引
            
        返回:
            RDKit 繪製的分子圖像
        """
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
        import io
        from PIL import Image
        
        if mol is None or not atom_to_group:
            print("無法視覺化：分子為空或無標記原子")
            return None
        
        # 設置分子名稱
        if not mol.HasProp("_Name"):
            mol.SetProp("_Name", Chem.MolToSmiles(mol))
        
        # 為每個官能基設置顏色
        colors = {
            "COOH": (0.0, 0.6, 1.0),           # 藍色
            "SulfonicAcid": (1.0, 0.6, 0.0),   # 橙色
            "PhosphonicAcid": (0.7, 0.7, 0.7), # 灰色
            "PrimaryAmine": (1.0, 0.4, 0.4),   # 紅色
            "SecondaryAmine": (0.7, 0.3, 0.9), # 紫色
            "TertiaryAmine": (0.0, 0.8, 0.0),  # 綠色
            "Phenol": (0.8, 0.8, 0.0),         # 黃色
            "Alcohol": (0.9, 0.5, 0.8),        # 粉紅色
            "AliphaticRingAlcohol": (0.7, 0.3, 0.6),  # 深粉紅色
            "Imidazole": (0.5, 0.8, 0.8),      # 青色
            "ImidazoleLike": (0.3, 0.6, 0.6),  # 深青色
            "Pyridine": (0.5, 0.5, 1.0),       # 淺藍
            "Thiol": (0.6, 0.4, 0.2),          # 棕色
            "NHydroxylamine": (0.8, 0.6, 0.4), # 棕褐色
            "PyrazoleNitrogen": (0.4, 0.8, 0.6), # 湖綠色
            "Indole": (0.3, 0.5, 0.3),         # 深綠色
            "Phosphate": (0.7, 0.7, 0.7),       # 灰色
            "Aniline": (1.0, 0.2, 0.4)         # 深紅色
        }
        
        # 準備高亮和顏色
        highlight_atoms = list(atom_to_group.keys())
        highlight_colors = {}
        
        for atom_idx, group_type in atom_to_group.items():
            if group_type in colors:
                highlight_colors[atom_idx] = colors[group_type]
            else:
                highlight_colors[atom_idx] = (0.5, 0.5, 0.5)  # 默認灰色
        
        # 建立畫布
        drawer = rdMolDraw2D.MolDraw2DCairo(600, 400)
        drawer.drawOptions().addAtomIndices = include_atom_indices
        
        # 畫分子
        drawer.DrawMolecule(
            mol,
            highlightAtoms=highlight_atoms,
            highlightAtomColors=highlight_colors
        )
        drawer.FinishDrawing()
        
        # 獲取圖像數據
        png_data = drawer.GetDrawingText()
        
        # 如果指定了輸出路徑，保存圖像
        if output_path:
            with open(output_path, 'wb') as f:
                f.write(png_data)
            print(f"分子圖像已保存至 {output_path}")
        
        # 將 PNG 數據轉換為 PIL 圖像以顯示（如果在 Jupyter 環境中）
        try:
            pil_img = Image.open(io.BytesIO(png_data))
            return pil_img
        except Exception as e:
            print(f"無法轉換為 PIL 圖像: {e}")
            return None

# --- 測試代碼 ---
if __name__ == "__main__":
    # 測試一些 SMILES 的標記功能
    test_smiles = [
        "CC(=O)O",                # 乙酸
        "CCN",                    # 乙胺
        "CCO",                    # 乙醇
        "c1ccccc1O",              # 酚
        "c1ccccn1",               # 吡啶
        "CN(C)C",                 # 三甲胺
        "CC(C)(C)N",              # 叔丁胺
        "N[C@@H](C)C(=O)O",       # 丙氨酸
        "O=S(=O)(O)c1ccccc1",     # 苯磺酸
        "c1c[nH]c2ccccc12",       # 吲哚
        "c1c[nH]cn1",             # 咪唑
        "c1cnc[nH]1",             # 咪唑異構體
        "CC[SH]",                 # 乙硫醇
        "ONC1=CC=CC=C1",          # N-羥基苯胺
        "c1ccc(cc1)P(=O)(O)O",    # 苯基磷酸
        "Oc1ccc([N+](=O)[O-])cc1", # 4-硝基酚 (測試硝基和酚的同時存在)
        "c1ccccc1N",              # 苯胺 (測試新添加的Aniline類型)
        "c1ccc2[nH]ccc2c1",       # 吲哚 (測試另一種表示法)
        "Nc1cccc2ccccc12"         # 1-荼胺 (測試NH2接在多環芳香體系上)
    ]
    
    labeler = FunctionalGroupLabeler()
    
    import os
    # 創建輸出目錄（如果不存在）
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../output/labeled_molecules")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, smiles in enumerate(test_smiles):
        print(f"\n處理分子 {i+1}/{len(test_smiles)}: {smiles}")
        mol, atom_to_group = labeler.label_molecule(smiles)
        
        if mol and atom_to_group:
            print(f"標記結果:")
            for atom_idx, group_type in atom_to_group.items():
                atom = mol.GetAtomWithIdx(atom_idx)
                print(f"  原子 {atom_idx} ({atom.GetSymbol()}) 被標記為 {group_type}")
            
            # 生成和保存可視化圖像
            output_path = os.path.join(output_dir, f"molecule_{i+1}.png")
            labeler.visualize_labeled_molecule(mol, atom_to_group, output_path=output_path)
        else:
            print(f"無法處理分子: {smiles}")
    
    print(f"\n所有分子的標記圖像已保存到 {output_dir} 目錄")
    