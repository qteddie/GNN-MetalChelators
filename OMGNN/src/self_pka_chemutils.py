# import re
# import sys 
# sys.path.append('/work/u7069586/zeff/src/zeff/')
# import zeff
# from mendeleev.fetch import fetch_table
# from mendeleev import element
import torch
import pubchempy as pcp
import numpy as np
import networkx as nx
from rdkit import Chem
from collections import defaultdict
# from rdkit.Chem.rdmolops import GetAdjacencyMatrix as gam

# !TODO Check ELEM_LIST
elem_list = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Cr', 'Mn', 'Fe', 'Co', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown'] # 26
TM_LIST = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Cn']
NM_LIST = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I']
VE_DICT = {'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8, 'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8}
VE_DICT.update({'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 6, 'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 2, 'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8})
VE_DICT.update({'Rb': 1, 'Sr': 2, 'Y': 3, 'Zr': 4, 'Nb': 5, 'Mo': 6, 'Tc': 7, 'Ru': 8, 'Rh': 9, 'Pd': 10, 'Ag': 11, 'Cd': 2, 'In': 3, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7, 'Xe': 8})
VE_DICT.update({'Cs': 1, 'Ba': 2, 'La': 3, 'Ce': 4, 'Pr': 5, 'Nd': 6, 'Pm': 7, 'Sm': 8, 'Eu': 9, 'Gd': 10, 'Tb': 11, 'Dy': 12, 'Ho': 13, 'Er': 14, 'Tm': 15, 'Yb': 16, 'Lu': 17})
VE_DICT.update({'Hf': 4, 'Ta': 5, 'W': 6, 'Re': 7, 'Os': 8, 'Ir': 9, 'Pt': 10, 'Au': 11, 'Hg': 2, 'Tl': 3, 'Pb': 4, 'Bi': 5, 'Po': 6, 'At': 7, 'Rn': 8})
VE_DICT.update({'Fr': 1, 'Ra': 2, 'Ac': 3, 'Th': 4, 'Pa': 5, 'U': 6, 'Np': 7, 'Pu': 8, 'Am': 9, 'Cm': 10, 'Bk': 11, 'Cf': 12, 'Es': 13, 'Fm': 14, 'Md': 15, 'No': 16, 'Lr': 17})
VE_DICT.update({'Rf': 4, 'Db': 5, 'Sg': 6, 'Bh': 21, 'Hs': 22, 'Mt': 23, 'Ds': 24, 'Rg': 25, 'Cn': 26, 'Nh': 27, 'Fl': 28, 'Mc': 29, 'Lv': 30, 'Ts': 31, 'Og': 32})


atom_eff_charge_dict = {'H': 0.1, 'He': 0.16875, 'Li': 0.12792, 'Be': 0.1912, 'B': 0.24214000000000002, 'C': 0.31358, 'N': 0.3834, 'O': 0.44532, 'F': 0.51, 'Ne': 0.57584, 'Na': 0.2507400000000001, 'Mg': 0.3307500000000001, 'Al': 0.40656000000000003, 'Si': 0.42852, 'P': 0.48864, 'S': 0.54819, 'Cl': 0.61161, 'Ar': 0.6764100000000002, 'K': 0.34952000000000005, 'Ca': 0.43979999999999997, 'Sc': 0.4632400000000001, 'Ti': 0.4816800000000001, 'V': 0.4981200000000001, 'Cr': 0.5133200000000002, 'Mn': 0.5283200000000001, 'Fe': 0.5434000000000001, 'Co': 0.55764, 'Ni': 0.5710799999999999, 'Cu': 0.5842399999999998, 'Zn': 0.5965199999999999, 'Ga': 0.6221599999999999, 'Ge': 0.6780400000000001, 'As': 0.7449200000000001, 'Se': 0.8287199999999999, 'Br': 0.9027999999999999, 'Kr': 0.9769199999999998, 'Rb': 0.4984499999999997, 'Sr': 0.60705, 'Y': 0.6256, 'Zr': 0.6445499999999996, 'Nb': 0.5921, 'Mo': 0.6106000000000003, 'Tc': 0.7226500000000002, 'Ru': 0.6484499999999997, 'Rh': 0.6639499999999998, 'Pd': 1.3617599999999996, 'Ag': 0.6755499999999999, 'Cd': 0.8192, 'In': 0.847, 'Sn': 0.9102000000000005, 'Sb': 0.9994500000000003, 'Te': 1.0808500000000003, 'I': 1.16115, 'Xe': 1.2424500000000003, 'Cs': 0.6363, 'Ba': 0.7575000000000003, 'Pr': 0.7746600000000001, 'Nd': 0.9306600000000004, 'Pm': 0.9395400000000003, 'Sm': 0.8011800000000001, 'Eu': 0.8121600000000001, 'Tb': 0.8300399999999997, 'Dy': 0.8343600000000002, 'Ho': 0.8439000000000001, 'Er': 0.8476199999999999, 'Tm': 0.8584200000000003, 'Yb': 0.8593199999999996, 'Lu': 0.8804400000000001, 'Hf': 0.9164400000000001, 'Ta': 0.9524999999999999, 'W': 0.9854399999999999, 'Re': 1.0116, 'Os': 1.0323000000000009, 'Ir': 1.0566599999999995, 'Pt': 1.0751400000000004, 'Au': 1.0938000000000003, 'Hg': 1.1153400000000004, 'Tl': 1.22538, 'Pb': 1.2393, 'Bi': 1.3339799999999997, 'Po': 1.4220600000000005, 'At': 1.5163200000000003, 'Rn': 1.6075800000000002}

# for atom_sym in elem_list:
#     try:
#         # Calculate the effective charge (0.1 * last Zeff Clementi value)
#         eff_charge = 0.1 * zeff.elem_data(atom_sym)["Zeff Clementi"].iloc[-1]
#         atom_eff_charge_dict[atom_sym] = eff_charge
#     except Exception as e:
#         print(f"Error processing element {atom_sym}: {e}")


def get_metal_oxidation_state(metal):
    oxidation_states = ''.join(filter(str.isdigit, metal))

    if len(oxidation_states) == 0:
        return 0
    else:
        return int(oxidation_states)
    
def calc_formal_charge(atom, oxidation_state, rdkit=True):
    bonds = atom.GetBonds()
    ve_bonds = sum([bond.GetBondTypeAsDouble() for bond in bonds])
    
    ve_imp_bonds = atom.GetImplicitValence()
    ve = VE_DICT.get(atom.GetSymbol(), 0)  # Default to 0 if not in VE_DICT
    
    if atom.GetSymbol() in NM_LIST:
        charge = int(ve + ve_bonds + ve_imp_bonds - 8)
    else:
        charge = oxidation_state - int(ve - ve_bonds - ve_imp_bonds)
    if atom.GetAtomicNum() > 10 and abs(charge) > 0:
        charge = -(abs(charge) % 2)
    if atom.GetSymbol() in TM_LIST:
        charge = oxidation_state

    return charge

# def onek_encoding_unk(x, allowable_set):
#     if x not in allowable_set:
#         x = allowable_set[-1]
#     return [x == s for s in allowable_set]

def onek_encoding_unk(value, allowable_set):
    if value in allowable_set:
        return [1 if v == value else 0 for v in allowable_set]
    else:
        return [0] * len(allowable_set)

def atom_features(atom, oxidation_state=None):
    atom_symbol_encoding = onek_encoding_unk(atom.GetSymbol(), elem_list)  # 118
    atom_sym = atom.GetSymbol()
    atom_eff_charge = [atom_eff_charge_dict.get(atom_sym, 0.1)]   #default 0.1 
    # atom_eff_charge = [0.1 * zeff.elem_data(atom_sym)["Zeff Clementi"].iloc[-1]]
    atom_degree_encoding = onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])  # 6
    formal_charge = atom.GetFormalCharge() if oxidation_state is None else oxidation_state
    formal_charge_encoding = onek_encoding_unk(formal_charge, [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 0])  # 13
    chiral_tag_encoding = onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])  # 4
    num_h_encoding = onek_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 5
    hybridization_encoding = onek_encoding_unk(
        int(atom.GetHybridization()), 
        [0, 1, 2, 3, 4, 5]  # Example encoding: [S, SP, SP2, SP3, SP3D, SP3D2]
    )  # 6
    is_aromatic = [atom.GetIsAromatic()]  # 1
    atomic_mass = [0.01 * atom.GetMass()]  # 1
    # return torch.Tensor(
    #     atom_eff_charge +
    #     atom_degree_encoding +
    #     formal_charge_encoding +
    #     chiral_tag_encoding +
    #     num_h_encoding +
    #     hybridization_encoding +
    #     is_aromatic +
    #     atomic_mass
    # )
    return torch.Tensor(
        atom_symbol_encoding +
        atom_degree_encoding +
        formal_charge_encoding+
        chiral_tag_encoding +
        num_h_encoding +
        hybridization_encoding +
        is_aromatic
        ) # 118+13 

def metal_features(metal):
    oxidation_state = get_metal_oxidation_state(metal)
    #remove number from metal
    metal_symbol = metal.split(str(oxidation_state))[0]
    mol = Chem.MolFromSmiles("[{}]".format(metal_symbol))
    atom = mol.GetAtomWithIdx(0) 
    edge_index = torch.tensor([[0],[0]], dtype=torch.long).cuda()
    batch1 = torch.tensor([0], dtype=torch.long).cuda()
    edge_attr = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0]], dtype=torch.float).cuda()

    return (atom_features(atom, oxidation_state)), (edge_index, batch1), edge_attr

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

def functional_group_features(group_type, atom=None):
    """
    為官能基創建特徵向量，格式與atom_features相同以便stack。
    
    Args:
        group_type: 官能基類型 (如 'COOH', 'Pyridine', 等)
        atom: 必須提供的RDKit原子對象
        
    Returns:
        torch.Tensor: 與atom_features格式相同的官能基特徵向量
    """
    if atom is None:
        raise ValueError("必須提供atom參數以確保與atom_features相容")
    
    # 獲取基本的原子特徵
    base_features = atom_features(atom)
    
    # 定義每種官能基的pKa範圍 (根據統計數據的Q1, Q2(中位數), Q3)
    pka_ranges = {
        'SulfonicAcid': (1.6425, 1.865, 2.785),   # Q1, 中位數, Q3
        'COOH': (2.89, 3.33, 4.07),
        'Aniline': (3.925, 4.47, 4.815),
        'Pyridine': (4.04, 5.005, 5.6175),
        'Phosphate': (1.7, 5.74, 6.91),
        'PhosphonicAcid': (1.9, 5.84, 6.61),
        'ImidazoleLike': (7.2, 7.22, 9.07),
        'Imidazole': (7.18, 7.63, 8.19),
        'Thiol': (6.5, 7.88, 9.3),
        'PrimaryAmine': (7.83, 8.41, 9.98),
        'TertiaryAmine': (8.065, 8.9, 10.08),
        'Phenol': (8.175, 9.15, 9.835),
        'NHydroxylamine': (8.91, 9.17, 9.77),
        'SecondaryAmine': (8.6075, 9.58, 10.4875),
        # 對於沒有統計數據的官能基，提供合理默認值
        'Alcohol': (15.0, 16.0, 17.0),
        'AliphaticRingAlcohol': (12.0, 13.0, 14.0),
        'PyrazoleNitrogen': (2.0, 2.5, 3.0),
        'Indole': (16.0, 16.5, 17.0),
    }
    
    # 如果官能基類型在已知範圍內，調整formal_charge部分的特徵
    # 這是原子特徵中的一部分，可以用於表示pKa相關特性
    if group_type in pka_ranges:
        # 獲取中位數pKa值
        _, median_pka, _ = pka_ranges[group_type]
        
        # 調整原子特徵中的電荷相關部分以反映官能基特性
        # 原子特徵的格式: 118(原子類型) + 6(度) + 13(形式電荷) + ...
        # 我們修改形式電荷部分來表示pKa特性
        formal_charge_start = 118 + 6  # 形式電荷特徵的起始索引
        formal_charge_end = formal_charge_start + 13  # 形式電荷特徵的結束索引
        
        # 創建一個新的形式電荷特徵向量，標記與pKa相關的位置
        # 根據pKa的範圍選擇合適的索引
        new_formal_charge = torch.zeros(13)
        if median_pka < 5.0:  # 酸性較強
            new_formal_charge[0] = 1.0  # 對應-4電荷位置
        elif 5.0 <= median_pka < 7.0:  # 中等酸性
            new_formal_charge[3] = 1.0  # 對應-1電荷位置
        elif 7.0 <= median_pka < 9.0:  # 弱酸性/弱鹼性
            new_formal_charge[12] = 1.0  # 對應0電荷位置
        elif 9.0 <= median_pka < 11.0:  # 中等鹼性
            new_formal_charge[4] = 1.0  # 對應+1電荷位置
        else:  # 強鹼性
            new_formal_charge[7] = 1.0  # 對應+4電荷位置
        
        # 替換原始特徵中的形式電荷部分
        features_list = base_features.tolist()
        features_list[formal_charge_start:formal_charge_end] = new_formal_charge.tolist()
        
        # 修改hybridization部分來表示官能基類型
        hybridization_start = formal_charge_end + 4 + 5  # 雜化狀態特徵的起始索引
        hybridization_end = hybridization_start + 6  # 雜化狀態特徵的結束索引
        
        # 根據官能基類型創建特殊的雜化狀態編碼
        new_hybridization = torch.zeros(6)
        if group_type in ['COOH', 'SulfonicAcid', 'PhosphonicAcid', 'Phosphate']:
            new_hybridization[1] = 1.0  # 代表酸性官能基
        elif group_type in ['Phenol', 'Thiol']:
            new_hybridization[2] = 1.0  # 代表酚類/硫醇類
        elif group_type in ['PrimaryAmine', 'SecondaryAmine', 'TertiaryAmine']:
            new_hybridization[3] = 1.0  # 代表胺類
        elif group_type in ['Pyridine', 'Imidazole', 'ImidazoleLike']:
            new_hybridization[4] = 1.0  # 代表含氮雜環
        else:
            new_hybridization[5] = 1.0  # 代表其他官能基
        
        # 替換原始特徵中的雜化狀態部分
        features_list[hybridization_start:hybridization_end] = new_hybridization.tolist()
        
        return torch.Tensor(features_list)
    
    # 如果官能基類型未知，直接返回原始原子特徵
    return base_features

def tensorize_for_pka(smiles_batch, functional_groups=None):
    """
    將分子轉換為圖形表示，以便用於pKa預測任務。
    這個函數處理一批SMILES字符串，為每個分子創建原子特徵、邊索引和邊特徵。
    
    Args:
        smiles_batch: 分子SMILES字符串列表
        functional_groups: 可選的官能基信息，格式為 [{'type': '官能基類型', 'indices': [原子索引]}]
        
    Returns:
        tuple: 包含(原子特徵, 邊索引, 邊特徵)的元組
    """
    if isinstance(smiles_batch, str):
        smiles_batch = [smiles_batch]
    
    mol = Chem.MolFromSmiles(smiles_batch[0])
    if mol is None:
        raise ValueError(f"無法解析SMILES: {smiles_batch[0]}")
    
    # 獲取原子特徵
    fatoms = []
    
    # 如果提供了functional_groups，創建原子索引到官能基類型的映射
    fg_map = {}
    if functional_groups:
        for fg in functional_groups:
            fg_type = fg['type']
            for idx in fg['indices']:
                fg_map[idx] = fg_type
    
    for i, atom in enumerate(mol.GetAtoms()):
        # if fg_map and i in fg_map:
        #     # 如果這個原子是官能基的一部分，使用官能基特徵
        #     fatoms.append(functional_group_features(fg_map[i], atom))
        # else:
        fatoms.append(atom_features(atom))
    
    fatoms = torch.stack(fatoms, 0)
    
    # 獲取邊和邊特徵
    edges = []
    edge_features = []
    
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        edge_feature = bond_features(bond)
        
        # 添加兩個方向的邊
        edges.append([begin_idx, end_idx])
        edge_features.append(edge_feature)
        
        edges.append([end_idx, begin_idx])
        edge_features.append(edge_feature)
    
    # 如果分子沒有鍵（只有一個原子），添加自環
    if len(edges) == 0:
        for i in range(mol.GetNumAtoms()):
            edges.append([i, i])
            # 創建零特徵向量作為自環的特徵
            edge_features.append(torch.zeros(len(bond_features(None))))
    
    # 轉換為PyTorch張量
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_features)
    else:
        edge_index = torch.tensor([], dtype=torch.long).view(2, 0)
        edge_attr = torch.tensor([], dtype=torch.float)
    
    return fatoms, edge_index, edge_attr
