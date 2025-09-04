import pandas as pd

df = pd.read_csv("/work/u5066474/NTHU/LiveTransForM-main/MetalGCN/model/metal_dataset_sample_position_results.csv")


def mol_with_atom_idx(mol):
    m = Chem.Mol(mol)
    for atom in m.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    return m

from rdkit.Chem import PandasTools
from rdkit import Chem
PandasTools.AddMoleculeColumnToFrame(df, 'smiles', 'Molecule')
df['Molecule'] = df['Molecule'].apply(mol_with_atom_idx)

df.drop(columns=['num_predicted', 'num_binding_sites', 'overlap_count', 'overlap_ratio', 'total_atoms', 'prediction_status'], inplace=True)

df.to_html("/work/u5066474/NTHU/LiveTransForM-main/MetalGCN/model/metal_dataset_sample_position_results_with_molecules.html", index=False)

import pandas as pd
df_left = pd.read_csv("MetalGCN/data/pre_metal_t15_T25_I0.1_E3_m10_clean.csv")
df_right = pd.read_csv("MetalGCN/model/metal_dataset_sample_position_results.csv")
df_right = df_right[['SMILES', 'metal_ion', 'predicted_positions']]
df_res = pd.merge(df_left, df_right, on=['SMILES', 'metal_ion'], how='left')
df_res.to_csv("MetalGCN/model/pre_metal_t15_T25_I0.1_E3_m10_clean_merged.csv", index=False)


from ast import literal_eval
error_count = 0
error_idx = []
for i, row in df_res.iterrows():
    count = len(literal_eval(row['predicted_positions']))
    if count != row['pKa_num']:
        error_count += 1
        error_idx.append(i)
print(f"Error count: {error_count}")
print(f"Error idx: {error_idx}")

df_res.drop(error_idx, inplace=True)
df_res.to_csv("MetalGCN/model/pre_metal_t15_T25_I0.1_E3_m10_clean_merged_clean.csv", index=False)

mask = ( df_res['predicted_positions'].map(lambda s: len(literal_eval(s))) != df_res['pKa_num'])

error_idx = mask[mask].index.tolist()



# 測試batch

import pandas as pd

df = pd.read_csv("pre_metal_t15_T25_I0.1_E3_m10_clean_merged_clean.csv")
row = df.iloc[1]