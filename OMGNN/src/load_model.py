from self_pka_sample import load_model

import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools

def draw_html_from_csv(path: str):
    df = pd.read_csv(path)
    def mol_with_atom_idx(mol):
        m = Chem.Mol(mol)
        for atom in m.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()))
        return m
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol="smiles", molCol="Mol")
    df['Mol'] = df['Mol'].apply(mol_with_atom_idx)
    
    return df


import os
os.makedirs("dupont_img", exist_ok=True)
df = draw_html_from_csv("dupont.csv")
for i, row in df.iterrows():
    id = row['id']
    smiles = row['smiles']
    pka = row['pka']
    mol = row['Mol']
    print(id, smiles, pka)
    Chem.Draw.MolToFile(mol, f"dupont_img/{id}.png")
    # print("-"*100)
# ────────────────────────────────────────────────────────────
# 載入模型
model = load_model(version="pka_ver26")
model.eval()
df = pd.read_csv("dupont.csv")

df = draw_html_from_csv("dupont.csv")
df.to_html("dupont.html", index=False)

model.sample(smiles=df.iloc[10]['smiles'], file_path="dupont.csv")


# main function
def predict_pka():
    df = pd.read_csv("dupont.csv")
    res = []
    for i, row in df.iterrows():
        # if i == 2 or i == 5:
        #     continue
        if row['id'] == 5:
            continue
        out = model.sample(smiles=row['smiles'], file_path="dupont.csv")
        out.update({"id": i})
        res.append(out)

    cols = ['id', 'smiles', 'rmse', 'true_pka', 'pred_pka', 'rec']
    # cols = ['id', 'smiles', 'true_pka', 'pred_pka', 'rmse', 'true_cla', 'pred_cla', 'acc', 'prec', 'rec', 'f1']
    res_df = pd.DataFrame(res, columns=cols)
    res_df.to_csv("dupont_res.csv", index=False)

predict_pka()
df = draw_html_from_csv("dupont_res.csv")
df['rmse'] = df['rmse'].apply(lambda x: round(x, 2))
df['true_pka'] = df['true_pka'].apply(lambda x: [round(float(y), 2) for y in eval(x)])
df['pred_pka'] = df['pred_pka'].apply(lambda x: [round(float(y), 2) for y in eval(x)])
df.to_html("dupont_res.html", index=False)

df_eval = pd.read_csv("../results/pka_ver26/pka_combined_pka_ver26.csv")
rec_mean = df_eval['rec'].mean()


import pandas as pd
df = pd.read_csv("../results/pka_ver26/pka_combined_pka_ver26.csv")
list_of_smiles = pd.read_csv("poster.csv")['smiles'].tolist()

list_of_rmse = df[df['smiles'].isin(list_of_smiles)]['rmse']
list_of_rec = df[df['smiles'].isin(list_of_smiles)]['rec']

df_res = pd.DataFrame({"smiles": list_of_smiles, "rmse": list_of_rmse, "rec": list_of_rec})
df_res['rmse'] = df_res['rmse'].apply(lambda x: round(x, 2))
df_res['rec'] = df_res['rec'].apply(lambda x: round(x, 2))
df_res.to_csv("poster.csv", index=False)

df_poster = pd.read_csv("poster.csv")

# Alternative approach to avoid pandas patching issues
def create_mol_column(df, smiles_col="smiles", mol_col="Mol"):
    """Create molecule column without relying on PandasTools patching"""
    df[mol_col] = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(x) if pd.notna(x) else None)
    return df

def mol_with_atom_idx(mol):
    """Add atom indices to molecule"""
    if mol is None:
        return None
    m = Chem.Mol(mol)
    for atom in m.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    return m

# Create molecule column and add atom indices
df_poster = create_mol_column(df_poster)
df_poster['Mol'] = df_poster['Mol'].apply(mol_with_atom_idx)
import os
os.makedirs("poster_img", exist_ok=True)
for i, row in df_poster.iterrows():
    mol = row['Mol']
    Chem.Draw.MolToFile(mol, f"poster_img/{i}.png")
    # print("-"*100)
smiles = "N[C@@H](CO)C(O)=O"
mol = Chem.MolFromSmiles(smiles)
Chem.Draw.MolToFile(mol, f"poster_img/6.png")

# Read the original database
df_ori = pd.read_csv("../data/NIST_database_onlyH_6TypeEq_pos_match_max_fg_other.csv")

# Re-read the poster data to ensure we have the correct columns
df_poster = pd.read_csv("poster.csv")

# Check if 'smiles' column exists
if 'smiles' in df_poster.columns:
    # Method 1: Using map with dictionary (recommended)
    smiles_to_ligand = df_ori.drop_duplicates('SMILES').set_index('SMILES')['Ligand'].to_dict()
    df_poster['name'] = df_poster['smiles'].map(smiles_to_ligand)
    df_poster.to_csv("poster.csv", index=False)
else:
    print("Error: 'smiles' column not found in df_poster")
    print("Available columns:", df_poster.columns.tolist()) 
    
    
df_metal = pd.read_csv("/work/u5066474/NTHU/LiveTransForM-main/MetalGCN/output/metal_ver10/metal_ver10_evaluation_results.csv")
metal_ion_list = df_metal.set_index('smiles')['metal_ion'].to_dict()
smiles_to_err = df_metal.set_index('smiles')['abs_error'].to_dict()

df_poster['metal_ion'] = df_poster['smiles'].map(metal_ion_list)
df_poster['abs_err'] = df_poster['smiles'].map(smiles_to_err)
df_poster.to_csv("poster.csv", index=False)