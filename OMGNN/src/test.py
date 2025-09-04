


from self_pka_trainutils import load_config_by_version, pka_Dataloader
from self_pka_chemutils import tensorize_for_pka
from self_pka_models   import pka_GNN, pka_GNN_ver2, pka_GNN_ver3
import torch, torch.nn as nn
import pandas as pd

config_csv = "../data/pka_parameters.csv"
version = "pka_ver26"

cfg = load_config_by_version(config_csv, version)
print("\n=== Config ===")
for k, v in cfg.items(): 
    print(f"{k:18}: {v}")
print("==============\n")


cols = ["smiles", "pka_values", "pka_matrix"]
train_loader, test_loader = pka_Dataloader.data_loader_pka(
    file_path   = cfg["input"],
    columns     = cols,
    tensorize_fn= tensorize_for_pka,
    batch_size  = cfg["batch_size"],
    test_size   = cfg["test_size"],
    k_pe        = cfg["pe_dim"],
    max_dist    = cfg["max_dist"],
)
# for batch in train_loader:
#     print(batch.lap_pos.shape)
# if not hasattr(batch, "lap_pos"):
#     raise ValueError("batch 中缺少 lap_pos 欄位")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bond_dim = 11 + cfg["max_dist"] + 1 # 11 + 6 = 17
model  = pka_GNN_ver3(
            node_dim  = 153,
            bond_dim  = bond_dim,
            hidden_dim= 153,
            output_dim= 1,
            dropout   = cfg["dropout"],
            depth     = cfg["depth"],
            heads      = cfg["heads"],
            pe_dim     = cfg["pe_dim"],
         ).to(device)
# model.set_pka_normalization(pka_mean, pka_std)
ckpt_path = f"/work/u5066474/NTHU/LiveTransForM-main/OMGNN/results/pka_ver20/pka_ver20_best.pkl"
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

###


smiles = "FC1(F)CNCCNCCCNCCNC1"
MAPPING_FILE = "../output/pka_mapping_results.csv"
model.eval()
res = model.sample(smiles, MAPPING_FILE, device)



#===============畫pKa分布================#
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval
VERSION = "pka_ver26"
# df = pd.read_csv(f"../results/{VERSION}/pka_combined_{VERSION}.csv")
def draw_pka_distribution(VERSION):
    df = pd.read_csv(f"../results/{VERSION}/pka_combined_{VERSION}.csv")
    x = df["true_pka"].apply(literal_eval)
    x_list = [item for sublist in x for item in sublist]
    y = df["pred_pka"].apply(literal_eval)
    y_list = [item for sublist in y for item in sublist]
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x_list, bins=100, alpha=0.5, label="True pKa")
    ax.hist(y_list, bins=100, alpha=0.5, label="Predicted pKa")
    ax.set_xlabel("pKa")
    ax.set_ylabel("Frequency")
    ax.set_title("pKa Distribution")
    ax.legend()
    plt.savefig(f"../results/{VERSION}/pka_combined_{VERSION}.png")
    print(f"[PLOT] ../results/{VERSION}/pka_combined_{VERSION}.png saved")
def merge_mapping_results(VERSION):
    df_mapping = pd.read_csv("../output/pka_mapping_results.csv")
    tmp_str_list = df_mapping['mapping'].apply(eval)
    dict_list = []
    for dict_item in tmp_str_list:
        tmp_dict = {}
        for key, value in dict_item.items():
            tmp_dict[key] = value['type']
        dict_list.append(tmp_dict)
    df_mapping['func_mapping'] = dict_list
    # df_mapping.to_csv("../output/pka_mapping_results1.csv", index=False)
    df = pd.read_csv(f"../results/{VERSION}/pka_combined_{VERSION}.csv")
    df_merged = pd.merge(df, df_mapping, on='smiles', how='left')
    df_merged.drop(columns=['pka_values', 'functional_groups', 'mapping'], inplace=True)
    df_merged.to_csv(f"../results/{VERSION}/pka_combined_{VERSION}.csv", index=False)
    dict_list_1 = []
    for _, row in df_merged.iterrows():
        key = eval(row['pred_pka'])
        value = row['func_mapping'].values()
        tmp_dict = dict(zip(key, value))
        dict_list_1.append(tmp_dict)
    df_merged['func_mapping_pred'] = dict_list_1
    df_merged.to_csv(f"../results/{VERSION}/pka_mapping_results_{VERSION}.csv", index=False)
    print(f"[CSV] ../results/{VERSION}/pka_mapping_results_{VERSION}.csv saved")
    # 畫出RMSE的分布
    plt.figure(figsize=(10, 5))
    plt.title("RMSE of pKa", fontsize=24)
    plt.hist(df_merged['rmse'], bins=100)
    plt.xlabel("RMSE", fontsize=24)
    plt.ylabel("Frequency", fontsize=24)
    plt.tight_layout()
    plt.savefig(f"../results/{VERSION}/RMSE_plot{VERSION}.png")
    print(f"[PLOT] ../results/{VERSION}/RMSE_plot{VERSION}.png saved")
merge_mapping_results(VERSION)
def plot_rmse_distribution(VERSION):
    df_merged = pd.read_csv(f"../OMGNN/results/{VERSION}/pka_mapping_results_{VERSION}.csv")
    # 畫出RMSE的分布
    font_size = 18
    font_size2 = 14
    plt.figure(figsize=(10, 5))
    plt.title("RMSE of pKa", fontsize=font_size)
    plt.hist(df_merged['rmse'], bins=100)
    plt.xlabel("RMSE", fontsize=font_size)
    plt.ylabel("Frequency", fontsize=font_size)
    plt.xticks(fontsize=font_size2)
    plt.yticks(fontsize=font_size2)
    plt.tight_layout()
    plt.savefig(f"../final_project/RMSE_plot{VERSION}.png")
    print(f"[PLOT] ../final_project/RMSE_plot{VERSION}.png saved")
plot_rmse_distribution(VERSION)
    
draw_pka_distribution(VERSION)

#===============做pred的boxplot的boxplot================#
import matplotlib.pyplot as plt
from ast import literal_eval
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
VERSION = "pka_ver24"
def draw_boxplot(VERSION):
    df_merged_2 = pd.read_csv(f"../results/{VERSION}/pka_mapping_results_{VERSION}.csv")
    grouped = defaultdict(list)
    for func_dict in df_merged_2['func_mapping_pred']:
        for key, value in eval(func_dict).items():
            grouped[value].append(key)
    df_plot = pd.DataFrame([{"Functional Group": group, "pKa": pka} 
                            for group, pka_list in grouped.items() 
                            for pka in pka_list])
    df_plot['pKa'] = df_plot['pKa'].apply(lambda x: float(round(x, 2)))
    #===============建立互動式boxplot================#
    import plotly.graph_objects as go
    import plotly.express as px
    # 設定顏色（依 functional group 數量自動選擇）
    unique_groups = df_plot['Functional Group'].unique()
    n_group = len(unique_groups)
    colors = px.colors.qualitative.Vivid[:n_group]  

    fig = go.Figure()

    for group, color in zip(unique_groups, colors):
        y_vals = df_plot[df_plot['Functional Group'] == group]['pKa']
        fig.add_trace(go.Box(
            y=y_vals,
            name=group,
            boxpoints='all',  
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=color,
            marker_size=3,
            line_width=1
        ))

    fig.update_layout(
        title='Predicted pKa by Functional Group',
        yaxis=dict(
            title="pKa",
            autorange=True,
            showgrid=True,
            zeroline=True,
            dtick=1,
            gridcolor='rgb(230, 230, 230)',
            gridwidth=1,
            zerolinecolor='rgb(180, 180, 180)',
            zerolinewidth=1,
        ),
        margin=dict(l=40, r=30, b=80, t=100),
        paper_bgcolor='rgb(250, 250, 250)',
        plot_bgcolor='rgb(250, 250, 250)',
        showlegend=False
    )
    # fig.show()
    fig.write_html(f"../results/{VERSION}/pka_mapping_results_{VERSION}.html")
    print(f"[HTML] ../results/{VERSION}/pka_mapping_results_{VERSION}.html saved")
draw_boxplot(VERSION)

# 計算每個functional group的rmse
from collections import defaultdict
import pandas as pd
import numpy as np
VERSION = "pka_ver24"
def calculate_rmse_by_functional_group(VERSION):
    df = pd.read_csv(f"../results/{VERSION}/pka_mapping_results_{VERSION}.csv")
    group = defaultdict(list)
    for _, row in df.iterrows():
        true_dict = eval(row['func_mapping'])
        pred_dict = eval(row['func_mapping_pred'])
        true_list = list(map(float, true_dict.keys()))
        pred_list = list(map(float, pred_dict.keys()))
        length = len(true_list)
        rmse_list = []
        if len(true_list) != len(pred_list):
            
            # print(f"true_list: {true_list}")
            # print(f"pred_list: {pred_list}")
            # print(f"row: {row}")
            continue
        for i in range(length):
            rmse = np.sqrt(np.mean((np.array(true_list[i]) - np.array(pred_list[i]))**2))      
            rmse_list.append(rmse)
        func_list = []
        for key, value in true_dict.items():
            func_list.append(value)
        for i in range(len(func_list)):
            group[func_list[i]].append(rmse_list[i])


    group_df = pd.DataFrame({"Functional Group": func, "RMSE": rmse} for func, rmse_list in group.items() for rmse in rmse_list)
    group_df.to_csv(f"../results/{VERSION}/rmse_by_functional_group_{VERSION}.csv", index=False)
    print(f"[CSV] ../results/{VERSION}/rmse_by_functional_group_{VERSION}.csv saved")

    import plotly.graph_objects as go
    import plotly.express as px

    # 設定顏色（依 functional group 數量自動選擇）
    unique_groups = group_df['Functional Group'].unique()
    n_group = len(unique_groups)
    colors = px.colors.qualitative.Vivid[:n_group]  

    fig = go.Figure()

    for group, color in zip(unique_groups, colors):
        y_vals = group_df[group_df['Functional Group'] == group]['RMSE']
        fig.add_trace(go.Box(
            y=y_vals,
            name=group,
            boxpoints='all',  
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=color,
            marker_size=3,
            line_width=1
        ))

    fig.update_layout(
        title='Predicted pKa by Functional Group',
        yaxis=dict(
            title="RMSE",
            autorange=True,
            showgrid=True,
            zeroline=True,
            dtick=1,
            gridcolor='rgb(230, 230, 230)',
            gridwidth=1,
            zerolinecolor='rgb(180, 180, 180)',
            zerolinewidth=1,
        ),
        margin=dict(l=40, r=30, b=80, t=100),
        paper_bgcolor='rgb(250, 250, 250)',
        plot_bgcolor='rgb(250, 250, 250)',
        showlegend=False
    )
    # fig.show()
    fig.write_html(f"../results/{VERSION}/rmse_by_functional_group_{VERSION}.html")
    print(f"[HTML] ../results/{VERSION}/rmse_by_functional_group_{VERSION}.html saved")


#===============打印參數================#
import pandas as pd
df_param = pd.read_csv("../data/pka_parameters.csv")
VERSION = "pka_ver25"
df_param = df_param.loc[df_param['version'] == VERSION]
print(f"{df_param['version'].values[0]}: "
      f"hidden_dim={df_param['hidden_size'].values[0]}, "
      f"depth={df_param['depth'].values[0]}, "
      f"heads={df_param['heads'].values[0]}, "
      f"pe_dim={df_param['pe_dim'].values[0]}, "
      f"max_dist={df_param['max_dist'].values[0]}, "
      f"weight_decay={df_param['weight_decay'].values[0]}, "
      f"dropout={df_param['dropout'].values[0]}, "
      f"learning_rate={df_param['lr'].values[0]}, "
)










def mol_len(smi):
    count = 0
    for i in range(len(smi)):
        if smi[i].isalpha():
            count += 1
    return count

smi_list = df_valid['smiles'].tolist()
smi_len = [mol_len(smi) for smi in smi_list]
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 5))
plt.hist(smi_len, bins=100)
plt.xlabel("Molecular Length")
plt.xticks(range(0, 100, 5))
plt.xlim(0, 50)
plt.ylabel("Frequency")
plt.title("Molecular Length Distribution")
plt.tight_layout()
plt.savefig("mol_len.png")



true_pka = np.concatenate(true_lists)
pred_pka = np.concatenate(pred_lists)
counts    = true_lists.str.len()            # 每列有幾個 pKa
is_train  = np.repeat(df_valid["is_train"].values.astype(bool), counts)
title = "pKa Parity Plot"
out_png = f"../results/pka_ver22/pka_parity_pka_ver22_1.png"
parity_plot(true_pka, pred_pka, title=title, out_png=out_png, is_train=is_train)



# =================計算每個pKa的權重=================#
def create_pka_weight_table(VERSION):
    import pandas as pd
    import numpy as np
    from ast import literal_eval
    combined_csv = "../results/pka_ver25/pka_mapping_results_pka_ver25.csv"
    df_valid = pd.read_csv(combined_csv).dropna()
    # 改成只計算train dataset的權重
    df_valid = df_valid[df_valid["is_train"] == True]
    true_lists = df_valid["true_pka"].apply(literal_eval)
    true_pka = np.concatenate(true_lists)
    pka_all = true_pka
    # 2. 建 bin（這裡 0.5 為寬，可依需要改）
    bin_edges = np.array([0,2,3,4,5,6,7,8,9,10,11,14], dtype=np.float32)
    hist, _ = np.histogram(pka_all, bins=bin_edges)
    # 3. 轉頻率 → 取倒數 → 歸一化
    freq      = hist / hist.sum()
    inv_freq  = 1 / (freq + 1e-6)
    gamma = 0.7
    weights = inv_freq ** gamma
    weights   = weights / weights.mean()        # 平均權重=1
    # 4. 存成 numpy，再用 torch 轉 tensor
    np.savez('../data/pka_weight_table.npz',
            bin_edges=bin_edges.astype(np.float32),
            weights=weights.astype(np.float32))
    print(f"[NPZ] ../data/pka_weight_table.npz saved")





#===============計算pKa的RMSE，並且包含其他資訊================#
import pandas as pd
import numpy as np
from ast import literal_eval
VERSION = "pka_ver24"
def calculate_pka_rmse(VERSION):
    combined_csv = f"../results/{VERSION}/pka_mapping_results_{VERSION}.csv"
    df_valid = pd.read_csv(combined_csv).dropna()

    diff_list = []
    for _, row in df_valid.iterrows():
        true = eval(row['true_pka'])
        pred = eval(row['pred_pka'])
        smiles = row['smiles']
        func_dict = eval(row['func_mapping_pred'])
        pka_matrix = eval(row['pka_matrix'])
        for t, p, pm in zip(true, pred, pka_matrix):
            rmse = np.sqrt(np.mean((t - p)**2))
            func = func_dict[p]
            diff_list.append({
                "smiles": smiles, 
                "true_pka": t,
                "pred_pka": p,
                "rmse": rmse, 
                "func": func,
                "pos": pm[0],
                "mapping":pka_matrix
            })
    res_df = pd.DataFrame(diff_list)
    res_df = res_df.sort_values(by="rmse", ascending=True)
    res_df.to_csv(f"../results/{VERSION}/pka_diff_{VERSION}.csv", index=False)
    print(f"[CSV] ../results/{VERSION}/pka_diff_{VERSION}.csv saved")
calculate_pka_rmse(VERSION)
#===============================#

#=================分析outlier=================#
import pandas as pd
VERSION = "pka_ver25"
def draw_html_outlier(VERSION):
    df_outlier = pd.read_csv(f"../results/{VERSION}/pka_diff_{VERSION}.csv")
    # smiles wrap換行
    from rdkit.Chem import PandasTools
    from rdkit import Chem
    def mol_with_atom_idx(mol):
        m = Chem.Mol(mol)
        for atom in m.GetAtoms():
            atom.SetProp("atomNote", str(atom.GetIdx()))
        return m
    PandasTools.AddMoleculeColumnToFrame(df_outlier, smilesCol="smiles", molCol="Mol")
    df_outlier['Mol'] = df_outlier['Mol'].apply(mol_with_atom_idx)
    df_outlier.to_html(f"../results/{VERSION}/df_outlier_{VERSION}.html", index=False)
    print(f"[HTML] ../results/{VERSION}/df_outlier_{VERSION}.html saved")
draw_html_outlier(VERSION)

# 畫residual histogram
# 確認真的沒有heavy-tail
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def draw_residual_histogram(VERSION):
    
    df = pd.read_csv(f"../results/{VERSION}/pka_combined_{VERSION}.csv")
    x_list = [item for sublist in df["true_pka"].apply(literal_eval) for item in sublist]
    y_list = [item for sublist in df["pred_pka"].apply(literal_eval) for item in sublist]
    
    true_np = np.asarray(x_list, dtype=float)
    pred_np = np.asarray(y_list, dtype=float)
    if true_np.shape != pred_np.shape:
        raise ValueError("true_np and pred_np have different shapes")
    residual = pred_np - true_np
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residual, bins=100, alpha=0.5, color="tab:blue")
    ax.set_xlabel("Prediction − Ground truth (pKa)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Residual Distribution of {VERSION}")
    fig.tight_layout()
    plt.savefig(f"../results/{VERSION}/residual_distribution_{VERSION}.png")
    plt.close(fig)
    print(f"[PLOT]殘差分布 ../results/{VERSION}/residual_distribution_{VERSION}.png saved")
draw_residual_histogram(VERSION)

# 畫QQ-plot
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ast import literal_eval
from pathlib import Path
from scipy import stats
def draw_residual_qq(VERSION):
    df = pd.read_csv(f"../results/{VERSION}/pka_combined_{VERSION}.csv")
    x_list = [item for sublist in df["true_pka"].apply(literal_eval) for item in sublist]
    y_list = [item for sublist in df["pred_pka"].apply(literal_eval) for item in sublist]
    
    true_np = np.asarray(x_list, dtype=float)
    pred_np = np.asarray(y_list, dtype=float)
    if true_np.shape != pred_np.shape:
        raise ValueError("true_np and pred_np have different shapes")
    residual = pred_np - true_np
    fig, ax = plt.subplots(figsize=(8,8), dpi=300)
    stats.probplot(residual, dist="norm", plot=ax)
    ax.set_title(f"QQ-plot of residuals of {VERSION}")
    ax.set_xlabel("Theoretical Quantiles (Normal)")
    ax.set_ylabel("Ordered residuals")
    out_png = f"../results/{VERSION}/residual_qq_{VERSION}.png"
    fig.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)
    print(f"[PLOT]殘差QQ-plot ../results/{VERSION}/residual_qq_{VERSION}.png saved")
draw_residual_qq(VERSION)


import pandas as pd
df_test = pd.read_csv("../data/pka_parameters.csv")
df_test.columns