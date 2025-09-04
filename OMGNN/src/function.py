import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ast import literal_eval
from collections import defaultdict
VERSION = "pka_ver26"
def draw_pka_distribution(VERSION):
    df = pd.read_csv(f"../results/{VERSION}/pka_combined_{VERSION}.csv")
    x = df["true_pka"].apply(literal_eval)
    x_list = [item for sublist in x for item in sublist]
    y = df["pred_pka"].apply(literal_eval)
    y_list = [item for sublist in y for item in sublist]
    fig = plt.figure(figsize=(10, 5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(x_list, bins=100, alpha=0.5, label="True pKa", color="#303030")
    ax.hist(y_list, bins=100, alpha=0.5, label="Predicted pKa", color="#ff0050")
    ax.set_xlabel("pKa")
    ax.set_ylabel("Frequency")
    ax.set_title("pKa Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"../results/{VERSION}/pka_combined_{VERSION}.png")
    print(f"[PLOT]分布圖： ../results/{VERSION}/pka_combined_{VERSION}.png saved")

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
    print(f"[CSV]Merge： ../results/{VERSION}/pka_mapping_results_{VERSION}.csv saved")
    # 畫出RMSE的分布
    plt.figure(figsize=(10, 5))
    plt.title("RMSE of pKa")
    plt.hist(df_merged['rmse'], bins=100)
    plt.xlabel("RMSE")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"../results/{VERSION}/RMSE_plot{VERSION}.png")
    print(f"[PLOT]RMSE分布圖： ../results/{VERSION}/RMSE_plot{VERSION}.png saved")
    
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
    print(f"[HTML]互動式boxplot： ../results/{VERSION}/pka_mapping_results_{VERSION}.html saved")
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
    print(f"[CSV]RMSE： ../results/{VERSION}/rmse_by_functional_group_{VERSION}.csv saved")

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
    print(f"[HTML]RMSE boxplot： ../results/{VERSION}/rmse_by_functional_group_{VERSION}.html saved")
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
    print(f"[CSV]RMSE diff： ../results/{VERSION}/pka_diff_{VERSION}.csv saved")

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
    print(f"[HTML]outlier： ../results/{VERSION}/df_outlier_{VERSION}.html saved")
