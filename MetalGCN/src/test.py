#########################
import pandas as pd
df = pd.read_csv('../data/pre_metal_t15_T25_I0.1_E3_m11_clean.csv')
df = df[df['metal_ion'] != 'Fe3+']
df.to_csv('../data/pre_metal_t15_T25_I0.1_E3_m10_clean.csv', index=False)

#########################

import pickle
import sys
import os
import torch
import pandas as pd
import numpy as np
sys.path.append("../model")
VERSION = "metal_ver3"
from metal_pka_transfer import load_config_by_version, CustomMetalPKA_GNN, evaluate, load_preprocessed_metal_data
cfg = load_config_by_version("../data/parameters.csv", VERSION)
with open(cfg["dataloader_path"], "rb") as f:
    dl_train, dl_val, dl_test, mu, sigma, n_metals = pickle.load(f)
print(f"已加載預處理的資料: {cfg['dataloader_path']}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# metal_ion_list = []
# for batch in dl_test:
#     metal_ion_list.extend(batch.metal_id.cpu().numpy())
# print(len(metal_ion_list))
# print(metal_ion_list)
model = CustomMetalPKA_GNN(n_metals, cfg["metal_emb_dim"],
                            cfg["num_features"],  # node_dim
                            9,  # bond_dim (固定)
                            cfg["hidden_size"],
                            cfg["output_size"],
                            cfg["dropout"],
                            cfg["depth"],
                            cfg["heads"],
                            ).to(device)
model.set_pka_normalization(mu, sigma)

model.load_state_dict(torch.load(f"../output/{VERSION}/best_model.pt", weights_only=True))
model.eval()

# metrics_tr type is dict
metrics_tr = evaluate(model, dl_train, "Final", device, is_train_data=True)
metrics_te = evaluate(model, dl_test,  "Final", device, is_train_data=False)

# 2. 串接結果
smiles = metrics_tr["smiles"] + metrics_te["smiles"]
metal_ion = metrics_tr["metal_ion"] + metrics_te["metal_ion"]
y_true  = np.concatenate([metrics_tr["y_true"],  metrics_te["y_true"]])
y_pred  = np.concatenate([metrics_tr["y_pred"],  metrics_te["y_pred"]])
is_train = np.concatenate([metrics_tr["is_train"], metrics_te["is_train"]])
print(len(smiles), len(metal_ion), len(y_true), len(y_pred), len(is_train))

res = pd.DataFrame({
    'metal_ion': metal_ion,
    'smiles': smiles,
    'y_true': y_true,
    'y_pred': y_pred,
    'is_train': is_train,
})

res['y_true'] = res['y_true'].apply(lambda x: round(x, 2))
res['y_pred'] = res['y_pred'].apply(lambda x: round(x, 4))

from metal_pka_transfer import build_metal_vocab
df = pd.read_csv("../data/Metal_t15_T25_I0.1_E3_m11.csv")
metal2idx = build_metal_vocab(df)
print(metal2idx)
# 創建反向映射：從索引到金屬離子名稱
idx2metal = {v: k for k, v in metal2idx.items()}
print("idx2metal:", idx2metal)
from collections import Counter
print(Counter(res['metal_ion']))
# 使用反向映射將索引轉換為金屬離子名稱
res['metal_ion'] = res['metal_ion'].apply(lambda x: idx2metal[x])
res.to_csv(f"../output/{VERSION}/evaluation_results.csv", index=False)


def plot_distribution(VERSION):
    import matplotlib.pyplot as plt
    res = pd.read_csv(f"../output/{VERSION}/{VERSION}_evaluation_results.csv")
    plt.figure(figsize=(10, 5))
    plt.title("Error Distribution of pKa")
    plt.hist(res['error'], bins=100)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    output_path = os.path.join(cfg["output_dir"], f"{VERSION}_error_distribution.png")
    plt.savefig(output_path)
    print(f"[PLOT] Error Distribution: {output_path} saved")

    plt.figure(figsize=(10, 5))
    plt.title("AbsoluteError Distribution of pKa")
    plt.hist(res['abs_error'], bins=100)
    plt.xlabel("Absolute error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    output_path = os.path.join(cfg["output_dir"], f"{VERSION}_absolute_error_distribution.png")
    plt.savefig(output_path)
    print(f"[PLOT] Absolute Error Distribution: {output_path} saved")
VERSION = "metal_ver3"
plot_distribution(VERSION)

res[res['abs_error'] > 10].to_csv(f"../data/outlier.csv", index=False)



# 把outlier的資料從pre dataset刪除
import pandas as pd
df = pd.read_csv(f"../data/pre_metal_t15_T25_I0.1_E3_m11.csv")
df_outlier = pd.read_csv(f"../data/outlier.csv")

KEY = ["metal_ion", "SMILES"]
df_clean = (
    df
        .merge(df_outlier[KEY].drop_duplicates(), on=KEY, how="left", indicator=True)
        .query('_merge == "left_only"')
        .drop(columns="_merge")
)
df_clean.to_csv(f"../data/pre_metal_t15_T25_I0.1_E3_m11_clean.csv", index=False)



# =================計算每個pKa的權重=================#
def create_pka_weight_table(VERSION):
    import pandas as pd
    import numpy as np
    from ast import literal_eval
    combined_csv = f"../output/{VERSION}/{VERSION}_pka_mapping_results.csv"
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
    
    
    
# 測試跑模型
import os, torch, pickle, sys, numpy as np, pandas as pd
sys.path.append("../model")
from metal_pka_transfer import parse_args, create_weight_tables, set_seed, train, output_evaluation_results, plot_boxplot, parity_plot, CustomMetalPKA_GNN, evaluate
cfg = parse_args()
print("\n=== Config ===")
for k, v in cfg.items():
    print(f"{k:18}: {v}")
print("===============\n")
# 讀取權重表
bin_edges, bin_w, metal_w_dict = create_weight_tables(cfg["version"])

set_seed(cfg["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if os.path.exists(cfg["dataloader_path"]):
    with open(cfg["dataloader_path"], "rb") as f:
        dl_train, dl_val, dl_test, mu, sigma, n_metals = pickle.load(f)
    print(f"已加載預處理的資料: {cfg['dataloader_path']}")
else:
    print(f"未找到預處理資料，重新預處理資料: {cfg['dataloader_path']}")
    dl_train, dl_val, dl_test, mu, sigma, n_metals = load_preprocessed_metal_data(
    cfg, cfg["metal_csv"], cfg["batch_size"], metal_w_dict=metal_w_dict)
# 如果換資料集的話要把dataloader.pt刪掉，重新預處理資料

metal_weights_tensor = torch.ones(len(metal_w_dict), dtype=torch.float32)
bond_dim = 9 + cfg["max_dist"] + 1 # 9 + 6 = 15
model = CustomMetalPKA_GNN(n_metals, cfg["metal_emb_dim"],
                            cfg["num_features"],  # node_dim
                            bond_dim,
                            cfg["hidden_size"],
                            cfg["output_size"],
                            cfg["dropout"],
                            cfg["depth"],
                            cfg["heads"],
                            spd_dim=cfg["spd_dim"],
                            max_dist=cfg["max_dist"],
                            bin_edges=bin_edges,
                            bin_weights=bin_w,
                            metal_weights_tensor=metal_weights_tensor
                            ).to(device)
model.set_pka_normalization(mu, sigma)

output_dir = cfg.get("output_dir", "./outputs")
# df_loss   = []
# model = train(model, (dl_train, dl_val), cfg, device, output_dir, df_loss)

# 1. 取得 train / test 兩批評估結果
# model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=True))
model.load_state_dict(torch.load(f"../output/{cfg['version']}/best_model.pt", weights_only=True))
model.eval()
metrics_tr = evaluate(model, dl_train, "Final", device, is_train_data=True)
metrics_te = evaluate(model, dl_test,  "Final", device, is_train_data=False)
y_true  = np.concatenate([metrics_tr["y_true"],  metrics_te["y_true"]])
y_pred  = np.concatenate([metrics_tr["y_pred"],  metrics_te["y_pred"]])
is_train = np.concatenate([metrics_tr["is_train"], metrics_te["is_train"]])

output_evaluation_results(metrics_tr, metrics_te, cfg["version"], cfg)
parity_png = os.path.join(output_dir, f'{cfg["version"]}_parity_plot.png')
parity_plot(y_true, y_pred, is_train, parity_png, title=f"Parity Plot - Test RMSE {metrics_te['rmse']:.3f}")
plot_boxplot(cfg["version"], output_dir)



# __________________________________

import pandas as pd, numpy as np
version = "metal_ver11"
df = pd.read_csv("../output/metal_ver11/metal_ver11_evaluation_results.csv")
pred = df["y_pred"]
true = df["y_true"]
sigma = 5.611272089179448
mu = 5.987196829793783

err = np.abs(pred - true) / sigma
print(np.percentile(err, [50, 90, 99]))


print(f"cls:{l_cls.item():.4f}  reg:{alpha*l_reg.item():.4f}")


# __________________________________

import pandas as pd, numpy as np
df = pd.read_csv("../output/metal_ver10/metal_ver10_evaluation_results.csv")
bins = pd.qcut(df['y_true'], q=4)      # 四分位
print(df.groupby(bins)['abs_error'].agg(['count','mean','max']).round(2))


import pandas as pd
from ast import literal_eval
import numpy as np
df = pd.read_csv("../data/pre_metal_t15_T25_I0.1_E3_m10_clean.csv")
flat_list = df["pKa_value"].apply(literal_eval).explode().tolist()
import matplotlib.pyplot as plt
plt.hist(flat_list, bins=100)
plt.savefig("pka_distribution.png")


# __________________________________


import pandas as pd
df = pd.read_csv("/work/u5066474/NTHU/LiveTransForM-main/MetalGCN/output/metal_ver10/metal_ver10_evaluation_results.csv")
from parity_plot import parity_plot
threshold = 6
parity_plot(df[df["y_true"] < threshold]["y_true"], df[df["y_true"] < threshold]["y_pred"], df[df["y_true"] < threshold]["is_train"], "parity_plot.png")