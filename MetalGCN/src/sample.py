


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("../../MetalGCN/model")
import pandas as pd
from metal_pka_transfer import load_config_by_version


def load_model(version: str):
    import sys, pickle, torch, os
    sys.path.append("../model")
    from metal_pka_transfer import CustomMetalPKA_GNN, load_config_by_version, create_weight_tables
    cfg = load_config_by_version("../data/parameters.csv", version)
    with open(cfg["dataloader_path"], "rb") as f:
        dl_train, dl_val, dl_test, mu, sigma, n_metals = pickle.load(f)
        print(f"已加載預處理的資料: {cfg['dataloader_path']}")
    bin_edges, bin_w = create_weight_tables(cfg["version"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomMetalPKA_GNN(n_metals, cfg["metal_emb_dim"],
                               cfg["num_features"],  # node_dim
                               9, # 先測試沒有spd的版本
                               cfg["hidden_size"],
                               cfg["output_size"],
                               cfg["dropout"],
                               cfg["depth"],
                               cfg["heads"],
                                bin_edges=bin_edges,
                                bin_weights=bin_w,
                                huber_beta=cfg["huber_beta"],
                                reg_weight=cfg["reg_weight"],
                               ).to(device)
    model.load_state_dict(torch.load(os.path.join(cfg["output_dir"], "best_model.pt"), weights_only=True))
    return model


VERSION = "metal_ver15"

model = load_model(VERSION)
model.eval()
import pandas as pd
from ast import literal_eval
df = pd.read_csv("../../OMGNN/src/dupont_metal.csv")
smiles = df["SMILES"].tolist()[0]
file_path = "../../OMGNN/src/dupont_metal.csv"


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = load_config_by_version("../data/parameters.csv", VERSION)

res_list = []
for i, row in df.iterrows():
    smi = row["SMILES"]
    # print(smi)
    res = model.sample(smi, "../../OMGNN/src/dupont_metal.csv", cfg)
    res_list.append(res)
print(res_list)

res_df = pd.DataFrame(res_list)
res_df['rmse'] = res_df['rmse'].apply(lambda x: round(x, 2))
res_df.to_csv("metal_res.csv", index=False)






