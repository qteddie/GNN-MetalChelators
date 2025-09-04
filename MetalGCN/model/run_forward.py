# 1. import
# 2. run make_
metal2idx = {
    'Ag+': 0, 'Ca2+': 1, 'Cd2+': 2, 'Co2+': 3, 'Cu2+': 4,
    'Mg2+': 5, 'Mn2+': 6, 'Ni2+': 7, 'Pb2+': 8, 'Zn2+': 9
}

import pandas as pd
df = pd.read_csv("MetalGCN/model/pre_metal_t15_T25_I0.1_E3_m10_clean_merged_clean.csv")

batch, _, _ = make_data_obj(df.iloc[1],metal2idx)

# 3. run model init
# 4. run model forward