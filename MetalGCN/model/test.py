from MetalGCN.model.new_metal_pka_transfer import make_data_obj
import pandas as pd

df = pd.read_csv("../data/pre_metal_t15_T25_I0.1_E3_m10_clean.csv")

df.head()

row = df.iloc[0]

cfg = parse_args()

metal2idx = {
    'Ag+': 0, 'Ca2+': 1, 'Cd2+': 2, 'Co2+': 3, 'Cu2+': 4,
    'Mg2+': 5, 'Mn2+': 6, 'Ni2+': 7, 'Pb2+': 8, 'Zn2+': 9
}
make_data_obj(row, metal2idx, cfg)






