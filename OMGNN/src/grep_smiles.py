import pandas as pd, pubchempy as pcp
import re, time
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm                            # ← 進度條

# ---------- 1. 讀檔 ----------
df = pd.read_csv("../data/stability_constant_unique.csv")

# ---------- 2. 工具函式 ----------
def pubchem_props(name, sleep=0.3):
    """回傳 (MolecularFormula, IsomericSMILES)，失敗時 (None, None)"""
    try:
        cids = pcp.get_cids(name, namespace="name")
        if not cids:
            return None, None
        comp = pcp.Compound.from_cid(cids[0])
        time.sleep(sleep)        # 節流，≤5 req/sec
        return comp.molecular_formula, comp.canonical_smiles
    except Exception:
        return None, None

# ---------- 3. 化學式比對 ----------
element_pat = re.compile(r"([A-Z][a-z]?)(\d*)")
parse_formula = lambda f: {el: int(n or 1) + 0 for el, n in element_pat.findall(f or "")}
same_formula  = lambda f1, f2: parse_formula(f1) == parse_formula(f2)

# ---------- 4. 多執行緒查詢 + 進度條 ----------
names = df["Ligand"].tolist()
print(f"Start querying {len(names)} ligands with 5 threads...")

with ThreadPool(5) as pool:
    # imap_unordered 會邊產生結果邊更新，搭配 tqdm 最順暢
    results = list(tqdm(pool.imap_unordered(pubchem_props, names),
                        total=len(names), desc="PubChem"))

df["PubChemFormula"] = [r[0] for r in results]
df["PubChemSMILES"]  = [r[1] for r in results]
df["FormulaMatch"]   = df.apply(lambda r: same_formula(str(r["Formula"]), str(r["PubChemFormula"])), axis=1)

# ---------- 5. 輸出 ----------
out_file = "../data/ligands_with_smiles.csv"
df.to_csv(out_file, index=False)
print(f"Done! --> {out_file}")
