# 格式： 
# SMILES,pKa_num, pKa_value
# NCC(=O)O, 2, "3,9"
import pandas as pd
import numpy as np
import os

# --- Configuration ---
INPUT_CSV_PATH = '../data/NIST_database_onlyH_6TypeEq_pos_match_max_fg_other.csv'
OUTPUT_CSV_PATH = '../data/processed_pka_data.csv'
OUTPUT_COLUMNS = ['SMILES', 'pKa_num', 'pKa_value']

# --- Main Processing Logic ---
def process_pka_data(input_path: str, output_path: str):
    """
    Reads the NIST pKa data, filters by temperature (25°C) and ionic strength (0.1),
    and creates a new CSV with max_eq_num and sorted, comma-separated pKa values.
    """
    try:
        df = pd.read_csv(input_path)
    except:
        return

    # --- Validate required columns ---
    required_cols = ['SMILES', 'Equilibrium', 'Value', 'max_eq_num', 'Temperature (C)', 'Ionic strength']
    if not all(col in df.columns for col in required_cols):
        return

    # --- Data Cleaning ---
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.dropna(subset=['SMILES', 'Value'], inplace=True)

    # Filter data by temperature and ionic strength
    filtered_df = df[(df['Temperature (C)'] == 25) & (df['Ionic strength'] == 0.1)]
    
    if filtered_df.empty:
        filtered_df = df
    
    # Group by SMILES and get Value as a list
    grp = filtered_df.groupby('SMILES')['Value'].agg(list).reset_index()
    
    # Function to sort and format pKa values
    def aggregate_pka(values):
        sorted_pka = sorted(values)
        return ",".join([f"{pka:.2f}" for pka in sorted_pka])

    # Apply the aggregation function
    grp['pKa_value'] = grp['Value'].apply(aggregate_pka)
    grp.drop('Value', axis=1, inplace=True)

    # Get the max_eq_num (should be consistent per SMILES, take the first)
    df_max_eq = filtered_df.groupby('SMILES', as_index=False)['max_eq_num'].first()
    try:
        df_max_eq['max_eq_num'] = df_max_eq['max_eq_num'].astype(int)
    except:
        pass
    df_max_eq.rename(columns={'max_eq_num': 'pKa_num'}, inplace=True)

    # Merge the aggregated pKa strings with the max_eq_num
    df_final = pd.merge(df_max_eq, grp, on='SMILES', how='inner')

    # Ensure correct column order
    df_final = df_final[OUTPUT_COLUMNS]
    df_final = df_final.sort_values(by='pKa_num')

    # --- Write Output ---
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_final.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except:
        pass

# --- Run the processing ---
if __name__ == "__main__":
    process_pka_data(INPUT_CSV_PATH, OUTPUT_CSV_PATH) 