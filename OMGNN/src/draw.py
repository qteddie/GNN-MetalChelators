# plot_pka_distribution.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import numpy as np # Not strictly needed for this version
import os
import argparse
from typing import List, Optional, Dict, Any

# --- Helper Functions ---

def parse_comma_separated_floats(value_str: str) -> Optional[List[float]]:
    """Parses a comma separated string into a list of floats."""
    if pd.isna(value_str) or not isinstance(value_str, str) or value_str.strip() == '':
        return None
    try:
        # Already comma separated from the extraction script
        values = [float(val.strip()) for val in value_str.split(',') if val.strip()]
        return values if values else None
    except ValueError:
        # Handle cases where conversion to float fails
        print(f"Warning: Could not parse value string to floats: '{value_str}'")
        return None
    except Exception as e:
        print(f"Warning: Unexpected error parsing value string '{value_str}': {e}")
        return None

def parse_comma_separated_strings(group_str: str) -> Optional[List[str]]:
    """Parses a comma separated string into a list of strings."""
    if pd.isna(group_str) or not isinstance(group_str, str) or group_str.strip() == '':
        return None
    try:
        # Already comma separated from the extraction script
        groups = [g.strip() for g in group_str.split(',') if g.strip()]
        return groups if groups else None
    except Exception as e:
        print(f"Warning: Unexpected error parsing group string '{group_str}': {e}")
        return None

# --- Main Plotting Function ---

def plot_distributions_by_expected_count(data_file_path: str, output_directory: str = '.'):
    """
    Loads functional group analysis data, filters based on status and Expected_pKa_Count,
    performs 1-to-1 assignment of pKa values to prioritized groups where counts match,
    and generates separate box plots for each Expected_pKa_Count (1-6).
    """
    print(f"Loading data from: {data_file_path}")
    if not os.path.exists(data_file_path):
        print(f"Error: Input file not found: {data_file_path}")
        return

    try:
        # Specify dtype for potentially problematic columns if known, otherwise rely on inference
        df = pd.read_csv(data_file_path)
        print(f"Successfully loaded {len(df)} rows.")
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        return

    required_columns = ['Status', 'Prioritized_Group_Types', 'Reported_pKa_Values', 'Expected_pKa_Count']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file must contain the columns: {required_columns}")
        return

    # Safely convert Expected_pKa_Count to numeric, coercing errors to NaN
    # Use float first to handle potential non-integer strings, then convert to Int64
    df['Expected_pKa_Count_Numeric'] = pd.to_numeric(df['Expected_pKa_Count'], errors='coerce')
    # Convert to nullable integer type to handle NaNs gracefully if needed
    # df['Expected_pKa_Count_Numeric'] = df['Expected_pKa_Count_Numeric'].astype('Int64')

    # Define the range of expected counts to plot
    max_expected_count_to_plot = 6

    for expected_count in range(1, max_expected_count_to_plot + 1):
        print(f"\n--- Processing Expected_pKa_Count = {expected_count} ---")

        # Filter based on the numeric expected count and status
        # Use .copy() to avoid SettingWithCopyWarning later
        df_filtered = df[
            (df['Expected_pKa_Count_Numeric'] == expected_count) &
            (df['Status'].astype(str).str.startswith('OK') | df['Status'].astype(str).str.startswith('Selected'))
        ].copy()

        if df_filtered.empty:
            print(f"  No data found with Status 'OK' or 'Selected' for Expected_pKa_Count = {expected_count}.")
            continue

        plot_entries: List[Dict[str, Any]] = []
        processed_rows_for_count = 0
        skipped_mismatch = 0
        skipped_parsing = 0

        for _, row in df_filtered.iterrows():
            processed_rows_for_count += 1
            groups = parse_comma_separated_strings(row.get('Prioritized_Group_Types'))
            pkas = parse_comma_separated_floats(row.get('Reported_pKa_Values'))

            # --- Perform 1-to-1 assignment ONLY if counts match and are > 0 ---
            if groups and pkas and len(groups) == len(pkas):
                if len(groups) == expected_count: # Double-check counts match expectation
                    for group, pka in zip(groups, pkas):
                        plot_entries.append({'Functional_Group': group, 'pKa': pka})
                else:
                    # This case should ideally not happen if the extraction script logic is correct,
                    # but log it if it does.
                    print(f"  Warning: Row index {row.name} has matching group/pKa counts ({len(groups)}), but != Expected_pKa_Count ({expected_count}). Skipping.")
                    skipped_mismatch += 1
            elif groups or pkas:
                 # Counts mismatch or one list is missing
                 print(f"  Debug: Skipping row index {row.name} due to group/pKa parsing or count mismatch (Groups: {len(groups) if groups else 'None'}, pKas: {len(pkas) if pkas else 'None'}).")
                 skipped_mismatch += 1
            else:
                 # Both parsing failed or resulted in empty lists
                 skipped_parsing += 1 # Only count rows where both were unparsable/empty


        print(f"  Processed {processed_rows_for_count} rows for this count.")

        if not plot_entries:
            print(f"  Warning: No valid 1-to-1 group-pKa pairs found for Expected_pKa_Count = {expected_count} after matching.")
            if skipped_mismatch > 0:
                print(f"    - Skipped {skipped_mismatch} rows due to group/pKa count mismatch or mismatch with expected count.")
            if skipped_parsing > 0:
                print(f"    - Skipped {skipped_parsing} rows due to parsing issues for both groups and pKas.")
            continue # Skip plotting for this count

        plot_df = pd.DataFrame(plot_entries)
        print(f"  Generated {len(plot_df)} data points for plotting (1-to-1 matched pairs).")
        if skipped_mismatch > 0:
             print(f"  Note: Skipped {skipped_mismatch} rows due to group/pKa count mismatch or mismatch with expected count.")
        if skipped_parsing > 0:
             print(f"  Note: Skipped {skipped_parsing} rows due to parsing issues for both groups and pKas.")


        # --- Plotting --- 
        print("  Generating plot...")
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.figure(figsize=(18, 10), dpi=300)

            # Calculate the order of groups based on median pKa for this specific subset
            if not plot_df.empty:
                group_order = plot_df.groupby('Functional_Group')['pKa'].median().sort_values().index
            else:
                group_order = [] # No order if no data

            # Create the box plot using Seaborn
            ax = sns.boxplot(
                x='Functional_Group',
                y='pKa',
                data=plot_df,
                order=group_order if len(group_order) > 0 else None,
                palette='viridis'
            )

            # Improve plot aesthetics
            plt.title(f'pKa Distribution by Functional Group (Expected Count = {expected_count}, 1-to-1 Assignment)', fontsize=18)
            plt.xlabel('Functional Group', fontsize=14)
            plt.ylabel('Assigned pKa Value', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=11)
            plt.yticks(fontsize=11)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            # --- Save the plot --- 
            output_filename = os.path.join(output_directory, f'pka_distribution_expected_{expected_count}.png')
            os.makedirs(output_directory, exist_ok=True)
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"  Plot saved successfully to: {output_filename}")

        except Exception as e_plot:
             print(f"  Error generating or saving plot for Expected_pKa_Count = {expected_count}: {e_plot}")
        finally:
             plt.close() # Close the figure explicitly to free memory

# --- Script Execution ---

if __name__ == "__main__":
    # Default paths: Adjust input to the output of the updated extraction script
    default_input = os.path.join('OMGNN', 'data', 'functional_groups_smarts_analysis.csv')
    default_output = os.path.join('OMGNN', 'data', 'plots_by_expected_count') # New output subdir

    # Setup argument parser
    parser = argparse.ArgumentParser(
        description='Plot pKa distribution for functional groups, assigning pKa values 1-to-1 where counts match, generating separate plots per Expected_pKa_Count.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', default=default_input,
                        help=f'Path to the input CSV file (output from extract_functional_groups.py) (default: {default_input})')
    parser.add_argument('--output_dir', default=default_output,
                        help=f'Directory to save the output plots (default: {default_output})')
    args = parser.parse_args()

    # Run the plotting function
    plot_distributions_by_expected_count(args.input, args.output_dir)
