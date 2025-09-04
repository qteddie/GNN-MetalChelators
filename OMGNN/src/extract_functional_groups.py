import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdPartialCharges # Needed for some dependencies, even if not used directly here
from tqdm import tqdm
import argparse
import numpy as np
import logging
import os
import re
from typing import List, Dict, Tuple, Optional, Set, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s - %(message)s')
LOGGER = logging.getLogger("SMARTS_Group_Extractor")

# --- Functional Group Priority --- 
# Lower number indicates higher priority for selection when needed.
GROUP_PRIORITY = {
    # Strong Acids
    'sulfonic_acid_O': 1,
    'phosphonic_acid_O': 2,
    'carboxylic_acid_O': 3,
    'tetrazole_N': 4,        # Acidic N-H
    'sulfonamide_N': 5,      # Acidic N-H
    # Phenols/Thiols/Enols
    'thiophenol_S': 6,
    'phenol_O': 7,
    'enolic_hydroxyl_O': 8,
     # Other N-H Acids
    'triazole_N': 9,         # Acidic N-H
    'pyrrole_indole_N': 10,  # Acidic N-H (Includes indole)
    # Other O/S Acids
    'oxime_O': 11,
    'hydroxylamine_O': 12,
    'thiol_S': 13,           # Non-aromatic thiol
    'alcohol_O': 14,
    # Strong Bases (Amines)
    'tertiary_amine_N': 15,
    'secondary_amine_N': 16,
    'primary_amine_N': 17,
    # Heterocyclic Bases
    'imidazole_pyridine_type_N': 18, # Covers pyridine, imidazole basic N
    # Weakly Acidic/Basic
    'amide_N': 19,           # Can be very weakly acidic N-H or basic N
    # Defaults for any unlisted groups (low priority)
    'DEFAULT': 99
}

# Helper function for sorting based on priority
def get_priority(group_tuple: Tuple[int, str]) -> int:
    """Returns the priority score for a given functional group tuple."""
    group_type = group_tuple[1]
    return GROUP_PRIORITY.get(group_type, GROUP_PRIORITY['DEFAULT'])

# --- SMARTS Patterns for Functional Groups ---
# Maps group names (matching keys in GROUP_PRIORITY) to RDKit Mol objects from SMARTS.
# More specific patterns should ideally be checked first if order mattered significantly,
# but the current logic processes all and then prioritizes/deduplicates.
FUNCTIONAL_GROUP_SMARTS = {
    # Acids (O/S containing)
    'carboxylic_acid_O': '[CX3](=[OX1])[OX2H1]', # Oxygen of COOH (C(O)=O)
    'sulfonic_acid_O': '[S;X4](=[O;X1])(=[O;X1])[O;X2H1]', # Oxygen of SO3H
    'phosphonic_acid_O': '[P;X4](=[O;X1])([O;X2H1])[O;X2H1]', # Oxygens of PO3H2
    'phenol_O': '[OH1][c]', # Phenolic Oxygen
    'alcohol_O': '[OH1][C;!c;!$(C=O)]', # Alcoholic Oxygen (excluding enols explicitly)
    'enolic_hydroxyl_O': '[OH1]C=C', # Enolic Oxygen (explicitly)
    'thiol_S': '[SH]', # Thiol Sulfur
    'thiophenol_S': '[SH][c]', # Thiophenol Sulfur (more specific than thiol)
    'hydroxylamine_O': '[N;!R][OH1]', # Hydroxylamine Oxygen
    'oxime_O': '[N;!R]=[C][OH1]', # Oxime Oxygen
    # N-H Acids / Bases
    'amide_N': '[N;H1,H2][CX3](=O)', # Amide N-H (weakly acidic/basic)
    'sulfonamide_N': '[S;X4](=[O;X1])(=[O;X1])[N;H1,H2]', # Sulfonamide N-H (acidic)
    'pyrrole_indole_N': '[nH;r5,r6]', # Pyrrole/Indole type N-H (acidic)
    'tetrazole_N': '[nH;r5;$(n1nnnc1)]', # Tetrazole N-H (acidic)
    'triazole_N': '[nH;r5;$(n1[c,n]nnc1),$(n1n[c,n]nc1),$(n1nc[c,n]n1)]', # Triazole N-H (acidic)
    # Basic Nitrogens
    'primary_amine_N': '[NH2;!$(NC=O);!$(NS=O)]', # Primary Amine N (excluding amides/sulfonamides)
    'secondary_amine_N': '[NH1;!$(NC=O);!$(NS=O);!$(n)]', # Secondary Amine N (non-amide, non-heteroaromatic N-H)
    'tertiary_amine_N': '[NX3;H0;!$(NC=O);!$(NS=O);!$(N#[C,N]);!$([N+]);!$([n+])]', # Tertiary Amine N (excluding amides, nitriles, N-oxides etc.)
    'imidazole_pyridine_type_N': '[n;+0;D2;r5,r6]', # Imidazole/Pyridine basic Nitrogen (sp2, 2 connections, in 5/6-mem ring)
}

# Pre-compile SMARTS patterns for efficiency
COMPILED_SMARTS_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in FUNCTIONAL_GROUP_SMARTS.items()}

# --- Core Functions ---

def sanitize_smiles(smiles: str) -> Tuple[Optional[Chem.Mol], str]:
    """Attempts to sanitize a SMILES string and return both the Mol object and canonical SMILES."""
    mol = None
    canonical_smiles = smiles # Default to original if parsing fails
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                # Add Hs here after sanitization, crucial for SMARTS matching
                mol = Chem.AddHs(mol)
                canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                return mol, canonical_smiles
            except Exception as e_sanitize:
                LOGGER.debug(f"Sanitization/AddHs failed for '{smiles}': {e_sanitize}. Proceeding with unsanitized mol.")
                # Return the unsanitized mol if sanitization failed but parsing worked
                return mol, smiles # Use original smiles as canonical failed
        else:
            LOGGER.warning(f"RDkit MolFromSmiles returned None for: '{smiles}'")
            return None, smiles
    except Exception as e_parse:
        LOGGER.warning(f"Error parsing SMILES '{smiles}': {e_parse}")
        return None, smiles


def find_smarts_matches(mol: Chem.Mol) -> List[Tuple[int, str]]:
    """Identify functional groups in a molecule using pre-compiled SMARTS patterns."""
    if mol is None:
        return []

    # Ensure properties needed for matching are computed (especially SSSR)
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.GetSymmSSSR(mol)
    except Exception as e:
        # If properties fail, matching might be unreliable, but attempt anyway
        LOGGER.debug(f"Could not update property cache or get SSSR for molecule: {e}")

    matched_atoms: List[Tuple[int, str]] = []
    processed_indices: Set[int] = set()

    for group_name, pattern in COMPILED_SMARTS_PATTERNS.items():
        if pattern is None: # Should not happen with pre-compilation, but check anyway
            LOGGER.warning(f"Pattern for group '{group_name}' is None, skipping.")
            continue
        try:
            matches = mol.GetSubstructMatches(pattern)
            if not matches:
                continue

            LOGGER.debug(f"Pattern '{group_name}' matched indices: {matches}")

            for match_indices in matches:
                # Heuristic to find the key atom (O, N, S with H, or specific N)
                target_atom_idx = -1
                if group_name.endswith('_O'):
                    target_atom_idx = next((idx for idx in match_indices if mol.GetAtomWithIdx(idx).GetSymbol() == 'O'), -1)
                elif group_name.endswith('_N'):
                    target_atom_idx = next((idx for idx in match_indices if mol.GetAtomWithIdx(idx).GetSymbol() == 'N'), -1)
                elif group_name.endswith('_S'):
                    target_atom_idx = next((idx for idx in match_indices if mol.GetAtomWithIdx(idx).GetSymbol() == 'S'), -1)
                else:
                    # Default or specific logic if needed for non-standard names
                    target_atom_idx = match_indices[0] # Fallback, review if new patterns added
                    LOGGER.debug(f"Group '{group_name}' lacks clear O/N/S suffix, using first atom {target_atom_idx}.")

                if target_atom_idx == -1:
                    LOGGER.debug(f"Could not find target atom for group '{group_name}' in match {match_indices}. Skipping.")
                    continue

                # Add the match if the target atom hasn't been assigned a group yet
                if target_atom_idx not in processed_indices:
                    # Basic check: Does the identified atom make sense as acidic/basic?
                    atom = mol.GetAtomWithIdx(target_atom_idx)
                    is_plausible = False
                    atom_symbol = atom.GetSymbol()
                    num_hs = atom.GetTotalNumHs() # Get Hs after AddHs

                    if atom_symbol in ['O', 'S'] and num_hs > 0: # OH, SH
                        is_plausible = True
                    elif atom_symbol == 'N':
                        # Acidic N-H types (amide, sulfonamide, azoles)
                        if num_hs > 0 and any(t in group_name for t in ['amide', 'sulfonamide', 'pyrrole', 'tetrazole', 'triazole']):
                            is_plausible = True
                        # Basic N types (amines, pyridine/imidazole)
                        elif any(t in group_name for t in ['amine', 'imidazole_pyridine_type']):
                            # Tertiary/hetero N don't need H, primary/secondary imply H availability
                            is_plausible = True

                    if is_plausible:
                        LOGGER.debug(f"  -> Adding atom {target_atom_idx} ({atom.GetSymbol()}) as group '{group_name}'")
                        matched_atoms.append((target_atom_idx, group_name))
                        processed_indices.add(target_atom_idx)
                    else:
                         LOGGER.debug(f"  -> Atom {target_atom_idx} ({atom.GetSymbol()}) matched '{group_name}' but deemed not plausible pKa site.")

        except Exception as e:
            LOGGER.warning(f"Error matching SMARTS pattern '{group_name}': {e}")

    # Sort by atom index for consistent output
    matched_atoms.sort(key=lambda x: x[0])

    # Deduplication (e.g., thiophenol_S vs thiol_S for the same atom)
    # Keep only the first match found for each atom index after sorting
    # (Assumes more specific patterns are defined/matched appropriately if needed)
    final_matches = []
    final_indices = set()
    for idx, group in matched_atoms:
        if idx not in final_indices:
            final_matches.append((idx, group))
            final_indices.add(idx)
        else:
             LOGGER.debug(f"Atom index {idx} already assigned, skipping duplicate group '{group}'.")

    return final_matches


def parse_numeric_value(value: Any) -> Optional[int]:
    """Safely converts a value to a positive integer, returning None if invalid."""
    if pd.isna(value):
        return None
    try:
        num = int(value)
        return num if num >= 0 else None
    except (ValueError, TypeError):
        return None

def parse_comma_separated_floats(pka_str: str) -> Optional[List[float]]:
    """Parses a comma-separated string of pKa values into a list of floats."""
    if pd.isna(pka_str) or not isinstance(pka_str, str) or not pka_str.strip():
        return None
    try:
        # Split by comma, strip whitespace, convert to float, filter empty strings after split
        values = [float(val.strip()) for val in pka_str.split(',') if val.strip()]
        return values if values else None # Return None if list is empty after filtering
    except ValueError:
        LOGGER.warning(f"Could not parse pKa string to floats: '{pka_str}'")
        return None
    except Exception as e:
        LOGGER.error(f"Unexpected error parsing pKa string '{pka_str}': {e}")
        return None

# --- Data Processing Function ---

def process_molecule_data(input_path: str, output_path: str):
    """
    Reads input CSV, identifies functional groups using SMARTS patterns,
    selects groups based on priority if count exceeds expected pKa count,
    and writes results (using comma separators for lists) to the output CSV.
    """
    LOGGER.info(f"Starting data processing from: {input_path}")
    try:
        df = pd.read_csv(input_path)
        LOGGER.info(f"Successfully loaded {len(df)} rows.")
    except FileNotFoundError:
        LOGGER.error(f"Input file not found: {input_path}")
        return
    except Exception as e:
        LOGGER.error(f"Error reading input CSV: {e}")
        return

    output_records = []
    processed_count = 0
    smiles_parse_errors = 0
    pka_parse_errors = 0
    processing_errors = 0

    # Get logger verbosity level once
    is_verbose = LOGGER.isEnabledFor(logging.DEBUG)

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing SMILES"):
        smiles = row.get('SMILES')
        pka_num_expected_raw = row.get('pKa_num')
        pka_value_str = row.get('pKa_value')

        status = "OK"
        canonical_smiles_out = smiles # Use original if parsing fails
        all_found_indices_str = ""
        all_found_types_str = ""
        initial_found_count = 0
        prioritized_indices_str = ""
        prioritized_types_str = ""
        selected_group_count = 0

        # --- 1. Validate Input SMILES --- 
        if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
            status = "Missing or Invalid SMILES"
            smiles_parse_errors += 1
            mol = None
        else:
            # --- 2. Parse and Sanitize SMILES --- 
            mol, canonical_smiles_parsed = sanitize_smiles(smiles.strip())
            canonical_smiles_out = canonical_smiles_parsed # Use the canonical form if successful
            if mol is None:
                 status = "SMILES Parse Error"
                 smiles_parse_errors += 1

        # --- 3. Parse Expected pKa Count --- 
        pka_num_expected = parse_numeric_value(pka_num_expected_raw)
        if pka_num_expected is None and not pd.isna(pka_num_expected_raw):
             if status == "OK": status = "Invalid pKa_num Format"
             # Don't increment error count here, proceed with group finding
        elif pd.isna(pka_num_expected_raw):
            if status == "OK": status = "Missing pKa_num"

        # --- 4. Parse Reported pKa Values --- 
        pka_list = parse_comma_separated_floats(pka_value_str)
        pka_values_count = len(pka_list) if pka_list else 0
        if pka_list is None and not pd.isna(pka_value_str) and str(pka_value_str).strip():
            if status == "OK": status = "pKa Value Parse Error"
            pka_parse_errors += 1
        elif pka_list and pka_num_expected is not None and pka_values_count != pka_num_expected:
             if status == "OK": status = f"Inconsistent pKa Info (Expected={pka_num_expected}, FoundValues={pka_values_count})"
        elif pd.isna(pka_value_str) or not str(pka_value_str).strip():
             if status == "OK": status = "OK (pKa Value Missing)"


        # --- 5. Find Functional Groups via SMARTS (if Mol is valid) --- 
        all_found_groups: List[Tuple[int, str]] = []
        if mol is not None:
            try:
                all_found_groups = find_smarts_matches(mol)
                initial_found_count = len(all_found_groups)

                if initial_found_count > 0:
                    # Use COMMA as separator now
                    all_found_indices_str = ",".join(str(idx) for idx, _ in all_found_groups)
                    all_found_types_str = ",".join(gtype for _, gtype in all_found_groups)

            except Exception as e_match:
                LOGGER.error(f"Error finding SMARTS matches for SMILES '{smiles}' at index {index}: {e_match}", exc_info=is_verbose)
                if status == "OK": status = f"Group Finding Error: {type(e_match).__name__}"
                processing_errors += 1
                initial_found_count = -1 # Indicate error

        # --- 6. Select/Prioritize Groups based on Expected Count --- 
        selected_groups = []
        if initial_found_count > 0: # Only prioritize if groups were found
            if pka_num_expected is not None and initial_found_count > pka_num_expected:
                # Sort by priority (lower number first) then by index
                all_found_groups.sort(key=lambda x: (get_priority(x), x[0]))
                selected_groups = all_found_groups[:pka_num_expected]
                if status == "OK": status = f"Selected Top {len(selected_groups)} from {initial_found_count} (Priority)"
            else:
                 # Keep all found groups if count matches, is less than expected, or expected is unknown/invalid
                 selected_groups = all_found_groups
                 if status == "OK" and pka_num_expected is not None and initial_found_count < pka_num_expected:
                      status = f"Count Mismatch (Low) (Found {initial_found_count}, Expected {pka_num_expected})"
                 elif status == "OK" and pka_num_expected is None:
                      # Status remains "Missing pKa_num" or "Invalid pKa_num Format"
                      pass # Report all found groups
                 elif status.startswith("OK"): # Handles OK, OK (pKa Value Missing)
                     pass # Count matches expected or expected was invalid

            selected_group_count = len(selected_groups)
            if selected_group_count > 0:
                 # Use COMMA as separator now
                 prioritized_indices_str = ",".join(str(idx) for idx, _ in selected_groups)
                 prioritized_types_str = ",".join(gtype for _, gtype in selected_groups)

        elif initial_found_count == 0 and pka_num_expected is not None and pka_num_expected > 0:
            # Expected groups but found none
            if status == "OK": status = "Group Prediction Failed (Expected > 0)"
            # Keep selected_group_count = 0

        elif initial_found_count == -1: # Error during group finding
             selected_group_count = -1 # Propagate error indication


        # --- 7. Append Record --- 
        output_records.append({
            'Original_SMILES': smiles, # Keep original input SMILES
            # 'Canonical_SMILES': canonical_smiles_out, # Add canonical SMILES
            'All_Found_Atom_Indices': all_found_indices_str,
            'All_Found_Group_Types': all_found_types_str,
            'Found_Group_Count': initial_found_count if initial_found_count != -1 else 'Error',
            'Expected_pKa_Count': pka_num_expected if pka_num_expected is not None else (pka_num_expected_raw if not pd.isna(pka_num_expected_raw) else ''), # Show parsed or original invalid
            'Prioritized_Atom_Indices': prioritized_indices_str,
            'Prioritized_Group_Types': prioritized_types_str,
            'Selected_Group_Count': selected_group_count if selected_group_count != -1 else 'Error',
            'Reported_pKa_Values': pka_value_str if not pd.isna(pka_value_str) else '',
            'Parsed_pKa_Count': pka_values_count, # Add count of successfully parsed pKa values
            'Status': status
        })
        processed_count += 1

    # --- 8. Save Results --- 
    output_df = pd.DataFrame(output_records)

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
             os.makedirs(output_dir, exist_ok=True)
        output_df.to_csv(output_path, index=False)
        LOGGER.info(f"Processing complete. Results saved to: {output_path}")

        # Summarize outcomes
        LOGGER.info(f"--- Processing Summary ---")
        LOGGER.info(f"Total rows processed: {processed_count}")
        LOGGER.info(f"SMILES Parsing Errors: {smiles_parse_errors}")
        LOGGER.info(f"pKa Value Parsing Errors (for non-empty strings): {pka_parse_errors}")
        LOGGER.info(f"Group Finding/Processing Errors: {processing_errors}")
        status_counts = output_df['Status'].value_counts()
        LOGGER.info(f"Status Breakdown:\n{status_counts.to_string()}")

    except Exception as e:
        LOGGER.error(f"Error writing output CSV to {output_path}: {e}")

# --- Main Execution --- 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract functional groups using SMARTS, select based on priority and expected pKa count.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--input', default='../data/processed_pka_data.csv',
                        help='Path to the input CSV file (must contain SMILES, pKa_num, pKa_value columns).')
    parser.add_argument('--output', default='../data/functional_groups_smarts_analysis.csv', # Updated default output name
                        help='Path to save the output analysis CSV file.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging (DEBUG level).')

    args = parser.parse_args()

    # Set logging level based on verbosity flag
    log_level = logging.DEBUG if args.verbose else logging.INFO
    # Ensure logger level is updated if already configured
    LOGGER.setLevel(log_level)
    # Also update root logger level if needed, or configure handlers specifically
    logging.getLogger().setLevel(log_level)

    # Pre-compile SMARTS patterns on startup
    LOGGER.info("Compiling SMARTS patterns...")
    # Check if any pattern failed to compile
    failed_patterns = [name for name, mol in COMPILED_SMARTS_PATTERNS.items() if mol is None]
    if failed_patterns:
        LOGGER.error(f"FATAL: The following SMARTS patterns failed to compile: {failed_patterns}")
        exit(1)
    LOGGER.info("SMARTS patterns compiled successfully.")

    # Run the main processing function
    process_molecule_data(args.input, args.output) 