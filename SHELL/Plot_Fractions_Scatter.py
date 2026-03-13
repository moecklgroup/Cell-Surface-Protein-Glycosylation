import os
import sys
import subprocess
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'seaborn'])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import KDTree
from pathlib import Path
from tqdm import tqdm
from collections import Counter


# --- CONFIGURATION ---
root_dir = r"/Users/nazlicanyurekli/Desktop/2026-03-10_CD4 cell data /Analyzed Data/2026-02-02_CD4+T Cells Segmented-Clustered-Glyco"
search_folder_name = "90_Custom Centers"

group_1_keyword = "Stimulated"
group_2_keyword = "Non_stimulated"

lectin_library = ['WGA', 'SNA', 'PHAL', 'AAL', 'PSA']
anchor_channel = 'EGFR'
glyco_radius = 35
dimer_radius = 36
pixel_scale = 130

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
})

# --- HELPER FUNCTIONS ---
def nested_dict_to_tuple_list(dict_to_convert_to_tuple):
    return { key: [(k, *v) for k, v in sub_dict.items()] for key, sub_dict in dict_to_convert_to_tuple.items()}

def duplicates_removed_tuple_list(input_dict):
    tuple_list = list({tuple(sorted(t)) for value in input_dict.values() for t in value})
    return [tuple(sorted(tup)) for tup in tuple_list]

def flatten_tuple_list(input_dict):
    return [element for tup in input_dict for element in tup]

def find_duplicates_with_counts(input_list):
    element_counts = Counter(input_list)
    return {element: count for element, count in element_counts.items() if count > 1}

def eliminate_duplicates(neighbor_dictionary, duplicate_list):
    for duplicate_item in duplicate_list:
        smallest_value = float('inf')
        smallest_location = None
        for core_point, neighbors_sub_dict in neighbor_dictionary.items():
            for key, index_distance_tuple in neighbors_sub_dict.items():
                for idx, (item,value) in enumerate(index_distance_tuple):
                    if item == duplicate_item and value < smallest_value:
                        smallest_value = value
                        smallest_location = (core_point, key, idx)

        for core_point, neighbors_sub_dict in neighbor_dictionary.items():
            for key, index_distance_tuple in neighbors_sub_dict.items():
                if smallest_location and (core_point, key) == smallest_location[:2]:
                    neighbor_dictionary[core_point][key] = [(item,value) if (idx == smallest_location[2]) else None
                                                                  for idx, (item,value) in enumerate(index_distance_tuple)]
                    neighbor_dictionary[core_point][key] = [tup for tup in neighbor_dictionary[core_point][key] if tup is not None]
                else:
                    neighbor_dictionary[core_point][key] = [(item,value) for item, value in index_distance_tuple if item != duplicate_item]
    return neighbor_dictionary

def remove_distance(data):
    for main_key, sub_dict in data.items():
        for sub_key, tuple_list in sub_dict.items():
            sub_dict[sub_key] = [elem[0] for elem in tuple_list]
    return data

def run_glyco(data_dict, protein, pixel_scale, dimer_radius, glyco_radius, protein_present=True, consider_dimers=True):
    """
    Executes the exact authentic logic from the user's original glyco script.
    """
    library = sorted(data_dict.keys())
    
    # =====================================================================
    # STEP 1: DETECT DIMERS 
    # =====================================================================
    polymer_neighbor_master = {}
    distance_indexed_polymer_neighbor = {}
    assigned_points_dimer = {}
    if protein_present and consider_dimers:
        df_of_interest_key = protein
        com_name = f"neighbors_of_{df_of_interest_key}"
        polymer_neighbor_master[com_name] = {}
        distance_indexed_polymer_neighbor[com_name] = {}
        trees = {key: KDTree((df[['x', 'y']]*pixel_scale).values) for key, df in data_dict.items() if key == df_of_interest_key}
        assigned_points_dimer = {key: {} for key in trees}
        df_of_interest = data_dict[df_of_interest_key]
        for row_index_of_com, column in df_of_interest.iterrows():
            x1, y1 = column['x']*pixel_scale, column['y']*pixel_scale
            for current_family, current_family_members in trees.items():
                indices = current_family_members.query_ball_point([x1, y1], r=dimer_radius)
                filtered_indices = [num for num in indices if df_of_interest_key != current_family or num != row_index_of_com]
                if filtered_indices:
                    dist_index_pairs = [(np.linalg.norm(current_family_members.data[idx] - [x1, y1]), idx) for idx in filtered_indices]
                    dist_index_pairs.sort(key=lambda x: x[0])
                    for distance, idx in dist_index_pairs:
                        if idx not in assigned_points_dimer[current_family] or distance < assigned_points_dimer[current_family][idx][0]:
                            assigned_points_dimer[current_family][idx] = (distance, row_index_of_com)
        for neighbor_family, family_member in assigned_points_dimer.items():
            for idx, (distance, df_of_interest_idx) in family_member.items():
                polymer_neighbor_master[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append(f'{neighbor_family}_{idx}')
                distance_indexed_polymer_neighbor[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append((f'{neighbor_family}_{idx}', distance))

    polymer_tuple_list = nested_dict_to_tuple_list(polymer_neighbor_master)
    polymer_list_without_mirror_duplicates = duplicates_removed_tuple_list(polymer_tuple_list)
    flattened_polymer_list = [element for tup in polymer_list_without_mirror_duplicates for element in tup]
    unique_items_in_flattened_polymer_list = set(flattened_polymer_list)
    dimer_list = [t for t in polymer_list_without_mirror_duplicates if len(t) == 2]
    protein_considered_in_dimer = [element for tup in dimer_list for element in tup]

    # =====================================================================
    # STEP 2: GLYCOSYLATE DIMERS (Assign Glycans to Dimers)
    # =====================================================================
    tree_for_dimer = {key: KDTree((df[['x', 'y']]*pixel_scale).values) for key, df in data_dict.items() if key != protein}
    dimer_glycosylation = []
    for tuple_of_dimers in dimer_list:
        coalesced_neighbors = []
        for element in tuple_of_dimers:
            family, idx = element.split('_')
            idx = int(idx)
            dataframe = data_dict[family]
            x_val = dataframe.iloc[idx]['x'] * pixel_scale
            y_val = dataframe.iloc[idx]['y'] * pixel_scale
            for current_family, current_family_tree in tree_for_dimer.items():
                neighbor_idxs = current_family_tree.query_ball_point([x_val, y_val], r=glyco_radius)
                neighbors = [f"{current_family}_{neighbor_idx}" for neighbor_idx in neighbor_idxs]
                coalesced_neighbors.extend(neighbors)
        coalesced_neighbors = list(set(coalesced_neighbors))
        if coalesced_neighbors:
            dimer_glycosylation.append(tuple_of_dimers + tuple(coalesced_neighbors))

    demoflatten = flatten_tuple_list(dimer_glycosylation)
    demoduplicate = find_duplicates_with_counts(demoflatten)
    seen = set()
    dimer_glycosylation_glycan_unduplicated = [None] * len(dimer_glycosylation)
    for i in sorted(range(len(dimer_glycosylation)), key=lambda i: len(dimer_glycosylation[i])):
        dimer_glycosylation_glycan_unduplicated[i] = tuple(
            x for x in dimer_glycosylation[i]
            if x not in demoduplicate or (x not in seen and not seen.add(x))
        )
    demoflatten = flatten_tuple_list(dimer_glycosylation_glycan_unduplicated)
    demoduplicate = find_duplicates_with_counts(demoflatten)

    # [FIXED] Align original and cleaned lists to prevent zip misalignment in Phase 2 logic (matching SHELL fix)
    dimer_orig_aligned = []
    dimer_cleaned_aligned = []
    for tup in dimer_glycosylation_glycan_unduplicated:
        if tup is None: continue
        cleaned = tuple(sorted(element.split('_')[0] for element in tup))
        # Filter for exactly 2 proteins (dimers)
        if cleaned.count(protein) == 2:
            dimer_orig_aligned.append(tup)
            dimer_cleaned_aligned.append(cleaned)
            
    dimer_glycosylation_glycan_unduplicated = dimer_orig_aligned
    index_removed_dimer_glycosylation = dimer_cleaned_aligned


    # Collect glycans in dimers
    glycan_in_dimer = []
    for t in dimer_glycosylation_glycan_unduplicated:
        protein_matches = []
        for s in t:
            if isinstance(s, str):
                parts = s.split("_")
                if len(parts) == 2 and parts[0] == protein and parts[1].isdigit():
                    protein_matches.append(s)
        if len(protein_matches) == 2:
            for s in t:
                if s not in protein_matches:
                    glycan_in_dimer.append(s)

    # =====================================================================
    # STEP 3: MONOMER ASSIGNMENT (Protein-Glycan proximity)
    # =====================================================================
    neighbor_master = {}
    distance_indexed_neighbor = {}
    for df_key in data_dict:
        if protein_present:
            df_key = protein
        df_of_interest_key = df_key
        com_name = f"neighbors_of_{df_key}"
        neighbor_master[com_name] = {}
        distance_indexed_neighbor[com_name] = {}
        trees_mono = {key: KDTree((df[['x', 'y']]*pixel_scale).values) for key, df in data_dict.items()}
        assigned_points_mono = {key: {} for key in trees_mono}
        df_of_interest = data_dict[df_of_interest_key]
        for row_index_of_com, column in df_of_interest.iterrows():
            x1, y1 = column['x']*pixel_scale, column['y']*pixel_scale
            for current_family, current_family_members in trees_mono.items():
                indices = current_family_members.query_ball_point([x1, y1], r=glyco_radius)
                filtered_indices = [num for num in indices if df_key != current_family or num != row_index_of_com]
                if filtered_indices:
                    dist_index_pairs = [(np.linalg.norm(current_family_members.data[idx] - [x1, y1]), idx) for idx in filtered_indices]
                    dist_index_pairs.sort(key=lambda x: x[0])
                    for distance, idx in dist_index_pairs:
                        if idx not in assigned_points_mono[current_family] or distance < assigned_points_mono[current_family][idx][0]:
                            assigned_points_mono[current_family][idx] = (distance, row_index_of_com)
                            if not protein_present:
                                break
        for neighbor_family, family_member in assigned_points_mono.items():
            for idx, (distance, df_of_interest_idx) in family_member.items():
                neighbor_master[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append(f'{neighbor_family}_{idx}')
                distance_indexed_neighbor[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append((f'{neighbor_family}_{idx}', distance))
        if protein_present:
            break

    converted_dict_indexed = nested_dict_to_tuple_list(neighbor_master)
    list_of_tuples_without_duplicates = duplicates_removed_tuple_list(converted_dict_indexed)
    list_of_all = flatten_tuple_list(list_of_tuples_without_duplicates)
    duplicates = find_duplicates_with_counts(list_of_all)
    dictionary_without_duplicates = eliminate_duplicates(distance_indexed_neighbor, duplicates)
    distance_removed_unduplicated_dictionary = remove_distance(dictionary_without_duplicates)
    final_list_of_tuples = nested_dict_to_tuple_list(distance_removed_unduplicated_dictionary)
    final_pair_wise_duplicates_removed = duplicates_removed_tuple_list(final_list_of_tuples)
    protein_filtered_monomer_glycosylation = [t for t in final_pair_wise_duplicates_removed if not any(x in protein_considered_in_dimer for x in t)]
    glycan_filtered_monomer_glycosylation = [tuple(x for x in t if x not in glycan_in_dimer) for t in protein_filtered_monomer_glycosylation]
    result_list = [tuple(sorted(element.split("_")[0] for element in tup)) for tup in glycan_filtered_monomer_glycosylation]
    result_list[:] = [tuple(sorted(tup)) for tup in result_list]
    if protein_present and consider_dimers:
        result_list = [t for t in result_list if t.count(protein) < 2]

    return {
        'polymer_list': flattened_polymer_list,
        'unique_items_in_flattened_polymer_list': unique_items_in_flattened_polymer_list,
        'dimer_list': dimer_list,
        'protein_considered_in_dimer': protein_considered_in_dimer,
        'dimer_glycosylation_glycan_unduplicated': dimer_glycosylation_glycan_unduplicated,
        'index_removed_dimer_glycosylation': index_removed_dimer_glycosylation,
        'glycan_filtered_monomer_glycosylation': glycan_filtered_monomer_glycosylation,
        'result_list': result_list
    }

# --- MAIN ANALYSIS LOOP ---
results = []
target_folders = list(Path(root_dir).rglob(f"**/{search_folder_name}"))

print(f"Found {len(target_folders)} cells to analyze.")

for loc_folder in tqdm(target_folders, desc="Processing Cells"):
    path_str = str(loc_folder)
    if group_1_keyword in path_str:
        condition = "Stimulated"
    elif group_2_keyword in path_str:
        condition = "Non_stimulated"
    else:
        continue

    data_dict = {f.stem.split("_")[0]: pd.read_hdf(f, key='locs') for f in loc_folder.glob("*.hdf5")}
    if anchor_channel not in data_dict: continue

    df_anchor = data_dict[anchor_channel]
    x_nm, y_nm = df_anchor['x'].values * pixel_scale, df_anchor['y'].values * pixel_scale
    protein_xy = np.column_stack((x_nm, y_nm))

    lectin_data = {l: data_dict[l] for l in lectin_library if l in data_dict}

    # Execute exact authentic original glyco calculation
    glyco_results = run_glyco(data_dict, anchor_channel, pixel_scale, dimer_radius, glyco_radius, protein_present=True, consider_dimers=True)

    flattened_polymer_list = glyco_results['polymer_list']
    dimer_list = glyco_results['dimer_list']
    dimer_glycosylation_glycan_unduplicated = glyco_results['dimer_glycosylation_glycan_unduplicated']
    glycan_filtered_monomer_glycosylation = glyco_results['glycan_filtered_monomer_glycosylation']
    result_list = glyco_results['result_list']
    protein_considered_in_dimer = glyco_results['protein_considered_in_dimer']
    unique_items_in_flattened_polymer_list = glyco_results['unique_items_in_flattened_polymer_list']

    total_egfr = len(protein_xy)
    if total_egfr == 0: continue

    dimer_indices = set()
    for elem in protein_considered_in_dimer:
        parts = elem.split('_')
        if len(parts) == 2 and parts[1].isdigit():
            dimer_indices.add(int(parts[1]))

    monomer_indices = [i for i in range(len(protein_xy)) if i not in (set(int(e.split('_')[1]) for e in flattened_polymer_list if '_' in e and e.split('_')[1].isdigit()))]
    
    total_monomers = len(monomer_indices)
    frac_monomer = total_monomers / total_egfr

    is_glyco_monomer = np.zeros(len(protein_xy), dtype=bool)
    monomer_lectin_assignments = {i: [] for i in monomer_indices}
    for tup in glycan_filtered_monomer_glycosylation:
        pidx = [int(e.split('_')[1]) for e in tup if e.split('_')[0] == anchor_channel]
        if len(pidx) == 1 and pidx[0] in monomer_lectin_assignments:
            monomer_lectin_assignments[pidx[0]] = [e.split('_')[0] for e in tup if e.split('_')[0] != anchor_channel]
            if monomer_lectin_assignments[pidx[0]]:
                is_glyco_monomer[pidx[0]] = True

    total_glyco_monomers = is_glyco_monomer.sum()
    frac_glyco_monomer = total_glyco_monomers / total_monomers if total_monomers > 0 else 0

    total_dimer_proteins = len(dimer_indices)
    frac_dimer = total_dimer_proteins / total_egfr

    is_glyco_dimer = np.zeros(len(protein_xy), dtype=bool)
    for tup in dimer_glycosylation_glycan_unduplicated:
        if tup is None: continue
        pidx = [int(e.split('_')[1]) for e in tup if e.split('_')[0] == anchor_channel]
        if len(pidx) == 2:
            p1, p2 = pidx
            has_glyco = any(e.split('_')[0] != anchor_channel for e in tup)
            if has_glyco:
                if p1 < len(is_glyco_dimer): is_glyco_dimer[p1] = True
                if p2 < len(is_glyco_dimer): is_glyco_dimer[p2] = True

    total_glyco_dimers = is_glyco_dimer.sum()
    frac_glyco_dimer = total_glyco_dimers / total_dimer_proteins if total_dimer_proteins > 0 else 0

    results.append({
        'Cell': loc_folder.parent.name,
        'Condition': condition,
        'Raw_Monomers': total_monomers,
        'Raw_Dimers': total_dimer_proteins,
        'Fraction_Monomer': frac_monomer,
        'Fraction_Dimer': frac_dimer,
        'Fraction_Glyco_Monomer': frac_glyco_monomer,
        'Fraction_Glyco_Dimer': frac_glyco_dimer
    })

    # Track glycan classes per cell
    # Track glycan classes per cell
    cell_classes = []
    
    # Extract monomer classes
    for tup in glycan_filtered_monomer_glycosylation:
        lectins = [e.split('_')[0] for e in tup if e.split('_')[0] != anchor_channel]
        if lectins:
            # EXACT COMPOSITION (No set())
            class_name = "-".join(sorted(lectins))
            cell_classes.append(class_name)
            
    # Extract dimer classes
    for tup in dimer_glycosylation_glycan_unduplicated:
        if tup is None: continue
        lectins = [e.split('_')[0] for e in tup if e.split('_')[0] != anchor_channel]
        if lectins:
            # EXACT COMPOSITION (No set())
            class_name = "-".join(sorted(lectins))
            cell_classes.append(class_name)
    
    class_counts = Counter(cell_classes)
    for cls, count in class_counts.items():
        results[-1][f'Class_{cls}'] = count

df = pd.DataFrame(results)

# Reshape data for seaborn hue mapping
melted_fractions = []
melted_glycosylation = []

for r in results:
    melted_fractions.append({'Condition': r['Condition'], 'Subpopulation': 'Monomers', 'Fraction': r['Fraction_Monomer']})
    melted_fractions.append({'Condition': r['Condition'], 'Subpopulation': 'Dimers', 'Fraction': r['Fraction_Dimer']})

    melted_glycosylation.append({'Condition': r['Condition'], 'Subpopulation': 'Monomers', 'Fraction': r['Fraction_Glyco_Monomer']})
    melted_glycosylation.append({'Condition': r['Condition'], 'Subpopulation': 'Dimers', 'Fraction': r['Fraction_Glyco_Dimer']})

df_frac = pd.DataFrame(melted_fractions)
df_glyco = pd.DataFrame(melted_glycosylation)

# Calculate Mean Raw Counts and Fold Change
mean_counts = df.groupby('Condition')[['Raw_Monomers', 'Raw_Dimers']].mean().reset_index()

# Reshape for seaborn barplot
melted_means = mean_counts.melt(id_vars=['Condition'], value_vars=['Raw_Monomers', 'Raw_Dimers'], 
                                var_name='Subpopulation', value_name='Mean_Count')
melted_means['Subpopulation'] = melted_means['Subpopulation'].map({'Raw_Monomers': 'Monomers', 'Raw_Dimers': 'Dimers'})

# Calculate Fold Change (Stimulated / Non_stimulated)
ns_means = mean_counts[mean_counts['Condition'] == 'Non_stimulated'].iloc[0]
stim_means = mean_counts[mean_counts['Condition'] == 'Stimulated'].iloc[0]

fold_changes = {
    'Monomers': stim_means['Raw_Monomers'] / ns_means['Raw_Monomers'] if ns_means['Raw_Monomers'] > 0 else 0,
    'Dimers': stim_means['Raw_Dimers'] / ns_means['Raw_Dimers'] if ns_means['Raw_Dimers'] > 0 else 0
}
df_fc = pd.DataFrame(list(fold_changes.items()), columns=['Subpopulation', 'Fold_Change'])

# Top 5 Lectin Classes Analysis
class_columns = [col for col in df.columns if col.startswith('Class_')]
df[class_columns] = df[class_columns].fillna(0)

class_sums = df[class_columns].sum().sort_values(ascending=False)
top_5_classes = class_sums.head(5).index.tolist()

df_top5 = df[['Condition'] + top_5_classes].copy()
df_top5_melted = df_top5.melt(id_vars=['Condition'], value_vars=top_5_classes, 
                              var_name='Glycan_Class', value_name='Count')
df_top5_melted['Glycan_Class'] = df_top5_melted['Glycan_Class'].str.replace('Class_', '')

mean_top5 = df_top5_melted.groupby(['Condition', 'Glycan_Class'])['Count'].mean().reset_index()

# --- PLOTTING ---
fig = plt.figure(figsize=(15, 12))
gs = plt.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, :]) # Top 5 classes spans entire bottom row

sns.set_style("white")

order = ["Non_stimulated", "Stimulated"]
palette = {"Monomers": "blue", "Dimers": "red"}

# 1. Monomer/Dimer Fractions
# SETTING JITTER TO FALSE SO DOTS ARE ON THE EXACT SAME Y LINE
sns.stripplot(data=df_frac, x='Condition', y='Fraction', hue='Subpopulation', order=order,
              palette=palette, size=7, ax=ax1, jitter=False, dodge=False, alpha=0.6)
ax1.set_ylim(0, 1.05)
ax1.set_title("Fraction of monomers/dimers", fontsize=12)
ax1.set_ylabel("Fraction", fontsize=12)
ax1.set_xlabel("")
ax1.legend(title="")

# 2. Glycosylation Fractions
# SETTING JITTER TO FALSE SO DOTS ARE ON THE EXACT SAME Y LINE
sns.stripplot(data=df_glyco, x='Condition', y='Fraction', hue='Subpopulation', order=order,
              palette=palette, size=7, ax=ax2, jitter=False, dodge=False, alpha=0.6)
ax2.set_ylim(0, 1.05)
ax2.set_title("Fraction of glycosylation", fontsize=12)
ax2.set_ylabel("Fraction", fontsize=12)
ax2.set_xlabel("")
ax2.legend(title="")

# 3. Mean Raw Counts Bar Chart
sns.barplot(data=melted_means, x='Condition', y='Mean_Count', hue='Subpopulation', order=order,
            palette=palette, ax=ax3, edgecolor='black', linewidth=1.5, alpha=0.8)
ax3.set_title("Mean Raw Counts", fontsize=12)
ax3.set_ylabel("Mean Count", fontsize=12)
ax3.set_xlabel("")
ax3.legend(title="")

# 4. Fold Change Bar Chart
sns.barplot(data=df_fc, x='Subpopulation', y='Fold_Change', palette=palette, ax=ax4, edgecolor='black', linewidth=1.5, alpha=0.8)
ax4.axhline(1, color='black', linestyle='--', linewidth=1) # Reference line at Fold Change = 1
ax4.set_title("Fold Change (Stimulated / Non_stimulated)", fontsize=12)
ax4.set_ylabel("Fold Change", fontsize=12)
ax4.set_xlabel("")

# 5. Top 5 Lectin Classes Bar Chart
sns.barplot(data=mean_top5, x='Glycan_Class', y='Count', hue='Condition', hue_order=order,
            ax=ax5, edgecolor='black', linewidth=1.5, alpha=0.8)
ax5.set_title("Top 5 Lectin Classes (Mean Counts)", fontsize=14)
ax5.set_ylabel("Mean Count", fontsize=12)
ax5.set_xlabel("Lectin Class", fontsize=12)
ax5.tick_params(axis='x', rotation=45)
ax5.legend(title="Condition")

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.tick_params(axis='both', labelsize=11, direction='out', length=4, width=1.0, color='black')
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)

plt.tight_layout()
output_path_pdf = root_dir + "/Fraction_Scatterplots.pdf"
output_path_png = root_dir + "/Fraction_Scatterplots.png"
output_path_csv = root_dir + "/Fraction_Scatterplots_Data.csv"

plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
df.to_csv(output_path_csv, index=False)

print(f"Successfully saved figures to PDF and PNG, and data to CSV: {output_path_csv}")
