# -*- coding: utf-8 -*-
"""
Top 4 Glycan Classes Spatial Distribution Plotter

This script analyzes SMLM data to identify the top 4 glycan classes globally (based off the
Non-Stimulated group benchmark) and generates a multi-page PDF. For each cell, it generates
a 2x2 panel figure showing the physical coordinates of the EGFR receptors associated with those 4 classes.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.spatial import KDTree
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import yaml
from matplotlib.backends.backend_pdf import PdfPages

# ===========================================================================
# CONFIGURATION
# ===========================================================================
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
})

root_dir = r"/Users/nazlicanyurekli/Desktop/2026-03-10_CD4 cell data /Analyzed Data/2026-02-02_CD4+T Cells Segmented-Clustered-Glyco"
search_folder_name = "90_Custom Centers"
lectin_library = ['WGA', 'SNA', 'PHAL', 'AAL', 'PSA']
anchor_channel = 'EGFR'
glyco_radius = 35
dimer_radius = 36
pixel_scale = 130

group_1_keyword = "Stimulated"
group_2_keyword = "Non_stimulated"
color_map = {group_1_keyword: 'darkorange', group_2_keyword: '#1f77b4'}  # Orange and Blue

# ===========================================================================
# HELPER FUNCTIONS (Exact GlyCo Duplication)
# ===========================================================================
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
                for idx, (item, value) in enumerate(index_distance_tuple):
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
    Exact authentic dimer-priority glyco logic from Protein_Glyco_dimer_priority-2.py.
    """
    # STEP 1: DETECT DIMERS
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

    # STEP 2: GLYCOSYLATE DIMERS
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

    # STEP 3: MONOMER ASSIGNMENT
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

# ===========================================================================
# MAIN EXECUTION
# ===========================================================================
if __name__ == "__main__":
    print(f"Finding data in {root_dir}")
    target_folders = list(Path(root_dir).rglob(f"**/{search_folder_name}"))
    
    # Storage for finding Top 4 globally
    monomer_glyco_counts = {group_1_keyword: {}, group_2_keyword: {}}
    
    # Store coordinates for plotting later
    cell_data_store = []

    print("\n[1/3] Processing cells to extract exact EGFR coordinates & determine Top 4 Classes...")
    for loc_folder in tqdm(target_folders):
        path_str = str(loc_folder)
        group = group_1_keyword if group_1_keyword in path_str else group_2_keyword if group_2_keyword in path_str else None
        if not group: continue

        data_dict = {f.stem.split("_")[0]: pd.read_hdf(f, key='locs') for f in loc_folder.glob("*.hdf5")}
        if anchor_channel not in data_dict: continue

        # Normalization Area
        area_of_cell = None
        for yml_file in loc_folder.glob("*.yaml"):
            with open(yml_file, 'r') as f:
                for info in yaml.safe_load_all(f):
                    if isinstance(info, dict):
                        if "Total Picked Area (um^2)" in info:
                            area_of_cell = np.float32(info["Total Picked Area (um^2)"])
                        elif "Area (um^2)" in info:
                            area_of_cell = np.float32(info["Area (um^2)"])
        if not area_of_cell: continue

        df_anchor = data_dict[anchor_channel]
        protein_xy = np.column_stack((df_anchor['x'].values * pixel_scale, df_anchor['y'].values * pixel_scale))
        lectin_dict = {l: data_dict[l] for l in lectin_library if l in data_dict}

        # Run the single canonical dimer-priority glyco function
        glyco_results = run_glyco(data_dict, anchor_channel, pixel_scale, dimer_radius, glyco_radius,
                                  protein_present=True, consider_dimers=True)

        glycan_filtered_monomer_glycosylation = glyco_results['glycan_filtered_monomer_glycosylation']

        # Build class_to_coords from monomer glycosylation tuples
        df_anchor = data_dict[anchor_channel]
        protein_xy = np.column_stack((df_anchor['x'].values * pixel_scale, df_anchor['y'].values * pixel_scale))

        class_to_coords = {}
        for tup in glycan_filtered_monomer_glycosylation:
            channels = [element.split("_")[0] for element in tup]
            egfr_elements = [e for e, c in zip(tup, channels) if c == anchor_channel]
            lectins = [c for c in channels if c != anchor_channel]
            if len(egfr_elements) == 1 and len(lectins) > 0:
                p_idx = int(egfr_elements[0].split('_')[1])
                combo = "-".join(sorted(lectins))
                if combo not in class_to_coords:
                    class_to_coords[combo] = []
                class_to_coords[combo].append(protein_xy[p_idx])

        # Accumulate metrics for Top 4 globally
        for combo, coords in class_to_coords.items():
            norm_count = len(coords) / area_of_cell
            if combo not in monomer_glyco_counts[group]:
                monomer_glyco_counts[group][combo] = []
            monomer_glyco_counts[group][combo].append(norm_count)
            
        cell_data_store.append({
            'folder': loc_folder,
            'group': group,
            'name': loc_folder.parent.name,
            'class_coords': class_to_coords
        })

    # =======================================================================
    # FIND TOP 4 CLASSES (Benchmark: Non-Stimulated Group)
    # =======================================================================
    # Calculate global average density for each class across all Non-Stimulated cells
    ns_dict = monomer_glyco_counts.get(group_2_keyword, {})
    all_ns_averages = {cls: np.mean(vals) for cls, vals in ns_dict.items() if len(vals) > 0}
    
    # Sort classes by their mean normalized density (norm_count) descending
    sorted_by_density = sorted(all_ns_averages.items(), key=lambda x: x[1], reverse=True)
    
    # Select the Top 4
    top_4_classes = [item[0] for item in sorted_by_density[:4]]
    
    print(f"\n[2/3] Top 4 Classes determined dynamically from {group_2_keyword} benchmark:")
    if not top_4_classes:
        print("  WARNING: No glycan classes found in Non-Stimulated group! Using defaults.")
        top_4_classes = ['AAL','SNA','AAL-SNA','PSA']
    
    for i, cls in enumerate(top_4_classes, 1):
        avg_dens = all_ns_averages.get(cls, 0)
        print(f"  {i}. {cls} (Avg Density: {avg_dens:.4f})")

    # =======================================================================
    # GENERATE PDF PLOTS
    # =======================================================================
    out_dir = Path(root_dir) / "Top4_Spatial_Maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Exact colors from the image provided
    # The reference image has: 
    # Top-Left: Blue, Top-Right: Orange, Bottom-Left: Green, Bottom-Right: Red
    class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    
    for target_group in [group_1_keyword, group_2_keyword]:
        group_cells = [c for c in cell_data_store if c['group'] == target_group]
        
        if not group_cells:
            continue
            
        pdf_path = out_dir / f"Top4_Spatial_Maps_{target_group}.pdf"
        print(f"\n[3/3] Generating PDF: {pdf_path}")
        
        import math
        import re
        import matplotlib.gridspec as gridspec
        
        with PdfPages(pdf_path) as pdf:
            cells_per_page = 6
            cols_per_page = 2
            rows_per_page = 3
            
            # Sort cells strictly numerically by extracting the number from the name (e.g., 'Cell10' -> 10)
            def extract_cell_num(cell):
                match = re.search(r'\d+', cell['name'])
                return int(match.group()) if match else 999999
            
            group_cells = sorted(group_cells, key=extract_cell_num)
            
            total_cells = len(group_cells)
            total_pages = math.ceil(total_cells / cells_per_page)
            
            for page_idx in tqdm(range(total_pages), desc=f"Plotting {target_group} Pages"):
                # A4 size is 8.27 x 11.69 inches
                fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
                
                # Global Legend at the top (Only mentioned once!)
                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=class_colors[k], markersize=6) for k in range(4)]
                fig.legend(handles, top_4_classes, loc='upper center', bbox_to_anchor=(0.5, 0.99), frameon=False, 
                           fontsize=10, ncol=4, handletextpad=0.1, labelspacing=0.6, prop={'family':'Arial', 'size':10})
                
                # Global Title - Arial 10 regular, no bold
                fig.text(0.5, 0.96, f"{target_group} cells", ha='center', fontsize=10, family='Arial')
                
                # Outer grid: wspace=0.5 gives room for each cell's own y-axis without clash
                outer_grid = gridspec.GridSpec(rows_per_page, cols_per_page, figure=fig, top=0.93, bottom=0.06, left=0.12, right=0.95, wspace=0.5, hspace=0.45)
                
                start_idx = page_idx * cells_per_page
                end_idx = min(start_idx + cells_per_page, total_cells)
                
                for relative_idx, cell_data in enumerate(group_cells[start_idx:end_idx]):
                    name = cell_data['name']
                    classes_coords = cell_data['class_coords']
                    
                    row_idx = relative_idx // cols_per_page
                    col_idx = relative_idx % cols_per_page
                    
                    # Find EXACT bounds of THIS specific cell
                    all_x, all_y = [], []
                    for coords in classes_coords.values():
                        if len(coords) > 0:
                            arr = np.array(coords)
                            all_x.extend(arr[:,0])
                            all_y.extend(arr[:,1])
                            
                    if not all_x: continue
                    
                    all_x_um = np.array(all_x) / 1000
                    all_y_um = np.array(all_y) / 1000
                    
                    q1_x, q3_x = np.percentile(all_x_um, 10), np.percentile(all_x_um, 90)
                    iqr_x = q3_x - q1_x
                    valid_x = all_x_um[(all_x_um > q1_x - 1.5*iqr_x) & (all_x_um < q3_x + 1.5*iqr_x)]
                    
                    q1_y, q3_y = np.percentile(all_y_um, 10), np.percentile(all_y_um, 90)
                    iqr_y = q3_y - q1_y
                    valid_y = all_y_um[(all_y_um > q1_y - 1.5*iqr_y) & (all_y_um < q3_y + 1.5*iqr_y)]
                    
                    min_x_um = np.min(valid_x) if len(valid_x) > 0 else np.min(all_x_um)
                    min_y_um = np.min(valid_y) if len(valid_y) > 0 else np.min(all_y_um)
                    
                    max_x_shifted = np.max(valid_x) - min_x_um if len(valid_x) > 0 else np.max(all_x_um) - min_x_um
                    max_y_shifted = np.max(valid_y) - min_y_um if len(valid_y) > 0 else np.max(all_y_um) - min_y_um
                    
                    # Determine a single max dimension to ensure perfect square boxes for all panels
                    raw_max = max(max_x_shifted, max_y_shifted)
                    
                    if raw_max > 40: 
                        chart_max = np.ceil(raw_max / 10.0) * 10
                    elif raw_max > 10: 
                        chart_max = np.ceil(raw_max / 5.0) * 5
                    elif raw_max > 5: 
                        chart_max = np.ceil(raw_max / 2.0) * 2
                    else: 
                        chart_max = np.ceil(raw_max)
                    
                    x_chart_max = y_chart_max = chart_max # Keep squares the same size
                        
                    # Inner 2x2 panels per cell - small gap
                    inner_grid = outer_grid[row_idx, col_idx].subgridspec(2, 2, wspace=0.1, hspace=0.1)
                    
                    for i, cls in enumerate(top_4_classes):
                        ax = fig.add_subplot(inner_grid[i // 2, i % 2])
                        ax.set_aspect('equal') # CRITICAL: Prevent squeezing/stretching
                        color = class_colors[i]
                        
                        # Cell name - Arial 10 regular, no bold
                        if i == 0:
                            ax.text(0.0, 1.15, f"{name}", transform=ax.transAxes, fontsize=10, family='Arial', ha='left', va='bottom')
                        
                        if cls in classes_coords and len(classes_coords[cls]) > 0:
                            x_coords = np.array(classes_coords[cls])[:,0] / 1000
                            y_coords = np.array(classes_coords[cls])[:,1] / 1000
                            x_shifted = x_coords - min_x_um
                            y_shifted = y_coords - min_y_um
                            ax.scatter(x_shifted, y_shifted, color=color, s=7.5, alpha=1.0, linewidths=0)
                        
                        ax.set_xlim(0, x_chart_max)
                        ax.set_ylim(0, y_chart_max)
                        
                        global_chart_max = max(x_chart_max, y_chart_max)
                        if global_chart_max > 20: ticker = MultipleLocator(10)
                        elif global_chart_max > 10: ticker = MultipleLocator(5)
                        else: ticker = MultipleLocator(2)
                            
                        ax.xaxis.set_major_locator(ticker)
                        ax.yaxis.set_major_locator(ticker)
                        
                        ax.tick_params(direction='out', length=4, width=1.0, color='black')
                        
                        if i in [0, 1]: # Top row
                            ax.set_xticklabels([])
                            ax.tick_params(axis='x', which='both', bottom=False, top=False)
                        if i in [1, 3]: # Right column
                            ax.set_yticklabels([])
                            ax.tick_params(axis='y', which='both', left=False, right=False)
                        
                        if i in [2, 3]: # Bottom row
                            ax.tick_params(axis='x', labelsize=10)
                            ax.set_xlabel("x (µm)", fontsize=10, family='Arial')
                        if i in [0, 2]: # Left column
                            ax.tick_params(axis='y', labelsize=10)
                            ax.set_ylabel("y (µm)", fontsize=10, family='Arial')
                        
                        xticks_locs = ax.get_xticks()
                        xticks_locs = [x for x in xticks_locs if 0 < x < x_chart_max]
                        if len(xticks_locs) == 0:
                             xticks_locs = [x_chart_max / 2]
                        ax.set_xticks(xticks_locs)
                        if i in [2, 3]:
                            ax.set_xticklabels([str(int(x)) if x.is_integer() else str(x) for x in xticks_locs])
                        
                        yticks_locs = ax.get_yticks()
                        yticks_locs = [y for y in yticks_locs if 0 < y < y_chart_max]
                        if len(yticks_locs) == 0:
                             yticks_locs = [y_chart_max / 2]
                        ax.set_yticks(yticks_locs)
                        if i in [0, 2]:
                            ax.set_yticklabels([str(int(y)) if y.is_integer() else str(y) for y in yticks_locs])
                        
                        # Remove gray grid lines if any, make ONE black outline around the axis
                        ax.grid(False)
                        for spine in ax.spines.values():
                            spine.set_linewidth(1.0)
                            spine.set_color('black')
                            
                pdf.savefig(fig)
                plt.close(fig)

    print("\n✓ Top 4 Spatial Generation Complete!")

