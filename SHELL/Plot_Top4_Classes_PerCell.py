# -*- coding: utf-8 -*-
"""
Cell-Specific Top 4 Glycan Classes Spatial Distribution Plotter

This script analyzes SMLM data to identify the top 4 glycan classes for EACH cell individually
(based on density) and generates a single multi-page PDF for all cells in the root directory.
For each cell, it generates a 2x2 panel figure showing the physical coordinates of the EGFR 
receptors associated with its dominant 4 classes.

- Organization: 6 cells per A4 page (2 columns x 3 rows).
- Output: Single consolidated PDF + CSV summary of top classes per cell.
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
import math
import matplotlib.gridspec as gridspec

# = [NEW] Group Keywords for benchmarking
group_1_keyword = "Stimulated"
group_2_keyword = "Non-stimulated"
number_to_plot = 5

# ===========================================================================
# CONFIGURATION
# ===========================================================================
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
})

root_dir = r"/Volumes/Naz TIRF 3/2026-02-16_In Situ T cell Data/CD4+ T Cells/Non-stimulated/FoV1/PAINT/Cell1"
search_folder_name = "90_Custom Centers"
lectin_library = ['WGA', 'SNA', 'PHAL', 'AAL', 'PSA']
anchor_channel = 'EGFR'
glyco_radius = 35
dimer_radius = 36
pixel_scale = 130

# Exact colors for the 4 panels
class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # Blue, Orange, Green, Red

# ===========================================================================
# HELPER FUNCTIONS (Exact GlyCo Logic)
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
        protein_matches = [s for s in t if isinstance(s, str) and s.startswith(f"{protein}_")]
        if len(protein_matches) == 2:
            for s in t:
                if s not in protein_matches:
                    glycan_in_dimer.append(s)

    # STEP 3: MONOMER ASSIGNMENT
    neighbor_master = {}
    distance_indexed_neighbor = {}
    df_of_interest_key = protein
    com_name = f"neighbors_of_{protein}"
    neighbor_master[com_name] = {}
    distance_indexed_neighbor[com_name] = {}
    trees_mono = {key: KDTree((df[['x', 'y']]*pixel_scale).values) for key, df in data_dict.items()}
    assigned_points_mono = {key: {} for key in trees_mono}
    df_of_interest = data_dict[df_of_interest_key]
    for row_index_of_com, column in df_of_interest.iterrows():
        x1, y1 = column['x']*pixel_scale, column['y']*pixel_scale
        for current_family, current_family_members in trees_mono.items():
            indices = current_family_members.query_ball_point([x1, y1], r=glyco_radius)
            filtered_indices = [num for num in indices if protein != current_family or num != row_index_of_com]
            if filtered_indices:
                dist_index_pairs = [(np.linalg.norm(current_family_members.data[idx] - [x1, y1]), idx) for idx in filtered_indices]
                dist_index_pairs.sort(key=lambda x: x[0])
                for distance, idx in dist_index_pairs:
                    if idx not in assigned_points_mono[current_family] or distance < assigned_points_mono[current_family][idx][0]:
                        assigned_points_mono[current_family][idx] = (distance, row_index_of_com)
    
    for neighbor_family, family_member in assigned_points_mono.items():
        for idx, (distance, df_of_interest_idx) in family_member.items():
            neighbor_master[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append(f'{neighbor_family}_{idx}')
            distance_indexed_neighbor[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append((f'{neighbor_family}_{idx}', distance))

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
    
    return {
        'glycan_filtered_monomer_glycosylation': glycan_filtered_monomer_glycosylation
    }

# ===========================================================================
# MAIN EXECUTION
# ===========================================================================
if __name__ == "__main__":
    print(f"Finding data in {root_dir}")
    target_folders = sorted(list(Path(root_dir).rglob(f"**/{search_folder_name}")))
    
    cell_data_store = []
    global_monomer_glyco_norm = {} # For benchmarking

    print(f"\n[1/3] Phase 1: Benchmarking Top {number_to_plot} classes on {group_2_keyword} monomers...")
    for loc_folder in tqdm(target_folders):
        path_str = str(loc_folder)
        # Determine group
        current_group = group_1_keyword if group_1_keyword in path_str else group_2_keyword
        
        try:
            data_dict = {f.stem.split("_")[0]: pd.read_hdf(f, key='locs') for f in loc_folder.glob("*.hdf5")}
            if anchor_channel not in data_dict or len(data_dict) < 2: continue

            # Area Normalization
            area_of_cell = None
            for yml_file in loc_folder.glob("*.yaml"):
                with open(yml_file, 'r') as f:
                    for info in yaml.safe_load_all(f):
                        if isinstance(info, dict):
                            if "Total Picked Area (um^2)" in info or "Area (um^2)" in info:
                                area_of_cell = np.float32(info.get("Total Picked Area (um^2)", info.get("Area (um^2)")))
                                break
            if not area_of_cell: continue

            # Run Glyco
            glyco_results = run_glyco(data_dict, anchor_channel, pixel_scale, dimer_radius, glyco_radius)
            monomer_tuples = glyco_results['glycan_filtered_monomer_glycosylation']

            # Extract coordinates for ALL classes
            df_anchor = data_dict[anchor_channel]
            protein_xy = np.column_stack((df_anchor['x'].values * pixel_scale, df_anchor['y'].values * pixel_scale))
            
            class_to_coords = {}
            for tup in monomer_tuples:
                channels = [element.split("_")[0] for element in tup]
                egfr_elements = [e for e, c in zip(tup, channels) if c == anchor_channel]
                lectins = [c for c in channels if c != anchor_channel]
                if len(egfr_elements) == 1 and len(lectins) > 0:
                    p_idx = int(egfr_elements[0].split('_')[1])
                    combo = "-".join(sorted(lectins))
                    if combo not in class_to_coords: class_to_coords[combo] = []
                    class_to_coords[combo].append(protein_xy[p_idx])

            # Store densities for benchmark IF Non-stimulated
            if current_group == group_2_keyword:
                for cls, coords in class_to_coords.items():
                    dens = len(coords) / area_of_cell
                    if cls not in global_monomer_glyco_norm: global_monomer_glyco_norm[cls] = []
                    global_monomer_glyco_norm[cls].append(dens)

            cell_data_store.append({
                'name': loc_folder.parent.name,
                'group': current_group,
                'class_coords': class_to_coords,
                'area': area_of_cell
            })
        except Exception as e:
            print(f"Error processing {loc_folder}: {e}")

    # Determine Global Top 5 Classes
    benchmark_means = {cls: np.mean(vals) for cls, vals in global_monomer_glyco_norm.items()}
    sorted_benchmark = sorted(benchmark_means.items(), key=lambda x: x[1], reverse=True)
    top_benchmark_classes = [c[0] for c in sorted_benchmark[:min(number_to_plot, len(sorted_benchmark))]]
    
    print(f"\nGlobal Benchmark ({group_2_keyword} Monomers):")
    for i, cls in enumerate(top_benchmark_classes, 1):
        print(f"  {i}. {cls} (mean density: {benchmark_means[cls]:.4f})")

    # CSV Summary Preparation
    per_cell_top_classes = []
    for cell in cell_data_store:
        row = {'Cell': cell['name'], 'Group': cell['group']}
        for i, cls in enumerate(top_benchmark_classes, 1):
            row[f'Class{i}'] = cls
            row[f'Dens{i}'] = len(cell['class_coords'].get(cls, [])) / cell['area']
        per_cell_top_classes.append(row)

    # Save CSV summary
    output_dir = Path(root_dir) / "Top4_PerCell_Results"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(per_cell_top_classes).to_csv(output_dir / "Top4_Classes_Per_Cell.csv", index=False)

    # [3/3] PDF Plotting Logic
    cells_per_page = 6
    total_cells = len(cell_data_store)
    total_pages = math.ceil(total_cells / cells_per_page)

    with PdfPages(pdf_path) as pdf:
        for page_idx in tqdm(range(total_pages), desc="Plotting Pages"):
            fig = plt.figure(figsize=(8.27, 11.69), dpi=300) # A4
            fig.text(0.5, 0.96, f"Spatial Maps - Benchmark Top {number_to_plot} Classes (Non-Stim Monomers)", ha='center', fontsize=12, weight='bold')
            
            outer_grid = gridspec.GridSpec(3, 2, figure=fig, top=0.93, bottom=0.06, left=0.10, right=0.95, wspace=0.3, hspace=0.3)
            
            start_idx = page_idx * cells_per_page
            end_idx = min(start_idx + cells_per_page, total_cells)
            
            for rel_idx, cell in enumerate(cell_data_store[start_idx:end_idx]):
                row, col = rel_idx // 2, rel_idx % 2
                
                # Dynamic Bounds Calculation
                all_pts = []
                for cls in top_benchmark_classes:
                    if cls in cell['class_coords'] and cell['class_coords'][cls]:
                        all_pts.append(np.vstack(cell['class_coords'][cls]) / 1000)
                
                if not all_pts: continue
                pts = np.vstack(all_pts)
                q1x, q3x = np.percentile(pts[:,0], 10), np.percentile(pts[:,0], 90)
                iqrx = max(q3x - q1x, 1)
                valid_x = pts[:,0][(pts[:,0] > q1x - 1.5*iqrx) & (pts[:,0] < q3x + 1.5*iqrx)]
                q1y, q3y = np.percentile(pts[:,1], 10), np.percentile(pts[:,1], 90)
                iqry = max(q3y - q1y, 1)
                valid_y = pts[:,1][(pts[:,1] > q1y - 1.5*iqry) & (pts[:,1] < q3y + 1.5*iqry)]
                
                min_x, min_y = np.min(valid_x), np.min(valid_y)
                max_x_sh, max_y_sh = np.max(valid_x) - min_x, np.max(valid_y) - min_y
                chart_max = max(max_x_sh, max_y_sh)
                chart_max = np.ceil(chart_max / 5.0) * 5 if chart_max > 0 else 10
                
                # 3x2 Grid per Cell Block (for Top 5 classes)
                inner_grid = outer_grid[row, col].subgridspec(3, 2, wspace=0.05, hspace=0.05)
                
                for i, cls_name in enumerate(top_benchmark_classes):
                    ax = fig.add_subplot(inner_grid[i // 2, i % 2])
                    ax.set_aspect('equal')
                    color = plt.cm.tab10(i)
                    
                    if i == 0:
                        ax.text(0.0, 1.25, f"{cell['name']} ({cell['group']})", transform=ax.transAxes, fontsize=9, weight='bold')
                    
                    ax.text(0.05, 0.05, cls_name, transform=ax.transAxes, fontsize=6, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                    
                    if cls_name in cell['class_coords']:
                        c_pts = np.array(cell['class_coords'][cls_name]) / 1000
                        ax.scatter(c_pts[:,0] - min_x, c_pts[:,1] - min_y, color=color, s=2, alpha=0.8, edgecolors='none')
                    
                    ax.set_xlim(0, chart_max)
                    ax.set_ylim(0, chart_max)
                    ax.tick_params(direction='in', length=2, labelsize=6)
                    if i < 4: ax.set_xticklabels([])
                    if i % 2 == 1: ax.set_yticklabels([])
            
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n✓ Analysis Complete! Output at: {output_dir}")
