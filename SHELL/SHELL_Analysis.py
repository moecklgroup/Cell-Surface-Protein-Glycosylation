# === 1. IMPORTS & DEPENDENCY CHECK ===


# In[ ]:


# ===========================================================================

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 2026
Last updated: 2026-02-19
Shell-based Hierarchical Evaluation of Localizations in Layers (SHELL) Analysis
Author: Nyurekli

This code is intended to quantify the spatial distribution of glycan classes on the cell surface 
using single-molecule localization microscopy (SMLM) data,on cluster centers of segmented cells. Shell includes GlyCO analysis pipeline to define 
only glycans co-clustered with the target protein (EGFR) (i.e., lectin localizations assigned to a target protein 
(EGFR) within a 35 nm search radius) are included in the analysis — glycans not associated with the target protein
(EGFR) are excluded. The pipeline computes radial shell density profiles from the cell edge inward
to the cell center, separately for monomeric and dimeric target protein (EGFR) populations, and 
compares two experimental groups (such as Stimulated vs. Non-Stimulated CD4+ T cells).

WORKFLOW:
=========
STEP 1: Dimer/Monomer separation
        -> Dimers = mutual closest EGFR pairs within dimer_radius
        -> Monomers = Total EGFR centers - Dimers - Polymers (>2) (proteins NOT in any structure)
        -> Glycolysation is assigned to dimers first, and duplicate glycans are removed.
        -> Excluded lists of "proteins in a dimer" and "glycans assigned to a dimer" are built.

STEP 2: Monomer Glycosylation
        -> Each remaining lectin point can only be assigned to ONE remaining protein (closest)
        -> This applies to monomers, strictly filtering out any proteins/glycans already claimed by a dimer.

STEP 3: Top 5 defined by NON-STIMULATED MONOMERS (boolean filters)
        -> Uses glyco assignments from STEP 2 (after duplicate removal)
        -> Top 5 = highest mean occurrence glyco-classes from NON-STIM monomers
        -> SAME Top 5 applied to BOTH monomers AND dimers for radial analysis

STEP 4: Radial analysis for BOTH Monomers AND Dimers
        -> Total, Glycosylated, Non-Glycosylated
        -> Top 5 classes (same classes for monomer and dimer)

STEP 5: Mean normalized (0-1) radial plots
        -> STIM vs NON-STIM comparison
        -> For all monomer categories AND all dimer categories

IMPORTANT DISTINCTION:
- MASK: Cell border + erosion contours from ALL channels combined -> CONSTANT for each cell
- SELECTION: Boolean arrays to filter which proteins to count 

@author:Nyurekli
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import sem
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator, NullLocator
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from scipy.spatial import KDTree, Delaunay, ConvexHull
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt
from skimage import measure, draw, morphology
from datetime import datetime
from pathlib import Path
import math
from tqdm import tqdm
from collections import Counter, OrderedDict
import yaml
import json
from colorama import init, Style, Back
init(autoreset=True)

# Dependency Check
try:
    from shapely.geometry import Polygon, Point
except ImportError:
    print(" ERROR: 'shapely' not found. Please run: pip install shapely")
    raise

print("\n✓ Imports OK — all dependencies available")

# ----------------------------------------



# In[ ]:


# ===========================================================================

print("✓ All imports loaded successfully")


# === 2. HELPER FUNCTIONS ===


# In[ ]:


# ===========================================================================

# --- HELPER FUNCTIONS ---

def detect_cell_border(x, y, alpha=None):
    """
    Detect cell border by converting points to a binary image and finding contours.
    Uses a 'Union of Balls' approach (Dilation) to ensure all points are included.
    Arguments:
        x, y: coordinates in nm
        alpha: Unused (kept for compatibility)
    """
    # 1. Define Image Resolution (Pixel Size in nm)
    pixel_size = 130.0

    if len(x) < 3: return np.column_stack([x, y])

    # 2. Shift coordinates with MORE padding to prevent clipping
    min_x, min_y = np.min(x), np.min(y)
    pad = 20 
    width = int((np.max(x) - min_x) / pixel_size) + 2 * pad
    height = int((np.max(y) - min_y) / pixel_size) + 2 * pad

    mask = np.zeros((height, width), dtype=bool)

    # Convert to indices
    ix = ((x - min_x) / pixel_size).astype(int) + pad
    iy = ((y - min_y) / pixel_size).astype(int) + pad

    # 3. Set pixels
    mask[iy, ix] = True

    # 4. Dilate to create "balls" around every point
    # Radius 2 pixels = ~260nm. Tight fit but ensures connectivity for nearby detached points.
    selem = morphology.disk(2)
    mask = morphology.binary_dilation(mask, selem)

    # 5. Fill Holes (optional but good for solid cell body)
    # mask = morphology.binary_fill_holes(mask) 
    mask = morphology.binary_closing(mask, morphology.disk(2))

    # 6. Find Contours
    contours = measure.find_contours(mask, 0.5)

    if not contours:
        return np.column_stack([x, y])

    largest_contour = max(contours, key=len)

    # 7. Convert back
    contour_y = (largest_contour[:, 0] - pad) * pixel_size + min_y
    contour_x = (largest_contour[:, 1] - pad) * pixel_size + min_x

    return np.column_stack([contour_x, contour_y])

def create_binary_mask(border_polygon, resolution=10.0):
    """Create binary mask from border polygon for EDT calculation."""
    min_x, min_y = np.min(border_polygon, axis=0) - 100
    max_x, max_y = np.max(border_polygon, axis=0) + 100
    width = int((max_x - min_x) / resolution)
    height = int((max_y - min_y) / resolution)
    mask = np.zeros((height, width), dtype=np.uint8)
    rr, cc = draw.polygon((border_polygon[:, 1] - min_y) / resolution,
                          (border_polygon[:, 0] - min_x) / resolution, shape=mask.shape)
    mask[rr, cc] = 1
    return mask, min_x, min_y, resolution

def find_center_by_erosion(border_polygon, bin_size_nm=100, resolution=10.0):
    """
    Find cell center using EDT and generate erosion contours for radial analysis.
    These contours define the CONSTANT MASK for radial shell analysis.
    """
    mask, x_offset, y_offset, res = create_binary_mask(border_polygon, resolution=resolution)
    dt = distance_transform_edt(mask)
    max_idx = np.unravel_index(np.argmax(dt), dt.shape)
    center_x = max_idx[1] * res + x_offset
    center_y = max_idx[0] * res + y_offset

    max_dist = dt.max()
    levels = np.arange(res, max_dist, bin_size_nm / res)

    erosion_contours = []
    areas = []
    for level in levels:
        contours = measure.find_contours(dt >= level, 0.5)
        if contours:
            largest = max(contours, key=len)
            c_real = np.zeros_like(largest)
            c_real[:, 0] = largest[:, 1] * res + x_offset
            c_real[:, 1] = largest[:, 0] * res + y_offset
            erosion_contours.append(c_real)
            x, y = c_real[:, 0], c_real[:, 1]
            areas.append(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    return center_x, center_y, erosion_contours, areas

# ---GLYCO DUPLICATE REMOVAL FUNCTIONS ---

def print_red(text):
    print(f"{Back.RED}{text}{Style.RESET_ALL}")
    
    
def nested_dict_to_tuple_list(dict_to_convert_to_tuple):
    "convert sub dictionary in a nested dictionary to a list of tuples"
    converted_dict = { key: [(k, *v) for k, v in sub_dict.items()] for key, sub_dict in dict_to_convert_to_tuple.items()}
    return converted_dict

def duplicates_removed_tuple_list(input_dict):
    "Make into a single list of tuples without duplicates ie. (PSA_123, WGA_345) = (WGA_345, PSA_123). No more same combination after this point."
    "alphabetic ordering is done here"
    tuple_list=list({tuple(sorted(t)) for value in input_dict.values() for t in value})
    tuple_list[:] = [tuple(sorted(tup)) for tup in tuple_list]

    return tuple_list

def flatten_tuple_list(input_dict):
    "to flatten the list of tuples"
    flat_list= [element for tup in input_dict for element in tup]
    return flat_list

def find_duplicates_with_counts(input_list):
    "find duplicates from the list"
    element_counts = Counter(input_list)
    duplicates = {element: count for element, count in element_counts.items() if count > 1}    
    print_red(f"\n {len(duplicates)} duplicates found")
    return duplicates

def eliminate_duplicates(neighbor_dictionary, duplicate_list):
    "For the entire dictionary there should not be an element appearing twice. This function removes the duplicates by assigning the"
    "closest ones together, by acting on the dictionary indexed with distances"
    for duplicate_item in tqdm(duplicate_list, desc="Removing duplicates"): #iterate through each duplicated item
        smallest_value = float('inf') #reset the smallest value for each item
        #search for the smallest value in the nested dictionary
        for core_point, neighbors_sub_dict in neighbor_dictionary.items(): #diving inside the dictionary looking for the current duplicate element
            for key, index_distance_tuple in neighbors_sub_dict.items(): #diving inside the subdictionary looking for the current duplicate element
                for idx, (item,value) in enumerate(index_distance_tuple): #fetch items from the list of tuples by indexing them.
                    if item == duplicate_item and value<smallest_value:
                        smallest_value = value
                        smallest_location = (core_point, key, idx)
        
        for core_point, neighbors_sub_dict in neighbor_dictionary.items():
            for key, index_distance_tuple in neighbors_sub_dict.items():
                if (core_point, key) == smallest_location[:2]: #for the smallest location
                    neighbor_dictionary[core_point][key] = [(item,value) if (idx == smallest_location[2]) else None
                                                                  for idx, (item,value) in enumerate (index_distance_tuple)] #check the index in the list
                    
                    neighbor_dictionary[core_point][key] = [ tup for tup in neighbor_dictionary[core_point][key] if tup is not None] #Eliminating None valued keys
                else: #until the smallest location combiation of core and key is found this part ofthe loop is executed. Here the dictionary is updated if the item in the tuple pair is not the duplicate item
                    neighbor_dictionary[core_point][key] = [(item,value) for item, value in index_distance_tuple if item != duplicate_item]
    return neighbor_dictionary

def remove_distance(data):
    "function to remove distance values from the dictionary."
    # Iterate through each key-value pair in the outer dictionary
    for main_key, sub_dict in data.items():
        
        # Iterate through each key-value pair in the sub-dictionary
        for sub_key, tuple_list in sub_dict.items():
            sub_dict[sub_key] = [elem[0] for elem in tuple_list]
    
    return data

def run_glyco(data_dict, protein, pixel_scale, dimer_radius, glyco_radius, protein_present=True, consider_dimers=True):
    """
 avoid code duplication in Phase 1 and Phase 2.
    """
    library = sorted(data_dict.keys())
    
    # =====================================================================
    # STEP 1: DETECT DIMERS (Protein-Protein proximity within dimer_radius)
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
    # process tuples from smallest to largest
    for i in sorted(range(len(dimer_glycosylation)), key=lambda i: len(dimer_glycosylation[i])):
        dimer_glycosylation_glycan_unduplicated[i] = tuple(
            x for x in dimer_glycosylation[i]
            if x not in demoduplicate or (x not in seen and not seen.add(x))
        )

    demoflatten = flatten_tuple_list(dimer_glycosylation_glycan_unduplicated)
    demoduplicate = find_duplicates_with_counts(demoflatten)

    # Align original and cleaned lists to prevent zip misalignment in Phase 2
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



def assign_lectins_to_proteins(protein_coords, lectin_dict, radius, pixel_scale,
                               proteins_in_dimer=None, glycans_in_dimer=None):

    protein = anchor_channel

    # Build data_dict 
    data_dict = {protein: pd.DataFrame({'x': protein_coords[:, 0],
                                        'y': protein_coords[:, 1]})} 
    for name, df in lectin_dict.items():
        data_dict[name] = df

    # === GlyCo ===
    neighbor_master = {}
    distance_indexed_neighbor = {}

    # When protein_present=True, only search neighbors OF protein
    df_of_interest_key = protein
    com_name = f"neighbors_of_{df_of_interest_key}"
    neighbor_master[com_name] = {}
    distance_indexed_neighbor[com_name] = {}

    # Create KDTree for ALL channels
    trees = {}
    for key, df in data_dict.items():
        if key == protein:
            trees[key] = KDTree(df[['x', 'y']].values)         
        else:
            trees[key] = KDTree((df[['x', 'y']] * pixel_scale).values)  

    # Track assigned points
    assigned_points = {key: {} for key in trees}

    df_of_interest = data_dict[df_of_interest_key]

    # Iterate through points in protein channel
    for row_index_of_com, column in tqdm(df_of_interest.iterrows(), desc=f"GlyCo: Searching neighbors of {df_of_interest_key}", leave=False, total=len(df_of_interest)):
        x1 = column['x']  
        y1 = column['y']

        # Check neighbors in ALL channels
        for current_family, current_family_members in trees.items():
            indices = current_family_members.query_ball_point([x1, y1], r=radius)
            filtered_indices = [num for num in indices if df_of_interest_key != current_family or num != row_index_of_com]

            if filtered_indices:
                dist_index_pairs = [(np.linalg.norm(current_family_members.data[idx] - [x1, y1]), idx)
                                    for idx in filtered_indices]
                dist_index_pairs.sort(key=lambda x: x[0])

                for distance, idx in dist_index_pairs:
                    if idx not in assigned_points[current_family] or distance < assigned_points[current_family][idx][0]:
                        assigned_points[current_family][idx] = (distance, row_index_of_com)
                    

    # Populate neighbor_master
    for neighbor_family, family_member in assigned_points.items():
        for idx, (distance, df_of_interest_idx) in family_member.items():
            neighbor_master[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append(f'{neighbor_family}_{idx}')
            distance_indexed_neighbor[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append((f'{neighbor_family}_{idx}', distance))

    # Convert to tuple list
    converted_dict_indexed = nested_dict_to_tuple_list(neighbor_master)
    list_of_tuples_without_duplicates = duplicates_removed_tuple_list(converted_dict_indexed)

    # Find duplicates
    list_of_all = flatten_tuple_list(list_of_tuples_without_duplicates)
    duplicates = find_duplicates_with_counts(list_of_all)

    # Eliminate duplicates using distance
    dictionary_without_duplicates = eliminate_duplicates(distance_indexed_neighbor, duplicates)

    # Remove distance
    distance_removed_unduplicated_dictionary = remove_distance(dictionary_without_duplicates)

    # Final filtering
    final_list_of_tuples = nested_dict_to_tuple_list(distance_removed_unduplicated_dictionary)
    final_pair_wise_duplicates_removed = duplicates_removed_tuple_list(final_list_of_tuples)

    if proteins_in_dimer:
        protein_filtered = [t for t in final_pair_wise_duplicates_removed
                            if not any(x in proteins_in_dimer for x in t)]
    else:
        protein_filtered = final_pair_wise_duplicates_removed
    # Remove glycan elements already used by dimers from remaining tuples
    if glycans_in_dimer:
        glycan_filtered = [tuple(x for x in t if x not in glycans_in_dimer)
                           for t in protein_filtered]
    else:
        glycan_filtered = protein_filtered

    # Convert to result_list (channel names only, sorted)
    result_list = [tuple(sorted(element.split("_")[0] for element in tup)) for tup in glycan_filtered]
    result_list[:] = [tuple(sorted(tup)) for tup in result_list]

    # Remove classes with more than one protein (dimers)
    result_list = [t for t in result_list if t.count(protein) < 2]

    # === BUILD OUTPUT FOR ANALYSIS ===
    protein_glycosylation = {i: [] for i in range(len(protein_coords))}
    is_single_egfr_cluster = np.zeros(len(protein_coords), dtype=bool)

    for tup in glycan_filtered:
        channels = [element.split("_")[0] for element in tup]
        egfr_count = sum(1 for c in channels if c == protein)
        lectins = [c for c in channels if c != protein]

        # Single EGFR with lectins (monomer glycosylation)
        if egfr_count == 1 and len(lectins) > 0:
            for element in tup:
                if element.startswith(f'{protein}_'):
                    p_idx = int(element.split('_')[1])
                    is_single_egfr_cluster[p_idx] = True
                    protein_glycosylation[p_idx] = lectins
                    break
                
        elif egfr_count >= 2 and len(lectins) > 0:
            for element in tup:
                if element.startswith(f'{protein}_'):
                    p_idx = int(element.split('_')[1])
                    # Store lectins for this protein (will be combined later for dimer class)
                    protein_glycosylation[p_idx] = lectins

    glycosylated_count = is_single_egfr_cluster.sum()

    stats = {
        'glycosylated_proteins': glycosylated_count,
        'non_glycosylated_proteins': len(protein_coords) - glycosylated_count,
        'is_single_egfr_cluster': is_single_egfr_cluster
    }
    return protein_glycosylation, stats

def detect_dimers_and_monomers(protein_coords, dimer_radius):
    """
    Detects dimers and monomers exactly as implemented in GlyCo
    """
    df_of_interest_key = anchor_channel
    com_name = f"neighbors_of_{df_of_interest_key}"

    tree = KDTree(protein_coords)
    assigned_points = {df_of_interest_key: {}}

    for row_idx in tqdm(range(len(protein_coords)), desc=f"Searching for neighbors of {df_of_interest_key}", leave=False):
        x1, y1 = protein_coords[row_idx]
        indices = tree.query_ball_point([x1, y1], r=dimer_radius)
        filtered_indices = [num for num in indices if num != row_idx]

        if filtered_indices:
            dist_idx_pairs = [(np.linalg.norm(tree.data[idx] - [x1, y1]), idx) for idx in filtered_indices]
            dist_idx_pairs.sort(key=lambda x: x[0])
            for distance, idx in dist_idx_pairs:
                if idx not in assigned_points[df_of_interest_key] or distance < assigned_points[df_of_interest_key][idx][0]:
                    assigned_points[df_of_interest_key][idx] = (distance, row_idx)

    polymer_neighbor_master = {com_name: {}}
    for family, members in assigned_points.items():
        for idx, (distance, p_idx) in members.items():
            polymer_neighbor_master[com_name].setdefault(f'{df_of_interest_key}_{p_idx}', []).append(f'{family}_{idx}')

    polymer_tuple_list = nested_dict_to_tuple_list(polymer_neighbor_master)
    polymer_list_without_mirror_duplicates = duplicates_removed_tuple_list(polymer_tuple_list)
    flattened_polymer_list = [element for tup in polymer_list_without_mirror_duplicates for element in tup]

    # EVERY link in polymer_list_without_mirror_duplicates is treated as a dimer object in GlyCo
    dimer_pairs = [tuple(sorted([int(t[0].split('_')[1]), int(t[1].split('_')[1])])) for t in polymer_list_without_mirror_duplicates if len(t) == 2]

    # Monomers are proteins NOT in the set of proteins participating in ANY structure
    proteins_in_polymers = {int(item.split('_')[1]) for item in flattened_polymer_list}
    monomer_indices = [i for i in range(len(protein_coords)) if i not in proteins_in_polymers]

    return dimer_pairs, monomer_indices, proteins_in_polymers

def create_radial_shell_polygons(border_poly, erosion_contours):
    """
    Create shell polygons from border and erosion contours.
    These define the CONSTANT RADIAL MASK for density calculation.
    """
    shell_polys = [Polygon(border_poly)] + [Polygon(c) for c in erosion_contours]
    shell_areas = []
    for i in range(len(shell_polys)-1):
        area = shell_polys[i].area - shell_polys[i+1].area
        shell_areas.append(max(area, 1e-10))
    return shell_polys, shell_areas

def calculate_radial_density(pts_x, pts_y, shell_polys, shell_areas):
    """
    Calculate radial density of points using the CONSTANT shell mask.
    """
    if len(pts_x) == 0:
        return np.zeros(len(shell_polys)-1)

    pts = [Point(px, py) for px, py in zip(pts_x, pts_y)]
    counts = np.zeros(len(shell_polys)-1)

    for p in pts:
        for i in range(len(shell_polys)-1):
            if shell_polys[i].contains(p) and not shell_polys[i+1].contains(p):
                counts[i] += 1
                break

    return (counts / np.array(shell_areas)) * 1e6  # Convert to locs/um^2

def interpolate_to_standard(density, bin_centers_norm, standard_norm_radius):
    """
    Interpolate density to standard normalized radius for group averaging.
    """
    if len(density) < 2:
        return np.zeros(len(standard_norm_radius))

    f = interp1d(bin_centers_norm, density, kind='linear', bounds_error=False, fill_value=0)
    return f(standard_norm_radius)

print("\n✓ Helper functions defined")

# ----------------------------------------



# In[ ]:


# ===========================================================================

# === 3. CONFIGURATION ===


# In[ ]:


# ===========================================================================

# --- CONFIGURATION ---
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial'],
    'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
})

root_dir = r"/Users/nazlicanyurekli/Desktop/2026-03-10_CD4 cell data /Analyzed Data/2026-02-02_CD4+T Cells Segmented-Clustered-Glyco/Test"
search_folder_name = "90_Custom Centers"
lectin_library = ['WGA', 'SNA', 'PHAL', 'AAL', 'PSA']
anchor_channel = 'EGFR'   # Channel for protein analysis
mask_channel = 'ALL_CHANNELS'  # Cell border/mask from ALL channels combined - CONSTANT per cell
glyco_radius = 35
dimer_radius = 36
bin_size_nm = 250
pixel_scale = 130
alpha_shape_param = 0.0005
save_results = True
group_1_keyword, group_2_keyword = "Stimulated", "Non_stimulated"
normalization_points = 100
standard_norm_radius = np.linspace(0, 1, normalization_points)
number_to_plot = 5  # Top N classes to plot

# --- A4 FIGURE CONSTANTS & FORMATTING HELPER ---
A4_W, A4_H = 8.27, 11.69  # inches

def format_radial_ax(ax):
    """Enforce xlim=[0,1], y-axis ceil of actual data max, x-ticks at 0/0.5/1."""
    ax.set_xlim(0, 1)
    all_y = [y for line in ax.get_lines() for y in line.get_ydata()]
    ytop = float(np.ceil(max(all_y))) if all_y and max(all_y) > 0 else 1.0
    ax.set_ylim(0, ytop)
    
   
    ax.tick_params(direction='out', length=3.5, width=0.8, colors='black', top=False, right=False)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(['0', '0.5', '1'])
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(0.8)

    for line in ax.get_lines():
        line.set_clip_on(True)

print("\n✓ Configuration set")

# ----------------------------------------



# In[ ]:


# ===========================================================================

# === 4. OUTPUT FOLDER STRUCTURE + STORAGE INITIALISATION ===


# In[ ]:


# ===========================================================================

# ============================================================================
# CREATE SUMMARY OUPUT FOLDER
# ============================================================================

# For Top 5 identification per group (Benchmark on Non-stimulated Monomers)
monomer_glyco_normalized = {group_1_keyword: {}, group_2_keyword: {}}

# For mean radial plots - MONOMERS
radial_monomer_total = {group_1_keyword: [], group_2_keyword: []}
radial_monomer_glyco = {group_1_keyword: [], group_2_keyword: []}
radial_monomer_non_glyco = {group_1_keyword: [], group_2_keyword: []}

# For mean radial plots - DIMERS
radial_dimer_total = {group_1_keyword: [], group_2_keyword: []}
radial_dimer_glyco = {group_1_keyword: [], group_2_keyword: []}
radial_dimer_non_glyco = {group_1_keyword: [], group_2_keyword: []}

# For mean radial plots - TOTAL EGFR 
radial_total_egfr = {group_1_keyword: [], group_2_keyword: []}

# For mean radial plots - Top 5 classes 
radial_top5_monomer = {group_1_keyword: {}, group_2_keyword: {}}
radial_top5_dimer = {group_1_keyword: {}, group_2_keyword: {}}

# For bar chart statistics
top5_monomer_stats = {group_1_keyword: {}, group_2_keyword: {}}
top5_dimer_stats = {group_1_keyword: {}, group_2_keyword: {}}

# For summary figures and tables (PHASE 3)
cell_statistics = {group_1_keyword: [], group_2_keyword: []}
fig_masks = {group_1_keyword: [], group_2_keyword: []}

target_folders = list(Path(root_dir).rglob(f"**/{search_folder_name}"))
print(f"DEBUG: Found {len(target_folders)} target folders.")

# ============================================================================
# PHASE 1: IDENTIFY GLOBAL TOP 5 GLYCO-CLASSES (BASED ON MONOMERS ONLY)
# Common between STIM and NON-STIM monomers after glyco + duplicate removal
# ============================================================================
print(f"\n{'='*70}")
print("PHASE 1: Identifying Top 5 Glyco-Classes (Based on MONOMERS)")
print(f"        Finding Top 5 classes from {group_2_keyword} monomers ONLY")
print(f"{'='*70}\n")

print("\n✓ Output folders created & storage initialised")

# ----------------------------------------



# In[ ]:


# ===========================================================================

# === 5. PHASE 1 — IDENTIFY GLOBAL TOP 5 GLYCO-CLASSES ===


# In[ ]:


# ===========================================================================

# ============================================================================
# PHASE 1: IDENTIFY GLOBAL TOP 5 GLYCO-CLASSES (BASED ON MONOMERS ONLY)
# Common between STIM and NON-STIM monomers after glyco + duplicate removal
# ============================================================================
print(f"\n{'='*70}")
print("PHASE 1: Identifying Top 5 Glyco-Classes (Based on MONOMERS)")
print(f"        Finding Top 5 classes from {group_2_keyword} monomers ONLY")
print(f"{'='*70}\n")

for loc_folder in target_folders:
    path_str = str(loc_folder)
    current_group = group_1_keyword if group_1_keyword in path_str else group_2_keyword if group_2_keyword in path_str else None
    if not current_group: continue


    # Force keys to .upper() to match ground-truth behavior
    data_dict = {f.stem.split("_")[0].upper(): pd.read_hdf(f, key='locs') for f in loc_folder.glob("*.hdf5")}
    if anchor_channel.upper() not in data_dict: continue

    # Get area normalization
    area_of_cell = None
    possible_keys = ["Total Picked Area (um^2)", "Area (um^2)"]
    yaml_files = list(loc_folder.glob("*.yaml"))
    if yaml_files:
        with open(yaml_files[0], 'r') as f:
            try:
                for info in yaml.safe_load_all(f):
                    # Match original glyco: overwrite with LAST matching document (no break)
                    if isinstance(info, dict):
                        for key in possible_keys:
                            if key in info:
                                area_of_cell = np.float32(info[key])
                                break
            except Exception as e:
                print(f"Error parsing {yaml_files[0]}: {e}")

    if area_of_cell is None or area_of_cell <= 0:
        print(f"SKIPPING CELL: No valid area found in {loc_folder}")
        continue

    # Build library from loaded data
    library = sorted(data_dict.keys())
    protein = anchor_channel
    protein_present = True
    consider_dimers = True

    # =====================================================================
    # <<< GLYCO LOGIC (Phase 1) >>>
    # =====================================================================
    glyco_results = run_glyco(data_dict, protein, pixel_scale, dimer_radius, glyco_radius, protein_present, consider_dimers)
    
    result_list = glyco_results['result_list']
    index_removed_dimer_glycosylation = glyco_results['index_removed_dimer_glycosylation']

    # Build monomer counts for Phase 1 collection 
    monomer_counts_this_cell = Counter()
    for tup in result_list:
        # tup contains EGFR + lectin entries; keep only lectin names
        lectin_only = [e for e in tup if e != protein]
        if lectin_only:
            # Use exact composition (allow multiple AALs for example)
            combo = "-".join(sorted(lectin_only))
            monomer_counts_this_cell[combo] += 1

    # ONLY benchmark and print stats for the Non-stimulated group
    if current_group == group_2_keyword:
        print(f"  Processed {loc_folder.parent.name} | Monomers (Benchmark): {len(result_list)}")
        for combo, count in monomer_counts_this_cell.items():
            normalized_count = count / area_of_cell
            if combo not in monomer_glyco_normalized[group_2_keyword]:
                monomer_glyco_normalized[group_2_keyword][combo] = []
            monomer_glyco_normalized[group_2_keyword][combo].append(normalized_count)
    else:
        # Just a silent progress indicator for non-benchmark cells if we process them here
        pass

# Calculate mean normalized counts per group (Benchmark on Non-stimulated Monomers)
non_stim_mono_means = {
    combo: np.mean(values) for combo, values in monomer_glyco_normalized[group_2_keyword].items()
}

sorted_non_stim = sorted(non_stim_mono_means.items(), key=lambda x: x[1], reverse=True)
top_5_classes = [c[0] for c in sorted_non_stim[:min(number_to_plot, len(sorted_non_stim))]]

print(f"\n  Top {number_to_plot} Glyco-Classes ({group_2_keyword} Monomers):")
for i, combo in enumerate(top_5_classes, 1):
    print(f"    {i}. {combo} (mean density: {non_stim_mono_means[combo]:.4f})")

# Result stored in top_5_classes

# Initialize stats collection for top 5 classes
for combo in top_5_classes:
    top5_monomer_stats[group_1_keyword][combo] = []
    top5_monomer_stats[group_2_keyword][combo] = []
    top5_dimer_stats[group_1_keyword][combo] = []
    top5_dimer_stats[group_2_keyword][combo] = []
    radial_top5_monomer[group_1_keyword][combo] = []
    radial_top5_monomer[group_2_keyword][combo] = []
    radial_top5_dimer[group_1_keyword][combo] = []
    radial_top5_dimer[group_2_keyword][combo] = []

print("\n✓ Top 5 classes identified & storage initialised")

# ----------------------------------------



# In[ ]:


# ===========================================================================

# === 7. PHASE 2 — FULL RADIAL ANALYSIS (MONOMERS + DIMERS) ===


# In[ ]:


# ===========================================================================

# ============================================================================
# PHASE 2: PROCESS EACH CELL WITH FULL ANALYSIS (MONOMERS + DIMERS)
# ============================================================================
print(f"\n{'='*70}")
print("PHASE 2: Processing Each Cell - Radial Analysis for MONOMERS + DIMERS")
print(f"{'='*70}\n")

for loc_folder in target_folders:
    path_str = str(loc_folder)
    current_group = group_1_keyword if group_1_keyword in path_str else group_2_keyword if group_2_keyword in path_str else None
    if not current_group: continue


    cell_name = loc_folder.parent.name
    print(f"\n{'='*60}")
    print(f"Processing: {current_group} | {cell_name}")
    print(f"{'='*60}")

    data_dict = {f.stem.split("_")[0]: pd.read_hdf(f, key='locs') for f in loc_folder.glob("*.hdf5")}
    if anchor_channel not in data_dict: continue

    # Get area normalization
    area_of_cell = None
    possible_keys = ["Total Picked Area (um^2)", "Area (um^2)"]
    yaml_files = list(loc_folder.glob("*.yaml"))
    if yaml_files:
        with open(yaml_files[0], 'r') as f:
            try:
                for info in yaml.safe_load_all(f):
                    if isinstance(info, dict):
                        for key in possible_keys:
                            if key in info:
                                area_of_cell = np.float32(info[key])
                                break
            except Exception as e:
                print(f"Error parsing {yaml_files[0]}: {e}")

    if area_of_cell is None or area_of_cell <= 0:
        print(f"SKIPPING CELL: No valid area found in {loc_folder}")
        continue

    # ========================================================================
    # LOAD DATA
    # ========================================================================

    df_anchor = data_dict[anchor_channel]
    x_nm, y_nm = df_anchor['x'].values * pixel_scale, df_anchor['y'].values * pixel_scale
    protein_xy = np.column_stack((x_nm, y_nm))

    # ========================================================================
    # CREATE CONSTANT CELL MASK FROM ALL CHANNELS (UNION OF LOCALIZATIONS)
    # ========================================================================

    all_x = []
    all_y = []
    for key, df in data_dict.items():
        all_x.extend(df['x'].values)
        all_y.extend(df['y'].values)
    
    x_mask_nm = np.array(all_x) * pixel_scale
    y_mask_nm = np.array(all_y) * pixel_scale

    print(f"  -> Creating CONSTANT cell mask from ALL channels ({len(data_dict)} combined)...")
    border_poly = detect_cell_border(x_mask_nm, y_mask_nm, alpha=alpha_shape_param)

    print("  -> Finding center by erosion (defines radial shells)...")
    c_x, c_y, erosion_contours, e_areas = find_center_by_erosion(border_poly, bin_size_nm=bin_size_nm)

    print("  -> Creating CONSTANT radial shell polygons...")
    shell_polys, shell_areas = create_radial_shell_polygons(border_poly, erosion_contours)
    print(f"    Number of radial shells: {len(shell_areas)}")

    # ========================================================================
    # CANVAS SETUP
    # ========================================================================

    canvas_size_um = 35 
    offset = canvas_size_um / 2
    x_canvas, y_canvas = (x_nm - c_x)/1000 + offset, (y_nm - c_y)/1000 + offset
    border_canvas = np.column_stack([(border_poly[:,0]-c_x)/1000+offset, (border_poly[:,1]-c_y)/1000+offset])
    e_contours_canvas = [np.column_stack([(c[:,0]-c_x)/1000+offset, (c[:,1]-c_y)/1000+offset]) for c in erosion_contours]

    # =====================================================================
    # <<< Am I a Dimer? >>>
    # =====================================================================
    library = sorted(data_dict.keys())
    protein = anchor_channel
    protein_present = True
    consider_dimers = True

    print("  -> STEP 1 & 2: Executing authentic Glyco Dimer/Monomer analysis...")
    glyco_results = run_glyco(data_dict, protein, pixel_scale, dimer_radius, glyco_radius, protein_present, consider_dimers)
    
    flattened_polymer_list = glyco_results['polymer_list']
    dimer_list = glyco_results['dimer_list']
    dimer_glycosylation_glycan_unduplicated = glyco_results['dimer_glycosylation_glycan_unduplicated']
    index_removed_dimer_glycosylation = glyco_results['index_removed_dimer_glycosylation'] # [FIXED] Extract this here too!
    glycan_filtered_monomer_glycosylation = glyco_results['glycan_filtered_monomer_glycosylation']
    result_list = glyco_results['result_list']
    protein_considered_in_dimer = glyco_results['protein_considered_in_dimer']
    unique_items_in_flattened_polymer_list = glyco_results['unique_items_in_flattened_polymer_list']

    # Build protein_xy for canvas/radial use
    df_anchor = data_dict[anchor_channel]
    x_nm, y_nm = df_anchor['x'].values * pixel_scale, df_anchor['y'].values * pixel_scale
    protein_xy = np.column_stack((x_nm, y_nm))

    # Derive numeric dimer/monomer index sets from original string-based lists
    dimer_indices = set()
    for elem in protein_considered_in_dimer:
        parts = elem.split('_')
        if len(parts) == 2 and parts[1].isdigit():
            dimer_indices.add(int(parts[1]))

    monomer_protein_strs = set()
    for t in glycan_filtered_monomer_glycosylation:
        for s in t:
            parts = s.split('_')
            if len(parts) == 2 and parts[0] == protein and parts[1].isdigit():
                monomer_protein_strs.add(int(parts[1]))

    # Glycosylated monomer count
    number_of_glycosylated_monomers = len(result_list)

    print(f"    Total EGFR proteins: {len(protein_xy)}")
    print(f"    Glycosylated proteins (any lectin): {len(monomer_protein_strs) + len(dimer_indices)}")
    print(f"    Total EGFR centers: {len(protein_xy)}")
    print(f"    Dimer pairs: {len(dimer_list)} ({len(dimer_indices)} proteins in dimers)")
    print(f"    Polymers (>2): {len(unique_items_in_flattened_polymer_list) - len(dimer_indices)} proteins")
    print(f"    Monomers (Total - Dimers - Polymers): {len(protein_xy) - len(unique_items_in_flattened_polymer_list)}")

    # ========================================================================
    # STEP 3: CREATE PROTEIN SELECTIONS FOR MONOMERS AND DIMERS
    # ========================================================================
    print("  -> STEP 3: Creating protein selections (boolean filters)...")

    # MONOMER SELECTIONS
    is_monomer = np.zeros(len(protein_xy), dtype=bool)
    monomer_indices = [i for i in range(len(protein_xy)) if i not in (set(int(e.split('_')[1]) for e in flattened_polymer_list if '_' in e and e.split('_')[1].isdigit()))]
    for i in monomer_indices:
        is_monomer[i] = True

    # Glycosylated monomers: those in monomer_protein_strs
    is_glyco_monomer = np.zeros(len(protein_xy), dtype=bool)
    for idx in monomer_protein_strs:
        if idx < len(protein_xy):
            is_glyco_monomer[idx] = True
    is_non_glyco_monomer = is_monomer & ~is_glyco_monomer

    # DIMER SELECTIONS
    is_dimer = np.zeros(len(protein_xy), dtype=bool)
    for idx in dimer_indices:
        is_dimer[idx] = True

    # Glyco dimer: check which dimers have glycans from index_removed_dimer_glycosylation
    is_glyco_dimer = np.zeros(len(protein_xy), dtype=bool)
    is_non_glyco_dimer = np.zeros(len(protein_xy), dtype=bool)
    glycosylated_dimer_protein_strs = set()
    for tup in dimer_glycosylation_glycan_unduplicated:
        if tup is None:
            continue
        has_glycan = any(s.split('_')[0] != protein for s in tup if '_' in s and not s.split('_')[1:][0].isdigit() == False)
        lectin_entries = [s for s in tup if s.split('_')[0] != protein]
        if lectin_entries:
            for s in tup:
                parts = s.split('_')
                if len(parts) == 2 and parts[0] == protein and parts[1].isdigit():
                    glycosylated_dimer_protein_strs.add(int(parts[1]))
    for idx in dimer_indices:
        if idx in glycosylated_dimer_protein_strs:
            is_glyco_dimer[idx] = True
        else:
            is_non_glyco_dimer[idx] = True

    # VISUAL DIMER SELECTIONS (Visual: 1 protein per pair)
    is_visual_dimer = np.zeros(len(protein_xy), dtype=bool)
    is_visual_glyco_dimer = np.zeros(len(protein_xy), dtype=bool)
    is_visual_non_glyco_dimer = np.zeros(len(protein_xy), dtype=bool)
    
    # We use the first protein in each dimer_list pair to represent the whole pair visually
    for tuple_pair in dimer_list:
        p1_str = tuple_pair[0] # e.g. 'EGFR_123'
        p1_idx = int(p1_str.split('_')[1])
        is_visual_dimer[p1_idx] = True
        
        # Determine if this pair is glycosylated (if EITHER protein has a glycan)
        # Using glycosylated_dimer_protein_strs which contains indices of ALL glyco-proteins in dimers
        is_p1_glyco = p1_idx in glycosylated_dimer_protein_strs
        # Check the second protein too to be safe (though GlyCo usually assigns to both)
        p2_str = tuple_pair[1]
        p2_idx = int(p2_str.split('_')[1])
        is_p2_glyco = p2_idx in glycosylated_dimer_protein_strs
        
        if is_p1_glyco or is_p2_glyco:
            is_visual_glyco_dimer[p1_idx] = True
        else:
            is_visual_non_glyco_dimer[p1_idx] = True

    print("\n" + "="*40)
    print(f"SUMMARY FOR {cell_name}:")
    print(f"Total {anchor_channel} Proteins: {len(protein_xy)}")
    print(f"Total Monomers (Search): {is_monomer.sum()}")
    print(f"Glycosylated Monomers (Result): {is_glyco_monomer.sum()}")
    print(f"Total Dimers (Pairs): {len(dimer_list)}")
    print(f"Total Dimer Proteins (Centers): {int(is_dimer.sum())}")
    print("="*40 + "\n")

    # TOP 5 GLYCO-CLASS SELECTIONS
    print("  -> Creating Top 5 selections (using Benchmark Logic)...")

    top5_monomer_selections = {}
    top5_dimer_selections = {}
    top5_monomer_counts = Counter()
    top5_dimer_counts = Counter()

    # Monomer selections — from glycan_filtered_monomer_glycosylation
    for t in glycan_filtered_monomer_glycosylation:
        protein_idxs = [int(s.split('_')[1]) for s in t if s.split('_')[0] == protein and len(s.split('_')) > 1 and s.split('_')[1].isdigit()]
        lectin_labels = [s.split('_')[0] for s in t if s.split('_')[0] != protein]
        if len(protein_idxs) == 1 and lectin_labels:
            combo = "-".join(sorted(lectin_labels))
            top5_monomer_counts[combo] += 1
            if combo in top_5_classes:
                if combo not in top5_monomer_selections:
                    top5_monomer_selections[combo] = np.zeros(len(protein_xy), dtype=bool)
                top5_monomer_selections[combo][protein_idxs[0]] = True

    # Dimer selections — from index_removed_dimer_glycosylation
    for orig_tup, cleaned_tup in zip(dimer_glycosylation_glycan_unduplicated, index_removed_dimer_glycosylation):
        if orig_tup is None:
            continue
        lectin_names = [e for e in cleaned_tup if e != protein]
        if lectin_names:
            combo = "-".join(sorted(lectin_names))
            top5_dimer_counts[combo] += 1
            if combo in top_5_classes:
                if combo not in top5_dimer_selections:
                    top5_dimer_selections[combo] = np.zeros(len(protein_xy), dtype=bool)
                # VISUAL: plot only ONE protein index per pair for the Top 5 spatial dot mapping
                # Find the FIRST protein center in the original tuple to represent the pair
                for s in orig_tup:
                    parts = s.split('_')
                    if len(parts) == 2 and parts[0] == protein and parts[1].isdigit():
                        top5_dimer_selections[combo][int(parts[1])] = True
                        break # Flag ONLY ONE center per pair for 1-dot visualization

    # Store normalized counts for ALL tracked classes
    for combo in top_5_classes:
        top5_monomer_stats[current_group][combo].append(top5_monomer_counts[combo] / area_of_cell)
        top5_dimer_stats[current_group][combo].append(top5_dimer_counts[combo] / area_of_cell)

    print(f"    Top 5 class counts (Non-stim Monomer top 5):")
    for combo in top_5_classes:
        print(f"      {combo}: Monomers={top5_monomer_counts[combo]}, Dimers={top5_dimer_counts[combo]}")

    print("\n" + "="*40)
    print(f"SUMMARY FOR {cell_name}:")
    print(f"Total Proteins: {len(protein_xy)}")
    print(f"Total Monomers: {is_monomer.sum()}")
    print(f"Total Dimer Edges: {len(dimer_list)}")
    print(f"Glycosylated Monomers: {is_glyco_monomer.sum()}")
    print("="*40 + "\n")

    # ========================================================================
    # CALCULATE RADIAL DENSITIES
    # ========================================================================

    print("  -> Calculating radial densities (using CONSTANT ALL-CHANNELS mask)...")

    # MONOMER DENSITIES
    d_monomer = calculate_radial_density(x_nm[is_monomer], y_nm[is_monomer], shell_polys, shell_areas)
    d_glyco_monomer = calculate_radial_density(x_nm[is_glyco_monomer], y_nm[is_glyco_monomer], shell_polys, shell_areas)
    d_non_glyco_monomer = calculate_radial_density(x_nm[is_non_glyco_monomer], y_nm[is_non_glyco_monomer], shell_polys, shell_areas)

    # DIMER DENSITIES
    d_dimer = calculate_radial_density(x_nm[is_dimer], y_nm[is_dimer], shell_polys, shell_areas)
    d_glyco_dimer = calculate_radial_density(x_nm[is_glyco_dimer], y_nm[is_glyco_dimer], shell_polys, shell_areas)
    d_non_glyco_dimer = calculate_radial_density(x_nm[is_non_glyco_dimer], y_nm[is_non_glyco_dimer], shell_polys, shell_areas)

    # Radial densities for ALL tracked classes - MONOMERS
    d_top5_monomers = {}
    for combo in top_5_classes:
        if combo in top5_monomer_selections:
            selection = top5_monomer_selections[combo]
            d_top5_monomers[combo] = calculate_radial_density(x_nm[selection], y_nm[selection], shell_polys, shell_areas)
        else:
            d_top5_monomers[combo] = np.zeros(len(d_monomer))

    # Radial densities for ALL tracked classes - DIMERS
    d_top5_dimers = {}
    for combo in top_5_classes:
        if combo in top5_dimer_selections:
            selection = top5_dimer_selections[combo]
            d_top5_dimers[combo] = calculate_radial_density(x_nm[selection], y_nm[selection], shell_polys, shell_areas)
        else:
            d_top5_dimers[combo] = np.zeros(len(d_dimer))

    # Reconstruct all_protein_glycosylation dict for downstream radial/JSON sections
    # {protein_idx: [lectin1, lectin2, ...]} — built from original glyco results
    all_protein_glycosylation = {i: [] for i in range(len(protein_xy))}
    for t in glycan_filtered_monomer_glycosylation:
        pidxs = [s for s in t if s.split('_')[0] == protein and len(s.split('_')) > 1 and s.split('_')[1].isdigit()]
        lects = [s.split('_')[0] for s in t if s.split('_')[0] != protein]
        if len(pidxs) == 1:
            all_protein_glycosylation[int(pidxs[0].split('_')[1])] = lects
    # Also add dimer proteins' glycans
    for orig_tup in dimer_glycosylation_glycan_unduplicated:
        if orig_tup is None:
            continue
        pidxs = [s for s in orig_tup if s.split('_')[0] == protein and len(s.split('_')) > 1 and s.split('_')[1].isdigit()]
        lects = [s.split('_')[0] for s in orig_tup if s.split('_')[0] != protein]
        for pidx_str in pidxs:
            pidx = int(pidx_str.split('_')[1])
            if pidx < len(protein_xy):
                all_protein_glycosylation[pidx] = lects

    # TOTAL EGFR DENSITIES
    is_glycosylated_total = np.array([len(all_protein_glycosylation[i]) > 0 for i in range(len(protein_xy))])
    d_total_egfr = calculate_radial_density(x_nm, y_nm, shell_polys, shell_areas)
    d_total_glyco = calculate_radial_density(x_nm[is_glycosylated_total], y_nm[is_glycosylated_total], shell_polys, shell_areas)
    d_total_non_glyco = calculate_radial_density(x_nm[~is_glycosylated_total], y_nm[~is_glycosylated_total], shell_polys, shell_areas)

    d_top5_total = {}
    for combo in top_5_classes:
        # Count occurrences across ALL EGFR for the Total Dashboard
        mask = np.zeros(len(protein_xy), dtype=bool)
        for p_idx, lectins in all_protein_glycosylation.items():
            if "-".join(sorted(lectins)) == combo:
                mask[p_idx] = True
        d_top5_total[combo] = calculate_radial_density(x_nm[mask], y_nm[mask], shell_polys, shell_areas)

    # ORIGINAL CONVENTION
    bin_centers_norm = np.linspace(1, 0, len(d_monomer))
    edges = np.arange(len(d_monomer) + 0) * bin_size_nm

    # ========================================================================
    # STORE INTERPOLATED DATA FOR GROUP MEAN PLOTS
    # ========================================================================

    # Interpolate to standard normalized radius
    radial_monomer_total[current_group].append(interpolate_to_standard(d_monomer, bin_centers_norm, standard_norm_radius))
    radial_monomer_glyco[current_group].append(interpolate_to_standard(d_glyco_monomer, bin_centers_norm, standard_norm_radius))
    radial_monomer_non_glyco[current_group].append(interpolate_to_standard(d_non_glyco_monomer, bin_centers_norm, standard_norm_radius))

    radial_dimer_total[current_group].append(interpolate_to_standard(d_dimer, bin_centers_norm, standard_norm_radius))
    radial_dimer_glyco[current_group].append(interpolate_to_standard(d_glyco_dimer, bin_centers_norm, standard_norm_radius))
    radial_dimer_non_glyco[current_group].append(interpolate_to_standard(d_non_glyco_dimer, bin_centers_norm, standard_norm_radius))

    # Store Total EGFR radial data
    radial_total_egfr[current_group].append(interpolate_to_standard(d_total_egfr, bin_centers_norm, standard_norm_radius))

    for combo in top_5_classes:
        radial_top5_monomer[current_group][combo].append(
            interpolate_to_standard(d_top5_monomers[combo], bin_centers_norm, standard_norm_radius)
        )
        radial_top5_dimer[current_group][combo].append(
            interpolate_to_standard(d_top5_dimers[combo], bin_centers_norm, standard_norm_radius)
        )

    # ========================================================================
    # SAVE INDIVIDUAL CELL RESULTS
    # ========================================================================

    if save_results:
        print("  -> Generating output figures...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out = loc_folder / f"{timestamp}_SHELL_Analysis_Results"
        out.mkdir(parents=True, exist_ok=True)

        b_min_x, b_min_y = np.min(border_canvas, axis=0)
        b_max_x, b_max_y = np.max(border_canvas, axis=0)
        padding = 0.5

        # Save statistics
        stats_dict = {
            'total_proteins': len(protein_xy),
            'total_monomers': int(is_monomer.sum()),
            'glycosylated_monomers': int(is_glyco_monomer.sum()),
            'non_glycosylated_monomers': int(is_non_glyco_monomer.sum()),
            'total_dimer_proteins': int(is_dimer.sum()),
            'glycosylated_dimers': int(is_glyco_dimer.sum()),
            'non_glycosylated_dimers': int(is_non_glyco_dimer.sum()),
            'total_dimer_pairs': len(dimer_list),
            'area_um2': area_of_cell,
            'n_radial_shells': len(shell_areas),
            'mask_channel': mask_channel
        }
        stats_df = pd.DataFrame([stats_dict])
        stats_df.to_csv(out / "Overall_Statistics.csv", index=False)

        # ====================================================================
        # MONOMER DASHBOARD 
        # ====================================================================
        print("  -> Creating MONOMER dashboard (A4)...")
        from matplotlib.backends.backend_pdf import PdfPages as _PdfPages

        _bar_w = 1.0 / len(bin_centers_norm) if len(bin_centers_norm) > 0 else 0.01

        def _draw_dashboard_page(titles, selections, densities, colors, suptitle,
                                 landscape=False):
            # 3 rows: spatial map | raw distance bars | normalised bars
            n_cols  = len(titles)
            figsize = (11.69, 8.27) if landscape else (8.27, 11.69)
            fig, axes = plt.subplots(3, n_cols, figsize=figsize,
                                     gridspec_kw={'height_ratios': [3, 2, 2]})
            if n_cols == 1:
                axes = axes.reshape(3, 1)
            # raw distance axis: 0 = centre, max nm = edge
            n_shells  = len(densities[0])
            raw_dist_x = np.arange(n_shells - 1, -1, -1) * bin_size_nm
            bar_w_raw  = bin_size_nm * 0.8
            for i in range(n_cols):
                # ── ROW 0: Spatial map ──────────────────────────────────────
                ax_map = axes[0, i]
                for cnt in e_contours_canvas:
                    ax_map.plot(cnt[:, 0], cnt[:, 1], color='black', lw=0.2, alpha=0.12, zorder=1)
                ax_map.plot(border_canvas[:, 0], border_canvas[:, 1], color='black', lw=0.6, zorder=2)
                sel = selections[i]
                ax_map.scatter(x_canvas[sel], y_canvas[sel], s=0.8, c=colors[i], alpha=0.75, zorder=3)
                ax_map.scatter(offset, offset, c='yellow', s=25, marker='*',
                               edgecolor='black', linewidth=0.4, zorder=10)
                ax_map.set_xlim(b_min_x - padding, b_max_x + padding)
                ax_map.set_ylim(b_min_y - padding, b_max_y + padding)
                ax_map.set_aspect('equal')
                ax_map.set_title(f"{titles[i]}\n(n={int(sel.sum())})",
                                 fontfamily='Arial', fontsize=10)
                ax_map.axis('off')

                # ── ROW 1: Raw distance (0 = centre, max nm = edge) ─────────
                ax_raw = axes[1, i]
                ax_raw.bar(raw_dist_x, densities[i], width=bar_w_raw,
                           color=colors[i], alpha=0.75)
                ax_raw.set_xlabel("Distance from centre (nm)",
                                  fontfamily='Arial', fontsize=9)
                ax_raw.set_ylabel("Density (locs/µm²)" if i == 0 else "",
                                  fontfamily='Arial', fontsize=9)
                ax_raw.set_title("Raw distance  (0 = centre)",
                                 fontfamily='Arial', fontsize=9)
                ax_raw.tick_params(direction='out', length=3, width=0.7, colors='black')
                for spine in ax_raw.spines.values():
                    spine.set_color('black'); spine.set_linewidth(0.7)
                for lbl in ax_raw.get_xticklabels() + ax_raw.get_yticklabels():
                    lbl.set_fontfamily('Arial'); lbl.set_fontsize(8)

                # ── ROW 2: Normalised (0 = centre, 1 = edge) ────────────────
                ax_pr = axes[2, i]
                ax_pr.bar(bin_centers_norm, densities[i], width=_bar_w,
                          color=colors[i], alpha=0.75)
                ax_pr.set_xlabel("0 = centre   1 = edge",
                                 fontfamily='Arial', fontsize=9)
                ax_pr.set_ylabel("Density (locs/µm²)" if i == 0 else "",
                                 fontfamily='Arial', fontsize=9)
                ax_pr.set_title("Normalised radius",
                                fontfamily='Arial', fontsize=9)
                ax_pr.set_xlim(0, 1)
                ax_pr.set_xticks([0, 0.5, 1])
                ax_pr.set_xticklabels(['0', '0.5', '1'])
                ax_pr.tick_params(direction='out', length=3, width=0.7, colors='black')
                for spine in ax_pr.spines.values():
                    spine.set_color('black'); spine.set_linewidth(0.7)
                for lbl in ax_pr.get_xticklabels() + ax_pr.get_yticklabels():
                    lbl.set_fontfamily('Arial'); lbl.set_fontsize(8)
            fig.suptitle(suptitle, fontfamily='Arial', fontsize=11, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            return fig

        mono_main_titles = ["Total Monomers", "Glycosylated Monomers", "Non-Glyco Monomers"]
        mono_main_sels   = [is_monomer, is_glyco_monomer, is_non_glyco_monomer]
        mono_main_dens   = [d_monomer, d_glyco_monomer, d_non_glyco_monomer]
        mono_main_cols   = ['royalblue', 'forestgreen', 'firebrick']

        mono_top5_titles = [f"M: {c}" for c in top_5_classes]
        mono_top5_sels   = [top5_monomer_selections.get(c, np.zeros(len(protein_xy), dtype=bool))
                            for c in top_5_classes]
        mono_top5_dens   = [d_top5_monomers[c] for c in top_5_classes]
        mono_top5_cols   = ['cyan','magenta','orange','lime','purple',
                            'gold','dodgerblue','hotpink','turquoise','slateblue'][:len(top_5_classes)]

        with _PdfPages(out / "Monomer_Dashboard.pdf") as _pdf:
            _f = _draw_dashboard_page(mono_main_titles, mono_main_sels, mono_main_dens,
                                      mono_main_cols, f"MONOMER Analysis: {cell_name} — Main Categories")
            _pdf.savefig(_f, bbox_inches='tight'); plt.close(_f)
            if mono_top5_titles:
                _f2 = _draw_dashboard_page(mono_top5_titles, mono_top5_sels, mono_top5_dens,
                                           mono_top5_cols, f"MONOMER Analysis: {cell_name} — Top {len(top_5_classes)} Classes", landscape=True)
                _pdf.savefig(_f2, bbox_inches='tight'); plt.close(_f2)

        # ====================================================================
        # DIMER DASHBOARD 
        # ====================================================================
        print("  -> Creating DIMER dashboard (A4)...")

        dimer_main_titles = ["Total Dimers", "Glycosylated Dimers", "Non-Glyco Dimers"]
        # Use visual masks (Pair-based: 1 dot/count per dimer) for dashboard maps and 'n=' counts
        dimer_main_sels   = [is_visual_dimer, is_visual_glyco_dimer, is_visual_non_glyco_dimer]
        dimer_main_dens   = [d_dimer, d_glyco_dimer, d_non_glyco_dimer]
        dimer_main_cols   = ['#1f4e79', '#1e6b3a', '#7f0000']

        dimer_top5_titles = [f"D: {c}" for c in top_5_classes]
        dimer_top5_sels   = [top5_dimer_selections.get(c, np.zeros(len(protein_xy), dtype=bool))
                             for c in top_5_classes]
        dimer_top5_dens   = [d_top5_dimers[c] for c in top_5_classes]
        dimer_top5_cols   = ['teal','violet','darkorange','olive','indigo',
                             'goldenrod','royalblue','deeppink','mediumturquoise',
                             'darkslateblue'][:len(top_5_classes)]

        with _PdfPages(out / "Dimer_Dashboard.pdf") as _pdf:
            _f = _draw_dashboard_page(dimer_main_titles, dimer_main_sels, dimer_main_dens,
                                      dimer_main_cols, f"DIMER Analysis: {cell_name} — Main Categories")
            _pdf.savefig(_f, bbox_inches='tight'); plt.close(_f)
            if dimer_top5_titles:
                _f2 = _draw_dashboard_page(dimer_top5_titles, dimer_top5_sels, dimer_top5_dens,
                                           dimer_top5_cols, f"DIMER Analysis: {cell_name} — Top {len(top_5_classes)} Classes", landscape=True)
                _pdf.savefig(_f2, bbox_inches='tight'); plt.close(_f2)

        # ====================================================================
        # TOTAL EGFR DASHBOARD 
        # ====================================================================
        print("  -> Creating TOTAL EGFR dashboard (A4)...")

        total_main_titles = ["Total EGFR", "Glycosylated EGFR", "Non-Glyco EGFR"]
        total_main_sels   = [np.ones(len(protein_xy), dtype=bool),
                             is_glycosylated_total, ~is_glycosylated_total]
        total_main_dens   = [d_total_egfr, d_total_glyco, d_total_non_glyco]
        total_main_cols   = ['navy', 'darkgreen', 'maroon']

        total_top5_sels = []
        for _c in top_5_classes:
            _mask = np.zeros(len(protein_xy), dtype=bool)
            for _pi, _lecs in all_protein_glycosylation.items():
                if _lecs and "-".join(sorted(_lecs)) == _c:
                    _mask[_pi] = True
            total_top5_sels.append(_mask)

        total_top5_titles = [f"T: {c}" for c in top_5_classes]
        total_top5_dens   = [d_top5_total[c] for c in top_5_classes]
        total_top5_cols   = ['teal','purple','darkgoldenrod','darkcyan','brown',
                             'darkkhaki','mediumblue','crimson',
                             'lightseagreen','rebeccapurple'][:len(top_5_classes)]

        with _PdfPages(out / "Total_EGFR_Dashboard.pdf") as _pdf:
            _f = _draw_dashboard_page(total_main_titles, total_main_sels, total_main_dens,
                                      total_main_cols, f"TOTAL EGFR Analysis: {cell_name} — Main Categories")
            _pdf.savefig(_f, bbox_inches='tight'); plt.close(_f)
            if total_top5_titles:
                _f2 = _draw_dashboard_page(total_top5_titles, total_top5_sels, total_top5_dens,
                                           total_top5_cols, f"TOTAL EGFR Analysis: {cell_name} — Top {len(top_5_classes)} Classes", landscape=True)
                _pdf.savefig(_f2, bbox_inches='tight'); plt.close(_f2)

        # ====================================================================
        # SAVE JSON FOR LEGACY COMPARISON 
        # ====================================================================
        legacy_class_counts = Counter()
        for t in glycan_filtered_monomer_glycosylation:
            pidxs = [s for s in t if s.split('_')[0] == protein and len(s.split('_')) > 1 and s.split('_')[1].isdigit()]
            lects = [s.split('_')[0] for s in t if s.split('_')[0] != protein]
            if len(pidxs) == 1 and lects:
                combo = tuple(sorted([protein] + lects))
                if 2 <= len(combo) <= 6:
                    legacy_class_counts[combo] += 1

        sorted_legacy = sorted(legacy_class_counts.items(), key=lambda x: x[1], reverse=True)
        # Apply PCA filtering: only include classes with value > 0 (matching PCA.py pattern)
        class_per_area = {str(k): float(v / area_of_cell) for k, v in sorted_legacy if v > 0}

        # Save regular PCA JSON (matching GlyCo naming structure)
        legacy_json_path = loc_folder / f"{timestamp}_Lectin_Classes_ per_sq-microns_for_intercheck_{glyco_radius}nm.json"
        legacy_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(legacy_json_path, 'w') as f:
            json.dump(class_per_area, f, indent=4)
        
        # ====================================================================
        # ====================================================================
        # STORE CELL MASK DATA FOR SUMMARY FIGURES
        # ====================================================================
        cell_statistics[current_group].append({
            'cell_name': cell_name,
            'total_monomers': int(is_monomer.sum()),
            'total_dimers': len(dimer_list),
            'area_um2': area_of_cell
        })

        fig_masks[current_group].append({
            'cell_name': cell_name,
            'border_canvas': border_canvas.copy(),
            'e_contours_canvas': [c.copy() for c in e_contours_canvas],
            'x_canvas': x_canvas.copy(),
            'y_canvas': y_canvas.copy(),
            'is_monomer': is_monomer.copy(),
            'is_dimer': is_visual_dimer.copy(),
            'b_min_x': b_min_x,
            'b_max_x': b_max_x,
            'b_min_y': b_min_y,
            'b_max_y': b_max_y,
            'offset': offset
        })

        print(f"  -> Cell {cell_name} completed.")

print("\n✓ Phase 2 complete — all cells processed")

# ----------------------------------------



# In[ ]:


# ===========================================================================

# === 8. PHASE 3 — SUMMARY CELL MASKS ===


# In[ ]:


# ===========================================================================

# ============================================================================
# PHASE 3: GENERATE SUMMARY FIGURES
# ============================================================================
print(f"\n{'='*70}")
print("PHASE 3: Generating Summary Figures")
print(f"{'='*70}\n")

summary_out = Path(root_dir) / "Summary_Figures"
summary_out.mkdir(parents=True, exist_ok=True)

# ============================================================================
# SUMMARY CELL MASKS
# ============================================================================

n_cols = 4  

for group_name in [group_1_keyword, group_2_keyword]:
    masks_data = fig_masks[group_name]
    n_cells = len(masks_data)

    if n_cells == 0:
        print(f"  No cells found for {group_name}, skipping...")
        continue

    # Calculate rows needed for 4 columns
    n_rows = math.ceil(n_cells / n_cols)

    print(f"  -> Creating Summary Cell Masks for {group_name}: {n_cells} cells ({n_rows} rows x {n_cols} cols)")

    fig_summary, axs_summary = plt.subplots(n_rows, n_cols, figsize=(A4_W, A4_W * n_rows / n_cols))

    # Handle edge cases for subplot array dimensions
    if n_rows == 1 and n_cols == 1:
        axs_summary = np.array([[axs_summary]])
    elif n_rows == 1:
        axs_summary = axs_summary.reshape(1, -1)
    elif n_cols == 1:
        axs_summary = axs_summary.reshape(-1, 1)

    import re
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', s)]
    
    masks_data_sorted = sorted(masks_data, key=lambda x: natural_sort_key(x.get('cell_name', '')))

    for idx, cell_data in enumerate(masks_data_sorted):
        row = idx // n_cols
        col = idx % n_cols
        ax = axs_summary[row, col]

        # Plot erosion contourS
        for cnt in cell_data['e_contours_canvas']:
            ax.plot(cnt[:,0], cnt[:,1], color='gray', lw=0.2, alpha=0.2)

        # Plot cell border 
        ax.plot(cell_data['border_canvas'][:,0], cell_data['border_canvas'][:,1],
                color='black', lw=0.5)

        # Plot monomers in blue, dimers in red
        x_c = cell_data['x_canvas']
        y_c = cell_data['y_canvas']
        is_mono = cell_data['is_monomer']
        is_dim = cell_data['is_dimer']

        ax.scatter(x_c[is_mono], y_c[is_mono], s=0.5, c='blue', alpha=0.6, label='Monomer')
        ax.scatter(x_c[is_dim], y_c[is_dim], s=0.5, c='red', alpha=0.6, label='Dimer')

        # Plot center
        ax.scatter(cell_data['offset'], cell_data['offset'], c='yellow', s=30,
                   marker='*', edgecolor='black', linewidth=0.5, zorder=10)

        # Set limits 
        padding = 0.3
        ax.set_xlim(cell_data['b_min_x'] - padding, cell_data['b_max_x'] + padding)
        ax.set_ylim(cell_data['b_min_y'] - padding, cell_data['b_max_y'] + padding)
        ax.set_aspect('equal')
        # is_dim here is cell_data['is_dimer'], which is already the visual (pair-based) mask
        ax.set_title(f"{cell_data['cell_name']}\nM:{is_mono.sum()} D:{int(is_dim.sum())}", fontsize=9)
        ax.axis('off')

    # Hide empty subplots
    for idx in range(n_cells, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axs_summary[row, col].axis('off')

    fig_summary.suptitle(f"Cell Masks Summary - {group_name}\n(Blue=Monomers, Red=Dimers)",
                         fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_summary.savefig(summary_out / f"Summary_Cell_Masks_{group_name}.pdf", dpi=150)
    plt.close(fig_summary)
    print(f"    Saved: Summary_Cell_Masks_{group_name}.pdf")

print("\n✓ Summary cell masks saved")

# ----------------------------------------



# In[ ]:


# ===========================================================================

print("✓ Cell mask summary figures saved")


# === 9. PHASE 3 — RADIAL COMPARISON PLOTS ===


# In[ ]:


# ===========================================================================

# ============================================================================
# SUMMARY RADIAL PLOTS
# ============================================================================
print("\n  -> Creating Summary Radial Comparison Plots...")

# MONOMER RADIAL COMPARISON 
fig_radial_mono, axs_rm = plt.subplots(3, 3, figsize=(A4_W, A4_W))

categories_mono = [
    ('Total Monomers', radial_monomer_total),
    ('Glycosylated Monomers', radial_monomer_glyco),
    ('Non-Glyco Monomers', radial_monomer_non_glyco)
]

for col, (title, data_dict) in enumerate(categories_mono):
    # Row 0: Group 1 (Stim)
    if data_dict[group_1_keyword]:
        arr = np.array(data_dict[group_1_keyword])
        mean_vals = np.mean(arr, axis=0)
        # Stim = Orange
        axs_rm[0, col].plot(standard_norm_radius, mean_vals, color='orange', lw=2, label=group_1_keyword)
    axs_rm[0, col].set_title(f"{title}\n{group_1_keyword}")

    # Row 1: Group 2 (Non-stim)
    if data_dict[group_2_keyword]:
        arr = np.array(data_dict[group_2_keyword])
        mean_vals = np.mean(arr, axis=0)
        # Non-stim = Blue
        axs_rm[1, col].plot(standard_norm_radius, mean_vals, color='blue', lw=2, label=group_2_keyword)
    axs_rm[1, col].set_title(f"{title}\n{group_2_keyword}")

    # Row 2: Overlay
    if data_dict[group_1_keyword]:
        arr = np.array(data_dict[group_1_keyword])
        mean_vals = np.mean(arr, axis=0)
        axs_rm[2, col].plot(standard_norm_radius, mean_vals, color='orange', lw=2, label=group_1_keyword)
    if data_dict[group_2_keyword]:
        arr = np.array(data_dict[group_2_keyword])
        mean_vals = np.mean(arr, axis=0)
        axs_rm[2, col].plot(standard_norm_radius, mean_vals, color='blue', lw=2, label=group_2_keyword)
    axs_rm[2, col].set_title(f"{title}\nOverlay")
    axs_rm[2, col].set_xlabel("Normalized Radius")

for r in range(3):
    for c in range(3):
        format_radial_ax(axs_rm[r, c])
fig_radial_mono.suptitle("MONOMER Radial Density - Group Comparison", fontsize=10, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
_h, _l, _seen = [], [], set()
for ax in fig_radial_mono.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in _seen:
            _h.append(h); _l.append(l); _seen.add(l)
if _h:
    fig_radial_mono.legend(_h, _l, loc='upper left', fontsize=8, frameon=True, bbox_to_anchor=(0.01, 0.96))
fig_radial_mono.savefig(summary_out / "Summary_Radial_Monomers.pdf", dpi=150)
plt.close(fig_radial_mono)

# DIMER RADIAL COMPARISON 
fig_radial_dimer, axs_rd = plt.subplots(3, 3, figsize=(A4_W, A4_W))

categories_dimer = [
    ('Total Dimers', radial_dimer_total),
    ('Glycosylated Dimers', radial_dimer_glyco),
    ('Non-Glyco Dimers', radial_dimer_non_glyco)
]

for col, (title, data_dict) in enumerate(categories_dimer):
    # Row 0: Group 1 (Stim)
    if data_dict[group_1_keyword]:
        arr = np.array(data_dict[group_1_keyword])
        mean_vals = np.mean(arr, axis=0)
        axs_rd[0, col].plot(standard_norm_radius, mean_vals, color='darkorange', lw=2, label=group_1_keyword)
    axs_rd[0, col].set_title(f"{title}\n{group_1_keyword}")

    # Row 1: Group 2 (Non-stim)
    if data_dict[group_2_keyword]:
        arr = np.array(data_dict[group_2_keyword])
        mean_vals = np.mean(arr, axis=0)
        axs_rd[1, col].plot(standard_norm_radius, mean_vals, color='darkblue', lw=2, label=group_2_keyword)
    axs_rd[1, col].set_title(f"{title}\n{group_2_keyword}")

    # Row 2: Overlay
    if data_dict[group_1_keyword]:
        arr = np.array(data_dict[group_1_keyword])
        mean_vals = np.mean(arr, axis=0)
        axs_rd[2, col].plot(standard_norm_radius, mean_vals, color='darkorange', lw=2, label=group_1_keyword)
    if data_dict[group_2_keyword]:
        arr = np.array(data_dict[group_2_keyword])
        mean_vals = np.mean(arr, axis=0)
        axs_rd[2, col].plot(standard_norm_radius, mean_vals, color='darkblue', lw=2, label=group_2_keyword)
    axs_rd[2, col].set_title(f"{title}\nOverlay")
    axs_rd[2, col].set_xlabel("Normalized Radius")

for r in range(3):
    for c in range(3):
        format_radial_ax(axs_rd[r, c])
fig_radial_dimer.suptitle("DIMER Radial Density - Group Comparison", fontsize=10, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
_h, _l, _seen = [], [], set()
for ax in fig_radial_dimer.axes:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in _seen:
            _h.append(h); _l.append(l); _seen.add(l)
if _h:
    fig_radial_dimer.legend(_h, _l, loc='upper left', fontsize=8, frameon=True, bbox_to_anchor=(0.01, 0.96))
fig_radial_dimer.savefig(summary_out / "Summary_Radial_Dimers.pdf", dpi=150)
plt.close(fig_radial_dimer)

# ============================================================================
# STIM vs NON-STIM OVERLAPPED - Monomers, Dimers, Total
# ============================================================================
print(f"  -> Creating {group_1_keyword} vs {group_2_keyword} Overlapped Comparison (Monomers, Dimers, Total)...")

fig_overlap, axs_ov = plt.subplots(1, 3, figsize=(A4_W, A4_H * 0.27))

# Column 0: Total Monomers - Stim vs Non-Stim
if radial_monomer_total[group_1_keyword]:
    arr = np.array(radial_monomer_total[group_1_keyword])
    mean_vals = np.mean(arr, axis=0)
    axs_ov[0].plot(standard_norm_radius, mean_vals, color='orange', lw=2, label=f'{group_1_keyword} (n={len(arr)})')
if radial_monomer_total[group_2_keyword]:
    arr = np.array(radial_monomer_total[group_2_keyword])
    mean_vals = np.mean(arr, axis=0)
    axs_ov[0].plot(standard_norm_radius, mean_vals, color='blue', lw=2, label=f'{group_2_keyword} (n={len(arr)})')
axs_ov[0].set_title("Total Monomers", fontsize=10)
axs_ov[0].set_xlabel("0 = center, 1 = edge")
axs_ov[0].set_ylabel("Density (locs/µm²)")
format_radial_ax(axs_ov[0])

# Column 1: Total Dimers - Stim vs Non-Stim
if radial_dimer_total[group_1_keyword]:
    arr = np.array(radial_dimer_total[group_1_keyword])
    mean_vals = np.mean(arr, axis=0)
    axs_ov[1].plot(standard_norm_radius, mean_vals, color='orange', lw=2, label=f'{group_1_keyword} (n={len(arr)})')
if radial_dimer_total[group_2_keyword]:
    arr = np.array(radial_dimer_total[group_2_keyword])
    mean_vals = np.mean(arr, axis=0)
    axs_ov[1].plot(standard_norm_radius, mean_vals, color='blue', lw=2, label=f'{group_2_keyword} (n={len(arr)})')
axs_ov[1].set_title("Total Dimers", fontsize=10)
axs_ov[1].set_xlabel("0 = center, 1 = edge")
axs_ov[1].set_ylabel("Density (locs/µm²)")
format_radial_ax(axs_ov[1])

# Column 2: Total EGFR - Stim vs Non-Stim
if radial_total_egfr[group_1_keyword]:
    arr = np.array(radial_total_egfr[group_1_keyword])
    mean_vals = np.mean(arr, axis=0)
    axs_ov[2].plot(standard_norm_radius, mean_vals, color='orange', lw=2, label=f'{group_1_keyword} (n={len(arr)})')
if radial_total_egfr[group_2_keyword]:
    arr = np.array(radial_total_egfr[group_2_keyword])
    mean_vals = np.mean(arr, axis=0)
    axs_ov[2].plot(standard_norm_radius, mean_vals, color='blue', lw=2, label=f'{group_2_keyword} (n={len(arr)})')
axs_ov[2].set_title("Total EGFR", fontsize=10)
axs_ov[2].set_xlabel("0 = center, 1 = edge")
axs_ov[2].set_ylabel("Density (locs/µm²)")
format_radial_ax(axs_ov[2])

fig_overlap.suptitle(f"Radial Distribution Comparison: {group_1_keyword} vs {group_2_keyword}", fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.94])
_h, _l, _seen = [], [], set()
for ax in axs_ov:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in _seen:
            _h.append(h); _l.append(l); _seen.add(l)
if _h:
    fig_overlap.legend(_h, _l, loc='upper left', fontsize=8, frameon=True, bbox_to_anchor=(0.01, 0.94))
fig_overlap.savefig(summary_out / "Summary_Overlapped_Stim_NonStim.pdf", dpi=150)
plt.close(fig_overlap)
print("    Saved: Summary_Overlapped_Stim_NonStim.pdf")

print("\n✓ Radial comparison plots saved")

# ----------------------------------------



# In[ ]:


# ===========================================================================

# === 10. PHASE 3 — TOP 5 RADIAL SUMMARY & BAR CHARTS ===


# In[ ]:


# ===========================================================================

# ============================================================================
# FIGURE 2: TOP 5 GLYCAN CLASS RADIAL SUMMARY (PER-GROUP)
# ============================================================================
print("  -> Creating Top 5 Glycan Class Radial Summaries...")

def plot_top5_radial_summary(top5_radial_data, classes_to_plot, title_prefix, filename):
    """Plot Top 5 radial summary with stim/non-stim overlay for given classes."""
    if not classes_to_plot:
        return

    n_cls = len(classes_to_plot)
    fig, axs = plt.subplots(1, n_cls, figsize=(A4_W, A4_H * 0.22))

    # Handle 1-class case
    if n_cls == 1:
        axs = [axs]

    for col, cls in enumerate(classes_to_plot):
        if cls in top5_radial_data[group_1_keyword] and top5_radial_data[group_1_keyword][cls]:
            arr = np.array(top5_radial_data[group_1_keyword][cls])
            mean_vals = np.mean(arr, axis=0)
            axs[col].plot(standard_norm_radius, mean_vals, color='orange', lw=1.5,
                          label=f'{group_1_keyword} (n={len(arr)})', clip_on=True)

        if cls in top5_radial_data[group_2_keyword] and top5_radial_data[group_2_keyword][cls]:
            arr = np.array(top5_radial_data[group_2_keyword][cls])
            mean_vals = np.mean(arr, axis=0)
            axs[col].plot(standard_norm_radius, mean_vals, color='blue', lw=1.5,
                          label=f'{group_2_keyword} (n={len(arr)})', clip_on=True)

        axs[col].set_title(cls, fontsize=10)
        axs[col].set_xlabel("0 = center, 1 = edge")
        axs[col].set_ylabel("Density (locs/µm²)" if col == 0 else "")
        format_radial_ax(axs[col])

    fig.suptitle(title_prefix, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _h, _l, _seen = [], [], set()
    for ax in axs:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in _seen:
                _h.append(h); _l.append(l); _seen.add(l)
    if _h:
        fig.legend(_h, _l, loc='upper left', fontsize=8, frameon=True, bbox_to_anchor=(0.01, 0.93))

    fig.savefig(summary_out / filename, dpi=150)
    plt.close(fig)
    print(f"    Saved: {filename}")

plot_top5_radial_summary(radial_top5_monomer, top_5_classes,
    f"MONOMER - Top 5 Glycan Classes ({group_1_keyword} vs {group_2_keyword})",
    "Summary_Radial_Top5_Monomers.pdf")
plot_top5_radial_summary(radial_top5_dimer, top_5_classes,
    f"DIMER - Top 5 Glycan Classes ({group_1_keyword} vs {group_2_keyword})",
    "Summary_Radial_Top5_Dimers.pdf")

# ============================================================================
# NEW FIGURE 3: TOP 5 GLYCAN CLASS MEAN DISTRIBUTION BAR CHART (PER-GROUP)
# ============================================================================
print("  -> Creating Top 5 Glycan Class Bar Charts...")

def plot_top5_bar_chart(classes_to_plot, stats_data, title, filename):
    """Plot bar chart comparing stim vs non-stim for given classes."""
    if not classes_to_plot:
        return

    fig_bar, ax = plt.subplots(1, 1, figsize=(A4_W, A4_H * 0.28))
    x_positions = np.arange(len(classes_to_plot))
    width = 0.35

    means_g1 = [np.mean(stats_data[group_1_keyword][c]) if stats_data[group_1_keyword].get(c) else 0 for c in classes_to_plot]
    means_g2 = [np.mean(stats_data[group_2_keyword][c]) if stats_data[group_2_keyword].get(c) else 0 for c in classes_to_plot]

    rects1 = ax.bar(x_positions - width/2, means_g1, width, label=group_1_keyword,
                    color='moccasin', alpha=0.8, edgecolor='orange', lw=1)
    rects2 = ax.bar(x_positions + width/2, means_g2, width, label=group_2_keyword,
                    color='skyblue', alpha=0.8, edgecolor='blue', lw=1)

    ax.bar_label(rects1, fmt='%.3f', padding=3, fontsize=8)
    ax.bar_label(rects2, fmt='%.3f', padding=3, fontsize=8)

    for i, cls in enumerate(classes_to_plot):
        pts1 = stats_data[group_1_keyword].get(cls, [])
        if pts1:
            ax.scatter(np.full(len(pts1), x_positions[i] - width/2), pts1,
                       color='orange', s=20, alpha=0.7, zorder=5)
        pts2 = stats_data[group_2_keyword].get(cls, [])
        if pts2:
            ax.scatter(np.full(len(pts2), x_positions[i] + width/2), pts2,
                       color='blue', s=20, alpha=0.7, zorder=5)

    ax.set_ylabel('Occurrences per µm²')
    ax.set_title(title, fontsize=10)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(classes_to_plot, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    fig_bar.savefig(summary_out / filename, dpi=150)
    plt.close(fig_bar)
    print(f"    Saved: {filename}")

plot_top5_bar_chart(top_5_classes, top5_monomer_stats,
    f"MONOMER Top 5 Glycan Classes ({group_1_keyword} vs {group_2_keyword})",
    "Summary_Top5_BarChart_Monomer.pdf")
plot_top5_bar_chart(top_5_classes, top5_dimer_stats,
    f"DIMER Top 5 Glycan Classes ({group_1_keyword} vs {group_2_keyword})",
    "Summary_Top5_BarChart_Dimer.pdf")

print("\n✓ Top 5 radial summaries and bar charts saved")

# ----------------------------------------



# In[ ]:


# ===========================================================================

# === 11. PHASE 3 — CIRCULAR DIFFERENCE PLOTS ===


# In[ ]:


# ===========================================================================

# ============================================================================
# CIRCULAR DIFFERENCE PLOTS (Simple_Difference style)
# ============================================================================
print("  -> Creating Circular Difference Plots (Top 5 classes)...")

n_layers = 20

def plot_circular_diff(ax, layer_values, title, cmap, vmin, vmax):
    """Plot concentric rings colored by difference values."""
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    theta = np.linspace(0, 2*np.pi, 100)
    nl = len(layer_values)
    for i in range(nl):
        r_inner = i / nl
        r_outer = (i + 1) / nl
        x_inner = r_inner * np.cos(theta)
        y_inner = r_inner * np.sin(theta)
        x_outer = r_outer * np.cos(theta)
        y_outer = r_outer * np.sin(theta)
        color = cmap(norm(layer_values[i]))
        ax.fill(np.concatenate([x_outer, x_inner[::-1]]),
                np.concatenate([y_outer, y_inner[::-1]]),
                color=color, edgecolor='gray', linewidth=0.3, alpha=0.95)
    # Layer labels
    for label_layer in [0, 4, 9, 14, 19]:
        if label_layer < nl:
            r_mid = (label_layer + 0.5) / nl
            ax.text(r_mid, 0, str(label_layer), ha='center', va='center', fontsize=6,
                    fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', facecolor='white',
                    edgecolor='gray', linewidth=0.5, alpha=0.8))
    ax.text(0, 0, 'C', ha='center', va='center', fontsize=8, fontweight='bold',
            bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=1))
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10, pad=10)

cmap_div = LinearSegmentedColormap.from_list('blue_red',
    ['#08519c', '#4292c6', '#deebf7', '#ffffff', '#fee0d2', '#fc9272', '#de2d26'])

def generate_circular_diff_figure(radial_data, classes_to_plot, data_type, group_label, filename):
    """Generate a circular difference figure for given classes."""
    if not classes_to_plot:
        return

    n_classes = len(classes_to_plot)
    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 5.5))
    if n_classes == 1:
        axes = [axes]

    fig.suptitle(f'Radial Distribution Differences: {data_type} ({group_label} Top {n_classes})\n'
                 f'BLUE = {group_2_keyword} Higher  |  RED = {group_1_keyword} Higher\n'
                 f'({n_layers} layers: 0=center \u2192 {n_layers-1}=edge)',
                 fontweight='bold')

    # Compute differences for all classes to get shared colorscale
    all_diffs = []
    diff_profiles = {}
    for cls in classes_to_plot:
        stim_arrays = radial_data[group_1_keyword].get(cls, [])
        non_stim_arrays = radial_data[group_2_keyword].get(cls, [])

        if stim_arrays:
            stim_mean = np.mean(np.array(stim_arrays), axis=0)
        else:
            stim_mean = np.zeros(normalization_points)

        if non_stim_arrays:
            non_stim_mean = np.mean(np.array(non_stim_arrays), axis=0)
        else:
            non_stim_mean = np.zeros(normalization_points)

        x_old = np.linspace(0, 1, len(stim_mean))
        x_new = np.linspace(0, 1, n_layers)
        stim_layers = np.interp(x_new, x_old, stim_mean)
        non_stim_layers = np.interp(x_new, x_old, non_stim_mean)

        diff = stim_layers - non_stim_layers
        diff_profiles[cls] = diff
        all_diffs.extend(diff)

    max_abs = np.max(np.abs(all_diffs)) if all_diffs and np.max(np.abs(all_diffs)) > 0 else 1.0

    for idx, cls in enumerate(classes_to_plot):
        plot_circular_diff(axes[idx], diff_profiles[cls], cls, cmap_div, -max_abs, max_abs)

    # Add colorbar
    cax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    norm_cb = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    sm = ScalarMappable(cmap=cmap_div, norm=norm_cb)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label(f'\u2190 {group_2_keyword} Higher | {group_1_keyword} Higher \u2192',
                   fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.91, 0.88])
    out_path = summary_out / filename
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"    Saved: {filename}")

generate_circular_diff_figure(radial_top5_monomer, top_5_classes,
    "Monomer", f"{group_2_keyword} Mono", "Simple_Difference_Monomer.pdf")
generate_circular_diff_figure(radial_top5_dimer, top_5_classes,
    "Dimer", f"{group_2_keyword} Mono", "Simple_Difference_Dimer.pdf")

print("\n✓ Circular difference plots saved")

# ----------------------------------------



# In[ ]:


# ===========================================================================

# === 12. SAVE SUMMARY STATISTICS & RADIAL CSVs ===


# In[ ]:


# ===========================================================================

# ============================================================================
# SAVE SUMMARY STATISTICS TABLE
# ============================================================================
print("  -> Saving Summary Statistics...")

summary_stats = []
for group in [group_1_keyword, group_2_keyword]:
    for cell_stat in cell_statistics[group]:
        cell_stat['group'] = group
        summary_stats.append(cell_stat)

if summary_stats:
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(summary_out / "Summary_Cell_Statistics.csv", index=False)

# ============================================================================
# SAVE RADIAL PROFILES TO CSV (FOR EXTERNAL PLOTTING)
# ============================================================================
print("  -> Saving Radial Profile CSVs...")
radial_csv_dir = summary_out / "Radial_CSV_Data"
radial_csv_dir.mkdir(exist_ok=True)

def save_radial_csv(data_dict, prefix):
    for group, arrays in data_dict.items():
        if arrays:
            df = pd.DataFrame(arrays)
            df.columns = [f"Bin_{i}" for i in range(df.shape[1])]
            safe_group = group.replace(" ", "_")
            fname = f"Radial_{prefix}_{safe_group}.csv"
            df.to_csv(radial_csv_dir / fname, index=False)

save_radial_csv(radial_monomer_total, "Monomer_Total")
save_radial_csv(radial_monomer_glyco, "Monomer_Glyco")
save_radial_csv(radial_monomer_non_glyco, "Monomer_NonGlyco")
save_radial_csv(radial_dimer_total, "Dimer_Total")
save_radial_csv(radial_dimer_glyco, "Dimer_Glyco")
save_radial_csv(radial_dimer_non_glyco, "Dimer_NonGlyco")
save_radial_csv(radial_total_egfr, "Total_EGFR")

def save_top5_radial_csv(top5_dict, prefix):
    for group, class_dict in top5_dict.items():
        for class_name, arrays in class_dict.items():
            if arrays:
                df = pd.DataFrame(arrays)
                df.columns = [f"Bin_{i}" for i in range(df.shape[1])]
                safe_group = group.replace(" ", "_")
                safe_class = class_name.replace(" ", "_").replace("'", "").replace('"', "")
                fname = f"Radial_{prefix}_{safe_class}_{safe_group}.csv"
                df.to_csv(radial_csv_dir / fname, index=False)

save_top5_radial_csv(radial_top5_monomer, "Top5_Monomer")
save_top5_radial_csv(radial_top5_dimer, "Top5_Dimer")

print(f"Radial CSVs saved to: {radial_csv_dir}")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"Summary figures saved to: {summary_out}")
print(f"{'='*70}\n")

print("\n✓ Analysis complete — all files saved")

# ----------------------------------------

print("✓ Statistics and radial CSVs saved — pipeline complete!")

    