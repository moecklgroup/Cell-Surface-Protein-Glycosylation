# -*- coding: utf-8 -*-
"""
This code is a modified version of GlyCO to count the number of glycans around a target protein.
We assume that RESI was used to specifically label a certain glycan, and that all the glycan localizations are in one .hdf5 file
@author: modified by lsison, original GlyCo code by dmoonnu
"""
#import packages
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm
from collections import Counter
from matplotlib import pyplot as plt
import json
from datetime import datetime
import matplotlib
import yaml
#hide grids from all th axes
plt.rcParams["axes.grid"] = False

#set the font to arial
plt.rcParams['font.family'] = 'arial'
#Decide on whether you want to save the analysis results
save = True
#Boolean for controlling the representation of the glycan maps depending on cell sizes. 
#If you have a small cell, turn this on so that the map is represented in the middle of the plot and zoomed in.
zoom = True
#Set font size required on plots
label_font_size = 10
title_font_size = 10
tick_font_size = 10
FIGFORMAT = '.pdf'
#Is protein preset in your dataset
protein_present = True
#Enter the name of protein as in the file
protein = "EGFR"

#Enter the name of the glycan channel as in the file
glycan = 'DBCO'
 

# Set the graphics backend to Qt
matplotlib.use('Qt5Agg')
# kEY TO LOOK IN THE YAML FILE
key_for_area = "Total Picked Area (um^2)"

#Define search radii in nm
glycan_search_radius = 30
protein_search_radius = 36

#Path to the localization files folder with the glycan and protein localization files.
#Add/delete pathlocs variables to account for the number of cells to analyze
pathlocs1 = r"C:\Users\nsison\Documents\2025-08-01_Linus In Situ Glycosylation\MCF10A\EGFR\FOV2\CELL1\RESI"
pathlocs2 = r"C:\Users\nsison\Documents\2025-08-01_Linus In Situ Glycosylation\MCF10A\EGFR\FOV2\CELL2\RESI"
pathlocs3 = r"C:\Users\nsison\Documents\2025-08-01_Linus In Situ Glycosylation\MCF10A\EGFR\FOV2\CELL3\RESI"
pathlocs4 = r"C:\Users\nsison\Documents\2025-08-01_Linus In Situ Glycosylation\MCF10A\EGFR\FOV2\CELL4\RESI"

pathfiles = [pathlocs1, pathlocs2, pathlocs3, pathlocs4]

#Initialize master lists for plotting
all_protein_protein_counts = []
all_protein_protein_totals = []
all_protein_glycan_counts = []
all_protein_glycan_totals = []

all_monomer_pgcounts = []
all_monomer_pgtotal = []
all_dimer_pgcounts = []
all_dimer_pgtotal = []

#%% Helper Functions
#Count the number of neighbors for a target protein
def countNeighbors(neighbors):
    counts = Counter(len(lst) for lst in neighbors.values())
    counts = np.transpose(np.array(sorted(counts.items())))
    total_counts = np.sum(counts[1])
    return counts, total_counts

for pathLocsPoints in pathfiles:
    #Start of Loop
    localization_folder = Path(pathLocsPoints)
    #Search for yaml files in the folder (yaml files are created as a metadata for any reslts from picasso)
    #Fetch the first file. Becoz the aim is to get the pick area of the FOV under analysis. This pick area is same for all channels.
    yaml_file = (list(localization_folder.glob("*.yaml")))[0]
    #open the yaml file
    with open(yaml_file,'r') as file:
        #load the data. Multiple documents are present in a single file. 
        documents=yaml.safe_load_all(file)
        for info in documents:
            if isinstance(info, dict) and key_for_area in info:
                area_of_cell = np.float32(info[key_for_area])
    
    
    #%%HDF5 handling
    
    #iterating to look for hdf5 files in the folder
    for file in localization_folder.glob('**/*.hdf5'):   
        print(file)
    
    print("Files loaded successfully")
    #create a dictionary of the dataset
    data_dict={}
    library = []
    #Looping to create a dictionary with the dataset.
    for file in localization_folder.glob("**/*.hdf5"):
        #Get the file stem for naming the keys
        file_name = (file.stem.split("_")[0]).upper()
        #read the data 
        data = pd.read_hdf(file, key ='locs')
        #append the dict 
        data_dict[file_name] = data
        library.append(file_name)
        
    
    #%%HDF5
    #Dictionary for storing the neighbors with core point as the key
    neighbor_master={}
    #Dictionary to store the neighbors with core point as the key and each value as the list of tuple pair with the distance to to the considered core
    distance_indexed_neighbor={}
    #iterate thru each dataset
    
    #Dictionaries to store neighbors of each protein
    protein_protein_neighbors = {}
    protein_glycan_neighbors = {}
    
    #test counter
    
    for df_key in data_dict:  
        if protein_present:
            df_key = protein
        df_of_interest_key = df_key  
        com_name = f"neighbors_of_{df_key}" #center of mass name to be used as  the key in the dictionaries
        neighbor_master[com_name] = {}
        distance_indexed_neighbor[com_name]={}
        # Create a tree Excluding the key of interest. This is a note for myself. In later version of the code, we decided to look into the same channel
        # Convert each DataFrame's points to KDTree format, except the DataFrame of interest
        trees = {key: KDTree((df[['x', 'y']]*130).values) for key, df in data_dict.items()}
        # Keep track of assigned points in each dataset
        assigned_points = {key: {} for key in trees}  # Store closest distance and index associated
        # Retrieve the DataFrame of interest
        df_of_interest = data_dict[df_of_interest_key]
        # Iterate through points in the DataFrame of interest and find neighbors in other DataFrames
        #print(f"Current core is {df_key}\n\n")
        for row_index_of_com, column in tqdm(df_of_interest.iterrows(), desc=f"Searching for neighbors of {df_of_interest_key}"):
            #Go to first point in the dataframe chosen
            x1, y1 = column['x']*130, column['y']*130
               #now look for neighbors standing at this point
            # Check for neighbors in all other DataFrames using their KDTree. Iterate through each tree.
            for current_family, current_family_members in trees.items(): #Current family = key and curent family members=KDtree
                #Set search radius (different for proteins and glycans)
                if current_family == glycan:
                    radius = glycan_search_radius           
                elif current_family == protein:
                    radius = protein_search_radius
                
                #Generates a list of indices coresponding to the dataframe 
                indices = current_family_members.query_ball_point([x1, y1], r=radius)
                #Preventing same point as the neighbor of itself
                filtered_indices = [num for num in indices if df_key != current_family or num != row_index_of_com]
                    
                #Record nearest neighbors for proteins and glycans
                if current_family == glycan:
                    protein_glycan_neighbors[row_index_of_com] = filtered_indices   
                elif current_family == protein:
                    protein_protein_neighbors[row_index_of_com] = filtered_indices
                    
        if protein_present:
            break 
                
    #Count the number of glycans that are within the glycan-protein search radius for each protein
    #Record the total number of glycans that have a neighboring protein
    protein_glycan_counts, protein_glycan_total = countNeighbors(protein_glycan_neighbors)
    
    #Count the number of proteins that are within the protein-protein search radius for each protein
    #Record the total number of proteins
    protein_protein_counts, protein_protein_total = countNeighbors(protein_protein_neighbors)
    
    #%% Monomer-Dimer Analysis
    #For Monomers
    #Get the indices of protein monomers
    monomer_protein_protein_neighbors = {key: value for key, value in protein_protein_neighbors.items() if len(value) == 0}
    
    #Get glycan count entries for monomers
    keys_of_interest = monomer_protein_protein_neighbors.keys()
    monomer_protein_glycan_neighbors = {key: protein_glycan_neighbors[key] for key in keys_of_interest if key in protein_glycan_neighbors}
    
    #Count the number of glycans around each protein monomer
    monomer_protein_glycan_counts, monomer_protein_glycan_total = countNeighbors(monomer_protein_glycan_neighbors)
    
    #For Dimers
    #Get the indices of protein dimers
    dimer_protein_protein_neighbors = {key: value[0] for key, value in protein_protein_neighbors.items() if len(value) == 1}
    
    #Get glycan count entries for dimers
    keys_of_interest = dimer_protein_protein_neighbors.keys()
    dimer_protein_glycan_neighbors = {key: protein_glycan_neighbors[key] for key in keys_of_interest if key in protein_glycan_neighbors}
    
    checked_neighbors = []
    unique_dpg_neighbors = {}
    
    #Loop through list of dimers and record unique sugars in the radius of a dimer pair
    for idx in dimer_protein_protein_neighbors.keys():
        pair_idx = dimer_protein_protein_neighbors[idx]
        
        #Record registered pairs/dimers
        if pair_idx not in dimer_protein_protein_neighbors.keys():
            checked_neighbors.append(idx)
            checked_neighbors.append(pair_idx)
        
        #If pair has not been checked before, create a list of the unique glycans to each protein unit in the dimer.
        elif idx not in checked_neighbors or pair_idx not in checked_neighbors:
            unique_dimer_neighbors = []
            
            for element in dimer_protein_glycan_neighbors[idx]:
                unique_dimer_neighbors.append(element)
            for element in dimer_protein_glycan_neighbors[pair_idx]:
                if element not in unique_dimer_neighbors:
                    unique_dimer_neighbors.append(element)
            
            unique_dpg_neighbors[idx] = unique_dimer_neighbors
            
            checked_neighbors.append(idx)
            checked_neighbors.append(pair_idx)
    
    #Count the number of UNIQUE glycans that neighbor each protein dimer complex
    dimer_protein_glycan_counts, dimer_protein_glycan_total = countNeighbors(unique_dpg_neighbors)
    
    #Assign all relevant files to master lists
    protein_glycan_counts = monomer_protein_glycan_counts[:,:11] + dimer_protein_glycan_counts[:,:11]
    all_protein_protein_counts.append(protein_protein_counts)
    all_protein_protein_totals.append(protein_protein_total)
    all_protein_glycan_counts.append(monomer_protein_glycan_counts[:,:11] + dimer_protein_glycan_counts[:,:11])
    all_protein_glycan_totals.append(monomer_protein_glycan_total + dimer_protein_glycan_total)
    
    all_monomer_pgcounts.append(monomer_protein_glycan_counts)
    all_monomer_pgtotal.append(monomer_protein_glycan_total)
    
    all_dimer_pgcounts.append(dimer_protein_glycan_counts)
    all_dimer_pgtotal.append(dimer_protein_glycan_total)
    

#%% Figure Plotting

#Plot number of sialic acids per EGFR monomer or dimer
plt.figure()
for (pg_counts, pg_total) in zip(all_protein_glycan_counts, all_protein_glycan_totals):
    plt.plot(pg_counts[0][0:10]/2, pg_counts[1][0:10]/pg_total, 'ko')
plt.xticks(np.arange(0,10))
plt.xlabel('Number of Sialic acids per EGFR monomer/dimer')
plt.ylabel('frequency')
plt.show()

#Plot histogram for the number of protein neighbors within the defined protein search radius
plt.figure()
for (pp_counts, pp_total) in zip(all_protein_protein_counts, all_protein_protein_totals):
    plt.plot(pp_counts[0][0:2], pp_counts[1][0:2]/pp_total, 'ko')
    plt.plot(pp_counts[0][2], np.sum(pp_counts[1][2:])/pp_total, 'ko')
plt.yticks(np.arange(0,1.1,0.1))
plt.xticks(np.arange(0,3))
plt.xlabel(f'Number of EGFR neighbors within a {protein_search_radius} nm radius')
plt.ylabel('frequency')
plt.show()

#Plot number of glycan localizations for each protein monomer
plt.figure()
for (monomer_pg_counts, monomer_pg_total) in zip(all_monomer_pgcounts, all_monomer_pgtotal):
    plt.plot(monomer_pg_counts[0][0:10], monomer_pg_counts[1][0:10]/monomer_pg_total, 'ko')
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,0.3,0.05))
plt.ylim(-0.01,0.27)
plt.xlabel('Number of Sialic acids per EGFR monomer')
plt.ylabel('frequency')
plt.show()

#Plot number of glycan localizations for each protein dimer
plt.figure()
for (dimer_pg_counts, dimer_pg_total) in zip(all_dimer_pgcounts, all_dimer_pgtotal):
    plt.plot(dimer_pg_counts[0][0:10], dimer_pg_counts[1][0:10]/dimer_pg_total, 'ko')
plt.xticks(np.arange(0,10))
plt.yticks(np.arange(0,0.3,0.05))
plt.ylim(-0.01,0.27)
plt.xlabel('Number of Sialic acids per EGFR dimer')
plt.ylabel('frequency')
plt.show()

