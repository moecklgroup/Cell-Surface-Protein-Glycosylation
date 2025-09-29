# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 09:26:13 2024
This code is intended to cluster the multidimensional super resolution data within a given radius. Localization data(cluster centers) should be saved
in a folder with the filenames starting with the corresponding lectin followed by underscores. (eg: "AAL_and whatever necesssary").
This folder should be given as the input in the variable "localization folder". Also set the radius. The outputs are multidimensional 
cluster information as a joson file, bar chart of the top classes, and the scatter plot of top x class location.  

Codes are supposed to be run at the level of a single cell. ie. Folder containing cluster centers of individual cells
@author: dmoonnu
"""
###############################CUSTOM RENDER###################################
#User Inputs

pixelsize =130
min_locs =1
frame_analysis = True
#Nena Factor for clustering
xnena = 2.35

#Is protein preset in your dataset
protein_present = True
#Enter the name of protein as in the file
protein = "EGFR"
consider_dimers = True
glyco_radius =37
#Decide on whether you want to save the analysis results
save = True
#Boolean for controlling the representation of the glycan maps depending on cell sizes. 
#If you have a small cell, turn this on so that the map is represented in the middle of the plot and zoomed in.
zoom = True
FIGFORMAT = '.pdf'
#%%Imports
from custom_picasso import postprocess, io
from custom_picasso import clusterer
import numpy as np
import os 
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json

plt.rcParams['font.family'] = 'arial'
plt.rcParams["axes.grid"] = False
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.facecolor"]   = "white"


#%% Fetching info from the parameter file about analysis folder
variables_from_parameter = {}
with open("parameter_file.json", "r", encoding="utf-8") as f:
    variables_from_parameter = json.load(f)

# get the values from the dictionary
localization_folder = Path(variables_from_parameter.get('localization_folder'))
print(localization_folder)


locs_file = []
for hdf5 in localization_folder.glob('*.hdf5'):
    locs_file.append(str(hdf5))
    
    
#%% <<<<<<<<<<<<Perform clustering using modified packages from picasso>>>>>>>>
#Adjust the path to the localization hdf5 file
cluster_data_location= localization_folder/"90_Custom SMLM Clustered"
cluster_data_location.mkdir(exist_ok=True)
centers_location =localization_folder/"90_Custom Centers"
centers_location.mkdir(exist_ok=True)
for file in locs_file:
    #locs_path = file
    #fetch data fromthe file
    locs, info = io.load_locs(file)
    #Calculate NeNA Precision
    nena= postprocess.nena(locs, info)
    
    #Assign radius
    radius_xy = xnena*nena[1]
    print(f"Clustering radius = {radius_xy*pixelsize}nm")
    
    #Strings for creating new filename
    directory, filename = os.path.split(file)
    name, extension = os.path.splitext(filename)
    suffix= f"min_loc-{min_locs}_{xnena}XNeNA"
    
    output_name = f"{name}_clustered_{suffix}{extension}"
    output_name_centers = f"{name}_centers_{suffix}{extension}"
    
    
    #clustering
    clustered_locs=clusterer.cluster(locs, radius_xy, min_locs, frame_analysis)
    new_info = {"Number of Clusters": len(np.unique(clustered_locs.group)),
                "Number of Clustered Locs": len(clustered_locs),
                "Min. cluster size": min_locs,
                "Performed basic frame analysis": frame_analysis,
                "Percentage parameter for BFA": "90%",
                "Number of bins for BFA": 100,
                "Clustering radius xy (nm)": float(radius_xy * pixelsize),
                "NeNA precision(nm)": float(nena[1]*130),
                "Basic Frame analysis status": "First and last inclusive, 100bins used",
                "Percentage of localization clustered": (len(clustered_locs)/len(locs))*100
            }
    info = info + [new_info]
    outputpath_locs = os.path.join(str(cluster_data_location), output_name)
    io.save_locs(outputpath_locs, clustered_locs, info)
    print("Calculating Centers...")
    cluster_centers = clusterer.find_cluster_centers(clustered_locs,130) #pixel size is not used incase of 2D data.
    outputpath_centers = os.path.join(str(centers_location), output_name_centers)
    io.save_locs(outputpath_centers, cluster_centers, info)
    

#%%Deleting the current variable to avaid any sort of overlaps
#TODO - Fix this as this is not a good practice
for var in list(globals().keys()):
    if var not in ["dictionaryNames",
                   "FIGFORMAT",
                   "protein_present",
                   "glyco_radius",
                   "save",
                   "zoom",
                   "protein",
                   "consider_dimers",
                   "variables_from_parameter","centers_location", "__builtins__", "__name__", "__doc__", "__package__"]:
        del globals()[var]
#%%
#############################GlyCo#############################################
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
from colorama import init, Style,Back
init(autoreset=True)
from collections import OrderedDict

#hide grids from all th axes
plt.rcParams["axes.grid"] = False

#set the font to arial
plt.rcParams['font.family'] = 'arial'
#Decide on whether you want to save the analysis results
#Boolean for controlling the representation of the glycan maps depending on cell sizes. 
#If you have a small cell, turn this on so that the map is represented in the middle of the plot and zoomed in.
zoom = True
#Set font size required on plots
label_font_size = 10
title_font_size = 10

tick_font_size = 10
FIGFORMAT = '.pdf'
 

# Set the graphics backend to Qt
matplotlib.use('Qt5Agg')
#kEY TO LOOK IN THE YAML FILE
key_for_area = "Total Picked Area (um^2)"
# Radius for neighborhood in nanometers ( Biologically relevant distance to find the neighbouring glycan)
glyco_radius = 37
#set the number of classes o be mapped
number_to_plot =5 #tp x to plot
#Location to the folder containing cluster centers
#pathLocsPoints = r"E:\2025-01-14_DNS006_MPZPM\FOV2\PAINT\Cell1 new clustering\90_Custom Centers"
pathLocsPoints = str(centers_location)
#convert the location to a path object
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
for file in localization_folder.glob("*.hdf5"):   
    print(file)

print("Files loaded successfully")
#create a dictionary of the dataset
data_dict={}
library = []
#Looping to create a dictionary with the dataset.
for file in localization_folder.glob("*.hdf5"):
    #Get the file stem for naming the keys
    file_name = (file.stem.split("_")[0]).upper()
    #read the data 
    data = pd.read_hdf(file, key ='locs')
    #append the dict 
    data_dict[file_name] = data
    library.append(file_name)

library=sorted(library)    

#%%HDF5
#Dictionary for storing the neighbors with core point as the key
neighbor_master={}
#Dictionary to store the neighbors with core point as the key and each value as the list of tuple pair with the distance to to the considered core
distance_indexed_neighbor={}

#iterate thru each dataset


for df_key in data_dict:  
    if protein_present:
        df_key = protein
    df_of_interest_key = df_key  
    com_name = f"neighbors_of_{df_key}" #center of mass name to be used as  the key in the dictionaries
    neighbor_master[com_name] = {}
    distance_indexed_neighbor[com_name]={}
    # Create a tree Excluding the key of interest. This is a note for myself. In later version of the code, we decided to look into the same channel
    # Convert each DataFrame's points to KDTree format, except the DataFrame of interest
    #trees = {key: KDTree((df[['x', 'y']]*130).values) for key, df in data_dict.items() if key != df_of_interest_key}
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
            #Generates a list of indices coresponding to the dataframe 
            indices = current_family_members.query_ball_point([x1, y1], r=glyco_radius)
            #Preventing same point as the neighbor of itself
            filtered_indices = [num for num in indices if df_key != current_family or num != row_index_of_com]

     
            #If indices exist ie. if neighbors exist for the current point from the main loop, these neighbors are stored to a dictionary with the 'id' of the point from main
            #loop as the key and list of row number as the value. Only if neighbours present then the if condition is satisfied and things are stored to the neighbor dictionary.
            if  filtered_indices:
                
                # Calculate scalar distances for each index # indices has the row number of the current family members
                dist_index_pairs = [(np.linalg.norm(current_family_members.data[idx] - [x1, y1]), idx) for idx in  filtered_indices]# creates a list of tuples with(distance, indices)
               
                # Sort by distance
                dist_index_pairs.sort(key=lambda x: x[0])
                
                # Find the closest point not already assigned or update if closer
                for distance, idx in dist_index_pairs: #idx is the index of current family member
                    # Only update if idx is unassigned or found closer
                    if idx not in assigned_points[current_family] or distance < assigned_points[current_family][idx][0]:#idx is the index of current family member
                        assigned_points[current_family][idx] = (distance, row_index_of_com)
                        if not protein_present:
                            break  # Only take the closest unassigned neighbor
    
    # After collecting the closest neighbors, populate neighbor_master with final assignments
    for neighbor_family, family_member in assigned_points.items():
        for idx, (distance, df_of_interest_idx) in family_member.items():
            neighbor_master[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append(f'{neighbor_family}_{idx}')    
            distance_indexed_neighbor[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append((f'{neighbor_family}_{idx}', distance))    
    if protein_present:
        break  # only neighbors of protein are searched. 
        
#%%Functions
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

           
#%%convert sub dictionary in a nested dictionary to a list of tuples

converted_dict_indexed = nested_dict_to_tuple_list(neighbor_master)
# Check for more closeness on this converted_dict_indexed
#%% Make into a single list of tuples without duplicates ie. (PSA_123, WGA_345) = (WGA_345, PSA_123). No more same combination after this point.
#alphabetic ordering is done here


list_of_tuples_without_duplicates=duplicates_removed_tuple_list(converted_dict_indexed)
    
#%%Check for duplicate single elements. Because a single element should not appear more than once in the entire cell. ie. one element
#should be part of only one "glycan structure". It shouldn't be linked to two different neighbor groups

#flat_list = [element for tup in list_of_tuples_without_duplicates for element in tup]

list_of_all = flatten_tuple_list(list_of_tuples_without_duplicates)
#duplicates = set([x for x in flat_list if flat_list.count(x)>1])
#%%Counter for dupllicates
element_counts = Counter(list_of_all) #counts how many times an element appears in the list.

frequency_counts = Counter(element_counts.values()) #Counts how many times 1,2,3,4,5,6 which are the values in the element count dictionary appears
    
# Define categories: 1, 2, 3, and "more than 3"
categories = ["1", "2", "3", ">3"]
counts = [
    frequency_counts[1],  # Count of elements appearing once
    frequency_counts[2],  # Count of elements appearing twice
    frequency_counts[3],  # Count of elements appearing thric
    
    sum(count for freq, count in frequency_counts.items() if freq > 3)  # Count of elements appearing more than thrice
]

# Plot the histogram
plt.figure(figsize=(8, 5))
plt.bar(categories, counts, color="skyblue")
plt.xlabel("Occurrence Frequency")
plt.ylabel("Number of Elements")
plt.title("Histogram of Element Frequencies")
plt.show()

#%%find duplicates
duplicates = find_duplicates_with_counts(list_of_all)



#%%#For each of the duplicates found, check its positions in the unique combination form cell list
#After a list of duplicates is made using the neighbor master, this list is used to remove these duplicates from the ditance
#indexed neighbor master based on who is closer. Later, initital filtering steps are redone on the distance indexed neighbor
#dictionary to remove the pair wise duplicates and arrage the classes in alphabetic order. 
        

dictionary_without_duplicates = eliminate_duplicates(distance_indexed_neighbor, duplicates)

#%%Remove distance from the updated dictionary to perform the initial filtering again                   

distance_removed_unduplicated_dictionary = remove_distance(dictionary_without_duplicates)                        
#%%perform the initial filtering agin to remove pairwise duplicates

final_list_of_tuples = nested_dict_to_tuple_list(distance_removed_unduplicated_dictionary) #key value pairs to a list tuple
#%%
final_pair_wise_duplicates_removed = duplicates_removed_tuple_list(final_list_of_tuples)
#%%
#result_list = [tuple(element.split("_")[0] for element in tup) for tup in final_pair_wise_duplicates_removed]
result_list = [tuple(sorted(element.split("_")[0] for element in tup)) for tup in final_pair_wise_duplicates_removed]
result_list[:] = [tuple(sorted(tup)) for tup in result_list]

if protein_present and consider_dimers:
    result_list = [t for t in result_list if t.count(protein) < 2]
    
#%% Define possible combinations
from itertools import combinations_with_replacement

#lectins = ['WGA','SNA','PHAL','AAL','PSA','DBCO']
possible_combinations= []
for i in range(2, 7):
    # Generate "combinations with replacement" for the current size i
    combs = list(combinations_with_replacement(library, i))
    possible_combinations.extend(combs)  # Store the combinations in the list

#hERE NO DUPLICATION SINCE THE TUPLES ARE CEREATED WITHOUT DUPLICATES. BUT FOR COMBINATIONS FROM THE CELLS DUPLICATES OCCUR.
#%%Counting classes
class_counter = {}

# Convert the standard list to sets for easier comparison
#standard_set = [set(tup) for tup in possible_combinations]

# Iterate through the standard list
for standard_tup in tqdm(possible_combinations, desc= "Counting Classes"):
    # Convert the current tuple to a set
    standard_tup_set = sorted(standard_tup)
    # Count occurrences in the tuple_list
    count = sum(1 for tup in result_list if sorted(tup) == standard_tup_set)
    # Update the result dictionary
    class_counter[standard_tup] = count



#%%Plotting class distribution  with normalization

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#sorting in the reverse order and picking th etop 10 class
sorted_data = sorted(class_counter.items(), key=lambda clas: clas[1], reverse=True)[:10] 
#Sorting the lectin classes
sorted_data = [(tuple(sorted(tup[0])), tup[1]) for tup in sorted_data]
#Fetching the list of categories
categories_ = [str(item[0]) for item in sorted_data]
#Scaling with
area_normalized_values = [item[1] / area_of_cell for item in sorted_data]

# Apply min-max normalization
# min_value = min(area_normalized_values)
# max_value = max(area_normalized_values)
# final_normalized_values = [(value - min_value) / (max_value - min_value) for value in area_normalized_values]
#%%

categories_ = [str(item[0]) for item in sorted_data]
plt.figure(figsize=(2.5, 3))
colors =  plt.cm.tab10(range(10))
plt.bar(categories_, area_normalized_values, color=colors, edgecolor='black')
plt.xticks(rotation=45, ha='right')
#plt.xlabel('Categories', fontsize=label_font_size)
plt.ylabel('Count per μm\u00b2', fontsize=label_font_size)
plt.title('Lectin classes distribution', fontsize=title_font_size)
plt.xticks(fontsize = tick_font_size)
plt.yticks(fontsize = tick_font_size)
plt.show()
if save ==True:
    plt.savefig(localization_folder/f"{timestamp}_Lectin_Classes_ per_sq-microns_{glyco_radius}nm{FIGFORMAT}",bbox_inches='tight')   
     
#%%Save classes to json file

# Save class counts
if save == True:
    output_file_path = localization_folder / f"{timestamp}_Number_of_Lectin_Classes_{glyco_radius}nm.json"

sorted_counter = dict(sorted(class_counter.items(), key=lambda item: item[1], reverse=True))
num_classes = {str(key): value for key, value in sorted_counter.items()}

# Save to a JSON file
if save == True:
    with open(output_file_path, 'w') as json_file:
        json.dump(num_classes, json_file, indent=4)
#Save Class count per unit area   
if save == True:
    PCA_output_file_path = localization_folder / f"{timestamp}_Lectin_Classes_ per_sq-microns_for_PCA_{glyco_radius}nm.json"

sorted_counter = dict(sorted(class_counter.items(), key=lambda item: item[1], reverse=True))
class_per_area = {str(key): value/area_of_cell for key, value in sorted_counter.items()}

# Save to a JSON file
if save ==True:
    with open(PCA_output_file_path, 'w') as json_file:
        json.dump(class_per_area, json_file, indent=4)
    
    
#%%
location_dictionary = {}

# Iterate through sorted data
for classes in tqdm(sorted_data, desc="Finding location of the classes "):

    #Extract only the tuple with Lectins
    class_looking_for_location = classes[0]
    location_dictionary[class_looking_for_location] = []
    # Iterate through the indexed tuples
    for indexed_tuple in final_pair_wise_duplicates_removed:
        # Convert each element in indexed_tuple to a tuple
        processed_tuple = tuple(q.split("_")[0] for q in sorted(indexed_tuple))
  
        # Check if it matches the class we are looking for
        if processed_tuple == class_looking_for_location:
            element, index = indexed_tuple[0].split("_")
            dataframe = data_dict[element]
            index = int(index)  # Ensure index is integer if needed
            x_val = dataframe.at[index, "x"]
            y_val = dataframe.at[index, "y"]
         
            location_dictionary[class_looking_for_location].append((x_val, y_val))
            
#%%Plot the class locations

# Define pixel size in nanometers
pixel_size_nm = 130  # 130 nanometers per pixel
scale_factor = pixel_size_nm / 1000  # Convert to micrometers (µm)

# Calculate full field of view in micrometers
field_of_view = 576 * scale_factor  # Field of view in µm (74.88 µm)

# Define plot dimensions in inches for a 576x576 pixel field
dpi = 100
fig_width = 576 / dpi
fig_height = 576 / dpi

# Sort the dictionary by the length of the coordinate list (in descending order)
location_dictionary_sorted = sorted(location_dictionary.items(), key=lambda item: len(item[1]), reverse=True)


# Select the top 5 entries based on the number of elements in the list
data_to_plot = location_dictionary_sorted[:number_to_plot]

#%% Plotting
from matplotlib.ticker import MultipleLocator
plt.figure(figsize=(2.57,2.57), dpi=dpi)
# Assign unique colors for each key
colors = plt.cm.tab10(range(len(data_to_plot)))  # Generate distinct colors for the top 5

# Find global min for shifting
all_coords = [coord for _, coords in data_to_plot for coord in coords]
scaled_all_coords = [(x * scale_factor, y * scale_factor) for x, y in all_coords]
all_x_vals, all_y_vals = zip(*scaled_all_coords)

global_x_min = min(all_x_vals)
global_y_min = min(all_y_vals)
global_y_max = max(all_y_vals) 
global_x_max = max(all_x_vals) 


for (key, coords), color in zip(data_to_plot, colors):
    # Scale coordinates to micrometers
    # scaled_coords = [(x * scale_factor, y * scale_factor) for x, y in coords]
    
    # x_vals, y_vals = zip(*scaled_coords)  # Unpack x and y coordinates
    #Shift and scale coordinates
    shifted_coords = [((x * scale_factor) - global_x_min,
                      global_y_max-(y * scale_factor))
                     for x, y in coords]
    x_vals, y_vals = zip(*shifted_coords)
    
    # Set a fixed spot size for all points (e.g., 50 points²)
    plt.scatter(x_vals, y_vals, label=str(key), color=color, s=5)  # Scatter plot
#just in case if the cell is smaller we have to put it to the middle zoomed in
if zoom==True:
    padding =0
    x_range = global_x_max - global_x_min
    y_range = global_y_max - global_y_min
    #If ticks need to be in the same values as the original position on the camera
    # plt.xlim(x_min - padding * x_range, x_max + padding * x_range)
    # plt.ylim(y_min - padding * y_range, y_max + padding * y_range)
    plt.xlim(0,x_range)
    plt.ylim(0,y_range)
    # plt.xlim(0, x_range)
    # plt.ylim(0, y_range)
    
# Set axis limits to match the full field of view. for bigger cells covering the fulll FOV we dont have to zoom it.
elif zoom == False:
    plt.xlim(0, field_of_view)  # 0 to 74.88 µm
    plt.ylim(0, field_of_view)  # 0 to 74.88 µm

# Invert the y-axis to set the origin at the top-left to match the orientation of the reconstruction
#plt.gca().invert_yaxis()
ax = plt.gca()
tick_spacing = 10
ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
ax.yaxis.set_major_locator(MultipleLocator(tick_spacing))
# Add labels

plt.xlabel("x (µm)", fontsize = label_font_size)
plt.ylabel("y (µm)", fontsize = label_font_size)
plt.title(f"Top {number_to_plot} classes", fontsize = title_font_size)
plt.xticks(fontsize = tick_font_size)
plt.yticks(fontsize=tick_font_size)
plt.grid(False)

# Show the plot
plt.show()
if save == True:
    plt.savefig(localization_folder/f"{timestamp}_Class_location_{glyco_radius}nm{FIGFORMAT}",bbox_inches='tight') 

     
#%% this part of the code is developed exclusively to cater the need to consider the dimer glycosylation

polymer_neighbor_master = {}
distance_indexed_ploymer_neighbor = {}
assigned_points = {}
if protein_present and consider_dimers:
    df_of_interest_key = protein
    
    com_name = f"neighbors_of_{df_of_interest_key}" #center of mass name to be used as  the key in the dictionaries
    polymer_neighbor_master[com_name] = {}
    distance_indexed_ploymer_neighbor[com_name] = {}
    
    trees = {key: KDTree((df[['x', 'y']]*130).values) for key, df in data_dict.items() if key ==df_of_interest_key}
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
            #Generates a list of indices coresponding to the dataframe 
            indices = current_family_members.query_ball_point([x1, y1], r=glyco_radius)
            filtered_indices = [num for num in indices if df_of_interest_key != current_family or num != row_index_of_com]
           
            if  filtered_indices:
                
                # Calculate scalar distances for each index # indices has the row number of the current family members
                dist_index_pairs = [(np.linalg.norm(current_family_members.data[idx] - [x1, y1]), idx) for idx in  filtered_indices]# creates a list of tuples with(distance, indices)
               
                # Sort by distance
                dist_index_pairs.sort(key=lambda x: x[0])
                
                # Find the closest point not already assigned or update if closer
                for distance, idx in dist_index_pairs: #idx is the index of current family member
                    # Only update if idx is unassigned or found closer
                    if idx not in assigned_points[current_family] or distance < assigned_points[current_family][idx][0]:#idx is the index of current family member
                        assigned_points[current_family][idx] = (distance, row_index_of_com)
                        
                        
    for neighbor_family, family_member in assigned_points.items():
        for idx, (distance, df_of_interest_idx) in family_member.items():
            polymer_neighbor_master[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append(f'{neighbor_family}_{idx}')    
            distance_indexed_ploymer_neighbor[com_name].setdefault(f'{df_of_interest_key}_{df_of_interest_idx}', []).append((f'{neighbor_family}_{idx}', distance))    
        

polymer_tuple_list = nested_dict_to_tuple_list(polymer_neighbor_master)
ploymer_list_without_mirror_duplicates=duplicates_removed_tuple_list(polymer_tuple_list)
flattened_polymer_list= [element for tup in ploymer_list_without_mirror_duplicates for element in tup]
elemental_duplicates_in_polymers = find_duplicates_with_counts(flattened_polymer_list) # This is not currently removed

#filter out dimers from polymers detected
dimer_list= [t for t in ploymer_list_without_mirror_duplicates if len(t) == 2]

tree_for_dimer = {key: KDTree((df[['x', 'y']]*130).values) for key, df in data_dict.items() if key != protein}


dimer_glycosylation = []
for tuple_of_dimers in dimer_list:
    coalesced_neighbors = []

    for element in tuple_of_dimers:
        family, idx = element.split('_')
        idx = int(idx)
        dataframe = data_dict[family]
        # get coordinates of current point
        x_val = dataframe.iloc[idx]['x'] * 130
        y_val = dataframe.iloc[idx]['y'] * 130

        # search in all KDTree families
        for current_family, current_family_tree in tree_for_dimer.items():
            # neighbor indices for that family
            neighbor_idxs = current_family_tree.query_ball_point([x_val, y_val], r=glyco_radius)
            # build neighbor IDs
            neighbors = [f"{current_family}_{neighbor_idx}" for neighbor_idx in neighbor_idxs]
            coalesced_neighbors.extend(neighbors) # group together the neighbors from two monomers
    coalesced_neighbors = list(set(coalesced_neighbors))
    if coalesced_neighbors:
        glycosylated_dimer = tuple_of_dimers + tuple(coalesced_neighbors)
        dimer_glycosylation.append(glycosylated_dimer)

index_removed_dimer_glycosylation =[]
for tup in dimer_glycosylation:
    # Remove numeric indices from each element
    cleaned_tuple = tuple(element.split('_')[0] for element in tup)
    index_removed_dimer_glycosylation.append(cleaned_tuple)

# Sort each tuple alphabetically
index_removed_dimer_glycosylation = [tuple(sorted(tup)) for tup in index_removed_dimer_glycosylation]

# Count frequency
frequency = Counter(index_removed_dimer_glycosylation)

# Normalize by area
dimers_per_sq_microns = {str(k): v / area_of_cell for k, v in frequency.items()}

# Sort by value descending
dimers_per_sq_microns_sorted = OrderedDict(
    sorted(dimers_per_sq_microns.items(), key=lambda x: x[1], reverse=True))
# Save to JSON
if save:
    dimer_glycosylation_file_path = localization_folder / f"{timestamp}_dimer_glycosylation_per_sq_microns.json"
    with open(dimer_glycosylation_file_path, 'w') as file:
        json.dump(dimers_per_sq_microns_sorted, file, indent=4)


#%%Deleting the current variables
#TODO - Fix this as this is not a good practice
for var in list(globals().keys()):
    if var not in ["dictionaryNames",
                   "FIGFORMAT",
                   "variables_from_parameter","pathLocsPoints", "__builtins__", "__name__", "__doc__", "__package__"]:
        del globals()[var]
#%%
#########################Nearest Neighbor Analysis#############################
from tqdm import tqdm
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import functionsAll as funct
import json
plt.rcParams["axes.grid"] = False

funct.save= True
#Choose if you want to annotate the peaks of histogram
funct.annotate = False
funct.FIGFORMAT = FIGFORMAT

# %% paths and name dictionary


# =============================================================================
# paths to the csv files to analyse
# 
# if the csv files ares the same for the points in search of a neig
# and those that make up the pool of potential neighbors, 
# then pathLocsPoints = 'path/to/folder'
# and pathLocsNeighbors = pathLocsPoints
# =============================================================================

# path to the csv files of the points in search of neigbors

pathLocsNeighbors = pathLocsPoints

# =============================================================================
# dictionary of equivalence: names of source files and futur names of new 
# created files and figures
# 
# to update when new lectins are used 
# =============================================================================



# =============================================================================
# histogram parameters
# =============================================================================

# histogram distances points to nearest neighbors in same channel
upper_limit = 300  # maximum display x axis
bin_size = 2  # bin size   
#%% import from hdf5

# makes 1 dictionary of arrays for the localizations of the points of all the channels
dictionaryLocalizationsPoints = funct.MultChannelsCallToDict_hdf5(pathLocsPoints, "locs")

orderedNames = list(dictionaryLocalizationsPoints.keys())
# if the paths are the same, there is no need to import the files again 
if pathLocsNeighbors == pathLocsPoints:
    dictionaryLocalizationsNeighbors = dict(dictionaryLocalizationsPoints)

else:
    dictionaryLocalizationsNeighbors = funct.MultChannelsCallToDict_hdf5(pathLocsNeighbors, "locs")


# %% new folder for analysis data

# =============================================================================
# crates new folder for the analysis data named as current date if no analysis 
# before on the same day
# if the folder already exists - nothing is done
# =============================================================================
timenow = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")

pathNewFolder = pathLocsPoints + '/' + str(timenow)
if not Path(pathNewFolder).exists():
    Path(pathNewFolder).mkdir()

# %% distance to first nearest neighbor for one in all six channels


maxima_x = pd.DataFrame(index=dictionaryLocalizationsPoints.keys(), columns=dictionaryLocalizationsPoints.keys())

for i in tqdm(dictionaryLocalizationsPoints.keys()):

    # calculate the nearest neighbor for every point in that channel in all the others including itself
    dictionaryDist = funct.nearestNeighborsOneInAll({i: dictionaryLocalizationsPoints[i]},
                                                    dictionaryLocalizationsNeighbors)

    nameCSV = timenow + '_DistNN_' + i + '_In_All'
    if list(dictionaryLocalizationsNeighbors.keys())[0].casefold().find('random') >= 0:
        nameCSV = nameCSV + 'random'
    if list(dictionaryLocalizationsNeighbors.keys())[0].casefold().find('centroid') >= 0:
        nameCSV = nameCSV + '_centroids'

    # save to csv (in the new folder inside original folder of the raw data)
    funct.dictionaryToCsv(dictionaryDist, pathNewFolder + '/' + nameCSV + '.csv')

    # =============================================================================
    #     distance to NN of one channel in itself is displayed separatly
    #     than the crosschannel distances to NN
    # =============================================================================    

    nameFIG = timenow + '_HistDistNN_' + i + '_In_All'
    if list(dictionaryLocalizationsNeighbors.keys())[0].casefold().find('random') >= 0:
        nameFIG = nameFIG + 'random'
    if list(dictionaryLocalizationsNeighbors.keys())[0].casefold().find('centroid') >= 0:
        nameFIG = nameFIG + '_centroids'
        
    #FIXME
    maxima_x = maxima_x.loc[orderedNames, orderedNames]    
    maxima_x = funct.displayHistFigure(
    dictionaryDist,  # Pass the entire dictionary without filtering
    rangeUp=upper_limit,  # Max X value for histogram display
    binsize=bin_size,  # Histogram bin size
    path=pathNewFolder + '/' + nameFIG,  # Save path for the figure
    maxima_matrix_x=maxima_x  # Passing previous maxima_x values
)
    maxima_x = maxima_x.loc[orderedNames, orderedNames]

    # =============================================================================
    #     the csv files of distance to NN and figures 
    #     are automatically saved in a new folder (1/per day) inside the original 
    #     folder of the point in search of a neighbor (pathLocsPoints)
    # =============================================================================

    continue
    # (for tqdm)

print(maxima_x)

nameFigMatrix = timenow + '_nN_matrix'
funct.plot_matrix_histogram(maxima_x, path=pathNewFolder + '/' + nameFigMatrix)

# save analysis parameters in txt file

# =============================================================================
# all the parameters used in the analysis are saved in a txt file
# the txt files name starts with the same timestamp than the new data files
# the txt file is saved with the new data files in the folder created for the day
# =============================================================================


parametersfilename = timenow + '_Parameters.txt'

# w tells python we are opening the file to write into it
outfile = open(pathNewFolder + '/' + parametersfilename, 'w')

outfile.write('Path to points for which to find neighbors : ' + pathLocsPoints + '\n\n')
outfile.write('Path to pools of potential neighbors : ' + pathLocsNeighbors + '\n\n')
outfile.write('Range histogram : 0-' + str(upper_limit) + ' (nm) \n\n')
outfile.write('bin size histogram : ' + str(bin_size) + ' (nm) \n\n')
outfile.close()  # Close the file when done
#plt.close('all')

#%%Combine the NN distance histogram peaks to a single file

keyword = "peaks"
pathNewFolder = Path(pathNewFolder)
peaks_combined_output_file = pathNewFolder / "Peaks_Combined.json"
combined_data = {}
for file in pathNewFolder.rglob(f"*{keyword}.json"):
    with file.open("r", encoding="utf-8") as f:
        data = json.load(f)
        combined_data.update(data)

# Write the combined data to a single JSON file
peaks_combined_output_file.write_text(json.dumps(combined_data, indent=4), encoding="utf-8")

