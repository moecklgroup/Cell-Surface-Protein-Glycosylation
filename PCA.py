# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:45:45 2025
This is a script to perform pricipal component analysis on a set of data. Any data input  for example, nn peaks distances or glyco numbers can
be used. The code normalizes the data using min max normaliztion. Shifting mean of the data set to origin is performed in the sklearn package.
This is supposed to run on a whole data set. That means either on a pannel dataset or a disease condotion dataset. It extracts the full dataset
@author: dmoonnu
"""
from sklearn.decomposition import PCA
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib
import numpy as np
#%%User Inputs
#path to parent folder
folders = r"C:\Users\dmoonnu\Documents\PCA Mannaz treat correct\New folder"
#name of subfolders denoting condition
folder_names = ["MCF10A", "MCF10A+TGFb","MCF10AT", "MCF10AT+TGFb"]
#folder_names = ["Tumor", "Regular"]
#Choose which data to plot
#data_to_plot ="NN"
data_to_plot ="NN"
#threed_view =[7,164]
threed_view =[-177,150]
save = False
legend = True
show_plot = True
FIGFORMAT = 'pdf'

#This controls the number of glyco classes considered
number_of_characters_to_consider = 36

matplotlib.use('Qt5Agg')
plt.rcParams['font.family'] = 'arial'

label_font_size = 13
title_font_size = 13
tick_font_size = 10
#%%matplotlib setting

 
#set coloring
colors = plt.cm.tab10(range(len(folder_names)))
#colors = list(plt.cm.tab10.colors[:3]) + ["magenta"]
color_map = dict(zip(folder_names, colors))
#color_map = {cell: colormap(i) for i, cell in enumerate(folder_names)}


#%%
if data_to_plot== "NN":
    keyword="Peaks_Combined"
if data_to_plot== "glyco":
    keyword="PCA"


#Number of axe to use in PCA
pca_axes = 3
#Boolean to decide whether to save or not

#orientation of the 3d plot [elevation,azimuth]




def find_json_files(base_dirs, keyword):
    """
    looks for json file in subfolders using a keyword  in the filename.

    Parameters
    ----------
    base_dirs : str
         
    keyword : str
        The default is "Peaks_Combined".

    Returns
    -------
    json_files : list of Path object


    """
    json_files = []

    for path in Path(base_dirs).rglob(f"*{keyword}*.json"):
        json_files.append(path)
    return json_files

def normalize_dataframe(df):
    "performs Min Max Normalization of the entire dataframe"
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    
    # Calculate global min and max from the entire DataFrame (only numerical columns)
    global_min = df[numerical_cols].min().min()
    global_max = df[numerical_cols].max().max()

    # Avoid division by zero if max == min
    if global_max != global_min:
        # Normalize only the numerical columns
        for col in numerical_cols:
            df[col] = (df[col] - global_min) / (global_max - global_min)
    else:
        # If all values are the same, set them to 0 (or any other strategy)
        df[numerical_cols] = 0
    
    return df

def perform_pca(df, n_components):
    numerical_cols = df.select_dtypes(include=['number']).columns
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(df[numerical_cols])

    for i in range(n_components):
        df[f'PC{i+1}'] = pca_components[:, i]

    return df, pca

def extract_key_value_pairs(json_files, folder_names,pca_axes):
    """
    Extract data from the list of of json files and append them to a dataframe with keys as the column name. Each file correspond to a cell.
    ie. Each row correspond to a cell.
    """
    rows = []
    
    for file in json_files:#iterat throgh each file
        try:
            with file.open('r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict): #Checks if the loaded data from json file is a dictionary
                    filtered_data = {k: v for k, v in data.items() if v}  # filtering Exclude zero-valued keys
                    
                    # Find the matching folder name
                    parent_folders = [p.name for p in file.parents] #check the file name for parent folder
                    #Fetching name of the subfolder
                    matched_name = next((name for name in folder_names if name in parent_folders), "Unknown") #Tries to match the fetched name with the given name of folders
                    
                    filtered_data["Cell Type"] = matched_name
                    rows.append(filtered_data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {file}: {e}")
    
    df = pd.DataFrame(rows)
    
    # Move "Cell Type" to the last column
    if "Cell Type" in df.columns:
        cols = [col for col in df.columns if col != "Cell Type"]
        cols.append("Cell Type")
        df = df[cols]
    
    # Normalize numerical columns using global min and max
    df = normalize_dataframe(df)
    df = df.fillna(0)
    if keyword == "PCA" or "Peaks_Combined":
        selected_cols = df.columns[:number_of_characters_to_consider].tolist()    
        # # Add the last_col at the end of cropped dataframe
        column_to_keep = "Cell Type"
        selected_cols.append(column_to_keep)
        df = df[selected_cols]
    #Apply PCA and reduce dimensionality to 2 components (or change as needed)
    df, pca = perform_pca(df, n_components=pca_axes)
    if pca_axes==2:
        plot_pca(df)
    elif pca_axes==3:
        legend=True #Does the pltting twice not to have the legend
        plot_pca_3d(df,legend)
        legend=False
        plot_pca_3d(df,legend)
    plot_scree(pca)

   
    
    return df, pca


def plot_pca_3d(df,legend):
    fig = plt.figure(figsize=(3.75, 3.75))
    #fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111, projection='3d')

    unique_sources = df["Cell Type"].unique()
    for source in unique_sources:
        source_data = df[df["Cell Type"] == source]
        color_view = color_map[source]
        ax.scatter(source_data["PC1"], source_data["PC2"], source_data["PC3"], label=source, s=100, color=color_view, alpha=0.7)
    if keyword == "PCA":
        fig_suffix = "GlyCo"
    elif keyword == "Peaks_Combined":
        fig_suffix= "NN peaks"
    ax.set_title(f'3D PCA plot of {fig_suffix}', fontsize = title_font_size)
    ax.set_xlabel('PC 1', fontsize = label_font_size)
    ax.set_ylabel('PC 2', fontsize = label_font_size)
    ax.set_zlabel('PC 3', fontsize = label_font_size)
    if legend:
        pca_legend = ax.legend(title="Cell Type", fontsize = 'medium',loc="upper left")
        pca_legend.get_title().set_fontsize(f'{label_font_size}')
    xstart, xend = ax.get_xlim()
    ystart, yend = ax.get_ylim()
    zstart, zend = ax.get_zlim()
    ax.xaxis.set_ticks(np.arange(xstart, xend,0.2))
    ax.yaxis.set_ticks(np.arange(ystart, yend,0.2))
    ax.zaxis.set_ticks(np.arange(zstart, zend,0.2))     
    #plt.xticks(labels="", fontsize = tick_font_size)
    # plt.yticks(fontsize=tick_font_size)
    ax.tick_params('x', labelsize=tick_font_size, labelbottom=False)
    ax.tick_params('z', labelsize=tick_font_size, labelleft=False)
    ax.tick_params('y', labelsize=tick_font_size, labelleft=False)
    # ax.tick_params('y', labelsize=tick_font_size, labelright=False)
    # ax.tick_params('y', labelsize=tick_font_size, labeltop=False)
    ax.view_init(elev=threed_view[0], azim=threed_view[1])  # Adjust orientation before saving
    plt.show()
    if save:
        if legend:
            plt.savefig(Path(folders)/f"PCA 3D with legend {fig_suffix}_{number_of_characters_to_consider}axes.{FIGFORMAT}", bbox_inches="tight")
            df.to_excel(Path(folders)/f"{fig_suffix}_{number_of_characters_to_consider}axes_dataset.xlsx", index=False)
        if not legend:
            plt.savefig(Path(folders)/f"PCA 3D without legend {fig_suffix}_{number_of_characters_to_consider}axes.{FIGFORMAT}", bbox_inches="tight")
    if not show_plot: #close the plots
        plt.close("all")


def plot_pca(df):
    # Plotting the PCA results (PC1 vs PC2)
    
    plt.figure(figsize=(10, 6))
    #sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cell Type", palette="Set1", s=100, alpha=0.7)

    if keyword == "PCA":
        fig_suffix = "GlyCO"
    elif keyword == "Peaks_Combined":
        fig_suffix= "NN Peaks"
    # Add labels and title
    plt.title(f"PCA Plot of {fig_suffix}", fontsize=title_font_size)
    plt.xlabel('PC 1', fontsize = label_font_size)
    plt.ylabel('PC 2', fontsize = label_font_size)

    # Show the plot
    pca_legend = plt.legend(title="Cell Type",fontsize='x-large')
    pca_legend.get_title().set_fontsize(f'{label_font_size}')
    plt.xticks(fontsize = tick_font_size)
    plt.yticks(fontsize=tick_font_size)
    plt.show()
    if save:
        plt.savefig(Path(folders)/f"PCA 2d {fig_suffix}.{FIGFORMAT}", bbox_inches="tight")
        
    if not show_plot: #close the plots
        plt.close("all")
def plot_scree(pca_model):
    # Plotting the Scree Plot
    plt.figure(figsize=(8, 6))
    
    # Get the explained variance for each principal component
    explained_variance = pca_model.explained_variance_ratio_
    
    # Plot the explained variance as a bar chart
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, color='b')
    
    # Add labels and title
    plt.title('Scree Plot')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    
    # Show the plot
    plt.show()
    if not show_plot: #close the plots
        plt.close("all")

def main():

    json_files = find_json_files(folders, keyword)
    
    # Extract and process the data
    result_df, pca_model = extract_key_value_pairs(json_files, folder_names,pca_axes)
    
    return result_df 

# Call main function
result_df = main()
