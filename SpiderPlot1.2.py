# -*- coding: utf-8 -*-
"""
Spider Plot Script: 
    
    Imports protein specific (user-defined) data and outputs a top N (user-defined) averaged lectin 
    distribution spider plot as well as individual cell lectin distribution spider plots. This version 
    uses the monomeric top N axes as the defined axes for dimer plots as well as individual cells.
    
Created on Tue Aug 12 10:10:39 2025

@author: sculpep
"""
#%%Imports
import numpy as np
import os
import json
import re
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
#%%User Inputs
#number of lectin classes (axes) shown
N=7
#filepath should have all PCA AND dimer .json files 
filepath="F:\Spatial glycoproteomics data\EGFR Corrected\EGFR\Spider plot"
#Decide on whether you want to save the analysis results
save = True
FIGFORMAT = '.pdf'

#Plotting Preferences
num_ticks = 5
label_font_size = 10
title_font_size = 11
tick_font_size = 10
fig_length = 3
fig_width = 3
plt.rcParams['font.family'] = 'arial'
plt.rcParams['axes.titlesize'] = title_font_size
plt.rcParams['axes.labelsize'] = label_font_size
plt.rcParams['xtick.labelsize'] = tick_font_size
plt.rcParams['ytick.labelsize'] = tick_font_size
#%% Fetching info from the parameter file about analysis folder
variables_from_parameter = {}

# Dictionary to hold all loaded data
all_pca_data = {}  # key = filename, value = data from JSON
all_dimer_data = {}

# List all files in the directory
files_in_directory = os.listdir(filepath)

# Loop through files and load JSON data
for file in files_in_directory:
    if not file.endswith('.json'):
        continue
    full_path = os.path.join(filepath, file)
    print(f"Loading file: {file}")
    with open(full_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if 'PCA' in file:
                all_pca_data[file] = data
            elif 'dimer' in file:
                all_dimer_data[file] = data
        except json.JSONDecodeError as e:
                print(f"Error parsing {file}: {e}")

pca_datasets = []
pca_file_names = []
pca_dataset_types = []

for filename, data in all_pca_data.items():
    dataset = [{"Key": k, "Value": v} for k, v in data.items()]
    pca_datasets.append(dataset)
    pca_file_names.append(filename)
    pca_dataset_types.append(tuple(sorted(data.keys())))
    
dimer_datasets = []
dimer_file_names = []
dimer_dataset_types = []
    
for filename, data in all_dimer_data.items():
    dataset = [{"Key": k, "Value": v} for k, v in data.items()]
    dimer_datasets.append(dataset)
    dimer_file_names.append(filename)
    dimer_dataset_types.append(tuple(sorted(data.keys())))
#%%Averaging Function
def average_datasets(datasets):
    combined = {}
    count = {}
    for dataset in datasets:
        for item in dataset:
            key = item['Key']
            value = item['Value']
            if key in combined:
                combined[key] += value
                count[key] += 1
            else:
                combined[key] = value
                count[key] = 1
                
    averaged = [{'Key': key, 'Value': combined[key] / count[key]} for key in combined]
    return averaged

#%% Radar Plot Creation Function
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.
    This function creates a RadarAxes projection and registers i and was found on Stack Overflow
    https://stackoverflow.com/questions/65514398/how-to-make-radar-spider-chart-with-pentagon-grid-using-matplotlib-and-python

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def plot_radar(data, title="Averaged Radar Chart", axis_labels=None, save_path=None, rmin=None, rmax=None, rticks=None):
    labels = [item['Key'] for item in data] if axis_labels is None else axis_labels
    values = [item['Value'] for item in data]
    num_vars = len(labels)

    theta = radar_factory(num_vars, frame='polygon')

    fig, ax = plt.subplots(figsize=(fig_length,fig_width), subplot_kw=dict(projection='radar'))

    if rmin is not None and rmax is not None:
        ax.set_ylim(rmin, rmax)

    # Plot data
    ax.plot(theta, values, 'o-', linewidth=2)
    ax.fill(theta, values, alpha=0.25)

    # Set labels
    ax.set_varlabels(labels)
    ax.set_title(title)

    # Set radial ticks safely
    if rticks is not None:
        ax.set_yticks(rticks)
        ax.set_yticklabels([str(rt) for rt in rticks])
    elif rmin is not None and rmax is not None:
        ax.set_yticks(np.linspace(rmin, rmax, num_ticks))
    else:
        # Only call get_ylim AFTER plotting, so auto-scaling has occurred
        current_rmin, current_rmax = ax.get_ylim()
        current_buffer = current_rmax * 0.1
        ax.set_yticks(np.linspace(current_rmin, current_rmax + current_buffer, num_ticks))

    plt.tight_layout()
    #bbox_inches='tight'
    if save_path:
        fig.savefig(save_path, format='pdf')
        print(f"Saved plot to {save_path}")
#%% Clean keys in order to make nice axes labels
def clean_key(key_str):
    if isinstance(key_str, tuple):
        parts = list(key_str)
    else:
        s = key_str.strip()
        if s.startswith('(') and s.endswith(')'):
            s = s[1:-1]
        parts = [e.strip().strip("'\"").upper() for e in s.split(',')]
    return tuple(parts)

def normalize_dimer_key(key, common_type):
    ck = clean_key(key)
    # Remove protein markers
    normalized = tuple(sorted([item for item in ck if item != common_type]))
    return normalized
#%% Plotting Averaged Monomer Data 
# Average and select top N
averaged_pca = average_datasets(pca_datasets)
N = min(N, len(averaged_pca))
top_N_averaged = sorted(averaged_pca, key=lambda x: x['Value'], reverse=True)[:N]
# Extract the keys of the Top N Averaged categories
top_N_keys = [item['Key'] for item in top_N_averaged]


# def clean_key(key_str):
#     key_str = key_str.strip()
#     if key_str.startswith('(') and key_str.endswith(')'):
#         key_str = key_str[1:-1]
#     parts = [
#         e.strip().strip("'\"").upper()
#         for e in key_str.split(',')
#     ]
#     return tuple(parts)

# Get the keys
top_keys = [clean_key(item['Key']) for item in top_N_averaged]
# Flatten and count elements
all_elements = [elem for key in top_keys for elem in key]
counts = Counter(all_elements)
# Identify the protein across all Top N
common_elements = [elem for elem, count in counts.items() if count == len(top_keys)]
if common_elements:
    common_type = common_elements[0]  # Use the first one
else:
    raise ValueError("No common element across top_N_averaged keys.")
# Strip any quotes from the common_type string
common_type = common_elements[0].strip("'\"").upper()
plot_title = common_type

# Create cleaned axes labels
axes_labels = [
    '+' + ' +'.join([e for e in key if e.strip("'\"").upper() != common_type])
    for key in top_keys
]

# Collect all numeric values from all datasets for scaling
all_values = []
for dataset in pca_datasets:
    for item in dataset:
        all_values.append(item['Value'])

# Compute min/max 
avg_max = max(item['Value'] for item in top_N_averaged)
# Set rmin/rmax and ticks
rmin = 0
rmax = round(avg_max, 2)
buffer = rmax * 0.1
rticks = np.linspace(rmin, rmax + buffer, num=num_ticks + 1).round(2).tolist()

# Plot Monomer Average
plot_radar(top_N_averaged,
           title=f"{plot_title} Monomer Average",
           axis_labels=axes_labels,
           save_path=os.path.join(filepath, f"average_top_{N}_{plot_title}_MONOMER.pdf"),
               rmin=rmin, rmax=rmax, rticks=rticks)
#%% Plotting each individual cell monomers 
monomer_cell_counter = 1
for pca_filename, dataset, dtype in zip(pca_file_names, pca_datasets, pca_dataset_types):
    # Extract FOV/Cell identifier
    match = re.search(r'FOV\d+Cell\d+', pca_filename)
    if match:
        fov_cell_id = match.group()
    else:
        fov_cell_id = f"Cell{monomer_cell_counter}"
        monomer_cell_counter += 1
    
    dataset_dict = {item['Key']: item['Value'] for item in dataset}
    aligned_values = [{'Key': key, 'Value': dataset_dict.get(key, 0)} for key in top_N_keys]  
    type_str = '_'.join(dtype)    
    save_file = os.path.join(filepath, f"{fov_cell_id}_top_{N}_{plot_title}_MONOMER.pdf")  
    plot_radar(aligned_values,
               title=f"{plot_title} Monomer {fov_cell_id}",
               axis_labels=axes_labels,
               save_path=save_file)
#%% Plotting Averaged Dimer Data   
#Need to clean keys to find data from Top N as defined from monomer data 
#EX. (EGFR,SNA) from monomeric data = (EGFR, EGFR,SNA) from dimeric data
normalized_top = [normalize_dimer_key(key, common_type) for key in top_N_keys]
#use the first matching original key or just top_N_keys
norm_to_orig = dict(zip(normalized_top, top_N_keys))
aligned_dimer_datasets = []
for dataset in dimer_datasets:
    dataset_dict = {
        normalize_dimer_key(item['Key'], common_type): item['Value']
        for item in dataset
    }
    # Align to normalized_top keys
    aligned = [{'Key': key, 'Value': dataset_dict.get(key, 0)} for key in normalized_top]
    aligned_dimer_datasets.append(aligned)
# Average across all aligned datasets
averaged_dimer = average_datasets(aligned_dimer_datasets)
# Prepare data for plotting with original labels
plot_data = []
for item in averaged_dimer:
    norm_key = item['Key']
    orig_key = norm_to_orig.get(norm_key, norm_key)  # fallback to norm_key if not found
    plot_data.append({'Key': orig_key, 'Value': item['Value']})

# Plot averaged dimer
plot_radar(plot_data,
           title=f"{plot_title} Dimer Average",
           axis_labels=axes_labels,
           save_path=os.path.join(filepath, f"average_top_{N}_{plot_title}_DIMER.pdf"))

#%% Plotting each individual cell dimers
dimer_cell_counter = 1
normalized_top = [normalize_dimer_key(key, common_type) for key in top_N_keys]
# Map normalized keys back to original keys
norm_to_orig = dict(zip(normalized_top, top_N_keys))
for dimer_filename, data in all_dimer_data.items():
    match = re.search(r'FOV\d+Cell\d+', dimer_filename)
    fov_cell_id = match.group() if match else f"Cell{dimer_cell_counter}"
    dimer_cell_counter += 1
    # Convert data dict to list of {'Key': key, 'Value': val} if needed
    dataset = [{"Key": k, "Value": v} for k, v in data.items()]
    # Normalize dataset keys and build dict
    dataset_dict = {
        normalize_dimer_key(item['Key'], common_type): item['Value']
        for item in dataset
    }
    # Align with normalized top keys
    aligned = [{'Key': key, 'Value': dataset_dict.get(key, 0)} for key in normalized_top]
    # Replace normalized keys with original keys for plotting labels
    plot_data = []
    for item in aligned:
        orig_key = norm_to_orig.get(item['Key'], item['Key'])
        plot_data.append({'Key': orig_key, 'Value': item['Value']})

    save_file = os.path.join(filepath, f"{fov_cell_id}_top_{N}_{plot_title}_DIMER.pdf")
    plot_radar(plot_data,
               title=f"{plot_title} Dimer {fov_cell_id}",
               axis_labels=axes_labels,
               save_path=save_file)