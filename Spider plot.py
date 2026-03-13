# -*- coding: utf-8 -*-
"""
Final Spider Plot Script
    
    Imports protein specific (user-defined) data and outputs a top N (user-defined) averaged lectin 
    distribution spider plot as well as individual cell lectin distribution spider plots. This version 
    uses the monomeric top N axes as the defined axes for dimer plots as well as individual cells.
    
@author: sculpep

This script supports:
1) Single dataset (Monomer + Dimer only)
2) NotStim + Stim datasets (each containing Monomer + Dimer)
"""

# =========================
# ====== IMPORTS ==========
# =========================

import numpy as np
import os
import json
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


# =========================
# ===== USER INPUTS =======
# =========================

# Number of lectin classes to display
N = 7

# Base directory:
# Case A: contains only JSON files (Monomer + Dimer)
# Case B: contains subfolders "Not_Stim" and "Stim"
base_dir = r"C:\Users\dmoonnu\Documents\CD4+ data analysis\New data after radius change\Spider plot"#this is Case B (for immune cell analysis)

# Protein must ALWAYS be user-defined
protein = "EGFR"

# Save settings
save_file = True
FIGFORMAT = ".pdf"

# Plot formatting, set to match individual plot size in illustrator 
num_ticks = 5
fig_length = 4
fig_width = 3

plt.rcParams["font.family"] = "arial"


# =========================
# === CONDITION DETECTION =
# =========================

conditions = {}

notstim_path = os.path.join(base_dir, "Not_Stim")
stim_path = os.path.join(base_dir, "Stim")

if os.path.isdir(notstim_path):
    conditions["NotStim"] = notstim_path

if os.path.isdir(stim_path):
    conditions["Stim"] = stim_path

# If no subfolders exist, treat base_dir as a single dataset
if not conditions:
    conditions["Single"] = base_dir


# =========================
# ===== HELPER FUNCTIONS ==
# =========================

def load_condition_data(condition_path):
    pca_datasets = []
    dimer_datasets = []

    for file in os.listdir(condition_path):
        if not file.endswith(".json"):
            continue

        with open(os.path.join(condition_path, file), "r") as f:
            data = json.load(f)

        dataset = [{"Key": k, "Value": v} for k, v in data.items()]

        if "PCA" in file:
            pca_datasets.append(dataset)
        elif "dimer" in file:
            dimer_datasets.append(dataset)

    return pca_datasets, dimer_datasets


def average_datasets(datasets):
    combined = {}
    count = {}

    for dataset in datasets:
        for item in dataset:
            key = item["Key"]
            value = item["Value"]

            if key in combined:
                combined[key] += value
                count[key] += 1
            else:
                combined[key] = value
                count[key] = 1

    return [{"Key": k, "Value": combined[k] / count[k]} for k in combined]


def clean_key(key_str):
    s = key_str.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1]

    parts = [e.strip().strip("'\"").upper() for e in s.split(",")]
    return tuple(parts)


def normalize_dimer_key(key, common_type):
    ck = clean_key(key)
    return tuple(sorted([item for item in ck if item != common_type]))


def compute_radial_scale(datasets):
    values = [item["Value"] for dataset in datasets for item in dataset]
    rmax = max(values)
    buffer = rmax * 0.1
    rmin = 0
    rticks = np.linspace(rmin, rmax + buffer, num_ticks + 1)
    return rmin, rmax, rticks


# =========================
# ===== LOAD ALL DATA =====
# =========================

all_data = {}

for cond, path in conditions.items():
    pca, dimer = load_condition_data(path)
    all_data[cond] = {
        "pca": pca,
        "dimer": dimer
    }


# =========================
# ===== AXIS DEFINITION ===
# =========================

if "NotStim" in all_data:
    reference_condition = "NotStim"
else:
    reference_condition = list(all_data.keys())[0]

ref_pca = all_data[reference_condition]["pca"]

# Average monomers
ref_avg = average_datasets(ref_pca)

# Select Top N
ref_avg_sorted = sorted(ref_avg, key=lambda x: x["Value"], reverse=True)
ref_top_N = ref_avg_sorted[:min(N, len(ref_avg_sorted))]
ref_top_keys = [item["Key"] for item in ref_top_N]

# Define protein (user-defined)
common_type = protein.strip().upper()

# Create axis labels
axes_labels = []
for key in ref_top_keys:
    cleaned = clean_key(key)
    labels = [e for e in cleaned if e != common_type]
    axes_labels.append("+" + " +".join(labels))


# =========================
# ===== ALIGN MONOMERS ====
# =========================

aligned_monomers = {}

for cond in all_data:
    aligned = []
    for dataset in all_data[cond]["pca"]:
        d = {item["Key"]: item["Value"] for item in dataset}
        aligned.append([
            {"Key": k, "Value": d.get(k, 0)}
            for k in ref_top_keys
        ])
    aligned_monomers[cond] = average_datasets(aligned)


# =========================
# ===== ALIGN DIMERS ======
# =========================

normalized_top = [
    normalize_dimer_key(k, common_type)
    for k in ref_top_keys
]

norm_to_orig = dict(zip(normalized_top, ref_top_keys))

aligned_dimers = {}

for cond in all_data:
    aligned = []
    for dataset in all_data[cond]["dimer"]:
        d = {
            normalize_dimer_key(item["Key"], common_type): item["Value"]
            for item in dataset
        }

        aligned.append([
            {"Key": norm_to_orig[nk], "Value": d.get(nk, 0)}
            for nk in normalized_top
        ])

    aligned_dimers[cond] = average_datasets(aligned)


# =========================
# ===== PLOTTING ==========
# =========================

def radar_factory(num_vars, frame='polygon'):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        # FORCE CLOSED LINE
        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                x, y = line.get_data()
                if x[0] != x[-1]:
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    line.set_data(x, y)
            return lines

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(*args, closed=closed, **kwargs)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        # Polygon frame
        def _gen_axes_patch(self):
            return RegularPolygon(
                (0.5, 0.5),
                num_vars,
                radius=.5,
                edgecolor="k"
            )

        def _gen_axes_spines(self):
            spine = Spine(
                axes=self,
                spine_type='circle',
                path=Path.unit_regular_polygon(num_vars)
            )
            spine.set_transform(
                Affine2D().scale(.5).translate(.5, .5) + self.transAxes
            )
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta


def plot_radar_multi(datasets, labels, title, save_path=None, shared_scale=None):
    num_vars = len(axes_labels)
    theta = radar_factory(num_vars)

    fig = plt.figure(figsize=(fig_length, fig_width))
    ax = fig.add_axes([0.15, 0.12, 0.7, 0.65], projection='radar')

    if shared_scale:
        rmin, rmax, rticks = shared_scale
        ax.set_ylim(rmin, rmax)
        ax.set_yticks(rticks)
    else:
        rmin, rmax, rticks = compute_radial_scale(datasets)
        ax.set_ylim(rmin, rmax)
        ax.set_yticks(rticks)

    for i, data in enumerate(datasets):
        values = [item["Value"] for item in data]
        ax.plot(theta, values, linewidth=2, marker="o", label=labels[i])
        ax.fill(theta, values, alpha=0.15)

    ax.set_thetagrids(np.degrees(theta), axes_labels)
    ax.set_title(title)
    #ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    if save_file and save_path:
        fig.savefig(save_path)

    plt.close(fig)


# =========================
# ===== AVERAGE PLOTS =====
# =========================

if len(aligned_monomers) > 1:
    # Overlay case
    mono_scale = compute_radial_scale(list(aligned_monomers.values()))
    dimer_scale = compute_radial_scale(list(aligned_dimers.values()))

    plot_radar_multi(
        list(aligned_monomers.values()),
        list(aligned_monomers.keys()),
        f"{common_type} Monomer Comparison",
        os.path.join(base_dir, f"{common_type}_MONOMER_Overlay{FIGFORMAT}"),
        shared_scale=mono_scale
    )

    plot_radar_multi(
        list(aligned_dimers.values()),
        list(aligned_dimers.keys()),
        f"{common_type} Dimer Comparison",
        os.path.join(base_dir, f"{common_type}_DIMER_Overlay{FIGFORMAT}"),
        shared_scale=dimer_scale
    )


# =========================
# ===== INDIVIDUAL PLOTS ==
# =========================

# =========================
# === INDIVIDUAL MONOMERS =
# =========================

for cond in all_data:

    for i, dataset in enumerate(all_data[cond]["pca"], start=1):

        # Align this individual dataset to reference axes
        d = {item["Key"]: item["Value"] for item in dataset}

        aligned = [
            {"Key": k, "Value": d.get(k, 0)}
            for k in ref_top_keys
        ]

        save_path = os.path.join(
            base_dir,
            f"{common_type}_MONOMER_{cond}_Cell{i}{FIGFORMAT}"
        )

        plot_radar_multi(
            [aligned],
            [f"{cond} Cell {i}"],
            f"{common_type} Monomer - {cond} Cell {i}",
            save_path=save_path
        )


# =========================
# ===== INDIVIDUAL DIMERS =
# =========================

for cond in all_data:

    for i, dataset in enumerate(all_data[cond]["dimer"], start=1):

        d = {
            normalize_dimer_key(item["Key"], common_type): item["Value"]
            for item in dataset
        }

        aligned = [
            {"Key": norm_to_orig[nk], "Value": d.get(nk, 0)}
            for nk in normalized_top
        ]

        save_path = os.path.join(
            base_dir,
            f"{common_type}_DIMER_{cond}_Cell{i}{FIGFORMAT}"
        )

        plot_radar_multi(
            [aligned],
            [f"{cond} Cell {i}"],
            f"{common_type} Dimer - {cond} Cell {i}",
            save_path=save_path
        )

