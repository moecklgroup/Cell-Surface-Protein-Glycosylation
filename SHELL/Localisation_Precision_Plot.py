# -*- coding: utf-8 -*-
"""
Localisation Precision Strip Plot — Stimulated vs Non-Stimulated
================================================================
Reads raw HDF5 localisation files directly from each cell folder
(not from 90_Custom Centers — those are clustered/centred outputs).

Each dot = median precision of one cell for that channel.
Two separate panels: Stimulated (orange) | Non-Stimulated (blue).

Output: <root_dir>/Localisation_Precision_by_Group.pdf

Author: N. Yurekli
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams.update({'font.family': 'Arial', 'font.size': 10})
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# ============================================================================
# CONFIGURATION  ← edit these before running
# ============================================================================
root_dir           = r"/Users/nazlicanyurekli/Desktop/2026-02-02_CD4+T Cells Segmented-Clustered-Glyco copy"          # path to your dataset folder
group_1_keyword    = "Stimulated"
group_2_keyword    = "Non_stimulated"
search_folder_name = "90_Custom Centers"   # used to locate cell folders
pixel_scale        = 130                   # nm per pixel

# Preferred channel display order (add/remove as needed)
CHANNELS_ORDER = ['WGA', 'SNA', 'PHAL', 'AAL', 'PSA', 'EGFR']
# ============================================================================

C1_COL = '#E87722'   # Stimulated    (orange)
C2_COL = '#4472C4'   # Non-Stimulated (same blue as original image)


def read_precision_nm(df, pixel_scale):
    """Return precision values in nm. Tries lprecision, lpx, lpy in order."""
    for col in ('lprecision', 'lpx', 'lpy'):
        if col in df.columns:
            return df[col].values * pixel_scale
    return None


def collect_precision(root_dir, group_1_keyword, group_2_keyword,
                      search_folder_name, pixel_scale):
    """
    For each cell, compute the per-channel MEDIAN precision.
    Returns: {group: {channel: [median_cell1, median_cell2, ...]}}
    """
    prec = {group_1_keyword: {}, group_2_keyword: {}}

    target_folders = list(Path(root_dir).rglob(f"**/{search_folder_name}"))
    print(f"Found {len(target_folders)} cell folders under '{search_folder_name}'")

    for loc_folder in target_folders:
        # raw HDF5 files sit one level up (directly in the cell folder)
        cell_folder = loc_folder.parent
        path_str    = str(cell_folder)

        if group_1_keyword in path_str:
            current_group = group_1_keyword
        elif group_2_keyword in path_str:
            current_group = group_2_keyword
        else:
            continue

        for hdf5_file in sorted(cell_folder.glob("*.hdf5")):
            channel = hdf5_file.stem.split("_")[0]   # e.g. EGFR, WGA, AAL
            try:
                df      = pd.read_hdf(hdf5_file, key='locs')
                prec_nm = read_precision_nm(df, pixel_scale)
                if prec_nm is not None and len(prec_nm) > 0:
                    # one dot per cell = median precision of all localisations
                    cell_median = float(np.median(prec_nm))
                    prec[current_group].setdefault(channel, []).append(cell_median)
            except Exception as e:
                print(f"  Warning: {hdf5_file.name}: {e}")

    return prec


def plot_precision(prec, group_1_keyword, group_2_keyword, out_path):
    """Two-panel strip chart matching the original figure style."""

    # Build channel order — keep preferred list, append any extras found
    extra = sorted(set(
        ch for grp in prec.values() for ch in grp if ch not in CHANNELS_ORDER))
    ordered_channels = [c for c in (CHANNELS_ORDER + extra)
                        if any(c in prec[g] for g in prec)]

    if not ordered_channels:
        print("No precision data found. Check that raw HDF5 files contain "
              "an 'lprecision' or 'lpx' column.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)

    for ax, (group, col) in zip(axes, [(group_1_keyword, C1_COL),
                                        (group_2_keyword, C2_COL)]):
        grp_data = prec[group]

        for xi, ch in enumerate(ordered_channels):
            vals = np.array(grp_data.get(ch, []))
            if vals.size == 0:
                continue
            # no horizontal jitter — dots aligned in one vertical column
            ax.scatter(np.full(len(vals), xi), vals,
                       color=col, s=30, linewidths=0, zorder=3)

        ax.set_xticks(range(len(ordered_channels)))
        ax.set_xticklabels(ordered_channels, fontfamily='Arial', fontsize=10,
                           rotation=45, ha='right')
        ax.set_xlabel('Channel', fontfamily='Arial', fontsize=10)
        ax.set_ylabel('Precision (nm)', fontfamily='Arial', fontsize=10)
        ax.set_title(f'{group}\nLocalisation Precision',
                     fontfamily='Arial', fontsize=10, color=col, fontweight='bold')
        ax.set_ylim(0, 15)
        ax.set_xlim(-0.5, len(ordered_channels) - 0.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.4, zorder=0)

        # Print summary
        all_vals = [v for ch in grp_data for v in grp_data[ch]]
        if all_vals:
            print(f"  {group}: median = {np.median(all_vals):.2f} nm  "
                  f"({sum(len(v) for v in grp_data.values())} cells, "
                  f"{len(grp_data)} channels)")

    fig.tight_layout()

    with PdfPages(out_path) as pdf:
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n✓ Saved → {out_path}")


if __name__ == "__main__":
    if not root_dir:
        raise ValueError("Set root_dir in the CONFIGURATION section before running.")

    out_path = Path(root_dir) / "Localisation_Precision_by_Group.pdf"

    prec = collect_precision(root_dir, group_1_keyword, group_2_keyword,
                             search_folder_name, pixel_scale)
    plot_precision(prec, group_1_keyword, group_2_keyword, out_path)
