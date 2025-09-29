# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:40:03 2024

@author: Chloe Bielawski
"""

# %% imports


from tqdm import tqdm
from pathlib import Path
import datetime
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import functionsAll as funct
import json
plt.rcParams["axes.grid"] = False

funct.save= False
#Choose if you want to annotate the peaks of histogram
funct.annotate = False
funct.FIGFORMAT = '.png'

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
pathLocsPoints = r"C:\Users\dmoonnu\Desktop\New code test data\Cell2"
# path to the csv files of the points - pool of potential neighbors 
pathLocsNeighbors = pathLocsPoints
# pathLocsPointspath = Path(pathLocsPoints)
# pathLocsNeighbors= Path(pathLocsPoints)
# =============================================================================
# dictionary of equivalence: names of source files and futur names of new 
# created files and figures
# 
# to update when new lectins are used 
# =============================================================================

# dictionaryNames = {'WGA': 'WGA',
#                    'sna': 'SNA',
#                    'phal': 'PHAL',
#                    'aal': 'AAL',
#                    'psa': 'PSA',
#                    'dbco': 'DBCO'}

#orderedNames = list(dictionaryNames.values())

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



# timestamp in the beginning of all new data files for identification
# is the same for all files of the same analysis (run of code)


# =============================================================================
# parameters for the histogram that can be chaged : 

#   max x for display of the histogram 
#         - 'rangeUp' parameter of the 'displayHistFigure' function
#         => using the variables upper_limit and upper_limit

#   size of the bins for the histogram 
#         - 'binsize' parameter of the 'displayHistFigure' function
#         => using the variables bin_size and binsizeCrossChannel
# =============================================================================

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
        
    # reorder the matrix
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

