# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 15:26:52 2024

@author: Chloe Bielawski
"""

# %% imports


from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from pathlib import Path
from astropy.stats import RipleysKEstimator
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as stat
import random
import math
import glob
#import cmasher as cmr
import os
from matplotlib.colors import ListedColormap, BoundaryNorm
import json
import scipy.spatial as spat
plt.rcParams['font.family'] = 'arial'


FIGFORMAT= '.pdf'

save = True
annotate = False
label_font_size = 15
title_font_size = 10
tick_font_size = 15


# %% clusteringDBSCAN


def clusteringDBSCAN(X, epsChoice, minSampleChoice):
    """
    Finds clusters in data from coordinates of points with sklearn.cluster’s DBSCAN. 
    DBSCAN takes all three parameters of the function and returns a label for each point. 
    This label is a number (zero and above) assigned to the cluster to which the point belongs.
    
    Displays the number of identified clusters and the number of outlier points. 
    Both of those values are extrapolated from the labels of the points.
    
    Only handles one channel at a time.
    
    Parameters: 
        X: Coordinates of the points to cluster (2D array, float).
        epsChoice: Maximum distance to a point for the other to be its neighbour. 
            Can be considered here the maximum radius of a cluster (float).
        minSampleChoice: Minimum number of neighbours a point must have to be a core point. 
            Can be considered here the minimum number of points in a cluster (int).
    
    Output: 
        labels: Labels of points. Reference the cluster they belong to with -1 for noise 
            points and numbers equal to or above zero for identified clusters (1D array, int).
        Text display.
    """

    db = DBSCAN(eps=epsChoice, min_samples=minSampleChoice).fit(X)

    labels = db.labels_
    # one number for each different cluster; -1 for noise

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # number of cluster without counting the noisy points (label = -1)
    n_noise_ = list(labels).count(-1)
    # count number of noise points (labels = -1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)

    return labels


def displayPointsSize(dictFunct, x, y):
    """
    Displays the points passed as parameter as dots in a figure which dimensions can be controlled by the user. 
    The coordinates of the points are passed as parameter using a dictionary of 2D arrays. 
    
    The different parts of the dictionary, corresponding to different channels, are displayed in separate figures.
    
    Handles multiple channels.
    
    Parameters:
        dictFunct: Coordinates of the points to display (dictionary of 2D arrays, float).
        x and y: Dimensions of the figure to plot (float).
    
    Output: 
        Graphic display.
    """

    if len(dictFunct) == 1:  # if only one figure to plot

        X = list(dictFunct.values())[0]  # array in position 0 in the dictionary
        plt.figure(figsize=(x, y))  # size of the figure (user input in call of function)
        plt.plot(X[:, 0], X[:, 1], '.', markersize=1)
        plt.title(list(dictFunct.keys())[0], fontsize=x)  # title of figure => font size propotional to fig size


    else:  # if multiple figures to plot

        for i in dictFunct.keys():  # i is the array for which we are tracing the histogramm

            plt.figure(figsize=(x, y))  # size of the figure (user input in call of function)

            X = dictFunct[i]  # i-eme array in dictionary
            plt.plot(X[:, 0], X[:, 1], '.', markersize=1)
            plt.title(i, fontsize=x)  # title of figure => font size propotional to fig size

    plt.show()


def displayPointsCentroids(dictFunct, dictCentroids, x, y, title):
    """
    Displays the points passed as parameter in an array of x y coordinates for both the points and the centroids. 
    
    The points and centroids of same index in their respective dictionaries are considered to belong 
    to the same channel and displayed in the same figure. The different parts of the 
    dictionaries are displayed in separate figures.
    
    Handles multiple channels.
    
    Parameters:
        dictFunct: Coordinates of the points to display (dictionary of 2D array, float).
        dictCentroids: Dictionary of 2D arrays containing the coordinates of the centroids of 
            clusters for the points of dictFunct (dictionary of 2D array, float).
        x, y: Dimensions of the figure (float).
        title: Title of the figures (1D array of strings with the same number of titles 
            than entries in the dictionaries).
    
    Output:
        Graphic display.
    """

    if len(dictFunct) == 1:  # if only one figure to plot

        X = list(dictFunct.values())[0]  # points to plot
        C = list(dictCentroids.values())[0]  # centroids for the points to plot
        plt.figure(figsize=(x, y))  # array in position 0 in the dictionary
        plt.plot(X[:, 0], X[:, 1], '.', color='black', markersize=2, label='points ' + list(dictFunct.keys())[0])  # points
        plt.plot(C[:, 0], C[:, 1], 'o', color='red', markersize=20,
                 markerfacecolor="none", markeredgewidth=3,
                 label='controids ' + list(dictCentroids.keys())[0])  # centroids

        plt.title(title[0], fontsize=x)
        # title of figure => font size propotional to fig size
        plt.legend(fontsize='x-large', loc='upper right')




    else:  # if multiple figures to plot

        for i in range(0, len(dictFunct)):  # i is the array for which we are tracing the histogramm
            X = dictFunct[list(dictFunct)[i]]  # i-eme array in dictionary
            C = dictCentroids[list(dictCentroids)[i]]  # i-eme array in dictionary

            plt.figure(figsize=(x, y))  # array in position 0 in the dictionary

            plt.plot(X[:, 0], X[:, 1], '.', markersize=2, label='points ' + list(dictFunct.keys())[i])  # points
            plt.plot(C[:, 0], C[:, 1], 'o', color='red', markersize=20,
                     markerfacecolor="none", markeredgewidth=3,
                     label='controids ' + list(dictCentroids.keys())[i])  # centroids

            plt.title(title[i], fontsize=x)
            # title of figure => font size propotional to fig size
            plt.legend(fontsize='x-large', loc='upper right')

    plt.show()


def displayHistFigure(dictFunct, rangeUp, binsize, path, maxima_matrix_x):
    """
    Displays histograms for the data passed as parameter. All the histograms are displayed in the same figure. 
    The beginning and number of bins are calculated inside the code from the binsize and maximum value of the studied dataset. 
    The number of occurrences and x value of the highest bin is displayed on the figure for all the histograms.
    
    The histograms are automatically saved as PNG files in path (path including the name of the new file).
    
    Handles multiple channels, but there are displayed in the same figure.
    
    Parameters:  
        dictFunct: Values for the histogram (dictionary of 1D arrays, float).
        rangeUp: Upper bound of the x range of displayed data in the histogram. 
            Is used for proper placement and display of the histogram bins (float).
        binsize: Size of histogram bins (float). 
        path: Path to folder where the figures will be saved, including the name of the new file (str). 
    
    Output:
        Display of histogram.
        PNG file.
    """
    output_file_path = path + "_peaks.json"
    #plt.figure(figsize=(15, 5))  # size of the figure
    plt.figure(figsize=(9, 3))  # size of the figure
    dict_of_peaks ={}
    for i in dictFunct.keys():  # i is the array for which we are tracing the histogramm

        y, x, _ = plt.hist(dictFunct[i],
                           bins=math.ceil(math.ceil(max(dictFunct[i])) / binsize),
                           histtype='step',
                           range=(0, math.ceil(max(dictFunct[i]))),
                           label=i)
        # the number of bins is the max value divided by the size of the bins (rounded up)
        # the range insure that the bins have the same edges throughout the channels
        #   it goes from 0 to the right edge of the last bin
        # the label for each histogram is it's name in the dictionary

        # print max x and count on the histogram
        xmax = x[np.argmax(y)]  # x value for the highest bin (max occurence)
        ymax = y.max()  # y (number of occurences value for the highest bin
        peak = x[np.argmax(y)] 
        channel_x, channel_y = i.split('-')
        #channel_x = channel_x.split('distNN_')[-1]
        maxima_matrix_x.loc[channel_x, channel_y] = xmax
        dict_of_peaks[f"{channel_x},{channel_y}"] = peak
        if annotate == True:
            plt.annotate("x={:.3f}, y={:.3f}".format(xmax, ymax), xy=(xmax, ymax))

    legend = plt.legend(fontsize='x-large', loc='upper right')
    legend.get_title().set_fontsize(f'{label_font_size}')
    plt.xlabel('Distance to nearest neighbor (nm)', fontsize = label_font_size)
    plt.ylabel('Frequency', fontsize = label_font_size)
    plt.xlim((0, rangeUp))  # x lim of the histogram display
    plt.xticks(fontsize = tick_font_size)
    plt.yticks(fontsize = tick_font_size)
    if save == True:
        
        plt.savefig(path + FIGFORMAT, bbox_inches='tight')  # save figure as png in specified path
        with open(output_file_path, 'w') as json_file:
            json.dump(dict_of_peaks, json_file, indent=4)
    
    return maxima_matrix_x


def plot_matrix_histogram(matrix, path):

    #plt.figure(figsize=(15, 15))
    plt.figure(figsize=(5,5))
    
    matrix_np = matrix.to_numpy().astype(float)
    original_matrix = np.copy(matrix_np)
    
    
    min_val = round(np.min(matrix_np))
    max_val = round(np.max(matrix_np))

    # Define the colormap
    cmap = plt.cm.coolwarm
    colors = cmap(np.arange(cmap.N))

    # Set color bounds for the original range and the added value
    bounds = list(np.linspace(min_val, max_val)) 

    norm = BoundaryNorm(bounds, cmap.N)
    #fig, ax = plt.subplots(figsize=(10,10))
    fig, ax = plt.subplots(figsize=(3.2,3.52))

    im = ax.imshow(matrix_np, cmap=cmap, norm=norm)
    
    #ax.grid(False)
    
    for (j,i), val in np.ndenumerate(original_matrix):
         ax.text(i, j, '{:.1f}'.format(val), ha='center', va='center', color='k', fontsize = 8.5)
            
    
    plt.xticks(np.arange(len(matrix.columns)), matrix.columns, fontsize = 10, rotation=45)
    plt.yticks(np.arange(len(matrix.index)), matrix.index, fontsize = 10)
    cbar = plt.colorbar(im, ax=ax)
    #cbar.set_label(label = r'maximum of nN distances (nm)', size = 'x-large', weight='bold')
    cbar.set_ticks(np.linspace(min_val, max_val,5))
    cbar.ax.tick_params(labelsize = 10)
    plt.tight_layout()
    if save == True:
        plt.savefig(os.path.join(path + FIGFORMAT),bbox_inches='tight', transparent=False)
    plt.close()



def calculateCentroids(labels, X):
    """
    Calculates the centroids of identified clusters from the coordinates of the points and their DBSCAN labels.
    
    Only handles one channel at a time.
    
    Parameters:
        labels: Cluster ID for each point of the dataset that has been clustered (1D array, int). 
            The outlier points are labelled as -1 while the clusters are named from zero. 
        X: Coordinates of points in the studied channel (2D array, float).
    
    Output:
        centroids: Coordinates of the centroids of the clusters in the studied channel (2D array, float).
    """

    unique_labels = set(labels)  # different clusters labels found by DBSCAN clustering

    centroids = np.ones((len(unique_labels) - (1 if (-1 in unique_labels) == True else 0),
                         2))  # create array of coordinates (2 columns) and (nb of clusters) lines
    # if there are noise points, len = len-1 sice we do not count them as cluster
    # else, stays len = len

    for k in unique_labels:
        if k != -1:  # labels = -1 means the point doesn't belong to a cluster (outlier) => no need for centroid
            points_of_cluster = X[labels == k]  # select point of one cluster
            centroids[k] = np.mean(points_of_cluster, axis=0)  # centroid of the selected cluster

    return centroids  # return the coordinates of the centroids of the clusters as an array


def nearestNeighborsClean(X, Y):
    """
    Searches for the nearest neighbour of each point in a dataset amongst the points of another dataset. 
    
    Calculates the distance between each point and its nearest neighbour.
    
    Only one channel at a time.
    
    Parameters:
        X: Coordinates of points that make up the pool of potential neighbours for the nearest neighbour search (2D array, float).
        Y: Coordinates of points we need to find the nearest neighbours of (2D array, float).
    
    Output:
        Distances: Distance to the nearest neighbour of each point of Y in X (1D array, float). 
        If the two datasets X and Y are the same, this function will return the second nearest 
            neighbours to each point, as the first neighbour would be the point itself.
    """

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)  # basic the array in which we search the NN
    # neighbors=2 bcs neighbors=1 is just the point
    distances, indices = nbrs.kneighbors(Y)  # search the NN of the points in Y in X

    if np.array_equal(X, Y):  # if the two arrays X and Y are the same
        return distances[:, 1]  # return the second column of distances (true NN and not 0 for point itself)
    else:
        return distances[:, 0]  # return first column, no problem


def nearestNeighborsMult(X, Y, nbNeighbors):
    """
    Searches the selected number of nearest neighbours for each point in one dataset amongst the points of another dataset. 
    
    Calculates the distance between each point and its nearest neighbours.
    
    One channel at a time only.
    
    Parameters:
        X: Coordinates of points that make up the pool of potential neighbours for the nearest neighbour search (2D array, float).
        Y: Coordinates of points we need to find the nearest neighbours of (2D array, float).
        nbNeighbors: Number of neighbours to find for each point of Y (int).
    
    Output:
        Distances: distance the nearest neighbours of each point of Y in X (nbNeighbors-D array, float). 
            If the two datasets X and Y are the same, this function will return the second to nbNeighbours+1 
            nearest neighbours to each point, as the first neighbour would be the point itself.
    """

    nbrs = NearestNeighbors(n_neighbors=nbNeighbors + 1, algorithm='auto').fit(X)
    # basic the array in which we search the NN
    # neighbors=2 bcs neighbors=1 is just the point
    distances, indices = nbrs.kneighbors(Y)  # search the NN of the points in Y in X

    if np.array_equal(X, Y):  # if the two arrays X and Y are the same
        return distances[:, 1:nbNeighbors + 1]
        # return the second column of distances (true NN and not 0 for point itself)
    else:
        return distances[:, 0:nbNeighbors]  # return first column, no problem


def nearestNeighborsOneInAll(dictPoints, dictLocs):
    """
    Searches the first nearest neighbour of each point in one dataset amongst the points of the datasets 
    of another dictionary, sequentially for all the datasets of that dictionary. 
    
    Calculates the distance between each point and its nearest neighbour.
    
    Only handles one channel in multiple.
    
    Parameters:
        dictPoints: Coordinates of points in for which we want the nearest neighbour found (dictionary with a unique 2D array, float).
        dictLocs: Coordinates of points that are the pool of potential neighbours for the nearest neighbour search (dictionary of 2D array, float).
        
    
    Output:
        dictDist: Distance the nearest neighbours of each point of dictPoints in each array of dictLocs (dictionary of 1D array, float).
    """

    dictDist = {}  # future dictionary of distances to NN for combinaisons of arrays

    for i in dictLocs.keys():

        nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(dictLocs[i])
        # basic the array in which we search the NN
        # neighbors=2 bcs neighbors=1 is just the point if we search in the same array

        distances, indices = nbrs.kneighbors(list(dictPoints.values())[0])
        # search the NN of the points in dictPoints inside ieme array of dictLocs

        # WARNING : THE KEYS HAVE TO BE THE SAME IF THE ARRAYS ARE THE SAME
        if list(dictPoints.keys())[0] == i:  # if the two arrays X and Y are the same - by name
            dictDist[list(dictPoints.keys())[0] + '-' + i] = distances[:, 1]
            # second column of distances (true NN and not 0 for point itself if same array)
        else:
            dictDist[list(dictPoints.keys())[0] + '-' + i] = distances[:, 0]
            # first column, no problem

    return dictDist


def dictionaryToCsv(dictFunct, nameCsv):
    """
    Converts a dictionary of arrays into a dataframe and then saves it as a unique CSV file. 
    To make this possible, the dimensions of the arrays are adjusted to the length of the longest one. 
    It is done by adding -1 to the end of the arrays until they reach the length of the longest one. 
    
    The CSV file is saved in the folder indicated by the path included in nameCsv.
    
    Handles multiple channels, but everything is saved to the same CSV file.
    
    Parameters:
        dictFunct: Dictionary of arrays to save as a unique CSV file (dictionary of 1D arrays).
        nameCsv: Name of the new CSV file (str), including or not the path to a specific folder in which it will be saved.
    
    Output:
        CSV file.
    """

    maxLength = max(len(dictFunct[i]) for i in dictFunct.keys())
    # length of the biggest array in the dictionary 

    dfDistances = pd.DataFrame()

    for i in dictFunct.keys():
        # for each of the channels in the dictionary
        dictFunct[i] = np.append(dictFunct[i], np.repeat(-1, maxLength - len(dictFunct[i])))
        # add -1 distances to fill the empty spaces of the arrays => put everything to the same length
        dfDistances.insert(len(dfDistances.columns), i, dictFunct[i])
        # add to the dataframe

    dfDistances.to_csv(nameCsv)  # save dataframe as csv
    
    
def dictionaryToHdf5(inputDict, savePath, dictionaryNames):
    """
    Saves localization data (numpy arrays) to hdf5 files

    Parameters
    ----------
    inputDict: The dictionary of localization data as numpy arrays
    savePath: The path to save hdf5 files
    dictionaryNames: The dictionary of names to label the hdf5 files
    """

    for key, value in inputDict.items():
        name = ""

        # Apply naming scheme based on dictionaryNames
        for i in dictionaryNames.keys():
            if key.casefold().find(i) >= 0:
                #if key.casefold().find('random') >= 0:
                #    name = dictionaryNames[i]+'random' 
                #else:
                name = dictionaryNames[i]

                #if key.casefold().find('centroid') >= 0:
                #    name = name + '_centroids'
       
        # Reverse transform of the to_numpy()*130 
        df = pd.DataFrame(value/130, columns=['x', 'y'])

        # Save the dataframe to hdf5 at savePath
        df.to_hdf(savePath, key='locs', mode='w')


def alpha_shape(points, alpha, only_outer=True):
    """
    
    Compute the alpha shape (concave hull) of a set of points.
    
    :param points: np.array of shape (n,2) points.
    
    :param alpha: alpha value.
    
    :param only_outer: boolean value to specify if we keep only the outer border
    
    or also inner edges.
    
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    
    the indices in the points array.
    """

    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """
        Add an edge between the i-th and j-th points,
        if not in the list already
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def partOfYourWorld(x, y, indexEdges, points):
    """
    Determinates if a point is inside or outside a predetermined shape. 
    The shape is delimited by the couples of points that make up its edges. 
    
    Only handles one channel at a time.
    
    Parameters: 
        X and Y: Coordinates of the point (float)
        indexEdges: Index of points in the dataset forming a part of the edge of the shape. 
            The couples of points are the two ends of a segment of the polyhedric edge (2D array, int).
        points: Coordinates of the points in the experimental dataset (2D array, float). 
            They must be the points the shape was extrapolated from.
    
    Output: 
        Boolean value indicating if the point is inside or outside the shape. 
            The function returns TRUE if the point is inside the shape and FALSE if it is outside. 
    """

    xEdgesInf = np.empty((50, 2), int);
    inf = 0  # edges below the point (xaxis)
    xEdgesSup = np.empty((50, 2), int);
    sup = 0  # edges above the point (xaxis)

    for i, j in indexEdges:  # for each segment composing the edge

        # x axis
        if min(points[i, 0], points[j, 0]) <= x < max(points[i, 0], points[j, 0]):
            # if the point is between the two ends of the segment
            if stat.mean([points[i, 1], points[j, 1]]) <= y:
                # if the point is below the segment (yaxis)
                xEdgesInf[inf] = [i, j]
                # add the segemnt to the array of segments below the point (xaxis)
                inf = inf + 1
            else:
                xEdgesSup[inf] = [i, j]
                # = above = 
                sup = sup + 1

    xEdgesInf = np.delete(xEdgesInf, np.s_[inf:len(xEdgesInf)], 0)
    xEdgesSup = np.delete(xEdgesSup, np.s_[inf:len(xEdgesSup)], 0)

    if ((len(xEdgesInf) % 2 == 0) or (len(xEdgesSup) % 2 == 0)):
        # if there is an even number of edges on any side of the point
        # then the point is outside the cell
        return False
    else:
        # if there is an odd number of edges on all sides of the point
        # then it is inside the cell
        return True


def randomPartOfYourWorld(indexEdges, points):
    """
    Draws a pool of random points between the minimum and maximum coordinates of points in the experimental dataset for both axes. 
    For every each of the points, it checks if the point is inside the shape and keep it if it is. 
    Repeats until there is the same number of random points and experimental points inside the shape. 
    
    The function will stop if all the points originally drawn have been assessed. 
    The default number of points drawn by this function is four times the number of points in the original dataset. 
    It should be enough to have the same number of random points than experimental points, but not guaranteed. 
    A warning message and ‘count of random points / count of experimental points’ inside the cell will be displayed in the console. 
    The factor of random points drawn can be adjusted using the variable factorExpRandPoints.
    
    Only handles one channel at a time.
    
    Parameters: 
        indexEdges: Index of points in the dataset forming a part of the edge of the shape. 
            The couples of points are the two ends of a segment of the polyhedric edge (2D array, int).
        points: Coordinates of points in the experimental dataset (2D array, float). 
            They must be the points the shape was extrapolated from.
    
    Output:
        randomPoints: Coordinates of random points inside the shape (2D array, float). 
            Except error in the number of points drawn, there will be the same number of random points than experimental points. 
        Text display if an error occurred in the number of random points originally drawn.
    """

    # parameters for random.randrange => ranges for coordinates of random point
    xmin = math.ceil(min(points[:, 0]));
    xmax = math.ceil(max(points[:, 0]))
    ymin = math.ceil(min(points[:, 1]));
    ymax = math.ceil(max(points[:, 1]))

    # empty array for the random points inside the cell
    randomPoints = np.empty((len(points), 2), float)

    # random draws outside the loop to reduce running time (a bit)

    factorExpRandPoints = 4

    hold = np.empty((len(points) * factorExpRandPoints, 2), float)
    for i in range(0, len(hold)):
        hold[i] = [random.randrange(xmin * 100, xmax * 100) / 100,
                   random.randrange(ymin * 100, ymax * 100) / 100]

    # checks if the points are inside or outside the shape

    count = -1

    for i in tqdm(range(0, len(points))):

        valBool = False
        # hold = np.empty((0,2), float)

        while valBool != True:
            count = count + 1

            valBool = partOfYourWorld(hold[count, 0], hold[count, 1], indexEdges, points)
            # valBool = True if the point is inside the cell

        randomPoints[i] = [hold[count, 0], hold[count, 1]]
        # add the point to the array of random points inside the cell

        if count == len(hold) - 1:
            print('not enough random points initially drawn')
            print('count of random points inside the shape : %d / %d' % (i, len(points)))
            break

        continue

    return randomPoints


def randomDistributionAll(points, alphaParameter):
    """
    Calculates the alpha shape for the points of an experimental dataset. 
    
    Generates an array of random points inside that shape. The number of random and experimental 
    points is equal unless the initial number of random points drawn is insufficient.
    
    Only handles one channel at a time.
    
    Parameters:
        points: Coordinates of points in the experimental dataset (2D array, float).
        alphaParameter: Parameter defining the precision of the edges for the alpha shape (int).
    
    Output:
        randomPoints: Coordinates of random points inside the shape (2D array, float). 
            Except error in the number of points drawn, there will be the same number of random points than experimental points. 
        Text display if an error occurred in the number of random points originally drawn.
    """

    edges = alpha_shape(points, alpha=alphaParameter, only_outer=True)  # alphaParameter = 700 is nice for now
    indexEdges = np.array(list(edges))  # convertion of edges from list to np array

    randomPoints = randomPartOfYourWorld(indexEdges, points)
    # array of random points inside a cell, the number of random and experimental points is equal

    # display figure with :
    # edges of experimetal cell for the alphaParameter
    # random points generated
    plt.figure(figsize=(50, 50))
    plt.plot(randomPoints[:, 0], randomPoints[:, 1], '.')
    for i, j in edges:
        plt.plot(points[[i, j], 0], points[[i, j], 1], color='black')
    plt.show()

    return randomPoints


def ripleyParametersForClustering(array, cuts, path):
    """
    Calculates the radius of clusters for a maximal clustering using Ripley’s H function. 
    The results of Ripley’s H function for the study areas are displayed in a unique figure for each of the channels to cluster.
    
    The function also calculates the minimum number of points in clusters for the radius of maximal clustering. 
    This automatic minimum number of points for clustering is the mean number of neighbours of a point in the radius of maximum clustering. 
    The number of neighbours for each point of a dataset is determined using spat.cKDTree(i).query_ball_point. 
    The final value is the mean of the results for each of the areas passed as parameters with cuts.
    
    Only handles one channel at a time.
    
    Parameters:
        array: Coordinates of the points to cluster (2D array, float).
        cuts: Coordinates of the bottom left corners of the study areas for Ripley’s H (2D array, float). 
            The dimension of the areas is 1500x1500nm.
        path: Path to folder where the figures will be saved, including the name of the new file (str). 
    
    Output:
        stat.mean(radius): Mean radius of the clusters calculated for each area with Ripley’s H.
        stat.mean(nb): Mean number of neighbours for points in each of the areas for the radius of maximum clustering.
        Graphic display (Ripley’s H functions graph).
    """

    radius = []
    nb = []

    # radii to test for ripley's function
    r = np.linspace(0, 400, 800)

    plt.figure(figsize=(15, 5))
    plt.plot(r, np.zeros(len(r)), '--', color='red')

    # for each of the areas defined in cuts
    for i, j in tqdm(cuts):

        # points of the array in the area (1500x1500 nm)
        dataTest = array[(array[:, 0] >= i) & (array[:, 0] <= i + 1500) &
                         (array[:, 1] >= j) & (array[:, 1] <= j + 1500)]

        # definition of the study area parameter for ripley's function
        x_minTest = min(dataTest[:, 0]);
        x_maxTest = max(dataTest[:, 0])
        y_minTest = min(dataTest[:, 1]);
        y_maxTest = max(dataTest[:, 1])
        areaTest = (x_maxTest - x_minTest) * (y_maxTest - y_minTest)

        if areaTest > 0:

            # implementing ripley's H function
            R = RipleysKEstimator(areaTest, x_maxTest, y_maxTest, x_minTest, y_minTest)
            ripleyH = R.Hfunction(data=dataTest, radii=r, mode='none')

            # the max of ripley's H function is the cluster radius of maximum clustering
            # = radius of clusters for DBSCAN
            if np.argmax(ripleyH) > 0:  # if clustered (clusterd if ripley's H above 0)

                radius = np.append(radius, r[np.argmax(ripleyH)])

                # nb = np.append(radius, math.ceil(stat.median(spat.cKDTree(dataTest).query_ball_point(dataTest, r = r[np.argmax(ripleyH)], return_length=True))))
                nb = np.append(nb, math.ceil(stat.mean(
                    spat.cKDTree(dataTest).query_ball_point(dataTest, r=r[np.argmax(ripleyH)], return_length=True))))
                # the min number of points is the median of the number of neighbors
                # calculated for each poit of the channel      

            plt.plot(r, ripleyH, label='Ripley H, study area ' + str(len(radius)))

        continue

    plt.legend(fontsize='large', loc='upper right')
    plt.xlabel('Radius (nm)')
    plt.ylabel('Ripley H')
    plt.savefig(path + '.png')  # save figure as png in specified path

    if len(radius) > 0:
        if len(nb) > 0:
            return stat.mean(radius), stat.mean(nb)
        else:
            return stat.mean(radius), 0
    elif len(nb) > 0:
        return 0, stat.mean(nb)
    else:
        return 0, 0


def MultChannelsCallToDict(path, dictionaryNames):
    """
    Imports the CSV files in path and makes an array from the coordinates in each file. 
    These arrays are added to a shared dictionary regrouping the coordinates of points for all imported channels. 
    The arrays are named after the lectin (or other) used to image that channel (e.g. R1WGA) using the dictionary of names provided. 
    
    Handles multiple channels. They are kept in different arrays but the same dictionary.
    
    Parameters:
        path: Path to folder containing the desired csv files (str).
        dictionaryNames: Dictionary of equivalences between names of files and future names of variables, 
            dictionaries and files (dictionary of str).
    
    Output:
        dictionaryLocalizations: Coordinates of points for each channel in path (dictionary of 2D arrays, float).
    """

    # array of all csv files in this folder (path to the files)
    filePaths = np.array(glob.glob(path + '/*.csv'))

    # new empty dictionary of localizations
    dictionaryLocalizations = {}

    for i in filePaths:
        # names of the files only (without extention or path)
        name = Path(i).stem

        namePath = Path(i).stem

        # import locations (csv)
        df = pd.read_csv(i)
        # coordinates of points only => in a np array
        df = df[['x [nm]', 'y [nm]']].to_numpy()

        # search for recognisable part in name of the dict and makes new names accordingly
        # names of lectins and state of randomness
        for i in dictionaryNames.keys():
            if namePath.casefold().find(i) >= 0:
                if namePath.casefold().find('random') >= 0:
                    name = dictionaryNames[i] + 'random'
                else:
                    name = dictionaryNames[i]

                if namePath.casefold().find('centroid') >= 0:
                    name = name + '_centroids'

                    # add arry to dictionary of localizations
        dictionaryLocalizations[name] = df

    return dictionaryLocalizations

def MultChannelsCallToDict_hdf5(path, keiy):
    """
    imports localization files with hdf5

    Parameters
    ----------
    path : TYPE
        DESCRIPTION.
    key : TYPE
        DESCRIPTION.
    dictionaryNames : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
   
    # filePaths = np.array(glob.glob(path+'/*.hdf5'))
    # print(filePaths)
    
    #new empty dictionary of localizations 
    path = Path(path)
    dictionaryLocalizations = {}
    library = []
    for file in path.glob("*.hdf5"):
        #Get the file stem for naming the keys
        file_name = file.stem.split("_")[0]
        #read the data 
        df = pd.read_hdf(file, key = keiy)
        df = df[['x','y']].to_numpy()*130
        #append the dict 
        dictionaryLocalizations[file_name.upper()] = df
        library.append(file_name)
    order = ["WGA", "SNA", "PHAL", "AAL", "PSA"]
    reordered_data = {
    **{key: dictionaryLocalizations[key] for key in order if key in dictionaryLocalizations},
    **{key: dictionaryLocalizations[key] for key in dictionaryLocalizations if key not in order}
}
    
    # for i in filePaths:
    #     #names of the files only (without extention or path)
    #     name = Path(i).stem
        
    #     namePath = Path(i).stem
        
    #     df = pd.read_hdf(i, key=keiy)
        
    #     df = df[['x','y']].to_numpy()*130
        
        
    #     #search for recognisable part in name of the dict and makes new names accordingly
    #     #names of lectins and state of randomness
    #     for i in dictionaryNames.keys():
    #         if namePath.casefold().find(i) >= 0:
    #             if namePath.casefold().find('random') >= 0:
    #                 name = dictionaryNames[i]+'random' 
    #             else:
    #                 name = dictionaryNames[i]
                    
    #             if namePath.casefold().find('centroid') >= 0:
    #                 name = name + '_centroids' 

    #     #add arry to dictionary of localizations 
    #     dictionaryLocalizations[name] = df
        
    return reordered_data

        
        
      
