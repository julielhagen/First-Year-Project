# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from cut import cut_im_by_mask

def color_dominance(image, mask, clusters = 1, include_percentages = False):
    '''Get the most dominent colors of the cut image that closest sorrounds the lesion using KMeans

    Args:
        image (numpy.ndarray): image to compute dominent colors for
        mask (numpy.ndarray): mask of lesion
        clusters (int, optional): amound of clusters and therefore dominent colors (defualt 1)
        include_percentages (bool, optional): whether to include the domination percentages for each color (defualt False)

    Return:  
        if include_percentages == True: 
            p_and_c (list): list of tuples, each containing the percentage and RGB array of the dominent color
        else: 
            dom_colors (array): array of RGB arrays of each dominent color.
    '''
    
    cut_im = cut_im_by_mask(image, mask) # Cut image to remove excess skin pixels
    flat_im = np.reshape(cut_im, (-1, 3)) # Flatten image to 2D array

    # Use KMeans to cluster image by colors
    k_means = KMeans(n_clusters=clusters, random_state=0)
    k_means.fit(flat_im)

    # Save cluster centers (dominant colors) in array
    dom_colors = np.array(k_means.cluster_centers_, dtype='float32') 

    if include_percentages:

        counts = np.unique(k_means.labels_, return_counts=True)[1] # Get count of each dominent color
        percentages = counts / flat_im.shape[0] # Get percentage of total image for each dominent color

        p_and_c = zip(percentages, dom_colors) # Percentage and colors
        p_and_c = sorted(p_and_c, reverse=True) # Sort in descending order

        return p_and_c
    
    return dom_colors

def plot_dominance(p_and_c):
    '''Plot dominance bar from percentage and count list.
    Necessitates percentages in input.'''

    bar = np.ones((50, 500, 3), dtype='float32')
    plt.figure(figsize=(12,8))
    plt.title('Proportions of Dominent Colors in the Image')
    start = 0
    i = 1
    for percentage, color in p_and_c:
        end = start + int(percentage * bar.shape[1])
        if i == len(p_and_c):
            bar[:, start:] = color[::-1]
        else:
            bar[:, start:end] = color[::-1]
        start = end
        i += 1

    plt.imshow(bar)
    plt.tick_params(
    axis='both',
    which='both',
    bottom=False,
    left=False,
    labelbottom=False,
    labelleft=False)