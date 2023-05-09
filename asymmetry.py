# Imports
import numpy as np
import matplotlib.pyplot as plt
from math import floor, ceil
from scipy.ndimage import rotate
from cut import cut_edges

def midpoint(mask):
    '''Find midpoint of mask array.'''
    row_mid = mask.shape[0] / 2
    col_mid = mask.shape[1] / 2
    return row_mid, col_mid

def plot_midpoint(mask):
    '''Cut mask and plot midpoint.'''
    cut_mask = cut_edges(mask)
    y, x = midpoint(cut_mask)
    plt.imshow(cut_mask, cmap="gray")
    plt.axvline(x = x, color = "r")
    plt.axhline(y = y, color = "r")

def asymmetry(mask):
    '''Calculate asymmetry score between 0 and 1 from vertical and horizontal axis
    on a binary mask, 0 being complete symmetry, 1 being complete asymmetry 
    (no pixels overlap when folding mask on x- and y-axis)

    Args:
        mask (numpy.ndarray): input mask
    
    Returns:
        asymmetry_score (float): Float between 0 and 1 indicating level of asymmetry.    
    '''

    row_mid, col_mid = midpoint(mask)

    # Split mask into halves hortizontally and vertically
    upper_half = mask[:ceil(row_mid), :]
    lower_half = mask[floor(row_mid):, :]
    left_half = mask[:, :ceil(col_mid)]
    right_half = mask[:, floor(col_mid):]

    # Flip one half for each axis
    flipped_lower = np.flip(lower_half, axis=0)
    flipped_right = np.flip(right_half, axis=1)

    # Use logical xor to find pixels where only one half is present
    hori_xor_area = np.logical_xor(upper_half, flipped_lower)
    vert_xor_area = np.logical_xor(left_half, flipped_right)

    # Compute sums of total pixels and pixels in asymmetry areas
    total_pxls = np.sum(mask)
    hori_asymmetry_pxls = np.sum(hori_xor_area)
    vert_asymmetry_pxls = np.sum(vert_xor_area)

    # Calculate asymmetry score
    asymmetry_score = (hori_asymmetry_pxls + vert_asymmetry_pxls) / (total_pxls * 2)

    return round(asymmetry_score, 4)

def rotation_asymmetry(mask, n: int):
    '''Rotate mask n times and calculate asymmetry score for each iteration.
    Rotates n times between 0 and 90 degrees, as 90 degree rotations do not change the
    asymmetry score, i.e., a 30 degree rotation is the same as a 120 degree rotation.

    Args:
        mask (numpy.ndarray): input mask  
        n (int): amount of rotations

    Returns:
        asymmetry_scores (dict): dict of asymmetry scores calculated from each rotation.    
    '''

    asymmetry_scores = {}

    for i in range(n):

        degrees = 90 * i / n

        rotated_mask = rotate(mask, degrees)
        cut_mask = cut_edges(rotated_mask)

        asymmetry_scores[degrees] = asymmetry(cut_mask)

    return asymmetry_scores

def plot_asymmetry(mask, rotations = 90):
    '''Plot asymmetry scores for each degree in steps 90/rotations.'''
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    plt.scatter(asymmetry_scores.keys(), asymmetry_scores.values())

def best_asymmetry(mask, rotations = 90):
    '''Return best (lowest) asymmetry score from mask.
    Optional argument (defualt 90) rotations decides amount of rotations in asymmetry calculation
    
    Args:
        mask (numpy.ndarray): mask to compute asymmetry score for
        rotations (int, optional): amount of rotations

    Returns:
        best_score (float): best asymmetry score.
    '''
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    best_score = min(asymmetry_scores.values())

    return best_score

def worst_asymmetry(mask, rotations = 90):
    '''Return worst (highest) asymmetry score from mask.
    Optional argument (defualt 90) rotations decides amount of rotations in asymmetry calculation
    
    Args:
        mask (numpy.ndarray): mask to compute asymmetry score for
        rotations (int, optional): amount of rotations

    Returns:
        worst_score (float): worst asymmetry score.
    '''
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    worst_score = max(asymmetry_scores.values())

    return worst_score

def mean_asymmetry(mask, rotations = 90):
    '''Return mean asymmetry score from mask.
    Optional argument (defualt 90) rotations decides amount of rotations in asymmetry calculation
    
    Args:
        mask (numpy.ndarray): mask to compute asymmetry score for
        rotations (int, optional): amount of rotations

    Returns:
        mean_score (float): mean asymmetry score.
    '''
    asymmetry_scores = rotation_asymmetry(mask, rotations)
    mean_score = sum(asymmetry_scores.values()) / len(asymmetry_scores)

    return mean_score
