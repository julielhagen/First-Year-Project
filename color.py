# Imports
import numpy as np
from statistics import variance, stdev
from skimage.segmentation import slic

def segments(image, mask, n_segments = 50, compactness = 0.1):
    '''Get color segments of lesion from SLIC algorithm. 
    Optional argument n_segments (defualt 50) defines desired amount of segments.
    Optional argument compactness (defualt 0.1) defines balance between color 
    and position.

    Args:
        image (numpy.ndarray): image to segment
        mask (numpy.ndarray):  image mask
        n_segments (int, optional): desired amount of segments
        compactness (float, optional): compactness score, decides balance between
            color and and position

    Returns:
        slic_segments (numpy.ndarray): SLIC color segments.
    '''
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)
    
    return slic_segments

def rgb_means_get(image, slic_segments):
    '''Get mean RGB values for each segment in a SLIC segmented image.

    Args:
        image (numpy.ndarray): original image
        slic_segments (numpy.ndarray): SLIC segmentation

    Returns:
        rgb_means (list): RGB mean values for each segment.
    '''

    max_segment_id = np.unique(slic_segments)[-1]

    rgb_means = []
    for i in range(1, max_segment_id + 1):

        #Create masked image where only specific segment is active
        segment = image.copy()
        segment[slic_segments != i] = 0

        #Get average RGB values from segment
        rgb_mean = np.mean(segment, axis = (0, 1), where = (segment != 0))
        
        rgb_means.append(rgb_mean)    
        
    return rgb_means

def color_var(image, mask, n_segments = 50, compactness = .1):
    '''Segment image by color using SLIC segmentation and get variance
    of the mean color values of each segment for red, green and blue channels. 

    Args:
        image (numpy.ndarray): image to compute color variance for
        mask (numpy.ndarray): image mask
        n_segments (int, optional): desired amount of segments
        compactness (float, optional): compactness score, decides balance between
            color and and position
    
    Returns:
        red_var (float): variance in red channel segment means
        green_var (float): variance in green channel segment means
        blue_var (float): variance in green channel segment means.
    '''

    slic_segments = segments(image, mask, n_segments, compactness)
    rgb_means = rgb_means_get(image, slic_segments)

    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) == 2:
        return 0, 0, 0

    #Seperate and collect channel means together in lists
    red = []
    green = []
    blue = []
    for rgb_mean in rgb_means:
        red.append(rgb_mean[0])
        green.append(rgb_mean[1])
        blue.append(rgb_mean[2])

    #Total mean of means for each channel
    red_mean = sum(red) / len(red)
    green_mean = sum(green) / len(green)
    blue_mean = sum(blue) / len(blue)

    #Compute variance for each channel seperately
    red_var = variance(red, red_mean)
    green_var = variance(green, green_mean)
    blue_var = variance(blue, blue_mean)

    return red_var, green_var, blue_var

def color_sd(image, mask, n_segments = 50, compactness = .1):
    '''Segment image by color using SLIC segmentation and get standard deviation
    of the mean color values of each segment for red, green and blue channels.

    Args:
        image (numpy.ndarray): image to compute color standard deviation for
        mask (numpy.ndarray): image mask
        n_segments (int, optional): desired amount of segments
        compactness (float, optional): compactness score, decides balance between
            color and and position
    
    Returns:
        red_sd (float): standard deviation in red channel segment means
        green_sd (float): standard deviation in green channel segment means
        blue_sd (float): standard deviation in green channel segment means
    '''

    slic_segments = segments(image, mask, n_segments, compactness)
    rgb_means = rgb_means_get(image, slic_segments)

    # If there is only 1 slic segment, return (0, 0, 0)
    if len(np.unique(slic_segments)) == 2:
        return 0, 0, 0

    #Seperate and collect channel means together in lists
    red = []
    green = []
    blue = []
    for rgb_mean in rgb_means:
        red.append(rgb_mean[0])
        green.append(rgb_mean[1])
        blue.append(rgb_mean[2])

    #Total mean of means for each channel
    red_mean = sum(red) / len(red)
    green_mean = sum(green) / len(green)
    blue_mean = sum(blue) / len(blue)

    #Compute standard deviation for each channel seperately
    red_sd = stdev(red, red_mean)
    green_sd = stdev(green, green_mean)
    blue_sd = stdev(blue, blue_mean)

    return red_sd, green_sd, blue_sd
