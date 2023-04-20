import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale

def prep_im_and_gt(im_id, im_dir_path, gt_dir_path, scalar = 1):
    '''Prepare image and corresponding ground truth segmentation from test images. 
    Paths to directories containing image and ground truth files required.
    If parameter scalar is passed, output image will be scaled by it. Defualt 1 retains original size.

    Args:
        im_id (str): image ID
        im_dir_path (str): image directory path 
        gt_dir_path (str): ground thruth directory path
        scalar (float, optional): rescale coefficient

    Returns:
        im (numpy.ndarray): image
        gt (numpy.ndarray): ground truth segmentation.
    '''

    # Read and resize image
    im = plt.imread(im_dir_path + im_id + ".png")[:, :, :3] #Some images have fourth, empty color chanel which we slice of here
    im = rescale(im, scalar, anti_aliasing=True)

    #Read and resize ground truth segmentation
    gt = plt.imread(gt_dir_path + im_id + "_GT.png")
    gt = rescale(gt, scalar, anti_aliasing=False)

    #Return GT to binary
    binary_gt = np.zeros_like(gt)
    binary_gt[gt > .5] = 1
    gt = binary_gt.astype(int)

    return im, gt

def prep_im(im_id, im_dir_path = "", scalar = 1):
    '''Prepare image from im_id and optional dictory path.
    If directory path is not passed, the whole filepath, including filetype notation, 
    should be given as im_id. If parameter scalar is passed, output image will be scaled by it. 
    Defualt 1 retains original size.
    
    Args:
        im_id (str): image ID
        im_dir_path (str, optional): image directory path
        scalar (float, optional): rescale coefficient

    Returns:
        im (numpy.ndarray): image.
    '''

    # Read and resize image
    if im_dir_path == "":
        im = plt.imread(im_id)[:, :, :3] #Some images have fourth, empty color chanel which we slice of here
    else:
        im = plt.imread(im_dir_path + im_id + ".png")[:, :, :3] #Some images have fourth, empty color chanel which we slice of here
    im = rescale(im, scalar, anti_aliasing=True)

    return im

def prep_gt(im_id, gt_dir_path = "", scalar = 1):
    '''Prepare ground truth segmentaion from im_id and optional dictory path.
    If directory path is not passed, the whole filepath, including filetype notation, 
    should be given as im_id. If parameter scalar is passed, output image will be scaled by it. 
    Defualt 1 retains original size.
    
    Args:
        im_id (str): image ID
        gt_dir_path (str, optional): ground truth directory path
        scalar (float, optional): rescale coefficient

    Returns:
        gt (numpy.ndarray): ground truth segmentation.
    '''

    # Read and resize image
    if gt_dir_path == "":
        gt = plt.imread(im_id) #Some images have fourth, empty color chanel which we slice of here
    else:
        gt = plt.imread(gt_dir_path + im_id + "_GT.png")
    gt = rescale(gt, scalar, anti_aliasing=False)

    # Return GT to binary
    binary_gt = np.zeros_like(gt)
    binary_gt[gt > .5] = 1
    gt = binary_gt.astype(int)

    return gt