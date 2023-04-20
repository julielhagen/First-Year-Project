***************
*** IMPORTS ***
***************

import numpy as np
import matplotlib.pyplot as plt


# Image functions
from skimage.color import rgb2gray
from skimage import filters
from skimage.transform import resize
from skimage.filters import gaussian

**************
*** IMAGES ***
**************

def prepare_im(im_id):

  im = plt.imread('example_imgs/' + im_id + '.png')
  im = resize(im, (im.shape[0] // 4, im.shape[1] // 4), anti_aliasing=True)
 
  gt = plt.imread('example_segmentation/' + im_id + '_GT.png')
  gt = resize(gt, (gt.shape[0] // 4, gt.shape[1] // 4), anti_aliasing=False) #Setting it to True creates values that are not 0 or 1


  return im, gt


*****************
*** ASYMMETRY ***
*****************



