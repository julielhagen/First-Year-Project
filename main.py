### UNFINISHED !!!

###############
### IMPORTS ###
###############

import os

from MyModel import ProcessImages


#################
### COMSTANTS ###
#################

file_data = 'metadata_withmasks.csv'
image_folder = 'images' + os.sep
mask_folder = 'images_masks' + os.sep
file_features = 'feature_data.csv'

ProcessImages(file_data, image_folder, mask_folder, file_features)