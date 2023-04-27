import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
from os.path import isfile, join
import os

# Load the data
image_folder_path = "test_images"
n_images = 100
paths = [f for f in listdir(image_folder_path) if isfile(join(image_folder_path, f))][:n_images]

print(paths)

