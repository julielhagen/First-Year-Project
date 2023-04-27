import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
from os.path import isfile, join
import os

# Load the data
image_folder_path = "imgs_part_1"
n_images = 100
paths = [f for f in listdir(image_folder_path) if isfile(join(image_folder_path, f))][:n_images]


# TODO: load labels for images
# y = [lab_of_image1, ...]
data = np.array([i.strip().split(',') for i in open('metadata.csv')][:n_images])
mask = data == ''
data[np.where(mask)] = np.nan

print(data)

# Load images
#images = []
#for im_path in paths:
#  image = Image.open(join(image_folder_path, im_path))
#  arr = np.asarray(image)
#  arr = arr[:180, :180, :3] # Make sure the image has the same size
#  images.append(arr)

# Turn the list of images (a.k.a. list of 3D np arrays) into a 4D np array
#X = np.stack(images, axis = 0)

# Flatten it, now each row represents a single image
#dim1, dim2, chan = arr.shape
#n_features = chan*dim1*dim2
#X = X.reshape((len(images), n_features)) # flattened --> this goes to PCA 

# Init the model (a.k.a. specify the hyper-parameters e.g. number of components)
#final_n_features = 10 # Hyper-parameter - try different values
#pca = PCA(n_components=final_n_features)

# Train-test split

# Transformed features
#X_new = pca.fit_transform(X) # X_new has final_n_features --> this can be fed to the classfier model

# Define a classifer
#clf = KNeighborsClassifier(n_neighbors=3)

# Train it --> need to define y first
# clf.fit(X_new, y)

# TODO: Predict on validation dataset and measure accuracy, f1-score
