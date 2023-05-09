# Imports
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
from skimage.transform import resize, rescale
from prep_image import prep_im

# Load images
image_folder_path = "imgs_part_1"
n_images = 200
paths = [f for f in listdir(image_folder_path) if isfile(join(image_folder_path, f))][:n_images]

images = []
for im_path in paths:
  image = prep_im(im_path, "imgs_part_1/", output_shape = (300,300))
  arr = np.asarray(image)
  images.append(arr)

# Load labels for images
data = np.array([i.strip().split(',') for i in open('metadata.csv')])

mask = data == ''
data[np.where(mask)] = np.nan

y = np.asarray([data[np.where(data[:,-2]==paths[i])[0][0],17] for i in range(len(paths))])

# Flatten it, now each row represents a single image
X = np.stack(images, axis = 0)

dim1, dim2, chan = arr.shape
n_features = chan*dim1*dim2
X = X.reshape((len(images), n_features)) # flattened --> this goes to PCA 

# Init the model (a.k.a. specify the hyper-parameters e.g. number of components)
final_n_features = 10 # Hyper-parameter - try different values
pca = PCA(n_components=final_n_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state =1,stratify = y)

# Transformed features
X_train_transformed = pca.fit_transform(X_train) # X_new has final_n_features --> this can be fed to the classfier model

X_test_transformed = pca.fit_transform(X_test) # X_new has final_n_features --> this can be fed to the classfier model

# Define a classifer
clf = KNeighborsClassifier(n_neighbors=10)

# Train it --> need to define y first
clf.fit(X_train_transformed, y_train)

# Predict on validation dataset and measure accuracy, f1-score
clf.predict(X_test_transformed);

print(clf.score(X_test_transformed, y_test))

