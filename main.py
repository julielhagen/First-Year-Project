***************
*** IMPORTS ***
***************

import os
from os.path import isfile, join

import pandas as pd

# Import assymmetry
# Import color
# Import compactness
# Import convexity


file_data = 'metadata.csv'
image_folder = 'imgs_scc'

# Image paths
n_images = 10
image_ids = [f for f in os.listdir(image_folder) if isfile(join(image_folder, f))][:n_images]

# Extract image ids and labels
df = pd.read_csv(file_data)

labels = np.asarray([data[np.where(data[:,-2]==paths[i])[0][0],17] for i in range(len(paths))])
image_ids = np.asarray([data[np.where(data[:,-2]==paths[i])[0][0],-2] for i in range(len(paths))])

# Load images
images = []
for im_path in paths:
  image = prep_im(im_path, "imgs_part_1/", output_shape = (300,300))
  arr = np.asarray(image)
  images.append(arr)


features = ['image_id', 'assymmetry', 'color_r', 'color_g', 'color_b', 'compactness', 'convexity', 'cancer_type']

**************************
*** FEATURE EXTRACTION ***
**************************
def extract_features():

	return assymmetry, color_r, color_g, color_b, compactness, convexity



******************
*** Classifier ***
******************

# Train classifier

# Test classifier
# Define a classifer
clf = KNeighborsClassifier(n_neighbors=10)

# Train it --> need to define y first
clf.fit(X_train_transformed, y_train)

# Predict on validation dataset and measure accuracy, f1-score
clf.predict(X_test_transformed);

print(clf.score(X_test_transformed, y_test))
