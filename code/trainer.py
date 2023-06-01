###############
### IMPORTS ###
###############

import os
from model import *

#################
### COMSTANTS ###
#################

file_data = '..' + os.sep + 'data' + os.sep + 'meta_data' + os.sep + 'metadata_withmasks.csv'
image_folder = '..' + os.sep + 'images' + os.sep + 'img' + os.sep
mask_folder = '..' + os.sep + 'images' + os.sep + 'mask' + os.sep
file_features = '..' + os.sep + 'data' + os.sep + 'feature_data' + os.sep + 'feature_data.csv'

feature_names = ['mean_assymmetry', 'best_asymmetry', 'worst_asymmetry', 'red_var', 'green_var', \
     'blue_var', 'hue_var', 'sat_var', 'val_var', 'dom_hue', 'dom_sat', 'dom_val', \
     'compactness', 'convexity', 'F1', 'F2', 'F3', 'F10', 'F11', 'F12']

##########################
###   PROCESS IMAGES   ###
##########################

# Extract features for all images with masks and save as csv.
# ProcessImages(file_data, image_folder, mask_folder, file_features, feature_names)

############################
###   TRAIN CLASSIFIER   ###
############################

# Metadata
df = pd.read_csv(file_data)
df = df[df['mask'] == 1]

# Labels
labels = df['diagnostic']

# Feature data
df_features = pd.read_csv(file_features)

# X and y
X = df_features[feature_names]
y = (labels == 'BCC') | (labels == 'SCC') | (labels == 'MEL') 

print('X shape: ', X.shape)

# Standardise scaler
train_scaler(X)
X_scaled = apply_scalar(X)
print('X_scaled shape:', X_scaled.shape)

# PCA
train_pca(X_scaled)
X_transformed = apply_pca(X_scaled)
print('X transformed shape (after PCA)', X_transformed.shape)

# Feature selection
train_feature_selector(X_transformed, y, 4)
X_transformed = apply_feature_selector(X_transformed)
print('X transformed shape (after feature selector)', X_transformed.shape)

# Train classifier
clf = [KNeighborsClassifier(n_neighbors = 7)]
trained_classifiers = train_clf(X_transformed, y, clf)

trained_clf = trained_classifiers[0]

# Save final classifier
pk.dump(trained_clf, open('classifier.pkl', 'wb'))
