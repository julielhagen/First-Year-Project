### UNFINISHED !!!

###############
### IMPORTS ###
###############

import os
from MyModel import *


#################
### COMSTANTS ###
#################

file_data = '..' + os.sep + 'metadata_withmasks.csv'
image_folder = '..' + os.sep + 'images' + os.sep
mask_folder = '..' + os.sep + 'images_masks' + os.sep
file_features = '..' + os.sep + 'feature_data.csv'


feature_names = ['mean_assymmetry', 'best_asymmetry', 'worst_asymmetry', 'red_var', 'green_var', \
     'blue_var', 'hue_var', 'sat_var', 'val_var', 'dom_hue', 'dom_sat', 'dom_val', \
     'compactness', 'convexity', 'F1', 'F2', 'F3', 'F10', 'F11', 'F12']


### PROCESS IMAGES ###

# Extract features for all images with masks and save as csv.
ProcessImages(file_data, image_folder, mask_folder, file_features, feature_names)


######################################

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


### TRAIN MODEL ###

# Split in train test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)


#groups = X_train['patient_id']
#X_train = X_train[feature_names]
#X_test = X_test[feature_names]

# Feature selection
#train_feature_selector(X_train, y_train, 10)
#X_train = apply_feature_selector(X_train)

# PCA
#train_pca(X_train)
#X_train_transformed = apply_pca(X_train)

#print(np.shape(X_train_transformed))

# Define classifiers
#classifiers = [KNeighborsClassifier(n_neighbors=i) for i in range(1, 20, 2)]

# Cross Validate
#cv_results = cross_validate_clf(X_train_transformed, y_train, classifiers, groups)
#print_results(cv_results)

# PCA
train_pca(X_train)
X_train_transformed = apply_pca(X_train)

# Feature selection
train_feature_selector(X_train_transformed, y_train, 3)
X_train_transformed = apply_feature_selector(X_train_transformed)

# Train classifier
clf = [KNeighborsClassifier(n_neighbors = 9)]
trained_classifiers = train_clf(X_train_transformed, y_train, clf)

#Evaluate model
X_test_transformed = apply_pca(X_test)
X_test_transformed = apply_feature_selector(X_test_transformed)

evaluation_results = evaluate_clf(X_test_transformed, y_test, trained_classifiers)

print_results(evaluation_results)

# Train and save final classifier


