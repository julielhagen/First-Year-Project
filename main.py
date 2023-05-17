### UNFINISHED !!!

###############
### IMPORTS ###
###############

import os

from MyModel import * #ProcessImages, std_X, train_pca, apply_pca, cross_validate, print_results,


#################
### COMSTANTS ###
#################

file_data = 'metadata_withmasks.csv'
image_folder = 'images' + os.sep
mask_folder = 'images_masks' + os.sep
file_features = 'feature_data.csv'

feature_names = ['assymmetry', 'red_var', 'green_var', 'blue_var', \
    'hue_var', 'sat_var', 'val_var', 'dom_hue', 'dom_sat', 'dom_val', \
    'compactness', 'convexity']


### PROCESS IMAGES ###

# Extract features for all images with masks and save as csv.

#ProcessImages(file_data, image_folder, mask_folder, file_features)


######################################

# Metadata
df = pd.read_csv(file_data)
df = df[df['mask'] == 1]

labels = df['diagnostic']

# Feature data
df_features = pd.read_csv(file_features)

# X and y
X = df_features
y =  (labels == 'BCC') | (labels == 'SCC') | (labels == 'MEL') 


### TRAIN MODEL ###

# Split in train test
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size = 0.2, random_state = 41, stratify = y)

X_train = X_train[feature_names]
X_test = X_test[feature_names]
groups = X_train['patient_id']

# PCA
train_pca(X_train)
X_train_transformed = apply_pca(X_train)

# Define classifiers
classifiers = [KNeighborsClassifier(n_neighbors=i) for i in range(1, 20, 2)]

# Cross Validate
cv_results = cross_validate(X_train_transformed, y, classifiers, groups)
print_results(evaluation_results)

# Train classifier
#trained_classifiers = train_clf(X_train, y_train, classifiers)

#Evaluate model
#X_test_transformed = apply_pca(X_test)
#evaluation_results = evaluate_clf(X_test, y_test, trained_classifiers)

#print_results(evaluation_results)


