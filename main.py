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



### PROCESS IMAGES ###

# Extract features for all images with masks and save as csv.

ProcessImages(file_data, image_folder, mask_folder, file_features)


### TRAIN MODEL ###

X_train, y_train, X_test, y_test = train_test_split([PAT_ID,features])

clf = train_model(X_train, y_train)


### EVALUATE MODEL ###

clf.predict(X_test, y_test)
clf.predict_proba(X_test, y_test)



