### Main model script ... Finish this part when script is done ...

###############
### IMPORTS ###
###############

# Standard Modules
import os
from os.path import isfile, join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Default packages for the minimum example
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score #example for measuring performance

# For saving/loading trained classifiers
import pickle

# Image preperation
from prep_image import prep_im_and_mask

# Feature extraction
from asymmetry import mean_asymmetry
from color import slic_segmentation, rgb_var, hsv_var, color_dominance
from compactness import compactness_score
from convexity import convexity_score

# Feature extraction (PCA)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Feature selection
from sklearn.feature_selection import SelectKBest,mutual_info_classif

###########################
### FEATURE EXTRACTIONS ###
###########################

def extract_features(im, im_mask):

	# Assymmetry
	asymmetry = mean_asymmetry(im_mask,4)

	# Color variance
	segments = slic_segmentation(im, im_mask)
	red_var, green_var, blue_var = rgb_var(im, segments)
	hue_var, sat_var, val_var = hsv_var(im, segments)

	# Color dominance
	dom_colors = color_dominance(im, im_mask, clusters=5, include_percentages=True) # Extract five most dominent colors, sorted by percentage of total area
	dom_hue, dom_sat, dom_val = dom_colors[0][1]     

	# Compactness
	compactness = compactness_score(im_mask)

	# Convexity
	convexity = convexity_score(im_mask)

	return np.array([asymmetry, red_var, green_var, blue_var, \
		hue_var, sat_var, val_var, dom_hue, dom_sat, dom_val, \
		compactness, convexity], dtype=np.float16)


########################
### IMAGE PROCESSING ###
########################

def ProcessImages(file_data, image_folder, mask_folder, file_features):
	# Import metadata from file
	df = pd.read_csv(file_data)

	# Remove images without masks
	df_mask = df['mask'] == 1
	df = df.loc[df_mask]

	# Features to extract
	feature_names = ['assymmetry', 'red_var', 'green_var', 'blue_var', \
		'hue_var', 'sat_var', 'val_var', 'dom_hue', 'dom_sat', 'dom_val', \
		'compactness', 'convexity']
	features_n = len(feature_names)
	
	features = np.zeros(shape = [len(df), features_n], dtype = np.float16)

	# Extract features
	images = []
	for i, id in enumerate(list(df['img_id'])):
	
		im, mask = prep_im_and_mask(id, image_folder, mask_folder)
		images.append(im)

		# Extract features
		x = extract_features(im, mask)
		features[i,:] = x

	# Save image_ids and features in a file
	df_features = pd.DataFrame(features, columns = feature_names)
	df_features.to_csv(file_features, index = False)

#########################
### FEATURE SELECTION ###
#########################

def feature_scores(train_X, train_y, k):
    '''Using SelectKBest to extract features from train_X, down to k features as output
    Returns a selector object (which is applied to X_train and X_test afterwards) 
	and the score for each feature.

    Args:
        train_X (pandas.DataFrame): Data Frame of features from X_train.
		train_y (pandas.DataFrame): Data Frame of target values from y_train.
        k (int): Number of features to output.

    Returns:
		feature_selector (selector object): 
        scores (numpy.ndarray): Array containg scores for each feature.    
    '''
    feature_selector = SelectKBest(mutual_info_classif, k=k)
    feature_selector.fit_transform(train_X, train_y)
    
    scores = feature_selector.scores_
    
    return feature_selector, scores

##########################
### FEATURE EXTRACTION ###
##########################

def PCA_(X, n: int):
    '''Using PCA to extract features from X, down to n features as output. 
    The features in X are first standardized, then

    Args:
        X (pandas.DataFrame): Data Frame of features.
        n (int): Number of features to keep.

    Returns:
        X_std_pca (numpy.ndarray): Array containg n standardized features.    
    '''

    std_slc = StandardScaler()
    X_std = std_slc.fit_transform(X)
    
    pca = PCA(n_components=n)
    X_std_pca = pca.fit_transform(X_std)
    return X_std_pca

########################
### TRAIN CLASSIFIER ###
########################

def train_classifier():

	# Extract metadata for images
	df = pd.read_csv(file_data)

	# Remove images without masks
	df_mask = df['mask'] == 1
	df = df.loc[df_mask]

	# Extract labels
	labels = np.array(df['diagnostic'])

	# Extract features
	feature_names = ['assymmetry', 'red_var', 'green_var', 'blue_var', \
		'hue_var', 'sat_var', 'val_var', 'dom_hue', 'dom_sat', 'dom_val', \
		'compactness', 'convexity']
	df_features = pd.read_csv(file_features)

	# Make dataset
	X = np.array(df_features[feature_names])
	y = (labels == 'BCC') | (labels == 'SCC') | (labels == 'MEL')
	patient_id = df['patient_id']

	# Train-test split
	num_folds = 5
	group_kfold = GroupKFold(n_splits=folds)
	group_kfold.get_n_splits(X, y, patient_id)

	sss = StratifiedShuffleSplit(n_splits = num_folds)
	sss.get_n_splits(X,y)

	skf = StratifiedKFold(n_splits=num_folds)

	#Different classifiers to test out
	classifiers = [
	    KNeighborsClassifier(1),
	    KNeighborsClassifier(5)
	]

	num_classifiers = len(classifiers)

	acc_val = np.empty([num_folds,num_classifiers])

	#for i, (train_index, val_index) in enumerate(sss.split(X, y, patient_id)):
	#for i, (train_index, val_index) in enumerate(group_kfold.split(X, y, patient_id)):
	for i, (train_index, val_index) in enumerate(skf.split(X, y, patient_id)):

		X_train = X[train_index,:]
		y_train = y[train_index]
		X_val = X[val_index,:]
		y_val = y[val_index]
	
		for j, clf in enumerate(classifiers): 
	
			#Train the classifier
			clf.fit(X_train,y_train)
	
			#Evaluate your metric of choice (accuracy is probably not the best choice)
			acc_val[i,j] = accuracy_score(y_val, clf.predict(X_val))