#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Data preprocessing functions for Hydra. 

TD: in the long run check if we need to implement I/O between functions in the workflow.

"""

from sklearn import preprocessing, decomposition
import numpy as np 
import pandas as pd
import pickle
import hydra_utils
import os

def dropLabels(collection, feature_type):
	"""
	Drop labels and return features + labels separately

	--args
	collection (pandas DF): dataset containing features + labels
	feature_type (str): determines which labels to drop

	--returns
	labels (pandas series): series containing all label names
	collection_features (pandas DF): dataset containing all feature data
	"""

	if feature_type == "1DCNN":
		labels = collection[["frame", "error", "freenrg", "overlap_score"]]
		collection_features = collection.drop(["frame", "error", "freenrg", "overlap_score"], axis=1)
	else:
		labels = collection[["error", "freenrg", "overlap_score"]]
		collection_features = collection.drop(["error", "freenrg", "overlap_score"], axis=1)

	return labels, collection_features



def normaliseDataset(path_to_raw_trainingset, path_to_save_loc, feature_type, chunksize):
	"""
	Normalise a provided dataset using pandas and SKLearn normalisation for rapid processing.
	Pickles the normalisation object for future external test sets.

	--args
	collection_iterable (iterable): pandas DF object containing an index and column names
	feature_type (str): describes which feature type is being processed

	--returns
	normalised_collection (pandas dataframe): normalised training set containing 
	an index and column names
	labels (list of series): vectors with target labels 
	collection.columns (list of strings): list of column names, including labels 
	collection_features.index (list of strings): list of index names, i.e. perturbations

	"""
	scaler = preprocessing.StandardScaler()

	if feature_type == "1DCNN": # all parameters here were found manually:
		n_components = 200
		print("This function takes ~10s to complete on 15K datapoints.\n")
	elif feature_type == "MOLPROPS":
		n_components = 750
		print("This function takes ~10m to complete on 15K datapoints.\n")
	elif feature_type == "PFP":
		n_components = 200
		print("This function takes ~10s to complete on 15K datapoints.\n")

	pca = decomposition.IncrementalPCA(n_components=n_components)
	

	###########################################################################################
	# we need to perform incremental standardization because of the large dataset:
	print("Making first pass (partial fitting)..")
	collection_iterable = hydra_utils.readHDF5Iterable(
										path_to_raw_trainingset, 
										chunksize=chunksize)


	for collection in collection_iterable:

		# omit labels from normalisation:
		labels, collection_features = dropLabels(collection, feature_type)

		# fit the normalisation:
		scaler.partial_fit(collection_features)
		

	###########################################################################################
	# Now with fully updated means + variances, make a second pass through the iterable and transform:
	print("Making second pass (partial transform + partial PCA fit)..")
	collection_iterable = hydra_utils.readHDF5Iterable(
									path_to_raw_trainingset, 
									chunksize=chunksize)

	for collection in collection_iterable:	
		# omit labels from normalisation:
		labels, collection_features = dropLabels(collection, feature_type)

		# transform:
		normalised_collection = pd.DataFrame(scaler.transform(collection_features))

		# now fit an incremental PCA to this chunk:
		pca.partial_fit(normalised_collection)
		
	
	# # uncomment to figure out ~ how many dims to retain for 95% VE:
	# ve_ratios = pca.explained_variance_ratio_
	# ve_counter = 0
	# ve_cumulative = 0
	# for ve in ve_ratios:
	# 	if not ve_cumulative >= 0.95:			
	# 		ve_cumulative += ve
	# 		ve_counter += 1
	# print("Keep", ve_counter, "to retain", ve_cumulative*100, "of variance explained.")

	###########################################################################################
	# now with the completed PCA object; go over iterable one last time;
	# apply normalisation and transform by PCA and save to individual files:
	print("Making third pass (normalise and PCA transform)..")
	collection_iterable = hydra_utils.readHDF5Iterable(
									path_to_raw_trainingset, 
									chunksize=chunksize)
	
	if os.path.exists(path_to_save_loc+feature_type+"/data.h5"):
		os.remove(path_to_save_loc+feature_type+"/data.h5")
	store = pd.HDFStore(path_to_save_loc+feature_type+"/data.h5")


	for collection in collection_iterable:
		
		# this is our final transform; save perturbation names:
		perturbation_indeces = collection.index

		# omit labels from normalisation:
		labels, collection_features = dropLabels(collection, feature_type)

		# normalise transform:
		normalised_collection = pd.DataFrame(scaler.transform(collection_features))

		# PCA transform to finish preprocessing:
		processed_collection = pca.transform(normalised_collection)

		# prettify the np matrix back into DF and append to HDF:
		num_PCA_dims = len(processed_collection[0])
		PCA_column_headers = [ "PC"+str(dim) for dim in range(num_PCA_dims)]

		pca_data_df = pd.DataFrame(
									processed_collection, 
									index=perturbation_indeces, 
									columns=PCA_column_headers
									)

		complete_preprocessed_df = pd.concat([pca_data_df, labels], 
												axis=1, sort=False)

		store.append(
					path_to_save_loc+feature_type+"/data.h5", 
					complete_preprocessed_df,
					format="table",
					index=False,
					)
	store.close()
	# finally, save both the standardscaler and PCA object for transforming test datasets:
	pickle.dump(scaler, open(path_to_save_loc+"PICKLES/"+feature_type+"_scaler.pkl","wb"))
	pickle.dump(pca, open(path_to_save_loc+"PICKLES/"+feature_type+"_pca.pkl","wb"))




if __name__ == "__main__":
	normaliseDataset(
					path_to_raw_trainingset="trainingsets/PFP_trainingset.h5",
					path_to_save_loc="trainingsets_prepared/",
					feature_type="PFP", 
					chunksize=6000)
