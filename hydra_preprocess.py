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


def normaliseDataset(collection_iterable, feature_type):
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

	# we need to perform incremental standardization because of the large dataset:
	print("Making first pass (partial fitting)..")
	for collection in collection_iterable:	
		# omit labels from normalisation:
		labels = collection[["frame", "error", "freenrg", "overlap_score"]]
		collection_features = collection.drop(["frame", "error", "freenrg", "overlap_score"], axis=1)

		# fit:
		scaler.partial_fit(collection_features)
		
	# Now with fully updated means + variances, make a second pass through the iterable and transform:
	print("Making second pass (partial transforming)..")
	for collection in collection_iterable:	
		# omit labels from normalisation:
		labels = collection[["frame", "error", "freenrg", "overlap_score"]]
		collection_features = collection.drop(["frame", "error", "freenrg", "overlap_score"], axis=1)

		# transform:
		normalised_collection = pd.DataFrame(scaler.transform(collection_features))




	# pickle the normaliser object for future transformations:
	pickle_path = "trainingsets/transformations/norm_"+feature_type+".pickle"
	pickle.dump(scaler, open(pickle_path, 'wb'))

	return normalised_collection, labels, collection_features.index


def reduceFeatures(training_data, feature_type, path_to_processed_training_set, 
						labels, 
						index_names):
	"""
	Reduce features using PCA of a provided dataset SKLearnrapid processing.
	Pickles the PCA object for future external test sets.

	--args
	training_data (pandas dataframe): normalised pandas DF object containing an index and column names
	feature_type (str): describes which feature type is being processed
	path_to_processed_training_set (str): path to which the processed training set can
	be written
	labels (list of series): vectors with target labels 
	collection_features.index (list of strings): list of index names, i.e. perturbations

	--returns
	train_postPCA (pandas dataframe): training set containing an index and column names,
	where now instead of normal features the columns are PCA dimensions.

	"""

	pca = decomposition.PCA(n_components=100)

	# Fit to and transform training set:			
	pca.fit(training_data)

	train_postPCA = pd.DataFrame(pca.transform(training_data))
	print("# of PCA features after reduction: "+str(len(train_postPCA.columns)))

	# pickle pca object to file so that external test sets can be transformed accordingly
	pickle_path = "trainingsets/transformations/PCA_"+feature_type+".pickle" 
	pickle.dump(pca, open(pickle_path, "wb"))

	# index is lost during reduction; retrieve together with label columns:
	train_postPCA.index = index_names
	train_postPCA = pd.concat([train_postPCA, labels], axis=1)


	# save to file:
	print("Writing to file..")
	train_postPCA.to_hdf(path_to_processed_training_set, key="df", format="table")


	return train_postPCA

if __name__ == "__main__":
	print("Reading..")
	collection_iterable = hydra_utils.readHDF5("trainingsets/1dcnn_trainingset_precise.h5")

	print("Normalising..")
	normalised_collection, labels, index_names = normaliseDataset(collection_iterable, "1dcnn")

	# print("Computing PCAs..")
	# reduceFeatures(normalised_collection, "1dcnn", "trainingsets/1dcnn_prepared_trainingset_precise.h5", labels, index_names)
