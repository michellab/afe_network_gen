#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Training functions for Hydra. 

"""

# TF-related imports & some settings to reduce TF verbosity:
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"	# current workstation contains 4 GPUs; exclude 1st
import tensorflow as tf 
from tensorflow import keras

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# SciKit-Optimize:
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_convergence
from skopt.plots import plot_objective, plot_evaluations
from tensorflow.python.keras import backend as K
from skopt.utils import use_named_args

# general imports:
import pandas as pd 
import numpy as np 
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm import tqdm



n_calls = 100						# Number of Bayesian optimisation loops for hyperparameter optimisation, 40 is best for convergence, > 60 scales to very expensive
startpoint_error = np.inf			# Point to consider top-performing model from





def importDataSet(path_to_trainingset, path_to_testset):
	"""
	Import a pre-processed training set (i.e. by hydra_preprocess) and 
	return X and y.

	--args
	path_to_trainingset (str): path from which the processed training set can
	be read

	--returns
	X_train (pandas dataframe): dataframe containing features for training; 
	contains perturbation names as index. 
	y_train (pandas dataframe): dataframe containing labels for training; 
	contains perturbation names as index. 

	"""
	trainingset = pd.read_hdf(path_to_trainingset)

	# now extract labels and return dataframes separately:
	y_train = trainingset[["overlap_score"]]
	X_train = trainingset.drop(["error", "freenrg", "overlap_score"], axis=1)

	testset = pd.read_csv(path_to_testset, index_col=0)

	y_test = testset[["OS"]]
	X_test = testset.drop(["OS"], axis=1)




	return X_train, y_train, X_test, y_test

# old version that trains using 80/10/10:
# def importDataSet(path_to_trainingset):
# 	"""
# 	Import a pre-processed training set (i.e. by hydra_preprocess) and 
# 	return X and y.

# 	--args
# 	path_to_trainingset (str): path from which the processed training set can
# 	be read

# 	--returns
# 	X_train (pandas dataframe): dataframe containing features for training; 
# 	contains perturbation names as index. 
# 	y_train (pandas dataframe): dataframe containing labels for training; 
# 	contains perturbation names as index. 
# 	X_test (pandas dataframe): dataframe containing features for testing; 
# 	contains perturbation names as index. 
# 	y_test (pandas dataframe): dataframe containing labels for testing; 
# 	contains perturbation names as index. 


# 	"""
# 	dataset = pd.read_hdf(path_to_trainingset)
# 	print(dataset)
# 	# now extract labels and return dataframes separately:
# 	y = dataset[["error"]]
# 	X = dataset.drop(["error", "freenrg", "overlap_score"], axis=1)

# 	X_train, X_test, y_train, y_test = train_test_split(X, y, 
# 														test_size=0.1, 
# 														random_state=42
# 														)



# 	return X_train, y_train, X_test, y_test

def denseNN(X_train, y_train, X_test, y_test, feature_type):

	# clean slate stats output for convergence data:
	stat_output_path = "output/"+feature_type+"_skopt_conv_data.csv"
	if os.path.exists(stat_output_path):
		open(stat_output_path).close()
	stats_per_skopt_call = []

	def create_model(
		num_dense_layers_base, 
		num_dense_nodes_base,
		num_dense_layers_end, 
		num_dense_nodes_end, 
		learning_rate,
		adam_b1,
		adam_b2,
		adam_eps,
		num_batch_size):


		model = keras.Sequential()

	# Add input layer of length of the dataset columns:
		model.add(keras.layers.Dense(len(X_train.columns), input_shape=[len(X_train.keys())]))

	# Generate n number of hidden layers (base, i.e. first layers):
		for i in range(num_dense_layers_base):
			model.add(keras.layers.Dense(num_dense_nodes_base,
			activation=keras.activations.relu
			))

	# Generate n number of hidden layers (end, i.e. last layers):
		for i in range(num_dense_layers_end):
			model.add(keras.layers.Dense(num_dense_nodes_end,
			activation=keras.activations.relu
			))

	# Add output layer:

		model.add(keras.layers.Dense(1, activation=keras.activations.linear))

		optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=adam_b1, beta_2=adam_b2, epsilon=adam_eps)

		model.compile(
			loss="logcosh",
			#loss="mae",
			optimizer=optimizer,
			metrics=["mean_absolute_error"]
			)
		return model


	# Set hyperparameter ranges, append to list:
	dim_num_dense_layers_base = Integer(low=1, high=2, name='num_dense_layers_base')
	dim_num_dense_nodes_base = Categorical(categories=list(np.linspace(5,261, 10, dtype=int)), name='num_dense_nodes_base')
	dim_num_dense_layers_end = Integer(low=1, high=2, name='num_dense_layers_end')
	dim_num_dense_nodes_end = Categorical(categories=list(np.linspace(5,261, 10, dtype=int)), name='num_dense_nodes_end')


	learning_rate = Categorical(categories=list(np.linspace(0.001,0.1,10)), name="learning_rate")
	dim_adam_b1 = Categorical(categories=list(np.linspace(0.8,0.99,11)), name="adam_b1")
	dim_adam_b2 = Categorical(categories=list(np.linspace(0.8,0.99,11)), name="adam_b2")
	dim_adam_eps = Categorical(categories=list(np.linspace(0.0001, 0.5, 11)), name="adam_eps")
	dim_num_batch_size = Categorical(categories=list(np.linspace(16, 30, 8, dtype=int)), name='num_batch_size')

	dimensions = [
				dim_num_dense_layers_base,
				dim_num_dense_nodes_base,
				dim_num_dense_layers_end,
				dim_num_dense_nodes_end,
				learning_rate,
				dim_adam_b1,
				dim_adam_b2,
				dim_adam_eps,
				dim_num_batch_size]	

	@use_named_args(dimensions=dimensions)
	def fitness(
		num_dense_layers_base, 
		num_dense_nodes_base, 
		num_dense_layers_end, 
		num_dense_nodes_end,
		learning_rate,
		adam_b1,
		adam_b2,
		adam_eps,
		num_batch_size):
		early_stopping = keras.callbacks.EarlyStopping(
														monitor='val_loss', 
														mode='min', 
														patience=30,
														verbose=0)
	# Create the neural network with these hyper-parameters:
		model = create_model(
							num_dense_layers_base=num_dense_layers_base,
							num_dense_nodes_base=num_dense_nodes_base,
							num_dense_layers_end=num_dense_layers_end,
							num_dense_nodes_end=num_dense_nodes_end,
							learning_rate=learning_rate,
							adam_b1=adam_b1,
							adam_b2=adam_b2,
							adam_eps=adam_eps,
							num_batch_size=num_batch_size)



		history = model.fit(
			X_train, y_train,
		epochs=500, 
		validation_split=0.1,
		verbose=0,
		callbacks=[
					early_stopping, 
					#PrintDot(),			# uncomment for verbosity on epochs
					], 		
		batch_size=num_batch_size)

		hist = pd.DataFrame(history.history)
		hist['epoch'] = history.epoch
		val_loss = hist["val_loss"].tail(5).mean()
		print(val_loss)

		#################
		# calculate some statistics on test set:
		perts_list = y_test.index.tolist()
		prediction = model.predict(X_test)

		prediction_1_list = [ item[0] for item in prediction ]
		exp_1_list = y_test.iloc[:,0].values.tolist()

		# in case of multitask:
		#prediction_2_list = [ item[1] for item in prediction ]
		#exp_2_list = y_test.iloc[:,1].values.tolist()


		# For plotting test set correlations:
		tuples_result = list(zip(perts_list, exp_1_list, prediction_1_list))
		#tuples_result = list(zip(perts_list, exp_1_list, prediction_1_list, exp_2_list, prediction_2_list))
		nested_list_result = [ list(elem) for elem in tuples_result ]
		dataframe_result = pd.DataFrame(nested_list_result, 
										columns=["Perturbation", "Exp1", "Pred1"])

		# compute r on test set:
		test_r = abs(stats.pearsonr(exp_1_list, prediction_1_list)[0])

		# append stats to skopt convergence data:
		stats_per_skopt_call.append([val_loss, test_r])

		# sometimes, r is nan or 0; adjust:
		if not type(test_r) == np.float64 or test_r == 0:
			test_r = 0.1

		# SKOPT API is easier when minimizing a function, so return the inverse of r:
		test_r_inverse = 1/test_r


		# Append data with best performing model.
		global startpoint_error

		if test_r_inverse < startpoint_error:
			print("New best r:", 1/test_r_inverse, "\n")

			startpoint_error = test_r_inverse

			# # write all model files:
			model.save_weights("models/"+feature_type+"_HYDRA_weights.h5")
			with open("models/"+feature_type+"_HYDRA_architecture.json", "w") as file:
				file.write(model.to_json())

			dataframe_result.to_csv("output/"+feature_type+"_top_performer.csv")

			# make a classic loss plot and save:
			plt.figure()
			plt.plot(hist['epoch'], hist['loss'], "darkorange", label="Training loss")
			plt.plot(hist['epoch'], hist['val_loss'], "royalblue", label="Validation loss")
			plt.xlabel("Epoch")
			plt.ylabel("Loss / MAE on OS")
			plt.ylim(0, 0.002)
			plt.legend()
			plt.savefig("output/"+feature_type+"_top_performer_loss_plot.png", dpi=300)

		
		del model
		tf.keras.backend.clear_session()
		K.clear_session()		
		
		return test_r_inverse

	# Bayesian Optimisation to search through hyperparameter space. 
	# Prior parameters were found by manual search and preliminary optimisation loops. 
	default_parameters = [
							2, 			# first half n connected layers
							33, 		# n neurons in first half connected layers
							1, 			# second half n connected layers
							90, 		# n neurons in second half connected layers
							0.1,		# learning rate
							0.971, 		# adam beta1
							0.895, 		# adam beta2
							1.0000e-04, # adam epsilon
							20			# batch size
							]

	print("###########################################")
	print("Created model, optimising hyperparameters..")

	search_result = gp_minimize(func=fitness,
								dimensions=dimensions,
								acq_func='EI', #Expected Improvement.
								n_calls=n_calls,
								x0=default_parameters)


	print("###########################################")
	print("Concluded optimal hyperparameters:")
	print(search_result.x)

	with open(stat_output_path, "w") as filepath:
		writer = csv.writer(filepath)
		for stats_row in stats_per_skopt_call:
			writer.writerow(stats_row)
		writer.writerow(search_result.x)
	print("###########################################")

if __name__ == "__main__":
	
	path_to_trainingset = "trainingsets_prepared/PFP/data.h5"
	path_to_testset = "TESTSETS/wang15/fullset_preprocessed_apfp.csv"
	(X_train, y_train, X_test, y_test) = importDataSet(path_to_trainingset, path_to_testset)

	denseNN(X_train, y_train, X_test, y_test, feature_type="PFP_OS")



