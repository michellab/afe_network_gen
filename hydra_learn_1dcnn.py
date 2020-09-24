#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Training functions for Hydra. 

"""

# TF-related imports & some settings to reduce TF verbosity:
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"	# current workstation contains 4 GPUs; exclude 1st
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



n_calls = 70						# Number of Bayesian optimisation loops for hyperparameter optimisation, 40 is best for convergence, > 60 scales to very expensive
startpoint_error = np.inf			# Point to consider top-performing model from






def importDataSet(path_to_trainingset, convolutional=False):
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
	X_test (pandas dataframe): dataframe containing features for testing; 
	contains perturbation names as index. 
	y_test (pandas dataframe): dataframe containing labels for testing; 
	contains perturbation names as index. 


	"""


	dataset = pd.read_hdf(path_to_trainingset).sample(frac=1)
	dataset_grouped = dataset.groupby(by=dataset.index)


	# Take a test set that is 10% of the complete dataset:
	slice_point = int(dataset_grouped.ngroups*0.9)
	train_indices = list(dataset_grouped.groups)[0:slice_point]
	test_indices = list(dataset_grouped.groups)[slice_point:]
	train = dataset.loc[train_indices]
	test = dataset.loc[test_indices]


	# now generate X and y dataframes:
	def extractXy(dataset):
		

		X = []
		y = []
		dataset_grouped = dataset.groupby(by=dataset.index)

		for group in tqdm(dataset_grouped, total=len(dataset_grouped)):
			# grab the metrics:
			group = group[1]
			error = group["error"].values[0]
			freenrg = group["freenrg"].values[0]
			overlap_score = group["overlap_score"].values[0]
			pert_name = group.index.values[0]
			
			# we need to reorder the columns in a bit:
			frame_order = group["frame"].values

			# now transpose so that APFP values are vertical:
			group = group.drop(["error", "freenrg", "overlap_score"], axis=1).transpose()

			# now fix column order and rebuild dataframe:
			group.columns = frame_order
			group = group[["ligandA", "morph", "ligandB"]]
			group = group.set_index([len(group)*[pert_name]])

			# remove last row (contains names of liganda/morph/ligandb):
			group = group[:-1]

			# generate labels dataframe that matches the shape of the features:
			labels = pd.DataFrame({"overlap_score" : overlap_score}, index=[pert_name])

			# append this dataframe to lists of dataframes (considerable speedup over df.append):
			X.append(group)
			y.append(labels)

		X = pd.concat(X)
		y = pd.concat(y)

		return X, y

			
	print("Tranforming training data:")
	X_train, y_train = extractXy(train)
	print("Tranforming test data:")
	X_test, y_test = extractXy(test)


	return X_train, y_train, X_test, y_test

path_to_trainingset = "trainingsets_prepared/1DCNN/data.h5"
(X_train, y_train, X_test, y_test) = importDataSet(path_to_trainingset)



def convNN(X_train, y_train, X_test, y_test, num_classes, feature_type):

	# clean slate stats output for convergence data:
	stat_output_path = "output/"+feature_type+"_skopt_conv_data.csv"
	if os.path.exists(stat_output_path):
		open(stat_output_path).close()
	stats_per_skopt_call = []
	# largely based on: https://towardsdatascience.com/human-activity-recognition-har-tutorial-with-keras-and-core-ml-part-1-8c05e365dfa0
	# and https://blog.goodaudience.com/introduction-to-1d-convolutional-neural-networks-in-keras-for-time-sequences-3a7ff801a2cf

	length_fp = X_test.groupby(by=X_test.index).agg(["count"]).values[0][0]



	# extrapolate some info from the data to make the correct form of CNN:
	def reshapeX(X, length):
			segments = [
						X["ligandA"].values,
						X["morph"].values,
						X["ligandB"].values
						]
			X = np.asarray(segments, dtype=np.float32).reshape(-1, length, 3)
			return X

	X_train = reshapeX(X_train, length_fp)
	y_train = np.asarray(y_train.values)
	num_time_periods, num_sensors = X_train.shape[1], X_train.shape[2]
	input_shape = num_time_periods*num_sensors
	X_train = X_train.reshape(X_train.shape[0], input_shape)

	# now do the same for the test set:
	X_test = reshapeX(X_test, length_fp)
	X_test = X_test.reshape(X_test.shape[0], input_shape)


	def create_model(
		conv_num_filters_1st,
		conv_num_filters_2nd,
		conv_num_filters_3rd,
		filter_size,
		final_1_dense_nodes,
		final_2_dense_nodes,
		dropout,
		learning_rate,
		adam_b1,
		adam_b2,
		adam_eps,
		num_batch_size):

		# now use Keras to build (fairly standard) CNN layers:
		model = keras.Sequential()
		model.add(keras.layers.Reshape((num_time_periods, num_sensors), input_shape=(input_shape,)))
		
		model.add(keras.layers.Conv1D(conv_num_filters_1st, [20], activation='relu', input_shape=(num_time_periods, num_sensors)))
		model.add(keras.layers.Conv1D(conv_num_filters_2nd, [20], activation='relu'))
		model.add(keras.layers.MaxPooling1D(3))
		#model.add(keras.layers.Dropout(dropout))

		model.add(keras.layers.Conv1D(conv_num_filters_3rd, [20], activation='relu'))
		model.add(keras.layers.GlobalAveragePooling1D())
		model.add(keras.layers.Dropout(dropout))
		model.add(keras.layers.Dense(final_1_dense_nodes, activation=keras.activations.relu))
		model.add(keras.layers.Dense(final_2_dense_nodes, activation=keras.activations.relu))

		# single linear output layer for regression, or multiple for multi-task learning:
		model.add(keras.layers.Dense(num_classes, activation='linear'))

		# define the optimizer like this so we can tune hyperparameters:
		optimizer = tf.keras.optimizers.Adam(lr=learning_rate, beta_1=adam_b1, 
											beta_2=adam_b2, epsilon=adam_eps)

		model.compile(loss='logcosh',
		                optimizer=optimizer)

		# uncomment to print model architecture for each SKOPT iteration:
		
		return model

	# Set hyperparameter ranges, append to list:
	# architecture-related ranges:
	dim_conv_num_filters_1st = Categorical(categories=list(np.linspace(10,200,15, dtype=np.int64)), name='conv_num_filters_1st')
	dim_conv_num_filters_2nd = Categorical(categories=list(np.linspace(10,200,15, dtype=np.int64)), name='conv_num_filters_2nd')
	dim_conv_num_filters_3rd = Categorical(categories=list(np.linspace(10,200,15, dtype=np.int64)), name='conv_num_filters_3rd')

	dim_filter_size = Categorical(categories=list(np.linspace(2,5,4, dtype=np.int64)), name='filter_size')
	dim_final_1_dense_nodes = Categorical(categories=list(np.linspace(1,261, 10, dtype=np.int64)), name='final_1_dense_nodes')
	dim_final_2_dense_nodes = Categorical(categories=list(np.linspace(1,261, 10, dtype=np.int64)), name='final_2_dense_nodes')
	dim_dropout = Categorical(categories=list(np.linspace(0.1,0.9,9, dtype=np.float32)), name="dropout")

	# training hyperparameter ranges:
	learning_rate = Categorical(categories=list(np.linspace(0.001,0.1,10, dtype=np.float32)), name="learning_rate")
	dim_adam_b1 = Categorical(categories=list(np.linspace(0.8,0.99,11, dtype=np.float32)), name="adam_b1")
	dim_adam_b2 = Categorical(categories=list(np.linspace(0.8,0.99,11, dtype=np.float32)), name="adam_b2")
	dim_adam_eps = Categorical(categories=list(np.linspace(0.0001, 0.5, 11, dtype=np.float32)), name="adam_eps")
	dim_num_batch_size = Categorical(categories=list(np.linspace(16, 30, 8, dtype=np.int64)), name='num_batch_size')

	dimensions = [
				dim_conv_num_filters_1st,
				dim_conv_num_filters_2nd,
				dim_conv_num_filters_3rd,

				dim_filter_size,
				dim_final_1_dense_nodes,
				dim_final_2_dense_nodes,
				dim_dropout,

				learning_rate,
				dim_adam_b1,
				dim_adam_b2,
				dim_adam_eps,
				dim_num_batch_size]	

	@use_named_args(dimensions=dimensions)
	def fitness(conv_num_filters_1st,
				conv_num_filters_2nd,
				conv_num_filters_3rd,

				filter_size,
				final_1_dense_nodes,
				final_2_dense_nodes,
				dropout,

				learning_rate,
				adam_b1,
				adam_b2,
				adam_eps,
				num_batch_size,
				):

		early_stopping = keras.callbacks.EarlyStopping(
														monitor='val_loss', 
														mode='min', 
														patience=30,
														verbose=0)
	# Create the neural network with these hyper-parameters:
		model = create_model(
							conv_num_filters_1st=conv_num_filters_1st,
							conv_num_filters_2nd=conv_num_filters_2nd,
							conv_num_filters_3rd=conv_num_filters_3rd,

							filter_size=filter_size,
							final_1_dense_nodes=final_1_dense_nodes,
							final_2_dense_nodes=final_2_dense_nodes,
							dropout=dropout,

							learning_rate=learning_rate,
							adam_b1=adam_b1,
							adam_b2=adam_b1,
							adam_eps=adam_eps,
							num_batch_size=num_batch_size)


		history = model.fit(X_train,
		                      y_train,
		                      batch_size=num_batch_size,
		                      epochs=500,
		                      callbacks=[early_stopping],
		                      validation_split=0.2,
		                      verbose=0)

		hist = pd.DataFrame(history.history)
		hist['epoch'] = history.epoch
		val_loss = hist["val_loss"].tail(5).mean()
		print(val_loss)

		# calculate some statistics on test set:
		prediction = model.predict(X_test)

		prediction_1_list = [ item[0] for item in prediction ]
		
		perts_list = y_test.index.tolist()
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
			# # Slightly hacky but TF's backend voids model parameters when the model is saved as a variable
			# # in order to retain the top performing model. From these temporary model files, all but the 
			# # top-performing model will be deleted from the system at the end of this script.
			if not os.path.exists("./opt_tmp"):
				os.makedirs("./opt_tmp")

			model.save_weights("opt_tmp/HYDRA_weights.h5")
			with open("opt_tmp/"+feature_type+"_HYDRA_architecture.json", "w") as file:
				file.write(model.to_json())


			dataframe_result.to_csv("output/"+feature_type+"_top_performer.csv")

			# make a classic loss plot and save:
			plt.figure()
			plt.plot(hist['epoch'], hist['loss'], "darkorange", label="Training loss")
			plt.plot(hist['epoch'], hist['val_loss'], "royalblue", label="Validation loss")
			plt.xlabel("Epoch")
			plt.ylabel("Loss / MAE on OS")
			plt.legend()
			plt.savefig("output/"+feature_type+"_top_performer_loss_plot.png", dpi=300)

		
		del model
		tf.keras.backend.clear_session()
		K.clear_session()		
		
		return test_r_inverse

	# Bayesian Optimisation to search through hyperparameter space. 
	# Prior parameters were found by manual search and preliminary optimisation loops. 
	default_parameters = [
							np.int64(10), 		# Number of filters in first convolutional layer
							np.int64(10), 		# Number of filters in second convolutional layer
							np.int64(10), 		# Number of filters in third convolutional layer

							np.int64(2), 			# Filter size
							np.int64(1), 		# Number of nodes in first fully-connected layer
							np.int64(1), 		# Number of nodes in second fully-connected layer
							np.float32(0.1), 		# Dropout rate

							np.float32(0.001), 		# Learning rate
							np.float32(0.8), 		# adam_beta1
							np.float32(0.8), 		# adam_beta2
							np.float32(0.5),     # adam_epsilon
							np.int64(16) 		# batch size
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

#print("Rethink SKOPT ranges; should we have increased channels? Are they even a thing in 1DCNN/Keras? Or same as n filters?")
convNN(X_train, y_train, X_test, y_test, num_classes=1, feature_type="1DCNNPFP")
