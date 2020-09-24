#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Plotting functions for Hydra. 

"""
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_absolute_error
import numpy as np



def computeStats(x, y):
	pearsonr = round(stats.pearsonr(x, y)[0], 3)
	tau = round(stats.kendalltau(x, y)[0], 3)
	mae = round(mean_absolute_error(x, y), 3)

	return pearsonr, tau, mae

def datToDensity(x, y):
	# Calculate the point density

	xy = np.vstack([x,y])
	z = stats.gaussian_kde(xy)(xy)

	# Sort the points by density, so that the densest points are plotted last
	x2 = []
	y2 = []
	z2 = []
	for value in sorted(zip(x, y, z), key=lambda pair: pair[2]):
		x2.append(value[0])
		y2.append(value[1])
		z2.append(value[2])
	return x2, y2, z2


df = pd.read_csv("output/top_performer.csv")

x_error = df["Exp1"].values
y_error = df["Pred1"].values
x_error, y_error, z_error = datToDensity(x_error, y_error)

x_freenrg = df["Exp2"].values
y_freenrg = df["Pred2"].values
x_freenrg, y_freenrg, z_freenrg = datToDensity(x_freenrg, y_freenrg)




pearsonr_error, tau_error, mae_error = computeStats(x_error, y_error)
pearsonr_freenrg, tau_freenrg, mae_freenrg = computeStats(x_freenrg, y_freenrg)

statistics = pd.DataFrame(data={
								"MBAR error": [pearsonr_error, tau_error, mae_error],
								"FreeNrg": [pearsonr_freenrg, tau_freenrg, mae_freenrg]
								},
								index=["Pearson R", "Kendall tau", "MUE"]
								)


fig, axs = plt.subplots(1, 2 ,constrained_layout=True)



axs[0].scatter(x_error, y_error, c=z_error)
axs[0].set_title("MBAR error prediction")
axs[0].set_xlabel("Experimental (kcal/mol)")
axs[0].set_ylabel("Prediction (kcal/mol)")

axs[1].scatter(x_freenrg, y_freenrg, c=z_freenrg)
axs[1].set_title("MBAR FreeNrg prediction")
axs[1].set_xlabel("Experimental (kcal/mol)")




print(statistics)
plt.show()

