#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Build training sets for Hydra by joining perturbation labels with their respective features.

#TD: rewrite so functions are universal to feature sets.


"""
import csv
import pandas as pd
import math
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles, rdFMCS

def extrapolateError(pert):
	"""
	Extrapolate a perturbation's error based on the number of changes in heavy atoms for the
	perturbation. This function is useful when MBAR doesn't output an error (due to e.g. 
	simulation errors) or when the error is returned as 0.0.
	The new error is computed as 10*number of heavy atoms that change in the perturbation.

	--args
	pert (str): name of the perturbation in form "mobley<index>~mobley<index>"

	--returns
	new_error (float): artifial error based on the number of changes in heavy atoms in
	the perturbation
	"""
	# first manipulate the name so that we can read in the indiv. ligand files:
	ligA = pert.split("~")[0]
	ligA_path = "FREESOLV/"+ligA+".pdb"

	ligB = pert.split("~")[1]
	ligB_path = "FREESOLV/"+ligB+".pdb"

	ligA_pdb = open(ligA_path, "r").read()
	ligB_pdb = open(ligB_path, "r").read()

	ligA_mol = rdmolfiles.MolFromPDBBlock(ligA_pdb)
	ligB_mol = rdmolfiles.MolFromPDBBlock(ligB_pdb)

	# compute the maximum common substructure between the two ligands:
	MCS_object = rdFMCS.FindMCS([ligA_mol, ligB_mol], completeRingsOnly=True)
	MCS_SMARTS = Chem.MolFromSmarts(MCS_object.smartsString)

	# delete the MCS from both ligands so we end up with the perturbed fragments:
	ligA_stripped = AllChem.DeleteSubstructs(ligA_mol, MCS_SMARTS)
	ligB_stripped = AllChem.DeleteSubstructs(ligB_mol, MCS_SMARTS)

	# the largest remaining fragment will have all atoms either removed or replaced,
	# so that fragment's number of atoms equals number perturbed. 
	if len(ligA_stripped.GetAtoms()) >= len(ligB_stripped.GetAtoms()):
		n_perturbed = len(ligA_mol.GetAtoms()) - MCS_object.numAtoms
	else:
		n_perturbed = len(ligB_mol.GetAtoms()) - MCS_object.numAtoms

	# sometimes MCS fails (various reasons); force the new error to still be 10:
	if n_perturbed == 0:
		n_perturbed = 1

	# generally the more atoms a perturbation has the higher its error will be. We multiply
	# n_perturbed with a factor 2 to introduce some level of continuity in the artificial errors:
	new_error = n_perturbed * 2
	return new_error


def buildTrainingSet(
				path_to_labels_file, 
				path_to_features_file, 
				path_to_trainingset_file,
				precise=False,
				):
	"""
	Build a training set by joining the features with labels

	--args
	path_to_labels_file (str): path to file containing labels
	path_to_features_file (str): path to file containing features
	path_to_trainingset_file (str): path to write resulting training set into

	--returns
	None

	"""

	# load featurised dataset into memory per line using the pandas generator:
	featurised_dataset = pd.read_hdf(path_to_features_file, chunksize=1)

	# clean slate the training set file:
	open(path_to_trainingset_file, "w").close()

	# # figure out how many perturbations we are processing:
	# with open(path_to_labels_file) as f:
	# 	num_perturbations = sum(1 for line in f)

	# load in the labels as a DF for fast index pairing:
	labels_df = pd.read_csv(path_to_labels_file, index_col=0, names=["error", "freenrg", "overlap_score"])
	if "1DCNN" in path_to_features_file:
		num_perturbations = len(labels_df)*3
	else:
		num_perturbations = len(labels_df)
	

	# per featurised datapoint, iterate:
	store = pd.HDFStore(path_to_trainingset_file)
	
	for features_frame in tqdm(featurised_dataset, total=num_perturbations):

		perturbation_name = features_frame.index.values[0]
		fep_info = labels_df.loc[perturbation_name]
		error = fep_info["error"]
		freenrg = fep_info["freenrg"]
		overlap_score = fep_info["overlap_score"]


		# insert artificial error value in case of absence; remove outliers (error >10 kcal/mol) if specified:
		if (math.isnan(float(error))) or (float(error) >= 10):
			if precise:
				error = None
			elif not precise:
				error = extrapolateError(perturbation_name)

		if freenrg and precise:
			if not -20 < float(freenrg) < 20:
				freenrg = None

		# if no overlap score is available for this perturbation, give it an OS of 0:
		if math.isnan(float(overlap_score)):
			if precise:
				overlap_score = None
			elif not precise:
				overlap_score = 0			

		# if the label values are not none, make this labelled datapoint and append to training set file:
		if error and overlap_score:

			# attach the labels:
			features_frame["error"] = round(float(error), 8)
			features_frame["freenrg"] = round(float(freenrg), 8)
			features_frame["overlap_score"] = round(float(overlap_score), 8)

			# write this perturbation's data to file:
			store.append(
					path_to_trainingset_file, 
					features_frame,
					format="table",
					index=False,
					)

		
if __name__ == "__main__":
	buildTrainingSet(
		"labels/mbar_labels.txt", 
		"features/1DCNN/free_featurised.h5", 
		"trainingsets/1DCNN_trainingset.h5", 
		precise=False
		)

