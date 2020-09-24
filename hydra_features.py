#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Featurisation functions for Hydra. 

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Feature sets:

Molecular properties (MolProps).
Perturbation fingerprints (APFP).
1DCNN (ECFP6).

"""
import hydra_utils

from rdkit import Chem
from rdkit.Chem import AllChem


import glob
import csv
import pandas as pd 
import numpy as np 
from tqdm import tqdm
import os.path as path


#########################################################################
### Molecular properties:											  ###
																	  ###
																	  ###
																	  ###
																	  ###
																	  ###

from mordred import Calculator, descriptors

def computeLigMolProps(
					freesolv_path="FREESOLV/", 
					working_dir="features/MOLPROPS/",
					target_columns=None, 
					verbose=False):
	"""
	Compute molecular properties for the molecules in given freesolv_path and write to file.

	--args
	freesolv_path (str): path to directory containing ligand files
	working_dir (str): path to directory to pickle into
	verbose (bool): whether or not to print featurisation info to stdout

	--returns
	freesolv_molprops (pandas dataframe): freesolv molecules with molecular properties

	"""
	mol_paths = glob.glob(freesolv_path+"*")
	
	# generate RDKit mol objects from paths:
	mols_rdkit = [ hydra_utils.retrieveMolecule(mol) for mol in mol_paths ]

	# generate molecule name from paths for indexing:
	mols_names = [ mol.replace(freesolv_path, "").split(".")[0] for mol in mol_paths ]

	
	# generate all descriptors available in mordred:
	calc = Calculator(descriptors, ignore_3D=False)
	print("Computing molecular properties:")
	freesolv_molprops = calc.pandas(mols_rdkit)

	# remove columns with bools or strings (not fit for subtraction protocol):
	if target_columns.any():
		# if variable is input the function is handling a testset and must 
		# keep the same columns as train dataset:
		freesolv_molprops = freesolv_molprops[target_columns]
	else:
		# if making a training dataset, decide which columns to retain:
		freesolv_molprops = freesolv_molprops.select_dtypes(include=["float64", "int64"])
	
	freesolv_molprops.index = mols_names

	# pickle dataframe to specified directory:
	freesolv_molprops.to_pickle(working_dir+"molprops.pickle")

	if verbose:
		print(freesolv_molprops)

	return freesolv_molprops

def computePertMolProps(
						perturbation_paths, 
						freesolv_molprops=None,
						free_path="SOLVATED/", 
						working_dir="features/MOLPROPS/"):
	"""
	Read featurised FreeSolv molecules and generate matches based on user input perturbations.
	Writes each perturbation features by appending it to the features.csv file.

	--args
	perturbation_paths (list): nested list with strings describing the perturbations. These
	combinations will be used to make pairwise extractions from freesolv_molprops

	freesolv_molprops (pandas dataframe; optional): dataframe object that contains the
	featurised FreeSolv dataset. If None, will attempt to pickle from working_dir

	free_path (str): path to directory containing perturbation directories

	working_dir (str): path to directory to pickle dataset from

	--returns
	None

	"""

	# test if input is there:
	if freesolv_molprops is None:
		try:
			freesolv_molprops = pd.read_pickle(working_dir+"molprops.pickle")
		except FileNotFoundError:
			print("Unable to load pickle file with per-ligand molprop data in absence of freesolv_molprops function input.")
			
	# clean slate featurised perturbations dataset; write column names:
	open(working_dir+"free_featurised.h5", "w").close()
	store = pd.HDFStore(working_dir+"free_featurised.h5") 

	# write list of column names to file for future testset feature generation:
	pd.DataFrame(freesolv_molprops.columns).transpose().to_csv(working_dir+"free_featurised.csv", header=False)

	# iterate over perturbations:
	for perturbation in tqdm(perturbation_paths):
		perturbation_name = perturbation.replace(free_path, "").split(".")[0]

		ligandA = perturbation_name.split("~")[0]
		ligandB = perturbation_name.split("~")[1]

		# extract molprops from per-ligand:
		ligandA_molprops = freesolv_molprops.loc[ligandA]
		ligandB_molprops = freesolv_molprops.loc[ligandB]

		# subtract and transform to dataframe:
		perturbation_molprops = ligandB_molprops.subtract(ligandA_molprops).to_frame(name=perturbation_name).transpose()

		# append to the molprops HDF5 file:
		store.append(
					working_dir+"free_featurised.h5", 
					perturbation_molprops,
					format="table",
					index=False,
					min_itemsize=500
					)
		# perturbation_molprops.to_hdf(
		# 					working_dir+"free_featurised.h5", 
		# 					mode="a",
		# 					key="df" 
		# 					)
	store.close()

#########################################################################
### Perturbation fingerprints (AP):									  ###
																	  ###
																	  ###
																	  ###
																	  ###
																	  ###

from rdkit.Chem import rdMolDescriptors

def computeLigPFPs(
					freesolv_path="FREESOLV", 
					working_dir="features/PFP/",
					verbose=False
					):
	"""
	Compute atom-pair fingerprints for the molecules in FreeSolv and write to file.
	Protocol based on https://www.ncbi.nlm.nih.gov/pubmed/25541888.

	--args
	freesolv_path (str): path to directory containing ligand files

	working_dir (str): path to directory to pickle into

	verbose (bool): whether or not to print featurisation info to stdout

	--returns
	freesolv_PFPs (pandas dataframe): freesolv molecules with atom-pair fingerprints

	"""
	def molToAPFP(mol):
		# create atom-pair bitstring of 256 bits 
	    mol.UpdatePropertyCache(strict=False)
	    FP = rdMolDescriptors.GetHashedAtomPairFingerprint(mol, 256)

	    # convert to array:
	    FP = np.array(list(FP))
	    return FP

	mol_paths = glob.glob(freesolv_path+"/*")
	
	# generate RDKit mol objects from paths:
	mols_rdkit = [ hydra_utils.retrieveMolecule(mol) for mol in mol_paths ]

	# generate molecule name from paths for indexing:
	mols_names = [ mol.replace(freesolv_path+"/", "").split(".")[0] for mol in mol_paths ]

	
	# generate atom-pair fingerprints of length 256:
	apfp_dataframe = pd.DataFrame()
	for mol, name in zip(mols_rdkit, mols_names):
		mol_fp = molToAPFP(mol)
		apfp_dataframe = pd.concat([apfp_dataframe, pd.DataFrame([mol_fp], index=[name])])


	# pickle dataframe to specified directory:
	apfp_dataframe.to_pickle(working_dir+"PFP.pickle")

	if verbose:
		print(apfp_dataframe)

	return apfp_dataframe


def computePertPFPs(
					perturbation_paths, 
					freesolv_PFP=None,
					free_path="SOLVATED", 
					working_dir="features/PFP/"):
	"""
	Read featurised FreeSolv molecules and generate matches based on user input perturbations.
	Writes each perturbation features by appending it to the features.csv file.

	--args
	perturbation_paths (list): nested list with strings describing the perturbations. These
	combinations will be used to make pairwise extractions from freesolv_PFP

	freesolv_PFP (pandas dataframe; optional): dataframe object that contains the
	featurised FreeSolv dataset. If None, will attempt to pickle from working_dir

	free_path (str): path to directory containing perturbation directories

	working_dir (str): path to directory to pickle dataset from

	--returns
	None

	"""

	# test if input is there:
	if not freesolv_PFP:
		try:
			freesolv_PFP = pd.read_pickle(working_dir+"PFP.pickle")
		except FileNotFoundError:
			print("Unable to load pickle file with freesolv PFP data in absence of freesolv_PFP function input.")
			
	# clean slate featurised perturbations dataset; write column names:
	open(working_dir+"free_featurised.h5", "w").close()
	store = pd.HDFStore(working_dir+"free_featurised.h5") 
	pd.DataFrame(freesolv_PFP.columns).transpose().to_csv(working_dir+"free_featurised.csv", header=False)

	# iterate over perturbations:
	for perturbation in tqdm(perturbation_paths):
		perturbation_name = perturbation.replace(free_path+"/", "").split(".")[0]
		ligandA = perturbation_name.split("~")[0]
		ligandB = perturbation_name.split("~")[1]

		# extract PFP from freesolv:
		ligandA_PFP = freesolv_PFP.loc[ligandA]
		ligandB_PFP = freesolv_PFP.loc[ligandB]

		# concatenate and transform to dataframe:
		perturbation_PFP = pd.concat([ligandB_PFP, ligandA_PFP]).to_frame(name=perturbation_name).transpose()

		# append to the PFP HDF5 file:
		store.append(
					working_dir+"free_featurised.h5", 
					perturbation_PFP,
					format="table",
					index=False
					)
	store.close()


#########################################################################
### 1DCNN:                                                            ###
																	  ###
																	  ###
																	  ###
																	  ###
																	  ###
from rdkit.Chem.rdmolops import FastFindRings


def fingerprintPDB(mol_path):
	"""
	Generate an ECFP6 fingerprint for a given input molecule path

	--args
	mol_path (str): path to ligand file

	--returns
	freesolv_ECFP6s (pandas dataframe): freesolv molecules with ECFP6 fingerprints

	"""

	mol = Chem.rdmolfiles.MolFromPDBFile(
										mol_path, 
										sanitize=False
										)


	# even though we can't sanitise, we still have to insert ring information:
	FastFindRings(mol)

	# make an APFP fingerprint and return it:
	FP = rdMolDescriptors.GetHashedAtomPairFingerprint(mol, 256)
	FP = np.array(list(FP))
	return FP					



def computeLig1DCNN(
					perturbation_paths, 
					freesolv_PFP=None,
					free_path="SOLVATED", 
					working_dir="features/1DCNN/"):
	"""
	Read featurised FreeSolv molecules and generate matches based on user input perturbations.
	Additionally, generate a PDB structure of the morph intermediate for all perturbations.
	Writes each perturbation features (i.e. a dataframe of PFP bitstrings
	for ligand A, morph and ligand B) by appending it to the features.csv file.

	--args
	perturbation_paths (list): nested list with strings describing the perturbations. These
	combinations will be used to make pairwise extractions from freesolv_ECFP6

	freesolv_PFP (pandas dataframe; optional): dataframe object that contains the
	featurised FreeSolv dataset. If None, will attempt to pickle from working_dir

	free_path (str): path to directory containing perturbation directories

	working_dir (str): path to directory to pickle dataset from

	--returns
	None

	"""

	print("Generating morph files:")
	fail_counter = 0
	for morph_path in tqdm(perturbation_paths):
		
		if not path.exists(morph_path+"/pre_morph.pdb"):
			# try to generate PDB from AMBER morph ligand:
			if path.exists(morph_path+"/free/lambda_0.0000/somd.prm7"):
				try:
					hydra_utils.writeMorphLigPDB(morph_path)
					# convert to RDKit-readable format:
					hydra_utils.insertDummyAtomsPDB(morph_path, verbose=False)
				except FileNotFoundError:
					fail_counter += 1
					pass
			else:
				fail_counter += 1
	print("Unable to generate morph molecules for "+str(fail_counter)+" perturbations.") 

	# test if input is there:
	if not freesolv_PFP:
		try:
			freesolv_PFP = pd.read_pickle("features/PFP/PFP.pickle")
		except FileNotFoundError:
			print("Unable to load pickle file with freesolv PFP data in absence of freesolv_PFP function input.")


	# clean slate featurised perturbations dataset; write column names:
	open(working_dir+"free_featurised.h5", "w").close()
	store = pd.HDFStore(working_dir+"free_featurised.h5")
	pd.DataFrame(freesolv_PFP.columns).transpose().to_csv(working_dir+"free_featurised.csv", header=False)

	# iterate over perturbations:
	print("Writing 1DCNN frames:")
	failed_frames = []
	for perturbation in tqdm(perturbation_paths):
		try:
			perturbation_name = perturbation.replace(free_path+"/", "").split(".")[0]
			ligandA = perturbation_name.split("~")[0]
			ligandB = perturbation_name.split("~")[1]

			# extract already-computed PFP from freesolv:
			ligandA_PFP = freesolv_PFP.loc[ligandA].values
			ligandB_PFP = freesolv_PFP.loc[ligandB].values


			# now generate the PFP for the morph pdb file:
			morph_PFP = fingerprintPDB(perturbation+"/morph.pdb")



			# concatenate the vectors into the frame intended for 1DCNN and requested by keras:
			PFPframe = pd.DataFrame([ligandA_PFP, morph_PFP, ligandB_PFP], index=3*[perturbation_name])
			PFPframe["frame"] =  ["ligandA", "morph", "ligandB"]

			# append to the 1DCNN file:
			store.append(
						working_dir+"free_featurised.h5", 
						PFPframe,
						format="table",
						index=False							
						)
		except OSError:
			failed_frames.append(perturbation)
	print("Failed to write "+str(len(failed_frames))+" frames:", failed_frames)
	store.close()




if __name__ == "__main__":

	perturbation_paths = glob.glob("SOLVATED/*")
	
	print("Writing molprops:")
	
	computePertMolProps(perturbation_paths)

	print("Writing PFPs:")
	computePertPFPs(perturbation_paths)

	computeLig1DCNN(perturbation_paths)




