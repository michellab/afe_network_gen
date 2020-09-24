#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

Utility functions for Hydra

"""

from rdkit.Chem import rdmolfiles
import subprocess
import csv
import multiprocessing as mp,os
import pandas as pd
from tqdm import tqdm
		

def retrieveMolecule(ligand_path):
	"""
	Returns RDKit molecule objects for requested path.

	-- args
	ligand_path (str): path leading to molecule pdb file

	-- returns
	RDKit molecule object
	"""
	mol = rdmolfiles.MolFromPDBFile(
									ligand_path, 
									sanitize=True
									)
	return mol


def writeMorphLigPDB(perturbation_path):
	"""
	Generate and write a pdb file using ambpdb.
	
	-- args
	perturbation_path (str): path to perturbation folder (Sire-style)

	-- returns None

	"""
	# run ambpdb to create a pdb file (includes Du atoms):

	subprocess.call(
				"ambpdb -p "+perturbation_path+"/free/lambda_0.0000/somd.prm7 \
				< "+perturbation_path+"/free/lambda_0.0000/somd.rst7 \
				> "+perturbation_path+"/pre_morph.pdb",
				shell=True
				)


def insertDummyAtomsPDB(perturbation_path, verbose=False):
	"""
	Edit a pre-morph pdb file so that dummy atoms are replaced with astatine.
	This allows RDKit to read in the molecule after which we switch back the atom type.
	PDB file is written but also returned for curation.


	-- args
	perturbation_path (str): path to perturbation folder (Sire-style)
	verbose (bool): whether or not to print PDB contents of morph 
	intermediates to stdout

	-- returns 
	final_pdb (list): nested list with the produced PDB entry. The user should check 
	whether "AT" is inserted at the correct rows. 

	"""	

	# read the pre-pdb file:
	final_pdb = []
	with open(perturbation_path+"/pre_morph.pdb", "r") as file:
		reader = csv.reader(file)
		for row in reader:

			# take only ligand entries:
			if "LIG" in row[0]:

				# if DU is present on the left-hand column:
				# (this is a bit hacky but easier than depending on an API)
				atomtype = row[0][12]+row[0][13]
				if atomtype == "DU":
					atom_to_replace = row[0][76]+row[0][77]

					# replace whatever atom is in the right-hand column with AT (which rdkit CAN parse):
					row = [row[0].replace(atom_to_replace, "AT")]

				# now append to the final PDB that contains only ligand and dummy atoms:
				final_pdb.append(row)
	

	# write the new pdb out to the final file:
	with open(perturbation_path+"/morph.pdb", "w") as file:
		writer = csv.writer(file)
		for row in final_pdb:
			if verbose == True:
				print(row[0])
			writer.writerow(row)

def readHDF5Iterable(path_to_trainingset, chunksize):
	"""
	Read in a training set using pandas' HDF5 utility

	--args
	path_to_trainingset (str): path to training set (HDF5) to read from 
	chunksize (int): number of items to read in per increment (recommended 5000 for large datasets)

	--returns
	training_set (iterable)

	"""
	training_set = pd.DataFrame()

	# use chunksize to save memory during reading:
	training_set_iterator = pd.read_hdf(path_to_trainingset, chunksize=chunksize)


	return training_set_iterator
