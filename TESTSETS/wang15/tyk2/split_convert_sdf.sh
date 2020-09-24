#!/bin/bash

# split a multi-mol ligands.sdf file into individual PDB files; rename files to ligand names (first line)

obabel ligands.sdf -O lig1.pdb -m

for ligfile in lig*.pdb; do
	newname=$(head -1 $ligfile | awk '{print $2}')
	mv $ligfile $newname.pdb
done
