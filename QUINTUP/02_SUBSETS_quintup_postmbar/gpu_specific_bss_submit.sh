#!/bin/bash

# runs all SOMD Free Energy analysis in this folder using a python script that
# grabs parameterised ligands, solvates the system and runs MD using BSS.

export CUDA_VISIBLE_DEVICES=1
export OPENMM_PLUGIN_DIR=/home/jscheen/biosimspace.app/lib/plugins/

COUNTER=1

#cp ../solv_freenrg_bss.py .
cat sorted.csv | while read pert_name; do
	/home/jscheen/biosimspace.app/bin/python3.7 solv_freenrg_bss.py $pert_name
	wait

	# move finished sim to index folder so the next replicate doesn't overwrite:
	pert=$(echo $pert_name | sed 's/,/~/g')
	mkdir sim_$COUNTER
	mv $pert sim_$COUNTER/$pert


	COUNTER=$[$COUNTER +1]
done	
