#!/bin/bash


top_dir=$(pwd)

find . -type d -name sim_* | while read line; do
	cd $line/mobley*/free/
	sim_folder=$(pwd)


	# run mbar:
	echo "Starting MBAR on "$sim_folder
	/home/jscheen/biosimspace.app/bin/analyse_freenrg mbar -i lambda*/simfile.dat --temperature 300.0 -p 95 --overlap --subsampling > freenrg-MBAR.dat 2> MBAR.out
	

	# move back to work dir:
	cd $top_dir
done
echo "Finished MBAR runs."
