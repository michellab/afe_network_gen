#!/bin/bash

pert=$1
rep=1

cat ../finished.txt | grep $pert | while read line; do
	while [ "$rep" -lt "6" ]; do
		pert_folder=$(echo $line | sed 's/\/free//g')
		cp -r .$pert_folder rep_$rep	
		rep=$((rep + 1))
	done
done
