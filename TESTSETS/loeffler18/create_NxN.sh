#!/bin/bash

# from a directory LIGANDS/ containing N pdb structures, creates NxN directories of names structureA~structureB.

cd LIGANDS/

for f in *; do name=$(basename $f .pdb); for f2 in *; do name2=$(basename $f2 .pdb); if [[ ! "$name1" == "$name2" ]]; then mkdir $name~$name2; fi; done; done

cd ../
