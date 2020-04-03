#!/bin/bash -l

module load python3

#$ -P lxcompl
#$ -N training_prepnet
#$ -o Testing_Datamax
#$ -e Test_Datamax_errors
#$ -l h_rt=10:00:00
#$ -m ea

python torchnet.py

