#!/bin/bash -l

module load python3

#$ -P lxcompl
#$ -N Test_200_Epoch2
#$ -j y
#$ -o training_results5
#$ -l h_rt=24:00:00
#$ -m ea

python torchnet.py

