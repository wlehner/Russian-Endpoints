#!/bin/bash -l

module load python3

#$ -P statnlp
#$ -N training_prepnet
#$ -o training_results
#$ -e training_errors
#$ -l h_rt=10:00:00
#$ -m ea

python torchnet.py

