#!/bin/bash -l

module load python3

#$ -P lxcompl
#$ -N training_prepnet
#$ -o training_results3
#$ -e training_errors3
#$ -l h_rt=10:00:00
#$ -m ea

python torchnet.py

