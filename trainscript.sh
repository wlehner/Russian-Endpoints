#!/bin/bash -l

module load python3

#$ -P lxcompl
#$ -N Script_Test2
#$ -j y
#$ -o training_results2
#$ -l h_rt=10:00:00
#$ -m ea

python torchnet.py

