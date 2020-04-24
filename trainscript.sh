#!/bin/bash -l

module load python3

#$ -P lxcompl
#$ -N learning_rate_test
#$ -j y
#$ -o training_results2
#$ -l h_rt=10:00:00
#$ -m ea

python torchnet.py

