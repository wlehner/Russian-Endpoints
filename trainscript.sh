#!/bin/bash -l

module load python3

#$ -P lxcompl
#$ -N testing_two_jobs
#$ -j y
#$ -o training_results1
#$ -l h_rt=10:00:00
#$ -m ea

python torchnet.py

