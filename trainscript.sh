#!/bin/bash -l

module load python3

#$ -P lxcompl
#$ -N testsave13
#$ -j y
#$ -o result_an_63.txt
#$ -l h_rt=12:00:00
#$ -m ea

python torchnet.py

