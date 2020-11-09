#!/bin/bash -l

module load python3

#$ -P lxcompl
#$ -N test79
#$ -j y
#$ -o result_an_79.txt
#$ -l h_rt=12:00:00
#$ -m ea

python torchnet.py

