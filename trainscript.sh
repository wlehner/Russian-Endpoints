#!/bin/bash -l

module load python3

#$ -P lxcompl
#$ -N train94
#$ -j y
#$ -o result_objall94.txt
#$ -l h_rt=24:00:00
#$ -m ea

python torchnet.py

