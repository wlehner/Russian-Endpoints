#!/bin/bash -l

module load python3

#$ -P lxcompl
#$ -N train85
#$ -j y
#$ -o result_train85.txt
#$ -l h_rt=12:00:00
#$ -m ea

python torchnet.py

