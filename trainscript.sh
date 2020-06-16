#!/bin/bash -l

module load python3

#$ -P lxcompl
#$ -N ShuffleTest2
#$ -j y
#$ -o TestShuffle2
#$ -l h_rt=20:00:00
#$ -m ea

python torchnet.py

