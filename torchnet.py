#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:17:13 2019

@author: walterlehner
"""
from __future__ import print_function
import torch
import conllu
from conllu import parse
import os

corpus_root= 'sample_ar/TEXTS'
fiction_root= 'sample_ar/TEXTS/Fiction/'

file1= 'ru_syntagrus-ud-dev.conllu'
file2= 'ru_syntagrus-ud-test.conllu'
file3= 'ru_syntagrus-ud-train.conllu'

#def constructconllu(sentence, objindex, prepindex):
vocabulary = []

def connllubiglist(file):
    corpus = pars(open(changefilename, "w",encoding="utf-8"))
    for sentence in corpus:
        connlluprocess(sentence)
        
def connlluprocess(sentence):
    example = []
    example[0] = []
    for word in sentence:
        base = word["lemma"]
        if base not in biglist:
            biglist.append(base)
        example[0].append(biglist.index(base))
                
def xmlbiglist(file):
    
    
#Construct Vocabulary from both corpuses
#Construct examples
#   Go through the sentences in Connllu
#       Use vocabulary to make list of words
#       Append a second dimension of grammatical tags
#       Produce appropriate 