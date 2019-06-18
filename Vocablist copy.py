#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:48:32 2019

@author: walterlehner
"""

from __future__ import print_function
from conllu import parse
import os, gensim, torch, conllu
from gensim.models import word2vec
from gensim.models import KeyedVectors
import numpy as np

corpus_root= 'sample_ar/TEXTS'
fiction_root= 'sample_ar/TEXTS/Fiction/'

file1= 'ru_syntagrus-ud-dev.conllu' #EMPTY???
file2= 'ru_syntagrus-ud-test.conllu'
file3= 'ru_syntagrus-ud-train.conllu'


#Takes Conllu File and Produces a List of Sentences as lists of Lemmas
def conllulemlist(file):
    result = []
    summ = 0
    corpus = parse(open(file, 'r',encoding ="utf-8").read())
    for sentence in corpus:
        summ += len(sentence)
        sentlist = []
        for word in sentence:
            sentlist.append(word["lemma"])
        result.append(sentlist)
    print(summ/(len(corpus)))
    return result


sentences = conllulemlist(file3)
sentences.extend(conllulemlist(file2))
sentences.extend(conllulemlist(file1))
model = gensim.models.Word2Vec(sentences, size=1, min_count=1)
model.save("word2vec.model")


