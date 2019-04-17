#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:17:13 2019

@author: walterlehner
"""
from __future__ import print_function
from conllu import parse
import os, gensim, torch, conllu
from gensim.models import word2vec
from gensim.models import KeyedVectors

corpus_root= 'sample_ar/TEXTS'
fiction_root= 'sample_ar/TEXTS/Fiction/'

file1= 'ru_syntagrus-ud-dev.conllu'
file2= 'ru_syntagrus-ud-test.conllu'
file3= 'ru_syntagrus-ud-train.conllu'

prepositions = ["в","на","за","к","из","с","от"]


#Takes Conllu File and Produces a List of Sentences as lists of Lemmas
def conllulemlist(file):
    result = []
    corpus = parse(open(file, 'r',encoding ="utf-8").read())
    for sentence in corpus:
        sentlist = []
        for word in sentence:
            sentlist.append(word["lemma"])
        result.append(sentlist)
    return result


sentences = conllulemlist(file3)
model = gensim.models.Word2Vec(sentences, min_count=1)
model.save("word2vec.model")
word_vectors = model.wv
#del model

#Takes Conllu Format and Produces a list of examples

def processconllu(file):
    corpus = parse(open(file, 'r',encoding ="utf-8").read())
    examples = []
    for sentence in corpus:
        sentprep = []
        for word in sentence:
            if word['lemma'] in prepositions:
                sentprep.append(word['lemma'])
        if sentprep:
            examples.append(processconlsent(sentence, sentprep))
    return examples
            
def processconlsent(sentence, preplist):
    pos = []
    number = []
    for word in sentence:
#POS
        if word['upostag']=='VERB':
            pos.append(1)
        elif word['upostag']== 'NOUN':
            pos.append(2)
        elif word['upostag']== 'ADJ':
            pos.append(3)
        elif word['upostag']== 'ADV':
            pos.append(4)
        elif word['upostag']== 'AUX':
            pos.append(5)
        elif word['upostag']== 'CCONJ':
            pos.append(6)
        elif word['upostag']== 'DET':
            pos.append(7)
        elif word['upostag']== 'INTJ':
            pos.append(8)
        elif word['upostag']== 'NUM':
            pos.append(9)
        elif word['upostag']== 'PART':
            pos.append(10)
        elif word['upostag']== 'PRON':
            pos.append(11)
        elif word['upostag']== 'PROPN':
            pos.append(12)
        elif word['upostag']== 'PUNCT':
            pos.append(13)
        elif word['upostag']== 'SCONJ':
            pos.append(14)
        elif word['upostag']== 'SYM':
            pos.append(15)
        elif word['upostag']== 'X':
            pos.append(16)
#NUMBER
        if word['number']== 'Sing':
            number.append(1)
        elif word['number']== 'Plur':
            number.append(2)
        
    
        

#vector = word_vectors["word"]
#wv = KeyedVectors.load("model.wv", mmap='r')
#Construct examples
#   Go through the sentences in Connllu
#       Use vocabulary to make list of words
#       Append a second dimension of grammatical tags
#       Produce appropriate 