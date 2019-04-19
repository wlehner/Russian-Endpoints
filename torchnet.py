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
        pos.append(processpos(word))
        
def processpos(word):
    if word['upostag']=='VERB':
        return 1
    elif word['upostag']== 'NOUN':
        return 2
    elif word['upostag']== 'ADJ':
        return 3
    elif word['upostag']== 'ADV':
        return 4
    elif word['upostag']== 'AUX':
        return 5
    elif word['upostag']== 'CCONJ':
        return 6
    elif word['upostag']== 'DET':
        return 7
    elif word['upostag']== 'INTJ':
        return 8
    elif word['upostag']== 'NUM':
        return 9
    elif word['upostag']== 'PART':
        return 10
    elif word['upostag']== 'PRON':
        return 11
    elif word['upostag']== 'PROPN':
        return 12
    elif word['upostag']== 'PUNCT':
        return 13
    elif word['upostag']== 'SCONJ':
        return 14
    elif word['upostag']== 'SYM':
        return 15
    elif word['upostag']== 'X':
        return 16

def processnum(word):
    if word['feats']['Number']== 'Sing':
        return 1
    elif word['feats']['Number']== 'Plur':
        return 2
    else:
        return 0

def processcase(word):
    if word['feats']['Case'] =='Acc':
        return 1
    elif word['feats']['Case'] =='Dat':
        return 2
    elif word['feats']['Case'] =='Gen':
        return 3
    elif word['feats']['Case'] =='Ins':
        return 4
    elif word['feats']['Case'] =='Loc':
        return 5
    elif word['feats']['Case'] =='Nom':
        return 6
    elif word['feats']['Case'] =='Par':
        return 7
    elif word['feats']['Case'] =='Voc':
        return 8
    else:
        return 0

def processgender(word):
    if word['feats']['Gender']=='Fem':
        return 1
    elif word['feats']['Gender']=='Masc':
        return 2
    elif word['feats']['Gender']=='Neut':
        return 3
    else:
        return 0
    
def processtense(word):
    if word['feats']['Tense']=='Fut':
        return 1
    elif word['feats']['Tense']=='Past':
        return 2
    elif word['feats']['Tense']=='Pres':
        return 3
    else:
        return 0

def processaspect(word):
    if word['feats']['Aspect']=='Imp':
        return 1
    elif word['feats']['Aspect']=='Perf':
        return 2
    else:
        return 0

def processmood(word):
    if word['feats']['Mood']=='Imp':
        return 1
    if word['feats']['Mood']=='Ind':
        return 2
    else:
        return 0

def processperson(word):
    if word['feats']['Person']=='1':
        return 1
    elif word['feats']['Person']=='2':
        return 2
    elif word['feats']['Person']=='3':
        return 3
    else:
        return 0
    
def processverbform(word):
    if word['VerbForm']=='':
        return



    
    


#TENSE
        
    
        

#vector = word_vectors["word"]
#wv = KeyedVectors.load("model.wv", mmap='r')
#Construct examples
#   Go through the sentences in Connllu
#       Use vocabulary to make list of words
#       Append a second dimension of grammatical tags
#       Produce appropriate 