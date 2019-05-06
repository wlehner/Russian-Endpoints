#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:17:13 2019

@author: walterlehner
"""
from __future__ import print_function
from conllu import parse
import os, gensim, torch, conllu
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np

corpus_root= 'sample_ar/TEXTS'
fiction_root= 'sample_ar/TEXTS/Fiction/'

file1= 'ru_syntagrus-ud-dev.conllu' #EMPTY???
file2= 'ru_syntagrus-ud-test.conllu'
file3= 'ru_syntagrus-ud-train.conllu'

#Dimensions
sentlen = 205

prepositions = ["в","на","за","к","из","с","от"]


model = Word2Vec.load("word2vec.model")
word_vectors = model.wv

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
            sentexamples = processconlsent(sentence, sentprep)
            if sentexamples:
                examples.append(sentexamples)
    return examples
            
def processconlsent(sentence, preplist):
    examples = []
    for prep in preplist:
        example = np.zeros(sentlen)
        index = 0
        for column, word in enumerate(sentence):
            example.appen[word_vectors[word['lemma']]]
            index+= 1
            featlist = processpos(word)
            if word['lemma'] == prep:
                featlist[1] = 1
            if(sentlen-len(featlist))> index:
                for feats in featlist:
                    example.append(feats)
                    index += 1
            else:
                break
        examples.append(example)
    return examples

#FEATURES: (lemma), POS, number, person, verbform, aspect, tense, voice, mood, case, gender, animacy,
def processpos(word):
    feats = []
    if word['upostag']=='VERB':
        feats = [1, processnum(word), processperson(word), processverbform(word), processaspect(word), 
                 processtense(word), processvoice(word), processmood(word), processcase(word), 
                 processgender(word), processanimacy(word)]
    elif word['upostag']== 'NOUN':
        feats = [2, processnum(word), processcase(word), processgender(word), processanimacy(word)]
    elif word['upostag']== 'ADJ':
        feats = [3, processnum(word), processcase(word), processgender(word), processanimacy(word)]
    elif word['upostag']== 'ADV':
        feats = [4]
    elif word['upostag']== 'AUX':
        feats = [5, processnum(word), processperson(word), processverbform(word), processaspect(word), 
              processtense(word), processvoice(word), processmood(word), processcase(word), 
              processgender(word)]
    elif word['upostag']== 'CCONJ':
        feats = [6]
    elif word['upostag']== 'DET':
        feats = [7, processnum(word), processcase(word), processgender(word), processanimacy(word)]
    elif word['upostag']== 'INTJ':
        feats = [8]
    elif word['upostag']== 'NUM':
        feats = [9, processcase(word), processgender(word), processanimacy(word)]
    elif word['upostag']== 'PART':
        feats = [10, processmood(word)]
    elif word['upostag']== 'PRON':
        feats = [11, processnum(word), processperson(word), processcase(word), processgender(word), 
                 processanimacy(word)]
    elif word['upostag']== 'PROPN':
        feats = [12, processnum(word), processcase(word), processgender(word), processanimacy(word)]
    elif word['upostag']== 'PUNCT':
        feats = [13]
    elif word['upostag']== 'SCONJ':
        feats = [14]
    elif word['upostag']== 'SYM':
        feats = [15]
    elif word['upostag']== 'X':
        feats = [16]
    elif word['upostag']== 'ADP':
        feats = [17, 0]
    return feats

def processnum(word):
    number = {'Sing':1, 'Plur':2}
    if word['feats']['Number']:
        return number[word['feats']['Number']]
    else:
        return 0

def processcase(word):
    case = {'Acc':1, 'Dat':2, 'Gen':3, 'Ins':4, 'Loc':5, 'Nom':6, 'Par':7, 'Voc':8}
    if word['feats']['Case'] in case:
        return case[word['feats']['Case']]
    else:
        return 0

def processgender(word):
    gender = {'Fem':1, 'Masc':2, 'Neut':3}
    if word['feats']['Gender'] in gender:
        return gender[word['feats']['Gender']]
    else:
        return 0
    
def processtense(word):
    tense = {'Fut':1, 'Past':2, 'Pres':3}
    if word['feats']['Tense'] in tense:
        return word['feats']['Tense']
    else:
        return 0
    
def processvoice(word):
    voice = {'Act':1, 'Mid':2, 'Pass':3}
    if word['feats']['Voice'] in voice:
        return voice[word['feats']['Voice']]
    else:
        return 0

def processaspect(word):
    aspect = {'Imp':1, 'Perf':2}
    if word['feats']['Aspect'] in aspect:
        return aspect[word['feats']['Aspect']]
    else:
        return 0

def processmood(word):
    mood = {'Imp':1, 'Ind':2}
    if word['feats']['Mood'] in mood:
        return mood[word['feats']['Mood']]
    else:
        return 0

def processperson(word):
    person = {1:1, 2:2, 3:3}
    if word['feats']['Person'] in person:
        return person[word['feats']['Person']]
    else:
        return 0
    
def processverbform(word):
    verbform = {'Conv':1,'Fin':2, 'Inf':3, 'Part':4}
    if word['feats']['VerbForm'] in verbform:
        return verbform[word['feats']['VerbForm']]
    else:
        return 0

def processanimacy(word):
    animacy = {'Anim':1, 'Inan':2}
    if word['feats']['Animacy'] in animacy:
        return animacy[word['feats']['Animacy']]
    else:
        return 0


#vector = word_vectors["word"]
#wv = KeyedVectors.load("model.wv", mmap='r')
#Construct examples
#   Go through the sentences in Connllu
#       Use vocabulary to make list of words
#       Append a second dimension of grammatical tags
#       Produce appropriate 