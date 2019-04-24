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
import numpy as np

corpus_root= 'sample_ar/TEXTS'
fiction_root= 'sample_ar/TEXTS/Fiction/'

file1= 'ru_syntagrus-ud-dev.conllu' #EMPTY???
file2= 'ru_syntagrus-ud-test.conllu'
file3= 'ru_syntagrus-ud-train.conllu'

#Dimensions
sentlen = 15
featsnum = 12

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
    for prep in preplist:
        example = np.zeros((featsnum,sentlen))
        column = 0
        for word in sentence:
            example[column,0] = word_vectors[word['lemma']]
            featlist = processpos(word)
            for x in range(featsnum-1):
                example[column,(x+1)] = featlist[x]
            column += 1

#FEATURES: (lemma)POS, number, person, verbform, aspect, tense, voice, mood, case, gender, animacy,
def processpos(word):
    feats = [0] * (featsnum-1)
    if word['upostag']=='VERB':
        feats[0] = 1
        feats[1] = processnum(word)
        feats[2] = processperson(word)
        feats[3] = processverbform(word)
        feats[4] = processaspect(word)
        feats[5] = processtense(word)
        feats[6] = processvoice(word)
        feats[7] = processmood(word)
        feats[8] = processcase(word)
        feats[9] = processgender(word)
        feats[10] = processanimacy(word)
        return feats
    elif word['upostag']== 'NOUN':
        feats[0] = 2
        feats[1] = processnum(word)
        feats[8] = processcase(word)
        feats[9] = processgender(word)
        feats[10] = processanimacy(word)
        return feats
    elif word['upostag']== 'ADJ':
        feats[0] = 3
        feats[1] = processnum(word)
        feats[8] = processcase(word)
        feats[9] = processgender(word)
        feats[10] = processanimacy(word)
        return feats
    elif word['upostag']== 'ADV':
        feats[0] = 4
        return feats
    elif word['upostag']== 'AUX':
        feats[0] = 5
        feats[1] = processnum(word)
        feats[2] = processperson(word)
        feats[3] = processverbform(word)
        feats[4] = processaspect(word)
        feats[5] = processtense(word)
        feats[6] = processvoice(word)
        feats[7] = processmood(word)
        feats[8] = processcase(word)
        feats[9] = processgender(word)
        return feats
    elif word['upostag']== 'CCONJ':
        feats[0] = 6
        return feats
    elif word['upostag']== 'DET':
        feats[0] = 7
        feats[1] = processnum(word)
        feats[8] = processcase(word)
        feats[9] = processgender(word)
        feats[10] = processanimacy(word)
        return feats
    elif word['upostag']== 'INTJ':
        feats[0] = 8
        return feats
    elif word['upostag']== 'NUM':
        feats[0] = 9
        feats[8] = processcase(word)
        feats[9] = processgender(word)
        feats[10] = processanimacy(word)
        return feats
    elif word['upostag']== 'PART':
        feats[0] = 10
        feats[7] = processmood(word)
        return feats
    elif word['upostag']== 'PRON':
        feats[0] = 11
        feats[1] = processnum(word)
        feats[2] = processperson(word)
        feats[8] = processcase(word)
        feats[9] = processgender(word)
        feats[10] = processanimacy(word)
        return feats
    elif word['upostag']== 'PROPN':
        feats[0] = 12
        feats[1] = processnum(word)
        feats[8] = processcase(word)
        feats[9] = processgender(word)
        feats[10] = processanimacy(word)
        return feats
    elif word['upostag']== 'PUNCT':
        feats[0] = 13
        return feats
    elif word['upostag']== 'SCONJ':
        feats[0] = 14
        return feats
    elif word['upostag']== 'SYM':
        feats[0] = 15
        return feats
    elif word['upostag']== 'X':
        feats[0] = 16
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