#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 22:46:34 2019

@author: walterlehner
"""

import nltk
#from io import open
#import chardet
#from nltk.corpus.reader.conll import ConllCorpusReader
#from nltk.corpus import universal_treebanks as ut
#import pyconll
import conllu
from conllu import parse
from conllu import parse_tree

file1= 'ru_syntagrus-ud-dev.conllu'
file2= 'ru_syntagrus-ud-test.conllu'
file3= 'ru_syntagrus-ud-train.conllu'
#file2= '/Users/walterlehner/Documents/MA_Project/UD_English-EWT-master/en_ewt-ud-dev.conllu'
changefilename= 'change.conllu'
motionfilename= 'motion.conllu'

changefile = open(changefilename, "w",encoding="utf-8")
motionfile = open(motionfilename, "w",encoding="utf-8")


verbs_of_change = ["стать","встать","поставить","лечь","класть",
                    "положить","уложить","садиться","посадить","сесть",
                    "усесться","вешать","повесить","вешаться",
                    "повеситься","прятаться","спрятаться","прятать",
                    "спрятать","собираться"]

verbs_of_motion = ["идти","ходить","ехать","ездить","бежать","бегать",
                   "брести","бродить","гнать","гонять","лезть","лазить",
                   "лететь","летать","плыть","плавать","ползти",
                   "ползать","везти","возить","нести","носить","вести",
                   "водить","тащить","таскать"]

def readcorpus(filename):
    return parse(open(filename, 'r',encoding ="utf-8").read())

def writeverbs(corp):
    for sentence in corp:
        for word in sentence:
            if word['upostag']=='VERB':
                if word["lemma"] in verbs_of_change:
                    changefile.write(sentence.serialize())
                elif word["lemma"] in verbs_of_motion:
                    motionfile.write(sentence.serialize())

#Processes the three conllu files
def writingcorpi():                  
    writeverbs(readcorpus(file1))
    writeverbs(readcorpus(file2))
    writeverbs(readcorpus(file3))

#Print the number of examples in each new file
def testcorpi():
    motioncorpus = readcorpus(motionfilename)
    changecorpus = readcorpus(changefilename)
    print("Examples of Verbs of Motion: ", len(motioncorpus))
    print("Examples with Change of State Verbs: ", len(changecorpus))
    
#writingcorpi()
#testcorpi()

def sentlengths(corp):
    lengths = [0]*215
    total = 0
    for sentence in corp:
        lengths[len(sentence)]+= 1
    for x in range(len(lengths)):
        total += x*lengths[x]
        print(x , ": " , lengths[x])
    print("Average: " , (total/len(corp)))
    
def sentfeatlengths(corp):
    lengths = [0]*300
    total = 0
    for sentence in corp:
        if len(sentence)<35:
            clength = 0
            for word in sentence:
                if word['upostag']=='VERB' or 'AUX': #Aux doesn't have animacy, should be okay
                    clength += 8
                elif word['upostag']== 'NOUN' or 'PROPN':
                    clength += 5
                elif word['upostag']== 'ADJ':
                    clength += 5
                elif word['upostag']== 'ADV':
                    clength += 1
                elif word['upostag']== 'CCONJ' or 'SCONJ':
                    clength += 1
                elif word['upostag']== 'DET':
                    clength += 5
                elif word['upostag']== 'INTJ':
                    clength += 1
                elif word['upostag']== 'NUM':
                    clength += 4
                elif word['upostag']== 'PART':
                    clength += 3
                elif word['upostag']== 'PRON':
                    clength += 6
                elif word['upostag']== 'ADP':
                    clength += 2
                else: # 'PUNCT' 'SYM' 'X'
                    clength += 1
            lengths[clength] += 1
    for x in range(len(lengths)):
        total += x*lengths[x]
        print(x, ": ", lengths[x])
    print("Average: ", (total/len(corp)))

        

sentfeatlengths(readcorpus(file2))
#sentlengths(readcorpus(file3))

