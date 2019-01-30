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

file= '/Users/walterlehner/Google Drive/workspace/ru_syntagrus-ud-dev.conllu'
#file2= '/Users/walterlehner/Documents/MA_Project/UD_English-EWT-master/en_ewt-ud-dev.conllu'
#data_stream = open(file,"r",encoding ="utf-8")
changefilename= '/Users/walterlehner/Google Drive/workspace/change.conllu'
motionfilename= '/Users/walterlehner/Google Drive/workspace/motion.conllu'

corpus = parse(open(file, 'r',encoding ="utf-8").read())
  
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


for sentence in corpus:
    for word in sentence:
        if word['upostag']=='VERB':
            x+=1
            if word["lemma"] in verbs_of_change:
                changefile.write(sentence.serialize())
            elif word["lemma"] in verbs_of_motion:
                motionfile.write(sentence.serialize())
                
motioncorpus = parse(open(motionfilename, 'r',encoding ="utf-8").read())
changecorpus = parse(open(changefilename, 'r',encoding ="utf-8").read())

print("Examples of Verbs of Motion: ", len(motioncorpus))
print("Examples with Change of State Verbs: ", len(changecorpus))
