#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:18:41 2018

@author: walterlehner
"""

import nltk
import chardet
#from nltk.corpus.reader import XMLCorpusReader
import os
#from tensorflow.keras import layers
import xml.etree.ElementTree as ET

corpus_root= 'sample_ar/TEXTS'
fiction_root= 'sample_ar/TEXTS/Fiction/'

fictionfiles = os.listdir(fiction_root)
corpusfiles = 'sample_ar/TEXTS/.*.xhtml'

#corpusfiction = XMLCorpusReader(fiction_root, fictionfiles)
#corpus = XMLCorpusReader(corpus_root, corpusfiles)

sentchangelist = []
sentmotionlist = []

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


#def copynode(node, parent):
    

def processdirectory(directory):
    filelist = os.listdir(directory)
    for file in filelist:
        filexml = ET.parse(directory + file)
        for sentence in filexml.getroot()[1].iter('se'):
            for word in sentence.iter('w'):
                if word[0].get('lex') in verbs_of_change:
                    sentchangelist.append(sentence)
                if word[0].get('lex') in verbs_of_motion:
                    sentmotionlist.append(sentence)
                    
processdirectory(fiction_root)
print(sentmotionlist)

#testwords = testfiction.findall("w")

#print(fictionfiles[1])

#print(testwords)

#print(corpusfiction.fileids())
#print(corpusfiction.words('ant3_31.xhtml'))
