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
sentlen = 205

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
                    
#processdirectory(fiction_root)
#print(sentmotionlist)

def processsentence(sentence):
    product = []
    for word in sentence.iter('w'):
        feats = processword(word)
        
# http://www.ruscorpora.ru/en/corpora-morph.html
# https://universaldependencies.org/treebanks/ru_syntagrus/index.html
        
def processword(word):
    featsfinal = []
    feats = word.get('gr').split(',' and '=')
    if feats[0] == 'V': #Verb
        featsfinal = [1]
    elif feats[0] == 'S': #Noun
        featsfinal = [2]
    elif feats[0] == 'A': #Adj
        featsfinal = [3]
    elif feats[0] == 'ADV' or 'ADV-PRO' or 'PRAEDIC' or 'PARENTH': #Adv, some ADV-PRO could be particles
        featsfinal = [4]                                #Predicatives are also questionable
#    elif feats[0] == 'S': #Aux-> kill lump with verb
#        featsfinal = [5]
    elif feats[0] == 'CONJ': #CConj and sconj
        featsfinal = [6]
    elif feats[0] == 'A-PRO': #Det, but could be Pronoun
        featsfinal = [7]
    elif feats[0] == 'INTJ': #Intj
        featsfinal = [8]
    elif feats[0] == 'NUM' or 'A-NUM': #NUM
        featsfinal = [9]
    elif feats[0] == 'PART': #Part
        featsfinal = [10]
    elif feats[0] == 'S-PRO': #Pron
        featsfinal = [11]
#    elif feats[0] == 'S': #Propn-> to kill, lump with noun
#        featsfinal = [12]
#    elif feats[0] == 'S': #Punc-> to kill, no equivalent
#        featsfinal = [13]
#    elif feats[0] == 'S': #Sconj-> to kill, lump with cconj
#        featsfinal = [14]
#    elif feats[0] == 'S': #Sym-< to kill, no equivalent
#        featsfinal = [15]
#    elif feats[0] == 'S': #X -> Turn into ambiguious case or 0?
#        featsfinal = [16]
    elif feats[0] == 'PR': #Adp Preposition
        featsfinal = [17]
    

def testprocess(directory):
    file = os.listdir(directory)[1]
    filexml = ET.parse(directory + file)
    root = filexml.getroot()
    sentence = root[1][1][1]
    word = sentence[2][0]
    print(word.get('lex'))
    print(word.get('gr').split(',' and '='))
    
testprocess(fiction_root)
        
    

            

#testwords = testfiction.findall("w")

#print(fictionfiles[1])

#print(testwords)

#print(corpusfiction.fileids())
#print(corpusfiction.words('ant3_31.xhtml'))
