#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:18:41 2018

@author: walterlehner
"""
#https://towardsdatascience.com/a-simple-starter-guide-to-build-a-neural-network-3c2cf07b8d7c
#https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#sphx-glr-beginner-nlp-deep-learning-tutorial-py

import nltk
import chardet
import re
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

#FEATURES: (lemma), POS, number, person, verbform, aspect, tense, voice, mood, case, gender, animacy,
        
def processword(word):
    featsfinal = []
    feats = re.split('[,=]',word.get('gr'))
    if feats[0] == 'V': #Verb
        featsfinal = [1, processnum(feats[3])]
    elif feats[0] == 'S': #Noun
        featsfinal = [2, processnum(feats[3])]
    elif feats[0] == 'A': #Adj
        featsfinal = [3, processnum(feats[1])]
    elif feats[0] == 'ADV' or 'ADV-PRO' or 'PRAEDIC' or 'PARENTH': #Adv, some ADV-PRO could be particles
        featsfinal = [4]                                #Predicatives are also questionable
    elif feats[0] == 'CONJ': #CConj and sconj
        featsfinal = [6]
    elif feats[0] == 'A-PRO': #Det, but could be Pronoun
        featsfinal = [7, processnum(feats[1])]
    elif feats[0] == 'INTJ': #Intj
        featsfinal = [8]
    elif feats[0] == 'NUM' or 'A-NUM': #NUM
        featsfinal = [9]
    elif feats[0] == 'PART': #Part
        featsfinal = [10]
    elif feats[0] == 'S-PRO': #Pron
        featsfinal = [11, processnum(feats[1])]
    elif feats[0] == 'PR': #Adp Preposition
        featsfinal = [12, 0]
    return featsfinal

#Process features methods are all going to take strings
def processnum(number):
    numbers = {'sg':1,'pl':2}
    if number in numbers:
        return numbers[number]
    else:
        return 0
    
def processcase(case):
    cases = {'nom':6, 'gen':3, 'dat':2, 'acc':1, 'ins':4, 'loc':5, # Lack of 'Par'? Need to look up
             'gen2':3, 'acc2':1, 'loc2':5, 'voc':8, 'adnum':0}     # all these weird cases
    if case in cases:
        return cases(case)
    else:
        return 0
    
def processanimacy(anim):
    anims = {'anim':1, 'inan':2}
    if anim in anims:
        return anims(anim)
    else:
        return 0
    
def processgender(gen):
    gens = {'f':1, 'm':2, 'm-f':3, 'n':3} #Need to check on common gender...
    if gen in gens:
        return gens(gen)
    else:
        return 0
    
def processmood(mood):
    moods = {'imper':1, 'imper2':1, 'indic':2}
    if mood in moods:
        return moods(mood)
    else:
        return 0
             
def processtense(tense):
    tenses = {'praet':2, 'praes':3, 'fut':1}
    if tense in tenses:
        return tenses(tense)
    else:
        return 0
    
def processaspect(aspect):
    aspects = {'pf':2, 'ipf':1}
    if aspect in aspects:
        return aspects(aspect)
    else:
        return 0
    
def processvoice(voice):
    voices = {'act':1, 'pass':3, 'med':2} #med seems to mean reflexive
    if voice in voices:
        return voices(voice)
    else:
        return 0
    
def processperson(person):
    persons = {'1p':1, '2p':2, '3p':3}
    if person in persons:
        return persons(person)
    else:
        return 0
    
def processverbform(form):
    forms = {'inf':2, 'partcp':3, 'ger':1}
    if form in forms:
        return forms(form)
    else:
        return 0

def testprocess(directory):
    file = os.listdir(directory)[2]
    filexml = ET.parse(directory + file)
    root = filexml.getroot()
    sentence = root[1][3][1]
    for x in range(len(sentence)):
        word = sentence[x][0]
        print(word.get('lex'))
        print(re.split('[,=]',word.get('gr')))
    
testprocess(fiction_root)
        
    

            

#testwords = testfiction.findall("w")

#print(fictionfiles[1])

#print(testwords)

#print(corpusfiction.fileids())
#print(corpusfiction.words('ant3_31.xhtml'))
