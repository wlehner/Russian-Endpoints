#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 17:18:41 2018

@author: walterlehner
"""
from googletrans import Translator
#import nltk
import torch, pickle
import torch.nn as nn
import numpy as np
#import torch.nn.functional as f
from gensim.models import Word2Vec

#import chardet
import re
#from nltk.corpus.reader import XMLCorpusReader
import os
#from tensorflow.keras import layers
import xml.etree.ElementTree as ET

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 
import six
#from google.cloud import translate_v2 as translate

#translator = Translator()
model = Word2Vec.load("word2vec4.model")

corpus_root= 'sample_ar/TEXTS'
fiction_root= 'sample_ar/TEXTS/fiction/'
public_root = corpus_root + '/public/'
science_root = corpus_root + '/science/'
speech_root = corpus_root + '/speech/'

loadobj = 'networks_folder/tnet_100hl200x1obj92.pkl' #'tnet_100hl200x1obj87.pkl'
loadsrc = 'networks_folder/tnet_100hl200x1src91.pkl' #'tnet_100hl200x1src88.pkl'

fictionfiles = os.listdir(fiction_root)
corpusfiles = 'sample_ar/TEXTS/.*.xhtml'
word_vectors = model.wv

input_size = 440 #154
hidden_size = 200  #154
output_size = 45 #Output Size and sentence length need to be seperated
hidden_layers = 1
poss_limit = 4

sentchangelist = []
sentmotionlist = []
sentlen = 440 #205

#trans_errors = 0

ceffile= 'ChangeEx_Fiction'
meffile= 'MotionEx_Fiction'
mesfile= 'MotionEx_Speech'
cesfile='ChangeEx_Science'
mesfile= 'MotionEx_Science'

class color():
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

#translate_client = translate.Client()

prepdir2 = {'в':'acc','на':'acc','под':'acc','за':'acc','к':'dat'}
preploc2 = {'в':'loc','на':'loc','под':'ins','за':'ins','перед':'ins','над':'ins','у':'gen'}
prepdir1 = {'в':'acc','на':'acc','под':'acc','за':'acc','к':'dat','из':'gen','с':'gen','от':'gen'} #Includes 'from'
preploc1 = {'в':'loc','на':'loc','под':'ins','за':'ins'} #incomplete
foreward = ['в','на','под','за','к','перед','над','у']

directprep = prepdir2
locatprep = preploc2
allprep = foreward

prepositions = ['в','на','под','за','к','перед','над','у','за','из','с','от']
#locational prep = ['в','на','под','за','перед','над','у']

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

def main_method():
    netload(loadobj, netobj)
    netload(loadsrc, netsrc)
    
class Net(nn.Module):
    def __init__(self, inputs, hiddens, hidden_layers2, outputs):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        #self.layers.append(nn.Linear(inputs, hiddens))
        self.fc1 = nn.Linear(inputs, hiddens)
        #self.layers.append(nn.ReLU())
        self.relu1 = nn.ReLU()
        for l in range(hidden_layers2):
            self.layers.append(nn.Linear(hiddens, hiddens))
            self.layers.append(nn.ReLU())
        #self.fc2 = nn.Linear(hiddens, hiddens)
        #self.layers.append(nn.ReLU())
        #self.relu2 = nn.ReLU()
        #self.layers.append(nn.Linear(hiddens, outputs))
        self.fc3 = nn.Linear(hiddens, outputs)

    def forward(self, x):
        #out = x
        out = self.fc1(x)
        out = self.relu1(out)
        for layer in self.layers:
            out = layer(out)
        #out = self.fc2(out)
        #out = self.relu2(out)
        out = self.fc3(out)
        return out
    
#Initialize NN    
netsrc = Net(inputs= input_size, hiddens= hidden_size, hidden_layers2= hidden_layers, outputs= output_size) #
netsrc = netsrc.float()

netobj = Net(inputs= input_size, hiddens= hidden_size, hidden_layers2= hidden_layers, outputs= output_size) #
netobj = netobj.float()

def netload(filename, nn):
    nn.load_state_dict(torch.load(filename))
    nn.eval()

def pickleload(filename):
    return pickle.load(open(filename,"rb"))

#def ru_translate(text):
    #if isinstance(text, six.binary_type):
        #text = text.decode("utf-8")
    #return translate_client.translate(text, target_language='en', source_language='ru')['translatedText']    
    
def test(directory):    
    file = os.listdir(directory)[2]
    filexml = ET.parse(directory + file) #ET.parse(fiction_root + (os.listdir(fiction_root)[2])).getroot()[1][3][3]
    root = filexml.getroot()
    sentence = root[1][3][3]
    for index, word in enumerate(sentence):
        print(index, word[0].get('lex'))
    return sentence
    

#def copynode(node, parent):
    
def printsection(biglist, num):
    start = num*10
    end = (num+1)*10
    printsentlist(biglist[start:end], start)
    #printgram(biglist[start:end])
        
#def transent(sentence):
    #bad_trans= ''
    #for word in sentence.iter('w'):
        #ru_word = word[0].get('lex')
        #ru_line += (' ' + ru_word)
        #bad_trans += (' ' + color.UNDERLINE + ru_translate(ru_word) + color.END + getgram(word))
    #print(ru_line)
    #return bad_trans
    
def printsentlist(sentences, offset):
    for index, sentence in enumerate(sentences):
        ru_line = ''
        for word in sentence.iter('w'):
            ru_line += word[0].get('lex') + ' '
        print((index+offset), ru_line)
        #print (transent(sentence), '\n')
        
def printgram(sentences):
    for index, sentence in enumerate(sentences):
        linegram = ''
        for word in sentence.iter('w'):
            if word[0].get('lex') in prepositions:
                linegram+= '###'
            linegram+= word[0].get('gr') + ' '
        print (index +1, linegram, '\n')

mlex = []

clex = [8]

def savelist(sentlist, numlist, filename):
    newlist = []
    for num in numlist:
        newlist.append(sentlist[num])
    pickle.dump(newlist, open(filename, "wb"))

def getgram(word):
    gram = word[0].get('gr')
    final = '-'
    if gram:
        if 'ADV' in gram:
            final = final + 'Adv'
        if 'PR' == gram[0:2]:
            final = final + 'Prep'
        if 'S' in gram:
            noungr = ['sg,','pl,']
            for ngr in noungr:
                if ngr in gram:
                    final = final + 'N,' + ngr + gram[-3:]
        if 'V' == gram[0]:
            verbgr = ['sg', 'pl,', '2p', '3p', '1p']
            final +=  'V,'
            for gr in verbgr:
                if gr in gram:
                    final = final + gr + ','
    return final

def getg(word):
    print(word[0].get('gr'))
    
#Makes a list of sentences with the relevant verbs
def processdirectory(directory):
    sentmotionlist = []
    sentchangelist = []
    posschange = []
    possmotion = []
    counter = 0
    filelist = os.listdir(directory)
    for file in filelist:
        filexml = ET.parse(directory + file)
        for sentence in filexml.getroot()[1].iter('se'):
            counter += 1
            for index, word in enumerate(sentence.iter('w')):
                if word[0].get('lex') in verbs_of_change:
                    sentchangelist.append((index, sentence))
                if word[0].get('lex') in verbs_of_motion:
                    sentmotionlist.append((index, sentence))
    posschange = checksentlist(sentchangelist)
    possmotion = checksentlist(sentmotionlist)
    print(counter, 'Sentences')
    return (posschange, possmotion)
#processdirectory(fiction_root)
#print(sentmotionlist)
    
def searchdirectory(directory):
    motionlists = {'ll':[], 'dd':[], 'ld':[], 'dl':[], 'o':[]}
    changelists = {'ll':[], 'dd':[], 'ld':[], 'dl':[], 'o':[]}
    filelist = os.listdir(directory)
    for file in filelist:
        filexml = ET.parse(directory + file)
        for sentence in filexml.getroot()[1].iter('se'):
            for index, word in enumerate(sentence.iter('w')):
                if word[0].get('lex') in verbs_of_change:
                    c = searchsent(index, sentence)
                    if c:
                        changelists[c].append(sentence)
                if word[0].get('lex') in verbs_of_motion:
                    m = searchsent(index, sentence)
                    if m:
                        motionlists[m].append(sentence)
    return(motionlists, changelists)
    
def searchlocatnest(directory): #Searches for Nested locational phrases
    locatnestlist = []
    filelist = os.listdir(directory)
    for file in filelist:
        filexml = ET.parse(directory + file)
        for sentence in filexml.getroot()[1].iter('se'):
            c = False
            for index, word in enumerate(sentence.iter('w')):
                if word[0].get('lex') in locatprep:
                    possobj = checkprepobj(sentence, index)
                    if any(possobj):
                        for obj in possobj:
                            possprep = checkword2(sentence, obj, locatprep)
                            if any(possprep):
                                for prep in possprep:
                                    if prep != index:
                                        c = True
            if c:
                locatnestlist.append(sentence)
    print('Possible Examples: ', len(locatnestlist))
    return locatnestlist

def searchlocatdest(directory):
    locatdestlist = []
    filelist = os.listdir(directory)
    for file in filelist:
        filexml = ET.parse(directory + file)
        for sentence in filexml.getroot()[1].iter('se'):
            c = False
            for index, word in enumerate(sentence.iter('w')):
                if word[0].get('lex') in verbs_of_change:
                    locpreps = checkword2(sentence, index, locatprep)
                    if any(locpreps):
                        for prep in locpreps:
                            preplex = getlex(sentence, prep)
                            possobj = checkprepobj(sentence, prep)
                            if any(possobj):                                
                                for obj in possobj:
                                    for index, word in enumerate(sentence.iter('w')):
                                        if index == obj:
                                            feats = re.split('[,=]',word[0].get('gr'))
                                            if locatprep[preplex] in feats:
                                                c = True
            if c:
                locatdestlist.append(sentence)
    print('Possible examples: ', len(locatdestlist))
    return locatdestlist

def searchlocatdest2(directory):
    locatdestlist = []
    filelist = os.listdir(directory)
    for file in filelist:
        filexml = ET.parse(directory + file)
        for sentence in filexml.getroot()[1].iter('se'):
            c = False
            for index, word in enumerate(sentence.iter('w')):
                if word[0].get('lex') in verbs_of_change:
                    locpreps = checkword2(sentence, index, locatprep)
                    if any(locpreps):
                        c = True
            if c:
                locatdestlist.append(sentence)
    print('Possible examples: ', len(locatdestlist))
    return locatdestlist
                                    
    
def searchsent(index, sentence):
    inner, outer = '', ''
    possprep = checkword2(sentence, index, allprep)
    if any(possprep):
        for prep in possprep:
            possobj = checkprepobj(sentence, prep)
            if any(possobj):
                for obj in possobj:
                    possprep2 = checkword2(sentence, obj, allprep)
                    if any(possprep2):
                        for prep2 in possprep2:
                            if prep != prep2:
                                possobj2 = checkprepobj2(sentence, prep2)
                                if possobj2:
                                    preplex = getlex(sentence, prep)
                                    preplex2 = getlex(sentence, prep2)
                                    for index, word in enumerate(sentence.iter('w')):
                                        if index == obj:
                                            feats = re.split('[,=]',word[0].get('gr'))
                                            if (preplex in directprep) and (directprep[preplex] in feats):
                                                inner = 'd'
                                            elif (preplex in locatprep) and (locatprep[preplex] in feats):
                                                inner = 'l'
                                        if index == possobj2:
                                            feats = re.split('[,=]',word[0].get('gr'))
                                            if (preplex2 in directprep) and (directprep[preplex2] in feats):
                                                outer = 'd'
                                            elif (preplex2 in locatprep) and (locatprep[preplex2] in feats):
                                                outer = 'l'
                                    if inner and outer:
                                        return inner + outer
                                return 'o'
    return False
                        
def lendictionlist(dictionlist):
        print('ll:', len(dictionlist['ll']))
        print('dd:', len(dictionlist['dd']))
        print('dl:', len(dictionlist['dl']))
        print('ld:', len(dictionlist['ld']))
        print('o:', len(dictionlist['o']))

def printtotal(directory):
        mol, chl = searchdirectory(directory)
        dictionlist = mol      
        sum1 =(len(dictionlist['ll']) + len(dictionlist['dd']) +len(dictionlist['ld']) + len(dictionlist['dl']) + len(dictionlist['o']))
        dictionlist = chl
        sum1+= (len(dictionlist['ll']) + len(dictionlist['dd']) +len(dictionlist['ld'])+ len(dictionlist['dl']) + len(dictionlist['o']))
        print(sum1)
        
def checksentlist(sentlist):
    posslist = []
    for index, sent in sentlist:
        possprep = checkword2(sent, index, allprep)
        s = False
        if possprep:
            for prep in possprep:
                preplex = getlex(sent, prep)
                possprep2 = checkword2(sent, prep, allprep)
                if possprep2:
                    for prep2 in possprep:
                        if prep != prep2:
                            preplex2 = getlex(sent, prep2)
                            if checkobject2(sent, preplex, preplex2):
                                s = True
        if s: #Each Sentence appears only once
            posslist.append(sent)
    return posslist

#takes sentence and prepositions, returns whether necessary cases are present
def checkobject2(sentence, preplex, preplex2): #tracks index already taken
    preplist = {'в':'acc','на':'acc','к':'dat','из':'gen','с':'gen','от':'gen'}
    case = preplist[preplex]
    case2 = preplist[preplex]
    first = False
    second = False
    if case and case2:
        for index, word in enumerate(sentence.iter('w')):
            feats = re.split('[,=]',word[0].get('gr'))
            if (case in feats) and not first:
                first = True
            elif (case2 in feats):
                second = True
        if first and second:
            return True
    return False

def getlex(sent, lindex):
    for index, word in enumerate(sent.iter('w')):
        if lindex == index:
            return word[0].get('lex')

def flatlist(vector):
    vlist = []
    for  x in vector:
        vlist.append(x)
    return vlist

def process_sentence(sentence, prepindex):
    product = []
    if len(sentence)*11 > input_size:
        #print('Too Long')
        return False
    for index, word in enumerate(sentence.iter('w')):
        ru_word = word[0].get('lex')
        if ru_word in word_vectors:
            product.append(flatlist(word_vectors[word['lemma']]))
        else:
            product.append(0)
        feats = processword(word)
        if prepindex == index:
            feats[1]= 1
        product.extend(feats)
    product.extend([0]*input_size)
    product = product[:input_size]
    return product

#Takes sentence and Location of Parent, return list of possible prepositions
def checkword2(sentence, wordindex, preplist):
    possprep = []
    for index, word in enumerate(sentence.iter('w')):
        lex = word[0].get('lex')
        if lex in preplist:
            tempquest = process_sentence(sentence, index)
            possibilities = []
            if tempquest:
                output = netsrc(torch.tensor(tempquest).float())
                possibilities = np.argpartition(output.detach().numpy(), -poss_limit)[-poss_limit:]
            if any(possibilities):
                for poss_index in possibilities:
                    if poss_index == wordindex:
                        possprep.append(index)
    return possprep

#takes sentence and location of preposition, returns possible objects
def checkprepobj(sentence, prepindex):
    possobj = []
    tempquest = process_sentence(sentence, prepindex)
    if tempquest:
        output = netobj(torch.tensor(tempquest).float())
        possobj = np.argpartition(output.detach().numpy(), -poss_limit)[-poss_limit:]
    return possobj

def checkprepobj2(sentence, prepindex):
    tempquest = process_sentence(sentence, prepindex)
    if tempquest:
        output = netobj(torch.tensor(tempquest).float())
        return output.max(0)[1]
        
#def ru_translate2(sentence_ru):
    #for i in range(3):
        #try:
            #return translator.translate(sentence_ru, src="ru", dest= "en").text
        #except:
            #global trans_errors
            #trans_errors += 1
    #return ('TF')
        
# http://www.ruscorpora.ru/en/corpora-morph.html
# https://universaldependencies.org/treebanks/ru_syntagrus/index.html

## POS, NUM, Person, Case, verbform, aspect, tense, animacy, voice, mood,
def processword(word):
    feats = re.split('[,=]',word[0].get('gr'))
    return [processpos(feats), processnum(feats), processperson(feats), processcase(feats), processverbform(feats), 
            processaspect(feats), processtense(feats), processanimacy(feats), processvoice(feats), processmood(feats)]

#Process features methods are all going to take lists of strings
#'VERB':1, 'AUX':1, 'NOUN':2, 'PROPN':2, 'ADJ':4, 'ADV':5, 'CCONJ':6, 'SCONJ':6, 'DET':7, 'INTJ':8, 'NUM':9, 'PART':10, 'PRON':3, 'ADP':11
def processpos(feats):
    speech_parts = {'V':1,'S':2, 'A':3,'ADV':4,'ADV-PRO':4, 'PRAEDIC':4, 'PARENTH':4, 
                    'CONJ':6, 'A-PRO':7, 'INTJ':8, 'NUM':9, 'A-NUM':9, 'PART':10, 'S-PRO':11, 'PR':12}
    for feat in feats:
        if feat in speech_parts:
            return speech_parts[feat]
    return 0

def processnum(feats):
    numbers = {'sg':1,'pl':2}
    for feat in feats:
        if feat in numbers:
            return numbers[feat]
    return 0
    
def processcase(feats):
    cases = {'nom':6, 'gen':3, 'dat':2, 'acc':1, 'ins':4, 'loc':5, # Lack of 'Par'? Need to look up
             'gen2':3, 'acc2':1, 'loc2':5, 'voc':8, 'adnum':0}     # all these weird cases
    for feat in feats:
        if feat in cases:
            return cases[feat]
    return 0
    
def processanimacy(feats):
    anims = {'anim':1, 'inan':2}
    for feat in feats:
        if feat in anims:
            return anims[feat]
    return 0
    
def processgender(feats):
    gens = {'f':1, 'm':2, 'm-f':3, 'n':3} #Need to check on common gender...
    for feat in feats:
        if feat in gens:
            return gens[feat]
    else:
        return 0
    
def processmood(feats):
    moods = {'imper':1, 'imper2':1, 'indic':2}
    for feat in feats:
        if feat in moods:
            return moods[feat]
    return 0
             
def processtense(feats):
    tenses = {'praet':2, 'praes':3, 'fut':1}
    for feat in feats:
        if feat in tenses:
            return tenses[feat]
    return 0
    
def processaspect(feats):
    aspects = {'pf':2, 'ipf':1}
    for feat in feats:
        if feat in aspects:
            return aspects[feat]
    return 0
    
def processvoice(feats):
    voices = {'act':1, 'pass':3, 'med':2} #med seems to mean reflexive
    for feat in feats:
        if feat in voices:
            return voices[feat]
    else:
        return 0
    
def processperson(feats):
    persons = {'1p':1, '2p':2, '3p':3}
    for feat in feats:
        if feat in persons:
            return persons[feat]
    else:
        return 0
    
def processverbform(feats):
    forms = {'inf':2, 'partcp':3, 'ger':1}
    for feat in feats:
        if feat in forms:
            return forms[feat]
    else:
        return 0

main_method()
