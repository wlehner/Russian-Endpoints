#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:17:13 2019

@author: walterlehner
"""
from __future__ import print_function
from conllu import parse
from conllu import parse_tree
import os, gensim, torch, conllu
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import torch.nn as nn
from googletrans import Translator

#Files
corpus_root= 'sample_ar/TEXTS'
fiction_root= 'sample_ar/TEXTS/Fiction/'
file1= 'ru_syntagrus-ud-dev.conllu' #EMPTY???
file2= 'ru_syntagrus-ud-test.conllu'
file3= 'ru_syntagrus-ud-train.conllu'

#Dimensions
input_size = 154
hidden_size = 154
output_size = 35

##NN Stuff
num_epochs = 2
batch_size = 100
learning_rate = 0.001

#Other
prepositions = ["в","на","за","к","из","с","от"]
model = Word2Vec.load("word2vec.model")
word_vectors = model.wv
translator = Translator()

def ru_translate(sentence_ru):
    return translator.translate(sentence_ru.metadata["text"], src="ru", dest= "en").text

def totensor(list):
    return torch.from_numpy(np.array(list))

#class model
class Net(nn.Module):
    def __init__(self, inputs, hiddens, outputs):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inputs, hiddens)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hiddens, outputs)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    

#Takes Conllu Format and Produces a list of examples
def processconllu(file):
    corpus = parse(open(file, 'r',encoding ="utf-8").read())
    examples = []
    for sentence in corpus:
        if len(sentence)< output_size:
            sentprep = []
            for word in sentence:
                if word['lemma'] in prepositions:
                    sentprep.append(word['id'])
            if sentprep:
                sentexamples = processconlsent(sentence, sentprep)
                if sentexamples:
                    examples.append(sentexamples)
    return examples

def searchtree(tree, preposition):
    if tree.children:
        for child in tree.children:
#            print(child.token['lemma'])
            if child.children:
                for grandchild in child.children:
                    if grandchild.token['id'] == preposition:
                        return (tree)
                x = searchtree(child, preposition)
                if x:
                    return x
                
def makeanswer(tree, preposition):
    node = searchtree(tree, preposition)
    if node:
        answer = [0]*output_size
        answer[node.token['id']]= 1
        return answer
    else:
        return False

def makequestion(sentence, preposition):
    question = []
    for word in sentence:
        if word['lemma'] in word_vectors:
            question.append(word_vectors[word['lemma']][0])
        else:
            question.append(0)
        featlist = processpos(word)
        if word['id'] == preposition:
            featlist[1] = 1
        if (len(question)+ len(featlist))< input_size:          
            for feature in featlist:
                question.append(feature)
        else:
            return False
    question.extend([0]*input_size)
    question[:input_size]
    return question
              
def processconlsent(sentence, preplist):
    examples = []
    for preposition in preplist:
        question = makequestion(sentence, preposition)
        answer = makeanswer(sentence.to_tree(), preposition)
        if question and answer:
            examples.append((totensor(question), totensor(answer)))
    return examples

#FEATURES: (lemma), POS, number, person, verbform, aspect, tense, voice, mood, case, gender, animacy,
def processpos(word):
#    print(word['lemma'])
    feats = []
    if word['upostag']== ('VERB' or 'AUX'): #Aux doesn't have animacy, should be okay
        feats = [1, processnum(word), processperson(word), processverbform(word), processaspect(word), 
                 processtense(word), processvoice(word), processmood(word), processcase(word), 
                 processgender(word), processanimacy(word)]
    elif word['upostag']== ('NOUN' or 'PROPN'):
        feats = [2, processnum(word), processcase(word), processgender(word), processanimacy(word)]
    elif word['upostag']== 'ADJ':
        feats = [3, processnum(word), processcase(word), processgender(word), processanimacy(word)]
    elif word['upostag']== 'ADV':
        feats = [4]
    elif word['upostag']== ('CCONJ' or 'SCONJ'):
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
    elif word['upostag']== 'ADP':
        feats = [12, 0]
    else: # 'PUNCT' 'SYM' 'X'
        feats = [13]

    return feats

def processnum(word):
    number = {'Sing':1, 'Plur':2}
    if word['feats'] != None and 'Number' in word['feats'] and word['feats']['Number'] in number:
        return number[word['feats']['Number']]
    else:
        return 0

def processcase(word):
    case = {'Acc':1, 'Dat':2, 'Gen':3, 'Ins':4, 'Loc':5, 'Nom':6, 'Par':7, 'Voc':8}
    if word['feats'] != None and 'Case' in word['feats'] and word['feats']['Case'] in case:
        return case[word['feats']['Case']]
    else:
        return 0

def processgender(word):
    gender = {'Fem':1, 'Masc':2, 'Neut':3}
    if word['feats'] != None and 'Gender' in word['feats'] and word['feats']['Gender'] in gender:
        return gender[word['feats']['Gender']]
    else:
        return 0
    
def processtense(word):
    tense = {'Fut':1, 'Past':2, 'Pres':3}
    if word['feats'] != None and 'Tense' in word['feats'] and word['feats']['Tense'] in tense:
        return tense[word['feats']['Tense']]
    else:
        return 0
    
def processvoice(word):
    voice = {'Act':1, 'Mid':2, 'Pass':3}
    if word['feats'] != None and 'Voice' in word['feats'] and word['feats']['Voice'] in voice:
        return voice[word['feats']['Voice']]
    else:
        return 0

def processaspect(word):
    aspect = {'Imp':1, 'Perf':2}
    if word['feats'] != None and 'Aspect' in word['feats'] and word['feats']['Aspect'] in aspect:
        return aspect[word['feats']['Aspect']]
    else:
        return 0

def processmood(word):
    mood = {'Imp':1, 'Ind':2}
    if word['feats'] != None and 'Mood' in word['feats'] and word['feats']['Mood'] in mood:
        return mood[word['feats']['Mood']]
    else:
        return 0

def processperson(word):
    person = {1:1, 2:2, 3:3}
    if word['feats'] != None and 'Person' in word['feats'] and word['feats']['Person'] in person:
        return person[word['feats']['Person']]
    else:
        return 0
    
def processverbform(word):
    verbform = {'Conv':2,'Fin':0, 'Inf':2, 'Part':3} #Fin is normal
    if word['feats'] != None and 'Verbform' in word['feats'] and word['feats']['VerbForm'] in verbform:
        return verbform[word['feats']['VerbForm']]
    else:
        return 0

def processanimacy(word):
    animacy = {'Anim':1, 'Inan':2}
    if word['feats'] != None and 'Animacy' in word['feats'] and word['feats']['Animacy'] in animacy:
        return animacy[word['feats']['Animacy']]
    else:
        return 0
    
#NN    
net = Net(inputs= input_size, hiddens= hidden_size, outputs= output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr= learning_rate)
    
def train(examplelist):
    for epoch in range(num_epochs):
        for i, (question, answer) in examplelist:
            optimizer.zero_grad()
            output = net(question)
            loss = criterion(output, answer)
            loss.backward()
            optimizer.step()
#            if (i+1) % 100 == 0:                              # Logging
#                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
            
examples = processconllu(file2)
train(examples)


def teststuff(file):
    corpus = parse(open(file, 'r',encoding ="utf-8").read())
    corpustree = parse_tree(open(file, 'r',encoding ="utf-8").read())
    sentence = corpus[12]
    sentencetree = corpustree[12]
#    sentencetree.print_tree()
#    mod = searchtree(sentencetree,8)
#    print(ru_translate(sentence))
#    print('Modified Word: ', mod.token['lemma'], ' - ', mod.token['id'])
#    print('Question: ', makequestion(sentence,8))
#    print('Answer: ', makeanswer(sentencetree,8))
#    print(sentence[3])
#    print(sentence[1]['lemma'], ': ',processpos(sentence[1]))
    for word in sentence:
        print (word)
        if 'feats' in word:
            print ('yes')
        else:
            print('no')
    
#teststuff(file1)


#List of Problems
#1- Punctuation
#2- Line up all features

#vector = word_vectors["word"]
#wv = KeyedVectors.load("model.wv", mmap='r')
#Construct examples
#   Go through the sentences in Connllu
#       Use vocabulary to make list of words
#       Append a second dimension of grammatical tags
#       Produce appropriate 