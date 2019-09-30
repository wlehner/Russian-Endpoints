#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:17:13 2019

@author: walterlehner
"""
from __future__ import print_function
from conllu import parse
from conllu import parse_tree
import os, gensim, torch, conllu, pickle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import torch.nn as nn
from googletrans import Translator

#Files
corpus_root= 'sample_ar/TEXTS'
fiction_root= 'sample_ar/TEXTS/Fiction/'
devfile= 'ru_syntagrus-ud-dev.conllu' #EMPTY???
testfile= 'ru_syntagrus-ud-test.conllu'
trainfile= 'ru_syntagrus-ud-train.conllu'

#Dimensions
input_size = 154 #154
hidden_size = 154 #154
output_size = 40 #Output Size and sentence length need to be seperated

##NN Stuff
num_epochs = 1
batch_size = 100
learning_rate = 0.001

#Other
prepositions = ["в","на","за","к","из","с","от"]
model = Word2Vec.load("word2vec.model")
word_vectors = model.wv
translator = Translator()



def main_method():
    training_set = processconllu(trainfile)
#    test_set = processconllu(testfile)
#    netload("torchnet.pkl")
    annotatedtrain(training_set, 10)
#    devtrain(training_set, test_set, 'Test')


def ru_translate(sentence_ru):
    return translator.translate(sentence_ru.metadata["text"], src="ru", dest= "en").text

def totensor(list):
#    return torch.from_numpy(np.array(list))
    return torch.tensor(list)

#class model
class Net(nn.Module):
    def __init__(self, inputs, hiddens, outputs):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inputs, hiddens)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hiddens, hiddens)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hiddens, outputs)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
    
#Initialize NN    
net = Net(inputs= input_size, hiddens= hidden_size, outputs= output_size)
net = net.float()

#criterion = nn.SmoothL1Loss()
criterion = nn.CrossEntropyLoss()
# Produces error: Dimension out of range (expected to be in range of [-1, 0], but got 1)
#https://visdap.blogspot.com/2018/12/pytorch-inputs-for-nncrossentropyloss.html
optimizer = torch.optim.Adam(net.parameters(), lr= learning_rate)

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
                    examples += sentexamples
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
    
def makeanswer_1(tree, preposition):
    node = searchtree(tree, preposition)
    if node:
        return node.token['id']
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
    question = question[:input_size]
    return question


           
def processconlsent(sentence, preplist):
    examples = []
    for preposition in preplist:
        question = makequestion(sentence, preposition)
        answer = makeanswer(sentence.to_tree(), preposition) #Define's Ansswer Shape
        if question and answer:
#            print('Question: ', question)
#            print('Answer: ', answer, '\n')
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

def netload(filename):
    net.load_state_dict(torch.load(filename))
    net.eval()
    
def train(examplelist):
    for epoch in range(num_epochs):
        for i, (question, answer) in enumerate(examplelist):
            optimizer.zero_grad()
            output = net(question.float())
            loss = criterion(output, answer.float())
            loss.backward()
            optimizer.step()
#            print('Successful Step')
            if (i+1) % 5000 == 0:                              # Logging
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(examplelist), loss))
    torch.save(net.state_dict(), 'torchnet.pkl')

def test(testlist, listname):
    correct = 0
    total = 0
    for question, answer in testlist:
        output = net(question.float())
        total += 1
        if torch.eq(output, answer.float()).all():
            correct += 1
    print('Accuracy of the network on the', listname, ' dataset: ', (100 * correct / total), '%')
    
def devtrain(trainlist, testlist, testname):
    train(trainlist)
    test(testlist, testname)

#Dev Tests    
def annotatedtest(testlist, listname, limit):
    testlist = testlist[:limit]
    print("Testing Network's Accuracy in the", listname, " dataset:")
    for question, answer in testlist:
        output = net(question.long())
        if torch.eq(output, answer.long()).all():
            print("Test as True")
        else:
            print("Test as False")
        print("Offical answer: ", answer)
        print("Network's answer: ", output)
        
def annotatedtrain(examplelist, limit):
    examplelist = examplelist[:limit]
    for question, answer in examplelist:
        optimizer.zero_grad()
        output = net(question.float())
        output.unsqueeze_(40)
        print("Output Length: ", output.size())
        print("Output: ", output)
        #answer.unsqueeze_(0)
        #answer = torch.max(answer, 1)[1]
        print("Answer Length: ", answer.size())
        print("Answer: ", answer)
        loss = criterion(output, answer)
        print("Successful Loss: ", loss)
        
#https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/3
#https://mlpipes.com/adding-a-dimension-to-a-tensor-in-pytorch/
#multi-target not supported at /Users/soumith/b101_2/2019_02_08/wheel_build_dirs/wheel_3.6/pytorch/aten/src/THNN/generic/ClassNLLCriterion.c:21
#https://www.programcreek.com/python/example/107644/torch.nn.CrossEntropyLoss
#https://github.com/asappresearch/sru/blob/master/classification/train_classifier.py
#Formating Help: https://github.com/htfy96/future-price-predictor/blob/master/model/cnnBeta.py
            
            


#devtrain(training_set, test_set, 'Test')
#test(training_set, 'Training')


def teststuff(file):
    corpus = parse(open(file, 'r',encoding ="utf-8").read())
#    corpustree = parse_tree(open(file, 'r',encoding ="utf-8").read())
    sentence = corpus[12]
#    sentencetree = corpustree[12]
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
            
main_method()


#List of Problems
#1- Punctuation
#2- Line up all features

#Loss Functions
#https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7

#Examples
#https://towardsdatascience.com/a-simple-starter-guide-to-build-a-neural-network-3c2cf07b8d7c
#https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#sphx-glr-beginner-nlp-deep-learning-tutorial-py
