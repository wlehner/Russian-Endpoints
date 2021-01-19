#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:17:13 2019

@author: walterlehner
"""
from __future__ import print_function
from conllu import parse
from conllu import parse_tree
import os, gensim, torch, conllu, pickle, random
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from datetime import datetime
from googletrans import Translator

goal = 'obj'#srcobj


#Files
corpus_root= 'sample_ar/TEXTS'
fiction_root= 'sample_ar/TEXTS/Fiction/'
development_set= 'ru_syntagrus-ud-dev.conllu'
testing_set= 'ru_syntagrus-ud-test.conllu'
training_set= 'ru_syntagrus-ud-train.conllu'
output_number = '94'
graphlossout  = 'pointstoplot' + output_number
sentencemap = 'sentmap' + output_number 
testexamples = 'testexam' + output_number
graph_bool = True
test_bool = False 

#Dimensions
input_size = 440 #154
hidden_size = 200  #154
output_size = 45 #Output Size and sentence length need to be seperated
class_num = output_size+1 #number of classes should be outputsize+1
hidden_layers = 1
offset = 100
annotation = 5
trans_errors = 0

##NN Stuff
num_epochs = 100
#batch_size = 32
learning_rate = 0.0001 #Default 1E-3
weight_decay = 0.00001 #Less than lr?
#betas = (0.0, 0.9) #Default 0.9,0.999


filefortraining = training_set
filefordev = development_set
filefortesting = testing_set

#Prepositions
orig  = ['\u0432','\u043D\u0430','\u0437\u0430','\u043A','\u0438\u0437','\u0441','\u043E\u0442'] 
#v, na, za, k, uz, c, ot
fore = ['\u0432','\u043d\u0430','\u043a','\u043f\u043e\u0434','\u0437\u0430',
               '\u043f\u0435\u0440\u0435\u0434','\u043d\u0430\u0434','\u0443']
#v, na, k, pod, za, pered, nad, y
allprep = ['\u0432','\u043D\u0430','\u0438\u0437','\u0441','\u043E\u0442','\u043a',
           '\u043f\u043e\u0434','\u0437\u0430','\u043f\u0435\u0440\u0435\u0434','\u043d\u0430\u0434','\u0443']
#v, na,  uz, c, ot, k, pod, za, pered, nad, y 
prepositions = allprep
prepstring = 'All'

#Other
model = Word2Vec.load("word2vec.model")
word_vectors = model.wv
translator = Translator()
log_freq = 100000
shuffle = True
test_index = []

picklenet = 'tnet_' +str(num_epochs) +'hl' +str(hidden_size) +'x' +str(hidden_layers)  +goal + output_number + '.pkl'

def main_method():
    main_training()
    #loadtesting()

def main_training():
    print('TRAINING:Started at:', datetime.now(), ' Goal:', goal,' Prepositions:', prepstring, ' Learning Rate:', learning_rate, ' Epochs:', num_epochs, 
          'Hidden Layer Size:', hidden_size, 'Number of Hidden Layers:', hidden_layers, 'Weight Decay:', weight_decay, 
          'Training with:', filefortraining, 'and' , filefordev, 'Data ID:', output_number)
    dev_set = processconllu(filefordev)
    train_set = processconllu(filefortraining)
    test_set, test_map  = processconllu_save(filefortesting)
    train_set = train_set + dev_set
    #netload('torchnet.pkl')
    train(train_set)
    print("Training Set Length", len(train_set))
    #print('Annotated Test on', filefortesting)
    #annotatedtest(test_set, test_map, annotation, offset)
    print('Testing on', filefortraining)
    test(train_set)
    print('Testing on', filefortesting)
    test(test_set)
    print('Completed at:', datetime.now())

def loadtesting():
    print('Testing on a Loaded Network. Started at: ', datetime.now(), 'Testing on', filefortesting, 'Data ID:', output_number)
    netload('torchnet.pkl')
    test_set, test_map= processconllu_save(filefortesting)
    dev_set = processconllu(filefordev)
    train_set = processconllu(filefortraining)
    train_set = train_set + dev_set
    print('Annotation Offset:', offset)
    annotatedtest(test_set, test_map, annotation, offset)
    print('Testing on', filefortesting)
    test(test_set)
    print('Tested on Training Data:', filefortraining, 'and', filefordev)
    test(train_set) 

def divide_set(set):
    tenth_size = round(len(set)/10)
    small_set = []
    big_set = []
    for i, (question1, answer1) in enumerate(set):
        if i < tenth_size:
            small_set.append((question1, answer1))
        else:
            big_set.append((question1, answer1))
        i += 1
    return small_set, big_set

def ru_translate(sentence_ru):
    for i in range(10):
        try:
            return translator.translate(sentence_ru, src="ru", dest= "en").text
        except:
            global trans_errors
            trans_errors += 1
    return ('Translation Failed')

def find_sent(sentid, corpus):
    for sentence in corpus:
        if sentence.metadata['sent_id']==sentid:
            return sentence
    return False

def totensor(list):
    return torch.tensor(list)

#class model
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
net = Net(inputs= input_size, hiddens= hidden_size, hidden_layers2= hidden_layers, outputs= output_size) #
net = net.float()

#criterion = nn.SmoothL1Loss() 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr= learning_rate, weight_decay= weight_decay)

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

def processconllu_save(file):
    corpus = parse(open(file, 'r',encoding ="utf-8").read())
    examples = []
    map_dic  = []
    for sentence in corpus:
        if len(sentence)< output_size:
            sentprep = []
            for word in sentence:
                if word['lemma'] in prepositions:
                    sentprep.append(word['id'])
            if sentprep:
                sentexamples, preps = processconlsent_save(sentence, sentprep)
                if sentexamples:
                    examples += sentexamples
                    for prep in preps:
                        map_dic.append((sentence.metadata['sent_id'], prep))
    if test_bool:
        pickle.dump(examples, open(testexamples, "wb"))
        pickle.dump(map_dic, open(sentencemap, "wb"))
    return examples, map_dic

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

def searchtree_obj(tree, preposition):
    if tree.children:
        for child in tree.children:
            if child.token['id'] == preposition:
                return (tree)
            x = searchtree_obj(child, preposition)
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

def makeanswer_2(tree, preposition):
    if goal == 'obj':
        node = searchtree_obj(tree, preposition)
    if goal == 'src':
        node = searchtree(tree, preposition)
    if node:
        return node.token['id']
    else:
        return False
    
def makequestion_1(sentence, preposition):
    question = []
    for word in sentence:
        if word['lemma'] in word_vectors:
            question.append(word_vectors[word['lemma']][0])
        else:
            question.append(0)
        featlist = processword(word)
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
        question = makequestion_1(sentence, preposition)  #THIS IS WHERE THE SWITCH BETWEEN STYLES IS
        answer = makeanswer_2(sentence.to_tree(), preposition)
        if question and answer:
            examples.append((totensor(question), totensor(answer)))
    return examples

def processconlsent_save(sentence, preplist):
    examples = []
    prep_fin = []
    for preposition in preplist:
        question = makequestion_1(sentence, preposition) 
        answer = makeanswer_2(sentence.to_tree(), preposition)
        if question and answer:
            examples.append((totensor(question), totensor(answer)))
            prep_fin.append(preposition)
    return examples, prep_fin

def processword(word):
# POS, NUM, Person, Case, verbform, aspect, tense, animacy, voice, mood, 
    return [processpartsos(word), processnum(word), processperson(word), processcase(word), processverbform(word), 
                 processaspect(word), processtense(word), processanimacy(word), processvoice(word), processmood(word),]

def processpartsos(word):
    partsos = {'VERB':1, 'AUX':1, 'NOUN':2, 'PROPN':2, 'ADJ':4, 'ADV':5, 'CCONJ':6, 'SCONJ':6, 'DET':7, 
               'INTJ':8, 'NUM':9, 'PART':10, 'PRON':3, 'ADP':11}
    if word['upostag'] != None and word['upostag']  in partsos:
        return partsos[word['upostag']]
    else:
        return 12

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
    count_x = 1
    losscount_list = []
    for epoch in range(num_epochs):
        losstotal = 0
        if shuffle == True:
            random.shuffle(examplelist)
        for i, (question, answer) in enumerate(examplelist):
            optimizer.zero_grad()
            output = net(question.float())
            output.unsqueeze_(0)
            y = answer.unsqueeze(0)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losscount_list.append(torch.IntTensor.item(loss))
            losstotal += loss
            if (i+1) % log_freq == 0:                              # Logging
                print('Epoch [%d/%d], Step [%d/%d], Avg Loss: %.4f' %(epoch+1, num_epochs, i+1, len(examplelist), losstotal/log_freq))
                losstotal = 0
    torch.save(net.state_dict(), picklenet)
    if graph_bool:
        pickle.dump(losscount_list, open(graphlossout, "wb"))#Loss Data is Saved

def test(testlist):
    correct = 0
    total = 0
    for question, answer in testlist:
        output = net(question.float())
        pred = output.max(0)[1] #Check out
        total += 1
        if torch.eq(pred, answer):
            correct += 1
    print('Accuracy of the network: ', (100 * correct / total), '%')
    

#Dev Tests    
def annotatedtest(testlist, testmap,  limit, off_set):
    corpus = parse(open(filefortesting, 'r',encoding ="utf-8").read())
    testlist = testlist[(0+off_set):(limit+off_set)]
    correct = 0
    total = 0
    acc = False
    #print("Annotated Test:")
    for i, (question, answer) in enumerate(testlist):
        print('Example:', (i+off_set))
        output = net(question.float())
        pred = output.max(0)[1]
        poss = np.argpartition(output.detach().numpy(), -4)[-4:]
        pred_num = output
        total += 1
        acc = torch.eq(pred, answer)
        if acc:
            correct += 1
        sentence_id, preposition = testmap[i+off_set]
        sentence = find_sent(sentence_id, corpus)
        firstq = question[0].item()
        firsts = word_vectors[sentence[0]['lemma']][0]
        if not np.isclose(firstq, firsts, rtol= 1e-05, atol=1e-08, equal_nan=False):
            print('Error: Different first words.',firstq, 'versus', firsts)
        ru_sent = sentence.metadata['text']
        ru_answer = ''
        ru_pred = ''
        ru_prep = ''
        out_index = pred-1
        prep_index = preposition-1
        answer_index = answer-1
        print('Russian: ', ru_sent)
        print('English: ', ru_translate(ru_sent))
        if (prep_index < len(sentence)):
            ru_prep = sentence[prep_index]['form']
            print('Preposition: ', preposition, '-', ru_prep, '-', ru_translate(ru_prep))
        else:
            print('Preposition out of bounds: ', preposition)
        if (answer_index < len(sentence)):
            ru_answer= sentence[answer_index]['form']
            print('Answer: ', ru_answer, ': ', ru_translate(ru_answer))
        else:
            print('Answer: Index outside sentence. Interesting....')
        if (out_index < len(sentence)):
            ru_pred = sentence[out_index]['form']
            print('Output: ', ru_pred,': ', ru_translate(ru_pred), ' Result: ', acc.item())
        else:
            print('Output: Index outside of sentence', 'Result: ', acc.item())
        print()
        for x in poss:
            poss_index = x-1
            if (poss_index < len(sentence)):
                ru_pred = sentence[poss_index]['form']
                print('Possible: ', ru_pred,': ', ru_translate(ru_pred), ' Result: ', acc.item())
            else:
                print('Possible: Index outside of sentence', 'Result: ', acc.item())
            print('Rated at:', output[x])

def findprep(question):
    for i in range(0, input_size, 11):
        if question([(i + 1)] == 11 and question[(i + 2)] == 1):
            return i
    
        
def annotatedtrain(examplelist, limit):
    print('Annotated Training')
    examplelist = examplelist[:limit]
    for epoch in range(num_epochs):
        for i, (question, answer) in enumerate(examplelist):
            optimizer.zero_grad()
            output = net(question.float())
            pred = output.max(0)[1]
            output.unsqueeze_(0)
            y = answer.unsqueeze(0)
            #loss = CrossEntropyLoss_2(output, answer)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if (i+1) % 1 == 0:
                print("Predicted Value:", pred.item(), " Answer:", answer, " Loss:", loss.item())
        
        
def CrossEntropyLoss_1(outputs, labels):
  batch_size = outputs.size()[0]            # batch_size
  outputs = f.log_softmax(outputs, dim=1)   # compute the log of softmax values
  outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
  return -torch.sum(outputs)#/num_examples

def CrossEntropyLoss_2(x, y):
    log_prob = -1.0 * f.log_softmax(x, 1)
    loss = log_prob.gather(1, y)
    loss = loss.mean()
    return loss
        
#https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/3
#https://mlpipes.com/adding-a-dimension-to-a-tensor-in-pytorch/
#multi-target not supported at /Users/soumith/b101_2/2019_02_08/wheel_build_dirs/wheel_3.6/pytorch/aten/src/THNN/generic/ClassNLLCriterion.c:21
#https://www.programcreek.com/python/example/107644/torch.nn.CrossEntropyLoss
#https://github.com/asappresearch/sru/blob/master/classification/train_classifier.py
#Formating Help: https://github.com/htfy96/future-price-predictor/blob/master/model/cnnBeta.py
            
main_method()


#List of Problems
#1- Punctuation
#2- Line up all features

#Loss Functions
#https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7

#Examples
#https://towardsdatascience.com/a-simple-starter-guide-to-build-a-neural-network-3c2cf07b8d7c
#https://pytorch.org/tutorials/beginner/nlp/deep_learning_tutorial.html#sphx-glr-beginner-nlp-deep-learning-tutorial-py
