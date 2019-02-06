#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:46:17 2018

@author: walterlehner
"""

import nltk
#from io import open
#import chardet
#from nltk.corpus.reader.conll import ConllCorpusReader
#from nltk.corpus import universal_treebanks as ut
#import pyconll
#import conllu
from conllu import parse
from conllu import parse_tree
from googletrans import Translator

changefilename= 'change.conllu'
motionfilename= 'motion.conllu'

motioncorpus = parse(open(motionfilename, 'r',encoding ="utf-8").read())
changecorpus = parse(open(changefilename, 'r',encoding ="utf-8").read())

motiontrees = parse_tree(open(motionfilename, 'r',encoding ="utf-8").read())
changetrees = parse_tree(open(changefilename, 'r',encoding ="utf-8").read())

def testprep(node):
    if(node.token["lemma"]==("в" or "на")):
        return True

def testloc(node):
    if node.token["feats"]["Case"]=="Loc":
        return True
    
def searchtree(tree):
    print("Current Node: ", tree.token["lemma"])
    testprep(tree)
    if tree.children:
        for child in tree.children:
            if testprep(child):
                if testloc(tree):
                    print("Locational Object")
                elif testdir(tree):
                    print("Directional Object")
            searchtree(child)

translator = Translator()
print(translator.translate(changecorpus[3].metadata["text"], src="ru", dest= "en").text)
searchtree(changetrees[3])

#changetrees[3].print_tree()
        
        