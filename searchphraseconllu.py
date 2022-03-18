#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:46:17 2018

@author: walterlehner
"""
import nltk
from conllu import parse
from conllu import parse_tree
from googletrans import Translator

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

development_set= 'ru_syntagrus-ud-dev.conllu'
testing_set= 'ru_syntagrus-ud-test.conllu'
training_set= 'ru_syntagrus-ud-train.conllu'

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ""
translator = Translator()

#motioncorpus = parse(open(motionfilename, 'r',encoding ="utf-8").read())
#changecorpus = parse(open(changefilename, 'r',encoding ="utf-8").read())

#motiontrees = parse_tree(open(motionfilename, 'r',encoding ="utf-8").read())
#changetrees = parse_tree(open(changefilename, 'r',encoding ="utf-8").read())

prepositions = ["в","на","за","к","из","с","от"]

def searchdfile(file):
    changesentlist = []
    motionsentlist = []
    corpus = parse(open(file, 'r',encoding ="utf-8").read())
    for sentence in corpus:
        c = False
        m = False
        for word in sentence:
            if word['lemma'] in verbs_of_change:
                c = True
            if word['lemma'] in verbs_of_motion:
                m = True
        if m:
            motionsentlist.append(sentence)
        if c:
            motionsentlist.append(sentence)
    return(motionsentlist, changesentlist)

def gettext(sentence_ru):
    return sentence_ru.metadata["text"]

def ru_translate(sentence_ru): #Requires translation be enabled
    return translator.translate(sentence_ru.metadata["text"], src="ru", dest= "en").text

def testprep(node):
    if node.token["lemma"] in prepositions:
        return node.token["lemma"]

def testcase(node, string):
    if node.token["feats"]["Case"]==string:
        return True

def testphrase(prepstring, casestring):
    if prepstring == ("в" or "на"):
        if casestring =="Loc":
            return "loc"
        elif casestring == "Acc":
            return "dir"
    elif prepstring == "за":
        if casestring == "Acc":
            return "dir"
        elif casestring == "Ins":
            return "loc"
    elif prepstring == ("к" or "из"):
        return "dir"
    elif prepstring == ("с" or "от"):
        if casestring == "Gen":
            return "dir"
        else:
#            print("Preposition: ", prepstring, "Case: ", casestring)
            return "other"
    return "other"
            
    
def testupos(node, string):
    if node.token["upostag"]==string:
        return True
    else:
        return False

    
def searchtree(tree, typelist):
#    print("Current Node: ", tree.token["lemma"])
    if tree.children:
        if testupos(tree, "NOUN"):
            first = False
            second = False
            for child in tree.children:
                if testprep(child):
                    first = True #Found First PP
                    prep1 = child.token["lemma"]
                    case1 = tree.token["feats"]["Case"]
                if testupos(child, "NOUN"):
                    for grandchild in child.children:
                        if testprep(grandchild):
                            second = True 
                            prep2 = grandchild.token["lemma"]
                            case2 = child.token["feats"]["Case"]
                if first and second:
                    firstphrase = testphrase(prep1, case1)
                    secondphrase = testphrase(prep2, case2)
                    if (firstphrase == "other" or secondphrase == "other"):
                        typelist.append("other")
                    if (firstphrase == "loc" and secondphrase == "dir"):
                        typelist.append("dirinloc")
                    if (firstphrase == "dir" and secondphrase == "loc"):
                        typelist.append("locindir")
                    if (firstphrase == "loc" and secondphrase == "loc"):
                        typelist.append("locinloc")
                    if (firstphrase == "dir" and secondphrase == "dir"):
                        typelist.append("dirindir")
                searchtree(child,typelist)
        else:
            for childe in tree.children:
                searchtree(childe,typelist)
    return typelist
                


def searchlist(sentlist):
    finalresult = {"locindir":[],"dirinloc":[],"locinloc":[],"dirindir":[],"other":[]}
    for sent in sentlist:
        tree = sent.to_tree()
        result = searchtree(tree,[])
        if result:
            if "locindir" in result:
                finalresult["locindir"].append(tree)
            elif "dirinloc" in result:
                finalresult["dirinloc"].append(tree)
            elif "locinloc" in result:
                finalresult["locinloc"].append(tree)
            elif "dirindir" in result:
                finalresult["dirindir"].append(tree)
            elif "other" in result:
                finalresult["other"].append(tree)
    return finalresult
            
def printresult(resultdict):
    print("Directional in Locational: ", len(resultdict["dirinloc"]))
    print("Locational in Directional: ", len(resultdict["locindir"]))
    print("Locational in Locational: ", len(resultdict["locinloc"]))
    print("Directional in Directional: ", len(resultdict["dirindir"]))
    print("Other situations: ", len(resultdict["other"]))
    
def printtranslate(sentencelist): #Requires translation be enabled
    for sentence in sentencelist:
        print(gettext(sentence))
        print(ru_translate(sentence) + "\n")
    

#Need to finish printout
def printout(): #Requires translation be enabled
    print("\nFor Change of State Verbs:")
    printresult(changeresult)
    print("\nLocational in Directional")
    printtranslate(changeresult["locindir"])
    print("\nDirectional in Locational")
    printtranslate(changeresult["dirinloc"])
    print("\nDirectional in Directional")
    printtranslate(changeresult["dirindir"])
