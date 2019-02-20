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

changefilename= 'change.conllu'
motionfilename= 'motion.conllu'

translator = Translator()

motioncorpus = parse(open(motionfilename, 'r',encoding ="utf-8").read())
changecorpus = parse(open(changefilename, 'r',encoding ="utf-8").read())

motiontrees = parse_tree(open(motionfilename, 'r',encoding ="utf-8").read())
changetrees = parse_tree(open(changefilename, 'r',encoding ="utf-8").read())

prepositions = ["в","на","за","к","из","с","от"]

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
                


def searchcorpus(manytrees):
    finalresult = {"locindir":[],"dirinloc":[],"locinloc":[],"dirindir":[],"other":[]}
    for tree in manytrees:
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
    
translator = Translator()
motionresult = searchcorpus(motiontrees)
changeresult = searchcorpus(changetrees)
print("For Verbs of Motion:")
printresult(motionresult)
print("For Change of State Verbs:")
printresult(changeresult)

#print(changecorpus[3][4])
#print(translator.translate(changecorpus[3].metadata["text"], src="ru", dest= "en").text)
#searchtree(changetrees[3])

#changetrees[3].print_tree()
        
        