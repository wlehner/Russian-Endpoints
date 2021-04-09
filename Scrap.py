#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 00:39:21 2021

@author: walterlehner
"""

from googletrans import Translator

#Including 'From' Prepositions {'в':'acc','на':'acc','за':'acc','к':'dat','из':'gen','с':'gen','от':'gen','под':'acc'}
#foreward = ['в','на','к','под','за','перед','над','у']
#uni_code = ['\u0432','\u043D\u0430','\u0438\u0437','\u0441','\u043E\u0442','\u043a', #в,на,из,с,от,к,под,за,перед,над,у 
#          '\u043f\u043e\u0434','\u0437\u0430','\u043f\u0435\u0440\u0435\u0434','\u043d\u0430\u0434','\u0443']
translator = Translator()

def ru_translate(sentence_ru):
    for i in range(10):
        try:
            return translator.translate(sentence_ru, src="ru", dest= "en").text
        except:
            global trans_errors
            trans_errors += 1
    return ('Translation Failed')