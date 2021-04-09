#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 12:38:07 2020

@author: walterlehner
"""

import matplotlib.pyplot as plt
import numpy as np
import bottleneck as bn
#import torch
import pickle
#import matplotlib as mpl
#mpl.rcParams['agg.path.chunksize'] = 10000

points = pickle.load( open( "pointstoplot46", "rb" ))
#points = pickle.load(open("plotdic9"))
#intpoints = []
#for point in points:
   #intpoints.append(torch.IntTensor.item(point)) 

epoch_num = 60
data_size = 57303 #50967 #57303
ave_len = 1000

averages = bn.move_mean(points, window=ave_len, min_count=1)

#print(points)
plt.plot(points, ',k')
plt.plot(averages, ',r')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.axis([0,len(points),0,6])
epochs = [0,5,10,15,20,25,30,35,40,45,50,55,60]
plt.xticks(np.arange(0,(epoch_num*data_size), step=data_size*5), epochs)
plt.savefig('scatterplot46.png', dpi = 1000)