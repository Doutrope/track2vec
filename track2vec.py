#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 6 11:37:12 2018

@author: sacha
"""

#========================================
# TRACK2VEC GENERATES TRACKS EMBEDDINGS
# BY LEARNING FROM TRACKS CO-OCCURENCES
#              IN PLAYLISTS.
# At the end, we have top n recos for 
# each tracks and metadata for each track
#========================================

import gensim
import numpy as np
import time
import datetime
import pandas as pd
import gc
import argparse

from itertools import product
from utils import generate_output, get_metric, get_playlists
from sklearn.model_selection import train_test_split

# general vars
parser = argparse.ArgumentParser()
parser.add_argument('-c','--country',type=str, help='playlist country', default='fr')
parser.add_argument('-p','--proportion', type=float, help='proportion of test users', default=0.1)
parser.add_argument('-t','--train',type=int, help='number of playlists (train+test) for training phase', default=5000)

args = parser.parse_args()
country = args.country
test_prop = args.proportion
nplaylists_train = args.train

nbRecos = 10 # for evaluation
nrecos_out = 30 # for output

print("#########################")
print("TRAINING TRACK2VEC MODEL")
print("country: " + country)

#================
# DATA PROCESSING

print("data processing..")
playlists, vocab, previews = get_playlists(country)
playlists = [x for x in playlists if len(x)>=3]

# 2. Split train test with light playlist size for speed
train, test = train_test_split(playlists[:nplaylists_train], test_size = test_prop)

# remove all dataset for cross valid 
del(playlists)
gc.collect()

print("nb playlists train : ", str(len(train)))
print("nb playlists test : ", str(len(test)))
print("mean playlists size : ", str(np.mean([len(plt) for plt in test])))

# get unique tracks in test
test_vocab = set([trk for plst in test for trk in plst])

#======================================
# MODEL LEARNING AND HYPER-PARAM TUNING

# 1. Hyper-Param Tuning
scores = pd.DataFrame(columns = ["country","time_minutes","model_vocab","nbRecos","window",
                              "size","negative","sample","metric"])

# TODO: add power by changing gensim files
sizes = [50,100,300]
negatives = [30,50,100]
windows = [5,30,50,100]
samples = [0.00001]

print("train with cross-valid...")
for negative,window,size,sample in product(negatives,windows,sizes,samples):
    
    start = time.time()
    model = gensim.models.Word2Vec(train,sg=1,negative=negative,size=size,
                                   window=window,sample=sample,min_count=3,workers=8,hs=0,iter=10)
    
    model.train(train,total_examples=len(train),epochs=model.iter)
    
    # 2: evaluate with recall-like metric
    metric = get_metric(test, model, nbRecos)

    end = time.time()
    elapsed = round((end-start)/60)
    
    row = {"country":country,
           "time_minutes":elapsed,
           "model_vocab":int(len(set(model.wv.vocab.keys()))),
           "nbRecos":int(nbRecos),
           "window":int(window),
           "size":int(size),
           "negative":int(negative),
           "sample":float(sample),
           "metric":round(metric,5)}
    
    # save scoresult
    now = datetime.datetime.now()
    date = str(now.day) + '_' + str(now.month) + '_' + str(now.year)
    
    scores = scores.append(row,ignore_index=True)
    scores.to_csv("models/result"+ date + ".csv",index = False)

del(model)
del(train)
gc.collect()

print("")
print("Recompute with all data and best params...")
best_params = scores[scores["metric"] == max(scores["metric"])]
playlists, vocab, previews = get_playlists(country)
playlists = [x for x in playlists if len(x)>=3]
model = gensim.models.Word2Vec(playlists,
                               size=int(best_params["size"]),
                               negative=int(best_params["negative"]),
                               sample=float(best_params["sample"]),
                               window=int(best_params["window"]),
                               workers=8,
                               iter=int(best_params["iter"]))
start = time.time()
model.train(playlists,total_examples=len(playlists),epochs=model.iter)
end = time.time()
best_model_vocab = set(model.wv.vocab.keys())

#==============
# SAVE OUTPUT

recos, metadata = generate_output(model,previews,nrecos_out)
metadata.to_csv('models/flow/metadata_vocab.csv',index=False)
recos.to_csv('models/flow/ordered_recos.csv',index=True)
model.save('models/flow/model')
