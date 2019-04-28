#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:09:32 2018

@author: sacha
"""

import shutil
import json
import os
from multiprocessing import Pool
import numpy as np 

def filter_user(uid):
    '''Filter one user folder'''
    
    try:
        user = json.load(open(os.path.join('data','user',uid,'file.json')))
        user = {'id':user['id'],
                'country':user['country'],
                'tracklist':user['tracklist']}
    except:
        shutil.rmtree(os.path.join('data','user',uid))
        return
    # if user playlists in folder, remove to free space
    if os.path.exists(os.path.join('data','user',uid,'playlists')):
        shutil.rmtree(os.path.join('data','user',uid,'playlists'))

def filter_usersdata_parallel():
    '''filter deezer users folder, to compute once'''
        
    # filter all folds //
    users = os.listdir('data/user')
    pool = Pool()
    pool.map(filter_user,users)    
    
def generate_output(model,previews,nb_recos):
    import pandas as pd
    
    '''
        Generate 2 outputs,
        1 df with the ordered recos for each track.
        1 df with the tracks metadata
    '''
    
    model_vocab = set(model.wv.vocab.keys())
    trk_recos = {}
    trk_infos = {}
    i=0
    for track in model_vocab:
    
        # We don't want the same artist in the recos than in the source track
        artist = track.split("#_#")[0]
        title = track.split("#_#")[1].replace(",", " ")
        sims = model.wv.most_similar(track,topn = 2 * nb_recos)
        trk_recos[track] = [sim[0] for sim in sims if artist not in sim[0]][0:nb_recos]
        trk_infos[i] = {'id':track, 'artist':artist, 'track':title, 'preview':previews[track]}        
        i=i+1
    
    ordered_recos = pd.DataFrame.from_dict(trk_recos,orient='index')
    ordered_recos.columns = ['reco' + str(x) for x in range(1,nb_recos+1)]
    
    metadata = pd.DataFrame.from_dict(trk_infos,orient='index')
    metadata.columns = ["id","artist","track","preview"]
    
    return ordered_recos, metadata

def get_plst_score(vocab, playlist, model, model_vocab, k):
    ''' get reco score for 1 playlist. 
        for each track in plst get top k recos, increment score of 1/plstLen 
        for each reco in playlist'''
        
    playlistLen = len(playlist)
    if playlistLen != 0:
        
        score = 0 
        
        # filter tracks that are not present in our model vocabulary from test playlists
        gen = [track for track in playlist if track in model_vocab]
        
        for track in gen:
            most_similars = model.wv.similar_by_vector(track,k)
            most_similars = [x[0] for x in most_similars]
            overlap = len([trk for trk in most_similars if trk in playlist])
            score += overlap/playlistLen
            
        return score

def get_metric(test, model, nRecos):
    '''Computes recall@k metric on test in //'''
    
    pool = Pool()    
    model_vocab = set(model.wv.vocab.keys())
    input_metric = [(model_vocab,x,model,model_vocab,nRecos) for x in test]
    
    scores = pool.starmap(get_plst_score,input_metric)
    
    return np.mean(scores)


def get_playlist(playlistpath,users):
    '''returns playlists, vocabulary and previews. 
        Filter only matching users'''
    try:
        playlist = json.load(open(playlistpath))
        ntracks = len(playlist["tracks"]["data"])
        vocab = set()
        prev = {}
        customPlaylist = []
    except:
        pass
    
    for i in range(ntracks):
            
        if playlist["tracks"]["data"][i]["readable"]:
            try:
                artistName = playlist["tracks"]["data"][i]["artist"]["name"]            
                trackName = playlist["tracks"]["data"][i]["title"]
                customId = str(artistName) + "#_#" + trackName
                vocab.add(customId)
                customPlaylist.append(customId)
                prev[customId] = playlist["tracks"]["data"][i]["preview"]
            except:
                pass
        else:
            try:
                artistName = playlist["tracks"]["data"][i]["alternative"]["artist"]["name"]
                trackName = playlist["tracks"]["data"][i]["alternative"]["title"]
                customId = str(artistName) + "#_#" + trackName
                vocab.add(customId)
                customPlaylist.append(customId)
                prev[customId] = playlist["tracks"]["data"][i]["alternative"]["preview"]
            except:
                pass
    return customPlaylist, vocab, prev
    

def load_user(userid,country):
    ''' '''
    user = json.load(open('data/user/' + userid + '/file.json'))
    if user['country'].lower() == country:
        return {user['id']:user['country']}

def get_playlists(country):
    '''Returns a list with the playlists with custom ids 
    for a specific chosen country. Also gives dict mapping and vocab'''
    
    pool = Pool()
    
    # get users chosen country
    users = [(userid,country) for userid in os.listdir('data/user')]
    usersCtry = pool.starmap(load_user,users)
    users = {}
    for us in usersCtry:
        if type(us)==dict:
            users.update(us)
    
    # get playlists vocab and previews
    playlists_users = [(os.path.join('data','playlist',idd,'file.json'),users) for idd in os.listdir('data/playlist')]
    plsts_vocab_prevs = pool.starmap(get_playlist,playlists_users)
    
    plsts = [x[0] for x in plsts_vocab_prevs]
    vocab = set.union(*[x[1] for x in plsts_vocab_prevs])
    previews = {}
    for prv in plsts_vocab_prevs:
        if type(prv[2])==dict:
            previews.update(prv[2])
            
    return plsts, vocab, previews
