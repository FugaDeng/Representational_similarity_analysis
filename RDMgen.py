# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:16:22 2020
modified 2021/04/15
@author: ld178
"""

# %%
import os
os.getcwd()
os.chdir('D:\Research_local\SchemRep\RSAmodel')
#import urllib3
import numpy as np
from PIL import Image
from cv2 import resize
from vgg16_hybrid_places_1365 import VGG16_Hybrid_1365
from tqdm import tqdm
from scipy.stats import spearmanr
from scipy import io
from keras import Model
import pandas as pd
#import wget
import matplotlib.pyplot as plt
# %%
model = VGG16_Hybrid_1365(weights='places')

# %%
tblStim=pd.read_csv(r"D:\Research_local\SchemRep\taskscripts\PTBtasks\fullStimList.csv")
tblStim.head()
# %%
filelist=tblStim['ObjectFile'].to_list()

imagePath=r'D:\Research_local\SchemRep\taskscripts\PTBtasks\updatedObjectsResampled'+chr(92)

# %%
padImagePath=r"D:\Research_local\SchemRep\taskscripts\PTBtasks\updatedObjectsResampled"+chr(92)

# %% get activations in layers (early, mid, late convLayers; last FC layer) 

tmpdir= padImagePath

ilayer=1
print(ilayer)
layer=model.layers[ilayer]
layer = layer.output
layer_model = Model(model.input, layer)

# this only applies to convolutional layers
num_chan=layer_model.output_shape[-1]
num_pix=layer_model.output_shape[-2]*layer_model.output_shape[-3]
num_img=len(filelist)

layerRDM=np.zeros((num_img,num_img,num_chan))

for i in tqdm(range(0, num_chan)):
    tmpmtx=np.zeros((num_pix,num_img))
    for j in range(0,num_img):
        image = Image.open( tmpdir + filelist[j] ) 
        image = np.array(image, dtype=np.uint8)
        image = resize(image, (224, 224))
        image = np.expand_dims(image, 0)
        activation1 = layer_model.predict(image)
        tmpmtx[:,j]=activation1[0,:,:,i].flatten()
    layerRDM[:,:,i],_ =spearmanr(tmpmtx)


savedir=r'D:\Research_local\SchemRep\RSAmodel\modelRDMs'+chr(92)
RDMvar={'R':layerRDM}
io.savemat(savedir+'layer'+str(ilayer)+'_RDM_padded.mat',RDMvar)
# %% -------------------------------------------------------------------------
# %% -------------------------------------------------------------------------
# %% -------------------------------------------------------------------------
# below is for creating semantic RDM
# %% get the word embedding file and create a embedding dictionary

#!wget http://nlp.stanford.edu/data/glove.6B.zip
#!unzip -q glove.6B.zip

path_to_glove_file = r"D:\Research_local\SchemRep\RSAmodel\glove.6B.300d.txt"

i=0
embeddings_index = {}

f=open(path_to_glove_file,encoding='utf8')
for line in f:
    i+=1
    word, coefs = line.split(maxsplit=1)
    coefs = np.fromstring(coefs, "f", sep=" ")
    embeddings_index[word] = coefs
f.close()

print("Found %s word vectors." % len(embeddings_index))

# %% calculate RDM for objects
# from scipy.stats import pearsonr
stim_word_emb=np.zeros((300,tblStim.shape[0]))

for i in tqdm(range(0,tblStim.shape[0])):
    # fix some words that have no exact match of keys in the embedding dictionary
    stimname = tblStim['Object'][i].lower()
    if stimname=='haircomb':
        stimname='hair comb'
    if stimname=='soccerball':
        stimname='soccer ball'
    
    stimname = stimname.split()
    
    for j in range(0,len(stimname)):
        try:
            stim_word_emb[:,i]+=embeddings_index[stimname[j]]
        except:
            print(stimname[j])
        

gloveRDM=np.corrcoef(stim_word_emb.T)
RDMvar_sem={'Rsem':gloveRDM}
io.savemat(savedir+'GLOVEsemantic_RDM.mat',RDMvar_sem)        
# %% calculate RDM for scenes, Congruent
# 
stim_scene_con_emb=np.zeros((300,tblStim.shape[0]))

for i in tqdm(range(0,tblStim.shape[0])):
    # fix some words that have no exact match of keys in the embedding dictionary
    stimname = tblStim['SceneCongruent'][i].lower()
    if stimname=='mcdonald\'s':
        stimname='mcdonalds'
    stimname = stimname.split()
    
    for j in range(0,len(stimname)):
        try:
            stim_scene_con_emb[:,i]+=embeddings_index[stimname[j]]
        except:
            print(stimname[j])

gloveRDM=np.corrcoef(stim_scene_con_emb.T)
RDMvar_scene_con={'Rsem_scene_con':gloveRDM}
io.savemat(savedir+'GLOVEsemantic_RDM_scene_con.mat',RDMvar_scene_con)
# %%
# %% calculate RDM for scenes, Neutral
# 
stim_scene_neu_emb=np.zeros((300,tblStim.shape[0]))

for i in tqdm(range(0,tblStim.shape[0])):
    # fix some words that have no exact match of keys in the embedding dictionary
    stimname = tblStim['SceneNeutral'][i].lower()
    if stimname=='mcdonald\'s':
        stimname='mcdonalds'
    stimname = stimname.split()
    
    for j in range(0,len(stimname)):
        try:
            stim_scene_neu_emb[:,i]+=embeddings_index[stimname[j]]
        except:
            print(stimname[j])

gloveRDM=np.corrcoef(stim_scene_neu_emb.T)
RDMvar_scene_neu={'Rsem_scene_neu':gloveRDM}
io.savemat(savedir+'GLOVEsemantic_RDM_scene_neu.mat',RDMvar_scene_neu)            
# %% calculate RDM for scenes, Incongruent
# 
stim_scene_inc_emb=np.zeros((300,tblStim.shape[0]))

for i in tqdm(range(0,tblStim.shape[0])):
    # fix some words that have no exact match of keys in the embedding dictionary
    stimname = tblStim['SceneIncongruent'][i].lower()
    if stimname=='mcdonald\'s':
        stimname='mcdonalds'
    stimname = stimname.split()
    
    for j in range(0,len(stimname)):
        try:
            stim_scene_inc_emb[:,i]+=embeddings_index[stimname[j]]
        except:
            print(stimname[j])

gloveRDM=np.corrcoef(stim_scene_inc_emb.T)
RDMvar_scene_inc={'Rsem_scene_inc':gloveRDM}
io.savemat(savedir+'GLOVEsemantic_RDM_scene_inc.mat',RDMvar_scene_inc)                  
# %% sanity check: is embedding similarity related to congruency?
os_embedding_sim=np.zeros((114,3))

for i in range(114):
    os_embedding_sim[i,2]=np.corrcoef(x=stim_word_emb[:,i],y=stim_scene_con_emb[:,i])[0,1]
    os_embedding_sim[i,1]=np.corrcoef(x=stim_word_emb[:,i],y=stim_scene_neu_emb[:,i])[0,1]
    os_embedding_sim[i,0]=np.corrcoef(x=stim_word_emb[:,i],y=stim_scene_inc_emb[:,i])[0,1]

plt.errorbar(x=[1,2,3],
             y=os_embedding_sim.mean(axis=0),
             yerr=os_embedding_sim.std(axis=0)/np.sqrt(114))
plt.xticks([1,2,3],labels=('INC','NEU','CON'))
plt.ylabel('Object-Scene embedding similarity')


# %%

