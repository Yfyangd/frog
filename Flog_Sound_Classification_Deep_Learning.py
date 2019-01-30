
# coding: utf-8

# In[1]:


import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Import Data
import IPython.display as ipd
fname = './wave/Butler_1.wav'
ipd.Audio(fname)

import librosa
import librosa.display
y, sr = librosa.load(fname)
plt.figure()
librosa.display.waveplot(y, sr=sr)
plt.title('Butler frog waveform')


file_list = []
label_list = []
for i in os.listdir('./wave2'):
    file_list.append(i)
    label_list.append(i.split('_')[0])

from pandas.core.frame import DataFrame
train= pd.DataFrame({'fname':file_list})

# Data Preprocess
from pydub import AudioSegment
file = 'American_bull_1.wav'
path = './wave2/'

import wave

def get_length(file):
    audio = wave.open(path+file)
    return audio.getnframes() / audio.getframerate()
get_length(file)

from joblib import Parallel, delayed
with Parallel(n_jobs=10, verbose=1) as ex:
    lengths = ex(delayed(get_length)(e) for e in train.fname)

train = train.query('length <= 10').reset_index(drop=True)

# Feature Extraction
melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.power_to_db(melspec) 
plt.figure(figsize=(10, 4))
librosa.display.specshow(logmelspec,y_axis='mel', x_axis='time')
plt.colorbar()
plt.title('Mel spectrogram')
plt.tight_layout()

melspec = librosa.feature.melspectrogram(y, sr, n_mels=128,fmax=8000)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(melspec,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram - max')
plt.tight_layout()

# Visualize the MFCC series
mfccs = librosa.feature.mfcc(y, sr, n_fft=1024, hop_length=512, n_mels=128)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC of Butler rice frog')
plt.tight_layout()

def obtain_mfcc(file, features=40):
    y, sr = librosa.load(path+file, res_type='kaiser_fast')
    return librosa.feature.mfcc(y, sr, n_mfcc=features)

def get_mfcc(file, n_mfcc=40, padding=None):
    y, sr = librosa.load(path+file, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=n_mfcc)
    if padding: mfcc = np.pad(mfcc, ((0, 0), (0, max(0, padding-mfcc.shape[1]))), 'constant')
    return mfcc.astype(np.float32)


mfcc = get_mfcc(file, padding=250)
print(mfcc.shape)
plt.figure(figsize=(12,5))
plt.imshow(mfcc, cmap='hot');

from functools import partial

n_mfcc = 40
padding = 431
fun = partial(get_mfcc, n_mfcc=n_mfcc, padding=padding)

with Parallel(n_jobs=10, verbose=1) as ex:
    mfcc_data = ex(delayed(partial(fun))(e) for e in train.fname)
    
# Juntamos la data en un solo array y agregamos una dimension
mfcc_data = np.stack(mfcc_data)[..., None]
mfcc_data.shape

# Data Preprocess for CNN Model

label_list = []
for i in train['fname']:
    label_list.append((i.split('_')[0]+i.split('_')[1]).split('.')[0]
                      .replace('1', '').replace('2', '').replace('3', '')
                      .replace('11', '').replace('12', '').replace('13', '').replace('14', '').replace('15', '')
                      .replace('21', '').replace('22', '').replace('23', '').replace('24', '').replace('25', '')
                      .replace('31', '').replace('32', '').replace('33', '').replace('34', '').replace('35', '')
                      .replace('4', '').replace('5', '')
                     )

train['label'] = label_list

lbl2idx = {lbl:idx for idx,lbl in enumerate(train.label.unique())}
idx2lbl = {idx:lbl for lbl,idx in lbl2idx.items()}
n_categories = len(lbl2idx)

train['y'] = train.label.map(lbl2idx)
y_train = train['y'].values
y_train = y_train.reshape(504,1)

from keras.utils import np_utils
y_label_OneHot = np_utils.to_categorical(y_train)

mfcc_data_scaled = (mfcc_data-np.min(mfcc_data))/(np.max(mfcc_data)-np.min(mfcc_data))

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Dropout, Input, GlobalAvgPool2D, GlobalMaxPool2D, concatenate, Flatten
from keras.optimizers import Adam, SGD
import keras.backend as K

bs = 50
lr = 0.003

m_in = Input([n_mfcc, padding, 1])
x = BatchNormalization()(m_in)

layers = [10, 20, 50, 100]
for i,l in enumerate(layers):
    strides = 1 if i == 0 else (2,2)
    x = Conv2D(l, 3, strides=strides, activation='relu', padding='same',
               use_bias=False, kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.02)(x)

x_avg = GlobalAvgPool2D()(x)
x_max = GlobalMaxPool2D()(x)

x = concatenate([x_avg, x_max])
x = Dense(1000, activation='relu', use_bias=False, kernel_initializer='he_uniform')(x)
x = Dropout(0.2)(x)
m_out = Dense(n_categories, activation='softmax')(x)

model1 = Model(m_in, m_out)
model1.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
model1.summary()

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(mfcc_data_scaled, y_label_OneHot, test_size=0.2, random_state=42)


log1 = model1.fit(mfcc_data, y_label_OneHot, bs, 15, validation_split=0.25)

# Backend Training
K.eval(model1.optimizer.lr.assign(lr/10))
log2 = model1.fit(x_train, y_train, batch_size=100, epochs=10, validation_data=[x_val, y_val])

def show_results(*logs):
    trn_loss, val_loss, trn_acc, val_acc = [], [], [], []
    
    for log in logs:
        trn_loss += log.history['loss']
        val_loss += log.history['val_loss']
        trn_acc += log.history['acc']
        val_acc += log.history['val_acc']
    
    fig, axes = plt.subplots(1, 2, figsize=(14,4))
    ax1, ax2 = axes
    ax1.plot(trn_loss, label='train')
    ax1.plot(val_loss, label='validation')
    ax1.set_xlabel('epoch'); ax1.set_ylabel('loss')
    ax2.plot(trn_acc, label='train')
    ax2.plot(val_acc, label='validation')
    ax2.set_xlabel('epoch'); ax2.set_ylabel('accuracy')
    for ax,title in zip(axes, ['Train', 'Accuracy']):
        ax.set_title(title, size=14)
        ax.legend()
        
show_results(log1, log2)

