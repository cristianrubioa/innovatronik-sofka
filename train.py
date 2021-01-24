#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 23:50:50 2019

@author: cristianrubio

"""

import joblib
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import re
import numpy as np
from gensim.models import Word2Vec
#from multiprocessing import Pool
import pandas as pd

# Quitar mierda de warnings
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


data = pd.read_excel('data/data.xlsx', sheet_name='Sheet1')
#data = pd.read_excel('ChatBOT.xlsx', sheet_name='Sheet2')

print('Cantidad de datos: ', len(data))


label = data.columns = list(map(str, data.columns))


def pre_process(text):
    text = str(text).lower()
    text = re.sub(r'\n', r' ', text)
    acentos = {'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
               'Á': 'A', 'E': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'}
    for x in acentos:
        if x in text:
            text = text.replace(x, acentos[x])
    return text


for i in label:
    data[i] = data[i].map(lambda x: pre_process(x))


data_v = data[label[0]].map(lambda x: x.split('.'))
data_e = data[label[1]].map(lambda x: x.split('.'))
#data_var = data[label[2]].map(lambda x: x.split('.'))
####
data_var = ['v' + str(x) for x in range(len(data))]

# Saber cuantas clases existen
data_intent = pd.Series(data_e)
num_intents = data_intent.value_counts()
print('Cantidad de intenciones: ', num_intents.size)
print(num_intents)

corpus = []
for i in (range(len(data_v))):
    for line in data_v[i]:
        words = [x for x in line.split()]
        corpus.append(words)

#text = [ [word1, word2, word3], [word1, word2, word3] ]

# Makia
rows = len(data_v)
columns = 10
for y in range(rows):
    for x in corpus[y]:
        if len(corpus[y]) < columns:
            corpus[y].append('0')

# iter=100,  workers=Pool()._processes
model = Word2Vec(sentences=corpus, size=100, sg=0, window=2, min_count=1)

model.save('data/word2vec_model')

model = Word2Vec.load('data/word2vec_model')

vector = model.wv  # Modelo entrenado, solo para consultas


# Convertir (10,100) a (1,1000) ## lo más makia
for x in range(len(data_v)):
    data_var[x] = np.zeros([1, 1000])
    for k in range(0, 10):
        data_var[x][0, k*100:(k+1)*100] = vector[corpus[x]][k]


# Matriz_Descriptores
X = np.concatenate(data_var, axis=0)

# Target
Y = pd.factorize(data[label[1]])[0]


#### Random Forest ####

# Load scikit's random forest classifier library

# n_estimators=10, max_depth=100, random_state=0,
# n_estimators=100 (en version 0.22 )
clf_RF = RandomForestClassifier(n_estimators=100)
clf_RF.fit(X, Y)

prediction = clf_RF.predict(X)
f1_RF = f1_score(Y, prediction, average=None)
print('Predicción: ', f1_RF)

suma = 0
for i in f1_RF:
    suma = suma + i
    percent = suma*100/len(f1_RF)

print("Porcentaje de predicción: ", percent, "%")

# Guardar Modelo
#from sklearn.externals import joblib#
joblib.dump(clf_RF, 'data/modelo_entrenado.pkl')
