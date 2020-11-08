# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
from flask_socketio import SocketIO, send, emit

import re
import xlrd
import joblib

import warnings
import numpy as np
from gensim.models import Word2Vec
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
socketio = SocketIO(app)


@app.route("/")
def index():
    return render_template('index.html')


@socketio.on('message')
# @app.route("/", methods=['GET', 'POST'])
def handle_message(msg):
    print("Message: " + msg)
    data = pd.read_excel('data/data.xlsx', sheet_name='Sheet1')
    label = data.columns = list(map(str, data.columns))

    data_array = data['intencion'].drop_duplicates().to_list()

    try:
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
        data_var = ['v' + str(x) for x in range(len(data))]

        cadena = msg
        cadena = pre_process(cadena)
        data_Test = cadena.split('.')

        # Corpus
        corpus_model = []
        for i in (range(len(data_v))):
            for line in data_v[i]:
                words = [x for x in line.split()]
                corpus_model.append(words)

        corpus_cadena = []
        for i in data_Test:
            for line in data_Test:
                words = [x for x in line.split()]
                corpus_cadena.append(words)

        ####

        def complete(data, corpus):
            rows = len(data)
            columns = 10
            for y in range(rows):
                for x in corpus[y]:
                    if len(corpus[y]) < columns:
                        corpus[y].append('0')
            return corpus

        corpus_train = complete(data_v, corpus_model)
        corpus_test = complete(data_Test, corpus_cadena)

        # Saber posición
        for x in range(len(corpus_train)):
            if (corpus_train[x] == corpus_test[0]):
                # por offset del excel,csv,numbers
                pos = corpus_train.index(corpus_test[0])
                value = corpus_train[pos]
                clase = data_e[pos]

        model = Word2Vec.load('data/word2vec_model')
        vector = model.wv

        #
        for x in range(len(data_v)):
            data_var[x] = np.zeros([1, 1000])
            for k in range(0, 10):
                data_var[x][0, k*100:(k+1)*100] = vector[corpus_train[x]][k]

        X_test = data_var[pos]

        clf_RF = joblib.load('data/modelo_entrenado.pkl')

        prediction = clf_RF.predict(X_test)
        print("Posición_data:", pos, " | Clase: ",
              clase, " | #Clase: ", prediction)
        # print(type(clase))
        data_r = pd.read_excel('data/data.xlsx', sheet_name='Sheet3')

        data_n = data_r[data_r['intencion'] == clase[0]]
        respuesta = data_n['respuesta'].sample()
        # res = respuesta.to_string()

        workbook = xlrd.open_workbook(
            'data/data.xlsx')
        worksheet = workbook.sheet_by_name('Sheet3')
        row_ = int(respuesta.index.values[0])
        res = worksheet.cell(row_+1, 2).value

        concat = res

        ctxt = {'txt': concat}
    except Exception as e:
        print(e)
        ctxt = {'txt': 'Entrada no valida'}

    emit('Hele', ctxt)

    return "Data sent. Please check your program log"


if __name__ == "__main__":
    socketio.run(app)
    # app.run(debug=True)
