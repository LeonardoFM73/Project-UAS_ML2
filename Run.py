import json 
import numpy as np
from datetime import datetime

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow import keras
import tensorflow as tf
tf.get_logger().setLevel('INFO')

import colorama 
colorama.init()
from colorama import Fore, Style

import pickle

import warnings

warnings.filterwarnings("ignore")

with open("intents.json") as file:
    data = json.load(file)


def chat():
    # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "keluar":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , np.random.choice(i['responses']))
                if i['tag'] == "Jadwal":
                    print(Fore.GREEN + "ChatBot: https://rumahsakit.unair.ac.id/website/jadwal-praktek-dokter-spesialis-poliklinik-vip-pasien-umum-2/")
                elif i['tag'] == 'Pendaftaran':
                    print(Fore.GREEN + "ChatBot: http://apps.rumahsakit.unair.ac.id/perdana/")


os.system('cls')
print(Fore.YELLOW + "Mulai mengetik dengan chatbot(ketik keluar untuk berhenti)!" + Style.RESET_ALL)
chat()