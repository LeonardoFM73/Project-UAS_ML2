import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import preparation

vocab_size = 1000
embedding_dim = 16
max_len = 20
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(preparation.training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(preparation.training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)