import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

intents_raw = open('intents.json').read()
intents = json.loads(intents_raw)

# Preprocessing all data
words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((words, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmetization
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters] # lemmetize each word in the words list and make it lower case IF it is not included in the ingore letters list
words = sorted(list(set(words))) # Sort all words in its list, reset variable
classes = sorted(list(set(classes))) # Sort all classes in its list, reset variable

# Dump lemmatized words and classes into pickle files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))