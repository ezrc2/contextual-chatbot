import json
import nltk
from nltk import tokenize
from nltk.stem.porter import PorterStemmer
import torch

with open('intents.json') as file:
    intents = json.load(file)

words = []
tags = []
documents = []
ignore = ['?', '!', '.', ',']

for intent in intents['intents']:
    tag = intent['tag']
    if tag not in tags:
        tags.append(tag)

    for pattern in intent['patterns']:
        tokenized = nltk.word_tokenize(pattern)
        words.extend(tokenized)
        documents.append((tag, tokenized))
        
for word in words:
    if word not in ignore:
        word = PorterStemmer().stem(word)
    else:
        words.remove(word)

words = sorted(set(words))
print(words)