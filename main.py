import json
import nltk
from nltk import tokenize
from nltk.stem.porter import PorterStemmer
import numpy as np

def stem(word_list):
    ignore = ['?', '!', '.', ',']
    stemmer = PorterStemmer()
    for word in word_list:
        if word not in ignore:
            word = stemmer.stem(word.lower())
        else:
            word_list.remove(word)

    return word_list

def bag_of_words(patterns, words):
    tokenized = stem(patterns)
    bag = np.zeros(len(words))
    for i, word in enumerate(words):
        if word in tokenized:
            bag[i] = 1

    return bag


def main():
    with open('intents.json') as file:
        intents = json.load(file)

    all_words = []
    tags = []
    documents = []

    for intent in intents['intents']:
        tag = intent['tag']
        if tag not in tags:
            tags.append(tag)

        for pattern in intent['patterns']:
            tokenized = nltk.word_tokenize(pattern)
            all_words.extend(tokenized)
            documents.append((tag, tokenized))
            
    all_words = sorted(set(all_words)) # remove duplicates

    X_train = []
    y_train = []
    for tag, patterns in documents:
        bag = bag_of_words(patterns, all_words)
        X_train.append(bag)
        y_train.append(tags.index(tag))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(X_train)

if __name__ == '__main__':
    main()