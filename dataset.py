import json
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from torch.utils.data import Dataset

class ChatbotDataset(Dataset):
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(X_train)

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]

    def __len__(self):
        return self.num_samples


def stem(word_list):
    ignore = ['?', '!', '.', ',']
    lemmatizer = WordNetLemmatizer()
    for word in word_list:
        if word not in ignore:
            word = lemmatizer.lemmatize(word.lower())
        else:
            word_list.remove(word)

    return word_list

def bag_of_words(patterns, words):
    tokenized = stem(patterns)
    bag = np.zeros(len(words), dtype='float32')
    for i, word in enumerate(words):
        if word in tokenized:
            bag[i] = 1

    return bag
        
def get_data():
    with open('intents.json') as file:
        intents = json.load(file)

    all_words = []
    classes = []
    documents = []

    for intent in intents['intents']:
        tag = intent['tag']
        if tag not in classes:
            classes.append(tag.lower())

        for pattern in intent['patterns']:
            tokenized = nltk.word_tokenize(pattern.lower())
            documents.append((tokenized, tag))
            for word in tokenized:
                if word not in all_words:
                    all_words.append(word)

    X_train = []
    y_train = []
    for patterns, tag in documents:
        bag = bag_of_words(patterns, all_words)
        X_train.append(bag)
        y_train.append(classes.index(tag))

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    return X_train, y_train, all_words, classes