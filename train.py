import json
import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import ChatbotData
from model import NeuralNetwork


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
    bag = np.zeros(len(words), dtype='float32')
    for i, word in enumerate(words):
        if word in tokenized:
            bag[i] = 1

    return bag

def train(dataloader, model, loss_fn, optimizer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        # forward
        pred = model(X)
        loss = loss_fn(pred, y)

        # back propogation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print (f'Loss: {loss.item():.4f}')


def main():
    with open('intents.json') as file:
        intents = json.load(file)

    all_words = []
    classes = []
    documents = []

    for intent in intents['intents']:
        tag = intent['tag']
        if tag not in classes:
            classes.append(tag)

        for pattern in intent['patterns']:
            tokenized = nltk.word_tokenize(pattern)
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

    batch_size = 32
    epochs = 1000

    train_dataloader = DataLoader(dataset=ChatbotData(X_train, y_train), batch_size=batch_size)
    model = NeuralNetwork(input_size=len(all_words), hidden_size=8, num_classes=len(classes))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for i in range(epochs):
        print(f"Epoch {i+1}\t", end=" ")
        train(train_dataloader, model, loss_fn, optimizer)
 

if __name__ == '__main__':
    main()