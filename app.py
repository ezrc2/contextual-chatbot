from flask import Flask, render_template, request
import torch
import json
import random
import nltk
from dataset import bag_of_words
from model import NeuralNetwork

app = Flask(__name__)


with open('intents.json') as file:
    intents = json.load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_data = torch.load('model.pth')
input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']
all_words = model_data['all_words']
classes = model_data['classes']
model_state = model_data['model_state']

model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

@app.route('/', methods=['GET'])
def main():
    return render_template('main.html')

@app.route('/', methods=['POST'])
def form():
    response = ''
    sentence = request.form['text']

    tokenized = nltk.word_tokenize(sentence)
    X = bag_of_words(tokenized, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, pred = torch.max(output, dim=1)
    tag = classes[pred.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][pred.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                response = random.choice(intent['responses'])
    else:
        response = "Sorry, I do not understand."

    return response


if __name__ == '__main__':
    app.run()