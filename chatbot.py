import json
import json
import torch
import nltk
from model import NeuralNetwork

with open('intents.json') as file:
    intents = json.load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_data = torch.load("model.pth")
input_size = model_data["input_size"]
hidden_size = model_data["hidden_size"]
output_size = model_data["output_size"]
all_words = model_data['all_words']
tags = model_data['tags']
model_state = model_data["model_state"]


model = NeuralNetwork(input_size, hidden_size, output_size)