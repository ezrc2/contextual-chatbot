import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ChatbotData(Dataset):
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(X_train)

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]

    def __len__(self):
        return self.num_samples


class NeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = nn.ReLU(x)
        x = self.lin2(x)
        x = nn.ReLU(x)
        x = self.lin3(x)
        return x

    def train(self):
        pass