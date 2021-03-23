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
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.layers(x)