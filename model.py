import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Chatbot(Dataset):
    
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.num_samples = len(X_train)

    def __getitem__(self, index):
        return self.X_train[index], self.y_train[index]

    def __len__(self):
        return self.num_samples