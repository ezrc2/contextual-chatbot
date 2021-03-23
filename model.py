import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import get_data
from dataset import ChatbotDataset


class NeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.layers(x)


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

        print(f'Loss: {loss.item():.4f}')


X_train, y_train, all_words, classes = get_data()
input_size = len(all_words)
hidden_size = 10
output_size = len(classes)

batch_size = 32
epochs = 1000
dataset = ChatbotDataset(X_train, y_train)

train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(epochs):
    print(f"Epoch {i+1}\t", end=" ")
    train(train_dataloader, model, loss_fn, optimizer)

model_data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "classes": classes,
    "all_words": all_words
}

torch.save(model_data, "model.pth")