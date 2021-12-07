import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.nn.modules.container import Sequential

device = 'cpu' 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam()

class FullyConnectedNN(nn.Module):
    
    def __init__(self, nbits):
        super(FullyConnectedNN, self).__init__()
        
        self.nbits = nbits
        self.linear_relu_stack = nn,Sequential(
                                                nn.Linear(self.nbits,512),
                                                nn.ReLU(),
                                                nn.Linear(512, 512),
                                                nn.ReLU(),
                                                nn.Linear(512, 1),
                                                nn.Sigmoid()
                                              )
        
    def forward(self, x):
        signal = self.linear_relu_stack(x)
        return signal    
    
    
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # 損失誤差を計算
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
if __name__ == '__main__':
    
    