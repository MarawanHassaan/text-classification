import torch
from torch import nn


#FNN
class Net(nn.Module):
    def __init__(self, input_size): 
        super(Net, self).__init__()
        
        self.hidden = nn.Linear(input_size, 100)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

        self.out = nn.Linear(100, 4)

    #forward step
    def forward(self, x):
        x = self.relu(self.hidden(x))
        return self.out(self.drop(x))
