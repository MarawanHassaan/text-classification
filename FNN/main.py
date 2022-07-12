import numpy as np
import pickle
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
from tqdm import tqdm_notebook as tqdm
from model import *

def to_tensor(X):
    #X is a sparse matrix, we first neet to transofrm it in the coo format
    #and then we can get the corresponding tensor
    X = X.tocoo()

    #contains all values inside X
    values = X.data

    #contains all indices with data of X
    indices = np.vstack((X.row, X.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = X.shape

    X = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return X

def accuracy(net, X_test, y_test):
    with torch.no_grad():
      net.eval()
      X_test, y_test = X_test.to(device), y_test.to(device)

      y_pred = net(X_test)
      correct = (y_pred.max(dim=1)[1] == y_test)
      return torch.mean(correct.float()).item()
    
    
#Loading data
X = pickle.load(open("data/vectorizer_balanced_4.sav", "rb"))
labels = pickle.load(open("data/labels_balanced_4.sav", "rb"))

X_train, X_test, y_train, y_test = train_test_split(X, labels, train_size=0.7)

#Transform X_train and y_train into tensors
X_train = to_tensor(X_train)
y_train = torch.from_numpy(np.array(y_train))

#Transofrm X_test and y_test into tensors
X_test = to_tensor(X_test)
y_test = torch.from_numpy(np.array(y_test))

#Check and get current device
use_cuda = torch.cuda.is_available()
print("CUDA available: " + str(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")

#Initialize network
net = Net(X.shape[1])
net.to(device)

loss = nn.CrossEntropyLoss()
opt = optim.Adam(params=net.parameters(), lr=0.001)

def train_step(x, y):
  net.train()

  y_pred = net(x)
  loss_epoch = loss(y_pred, y)
  loss_epoch.backward()
  
  opt.step()
  opt.zero_grad()

#Training
n_epochs = 100

X_train, y_train = X_train.to(device), y_train.to(device)

for epoch in range(n_epochs):
  net.train()
  train_step(X_train, y_train)


#Evaluating
#accuracy on the training set
training_acc = accuracy(net, X_train, y_train)

#accuracy on the test set
test_acc = accuracy(net, X_test, y_test)

print("Training accuracy: ", training_acc)
print("Test accuracy: ", test_acc)
