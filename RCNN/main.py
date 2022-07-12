from utils import *
from model import *
from config import Config
import numpy as np
import sys
import torch.optim as optim
from torch import nn
import torch

train_file = '../data/df_4.pkl'
    
w2v_file = '../data/glove.840B.300d.txt'

config = Config()
dataset = Dataset(config)
dataset.load_data(w2v_file, train_file)

# Create Model with specified optimizer and loss function
model = RCNN(config, len(dataset.vocab), dataset.word_embeddings)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)

opt = optim.SGD(model.parameters(), lr = 0.5)
loss = nn.NLLLoss()
model.add_optimizer(opt)
model.add_loss_op(loss)


# Training
train_losses = []
val_accuracies = []

model.train()
for i in range(config.max_epochs):
    print ("Epoch: {}".format(i))
    train_loss, val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
    train_losses.append(train_loss)
    val_accuracies.append(val_accuracy)

# Evaluate
#model.eval()
train_acc = evaluate_model(model, dataset.train_iterator)
val_acc = evaluate_model(model, dataset.val_iterator)

print ('Final Training Accuracy: {:.4f}'.format(train_acc))
print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
