import spacy
import torch
import pandas as pd
import numpy as np
from torchtext import data
from torchtext.vocab import Vectors
from sklearn.metrics import accuracy_score

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.train_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}
        
    def load_data(self, w2v_file, train_file):
        NLP = spacy.load('en')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]

        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=self.config.max_sen_len)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("text", TEXT),("label", LABEL)]

        # Load data from pd.DataFrame into torchtext.data.Dataset
        train_df = pd.read_pickle(train_file)
        train_examples = [data.Example.fromlist(i, datafields) for i in train_df.values.tolist()]
        train_data = data.Dataset(train_examples, datafields)

        train_data, val_data = train_data.split(split_ratio=0.8)

        TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab
        
        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size = self.config.batch_size,
            sort_key = lambda x: len(x.text),
            repeat = False,
            shuffle = True)
            
        self.val_iterator = data.BucketIterator(
            (val_data),
            batch_size = self.config.batch_size,
            sort_key = lambda x: len(x.text),
            repeat = False,
            shuffle = False)
        
        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} validation examples".format(len(val_data)))

def evaluate_model(model, iterator):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    all_preds = []
    all_y = []

    for idx, batch in enumerate(iterator):
        x = batch.text.to(device)
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1] + 1
        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
        
    score = accuracy_score(all_y, np.array(all_preds).flatten())
    return score

