# coding: utf-8
import os
import sys
from importlib import import_module
from sklearn.model_selection import train_test_split
import torch

# import local modules
from .utils import load_config, set_random_seed, get_device, load_vocab, build_vocab, get_data_iterator, Tokenizer, get_pretrained_embeddings

from .train_eval import train, evaluate, predict, init_network
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
sys.path.append(os.path.join(BASE_DIR,'../')) 
BASE_MODULE = "textgo.classifier"
from preprocess import Preprocess

class Model():
    def __init__(self, args):
        self.args = args
        self.device = get_device()
        set_random_seed(self.args['random_seed'])
        self.args['device'] = self.device
        self.tokenizer = Tokenizer(args['word_level'], args['preprocess'], args['lang'])
        self.vocab = load_vocab(self.args['vocab_path'])
        self.args['n_vocab'] = len(self.vocab)
        if self.args['embedding']=='random':
            self.args['embedding_pretrained'] = None
        else:
            self.args['embedding_pretrained'] = torch.tensor(get_pretrained_embeddings(self.args['embedding'], self.vocab, method='word2vec'))
            self.args['embed'] = self.args['embedding_pretrained'].size(1) 
        module = import_module(BASE_MODULE+'.'+args['model_name'])
        self.model = module.Model(self.args).to(self.device)
        
    def train(self, X_train, y_train, X_dev=None, y_dev=None, evaluate_test=False):
        if X_dev is None or y_dev is None: # train: dev = 8:2
            X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, random_state=self.args['random_seed'], test_size=0.2)
        if evaluate_test: # train: dev: test = 8:1:1
            X_dev, X_test, y_dev, y_test = train_test_split(X_dev, y_dev, random_state=self.args['random_seed'], test_size=0.5)
        train_iter = get_data_iterator(self.args, self.vocab, X_train, y_train)
        dev_iter = get_data_iterator(self.args, self.vocab, X_dev, y_dev)
        init_network(self.model, seed=self.args['random_seed'])
        print(self.model.parameters)
        train(self.args, self.model, train_iter, dev_iter)
        print("Evaluate on dev dataset:")
        evaluate(self.args, self.model, dev_iter)
        if evaluate_test:
            print("Evaluate on test dataset:")
            test_iter = get_data_iterator(self.args, self.vocab, X_test, y_test)
            test_report, test_acc, test_loss = evaluate(self.args, self.model, test_iter)
            return test_report, test_acc

    def predict(self, X, model_path='', model=None):
        if model is None:
            if model_path == '':
                model_path = self.args['save_path']
            model = self.load_model(model_path)
        data_iter = get_data_iterator(self.args, self.vocab, X)
        predclass = predict(self.args, model, data_iter)
        return predclass

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval() 
        return self.model
 
if __name__ == '__main__':
    # load data
    import pandas as pd
    data = pd.read_csv('data/2_categories_data.csv')  
    X_train = data['text'].tolist() 
    y_train = data['label'].tolist()
    # load config
    config_path = "./config.ini"
    model_name = "TextRNN_Att"
    args = load_config(config_path, model_name)
    args['model_name'] = model_name
    args['save_path'] = "output/%s.bin"%model_name
    # build vocab if vocab file does not exists 
    # load tokenizer
    #tokenizer = Tokenizer(args['word_level'], args['preprocess'], args['lang'])
    #vocab = build_vocab(X_train, tokenizer, args['vocab_path'], max_vocab_size=args['max_vocab_size'], min_freq=1)
    print(args)
    # train 
    clf = Model(args)
    clf.train(X_train, y_train, evaluate_test=True)
    predclass = clf.predict(X_train)    
