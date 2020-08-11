# coding: utf-8
import os
import sys
from importlib import import_module

# import local modules
from utils import load_config, build_vocab, Tokenizer


class Classifier():
    def __init__(self, args):
        self.args = args
        self.nn_models = ['TextCNN','TextRNN','TextRCNN','TextRNN_Att']
        if args['model_name'] in self.nn_models:
            module = import_module("nn_classifier")
        else:
            module = import_module(args['model_name'])
        self.model = module.Model(self.args)
        
    def train(self, X_train, y_train, X_dev=None, y_dev=None, evaluate_test=False):
        if self.args['model_name'] == 'FastText': # FastText不需要dev dataset
            if evaluate_test:
                test_report, test_acc = self.model.train(X_train, y_train, evaluate_test=True)
                return test_report, test_acc
            else:
                self.model.train(X_train, y_train, evaluate_test=False)
        else:
            if evaluate_test:
                test_report, test_acc = self.model.train(X_train, y_train, X_dev, y_dev, evaluate_test=True)
                return test_report, test_acc
            else:
                self.model.train(X_train, y_train, X_dev, y_dev, evaluate_test=False)

    def predict(self, X, model_path=''):
        predclass = self.model.predict(X, model_path=model_path)
        return predclass

if __name__ == '__main__':
    # load data
    import pandas as pd
    data = pd.read_csv('data/2_categories_data.csv')  
    X_train = data['text'].tolist() 
    y_train = data['label'].tolist()
    # load config
    config_path = "./config.ini"
    model_name = "Bert"
    args = load_config(config_path, model_name)
    args['model_name'] = model_name
    args['save_path'] = "output/%s"%model_name
    # build vocab if vocab file does not exists 
    # load tokenizer
    #tokenizer = Tokenizer(args['word_level'], args['preprocess'], args['lang'])
    #vocab = build_vocab(X_train, tokenizer, args['vocab_path'], max_vocab_size=args['max_vocab_size'], min_freq=1)
    print(args)
    # train 
    clf = Classifier(args)
    clf.train(X_train, y_train, evaluate_test=True)
    predclass = clf.predict(X_train)    
