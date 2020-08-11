# coding: utf-8
import os
import sys
import fasttext
from sklearn.metrics import classification_report, accuracy_score 
from sklearn.model_selection import train_test_split

# import local modules
from .utils import load_config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   
sys.path.append(os.path.join(BASE_DIR,'../'))  
from preprocess import Preprocess

class Model():
    def __init__(self, args):
        self.args = args
        self.tp = Preprocess(lang=args['lang'])
    
    def save2csv(self, X, y, path):
        '''Save data to csv format.
        Input:
            X: list of preprocessed text strings (sep by ' ').
            y: list of classification labels (string).
            path: string. File path.
        '''
        prefix = "__label__"
        with open(path, 'w') as f:
            for i in range(len(X)):
                f.write(prefix+str(y[i]))
                f.write(' ')
                f.write(X[i])
                f.write('\n')
        
    def train(self, X_train, y_train, evaluate_test=False):
        '''Train FastText classification model.
        Input:
            X_train: list of preprocessed text strings (sep by ' ')
            y_train: list of classification labels (int/string)
            evaluate_test: boolean. Whether evaluate on test dataset or not.
        Output:
            model: FastText model
        '''
        # Set params
        save_path = self.args['save_path']
        dim = self.args['embed']
        lr = self.args['learning_rate']
        num_epochs = self.args['num_epochs']
        wordNgrams = self.args['wordNgrams']
        loss_function = self.args['loss_function']
        # Preprocess data
        X_train = self.tp.preprocess(X_train)
        # Split train/test
        if evaluate_test: # 8:2
            X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, random_state=self.args['random_seed'], test_size=0.1)
        # Save train data for training
        tmp_dir = "./data"  
        if not os.path.exists(tmp_dir): 
            os.mkdir(tmp_dir)
        train_path = os.path.join(tmp_dir, "fasttext_train.txt")
        self.save2csv(X_train,y_train,train_path)
        # Train fasttext model
        model = fasttext.train_supervised(input=train_path, dim=dim, lr=lr, epoch=num_epochs, wordNgrams=wordNgrams,loss=loss_function)
        # Save model
        if save_path!='':
            model.save_model(save_path)
        # Evaluate
        print('Evaluate on train dataset:')
        self.evaluate(X_train,y_train,model)
        if evaluate_test:
            print('Evaluate on test dataset:')
            test_report, test_acc = self.evaluate(X_test,y_test,model)
            return test_report, test_acc

    def predict(self, X, model_path=''):
        '''Predict.
        Input:
            X: list of preprocessed text strings (sep by ' ')
            model: model path or FastText model object. if type is string, load model.
        Output:
            predpro: float, max predicted probability according to predclass
            predclass: string, predicted class
        '''
        # Load model
        if type(model_path) is str:
            if model_path=='':
                model_path = self.args['save_path']
            model = fasttext.load_model(model_path)
        else:
            model = model_path # model object
        # Preprocess data
        X = self.tp.preprocess(X)
        # Predict
        predpro = []
        predclass = []
        pred_res = model.predict(X)
        predclass = [p[0].replace('__label__','') for p in pred_res[0]]
        return predclass

    def evaluate(self, X, y, model):
        '''Evaluate model. 
        Input: 
            X: list of preprocessed text strings (sep by ' '). 
            y: list of classification labels (int/string). 
            model: FastText model object. 
        Output: 
            report_dict: dict. Classification report. 
            acc: float. Classification accuracy. 
        '''
        y = [str(label) for label in y]
        predclass = self.predict(X, model) 
        print(classification_report(y,predclass,digits=3)) 
        report_dict = classification_report(y,predclass,output_dict=True,digits=3)
        acc = accuracy_score(y,predclass)
        print('Acc: ',acc) 
        return report_dict, acc
     
if __name__ == '__main__':
    # load data  
    import pandas as pd  
    data = pd.read_csv('data/2_categories_data.csv')    
    X_train = data['text'].tolist()   
    y_train = data['label'].tolist()  
    # load config  
    config_path = "./config.ini"  
    model_name = "FastText"  
    args = load_config(config_path, model_name)  
    args['model_name'] = model_name  
    args['save_path'] = "output/%s"%model_name   
    print(args) 
    # train 
    clf = Model(args) 
    clf.train(X_train, y_train, evaluate_test=True) 
    predclass = clf.predict(X_train)   
