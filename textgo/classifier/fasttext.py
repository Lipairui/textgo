# coding: utf-8

import fasttext
from sklearn.metrics import classification_report, accuracy_score 

class FastText():
    def save2csv(self, X, y, path):
        '''Save data to csv format.
        Input:
            X: list of preprocessed text strings (sep by ' ').
            y: list of classification labels (string).
            path: string. File path.
        '''
        with open(path, 'w') as f:
            for i in range(len(X)):
                f.write(y[i])
                f.write(' ')
                f.write(X[i])
                f.write('\n')
        
    def train(self, X_train, y_train, input_path='fasttext_train.txt', output_path='', dim=100, lr=0.1, epoch=50, wordNgrams=2):
        '''Train FastText classification model.
        Input:
            X_train: list of preprocessed text strings (sep by ' ')
            y_train: list of classification labels (int/string)
            input_path: string, path to save training data (default='fasttext_train.txt')
            output_path: string, path to save the FastText model
            dim: size of word vectors
            lr: learning rate
            epoch: number of epochs 
            wordNgrams: max length of word ngram
        Output:
            model: FastText model
        '''
        y_train = [str(label) for label in y_train]
        y = ['__label__'+label for label in y_train]
        self.save2csv(X_train,y,input_path)
        model = fasttext.train_supervised(input=input_path, dim=dim, lr=lr, epoch=epoch, wordNgrams=wordNgrams)
        if output_path!='':
            model.save_model(output_path)
        self.evaluate(X_train,y_train,model)
        return model

    def predict(self, X, model):
        '''Predict.
        Input:
            X: list of preprocessed text strings (sep by ' ')
            model: model path or FastText model object. if type is string, load model.
        Output:
            predpro: float, max predicted probability according to predclass
            predclass: string, predicted class
        '''
        if type(model) is str:
            model = fasttext.load_model(model)
        predpro = []
        predclass = []
        pred_res = model.predict(X)
        predpro = [p[0] for p in pred_res[1]]
        predclass = [p[0].replace('__label__','') for p in pred_res[0]]
        return predpro, predclass

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
        predpro, predclass = self.predict(X, model) 
        print(classification_report(y,predclass)) 
        report_dict = classification_report(y,predclass,output_dict=True)
        acc = accuracy_score(y,predclass)
        print('Acc: ',acc) 
        return report_dict, acc
        
