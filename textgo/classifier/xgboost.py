# coding: utf-8

import numpy as np
import xgboost as xgb 
from sklearn.metrics import classification_report, accuracy_score 
 
import logging  
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"  
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)  
logger = logging.getLogger(__file__)


class XGBoost():
    
    def train(self, X_train, y_train, output_path='', params=None, num_rounds=100, feature_names=[]):
        '''Train xgboost model.
        Input: 
            X_train: numpy array of shape [n_samples, n_features], float
            y_train: numpy array or python list of shape [n_samples], int/float
            output_path: string, path to save the XGBoost model
            params: dict, parameters of XGBoost
            num_rounds: int, number of epochs
            num_class: int, number of classes
            feature_names: list of strings of shape [n_features], provided for feature importance
        Output:
            model: XGBoost model
        '''
        if type(y_train)==list:
            y_train = np.array(y_train).astype(float)
        X_train = X_train.astype(float)
        dtrain = xgb.DMatrix(X_train,y_train)
        if params is None:
            params = { 
            'objective': "binary:logistic",  
            #'objective': "multi:softmax", 
            #'num_class': 5, 
            'eval_metric': "logloss", #'merror' 
            #'eval_metric': "merror", 
            'max_depth': 5, 
            'min_child_weight': 1, 
            'gamma': 0, 
            'subsample': 0.8, 
            'colsample_bytree': 0.8, 
            'scale_pos_weight':1, 
            'verbose':False, 
            'silent':1 
            } 
            num_class = len(set(list(y_train)))
            if num_class>2:
                params['objective'] = "multi:softmax"
                params['num_class'] = num_class
                params['eval_metric'] = 'merror'

        # Train 
        logger.info('Cross validation...') 
        cv_res = xgb.cv(params,dtrain,num_boost_round=num_rounds,early_stopping_rounds=10,nfold=3, metrics=params['eval_metric'],show_stdv=True) 
        print(cv_res) 
        logger.info('Train...') 
        model = xgb.train(params,dtrain,num_boost_round=cv_res.shape[0]) 
        logger.info('Best iteration: %i',model.best_ntree_limit) 
 
        # Save model 
        if output_path!='':
            model.save_model(output_path) 
            logger.info('Model saved to: %s'%output_path) 
 
        logger.info('Train evaluation:') 
        self.evaluate(X_train,y_train,model)

        if len(feature_names)==X_train.shape[1]:
            # Feature importance 
            logger.info('Feature importance:')
            feature_names = X_train.columns.tolist() 
            feature_imps = self.get_xgb_imp(model,feature_names) 
            print(feature_imps)

        return model

    def predict(self, X, model):
        '''Predict.
        Input:
            X: numpy array of shape [n_samples, n_features].
            model: string (model path) or model object. If string, load model.
        Output:
            predpro: list of predicted probability
            predclass: list of predicted class
        '''
        if type(model) is str:
            xgb_model = xgb.Booster({'nthread':4})
            xgb_model.load_model(model)
        else:
            xgb_model = model
        dX = xgb.DMatrix(X.astype(float))
        predpro = xgb_model.predict(dX)
        if max(predpro)>1: # multi class
            predclass = predpro
        else: # 2 class
            predclass = [1 if p>=0.5 else 0 for p in predpro]
        return predpro, predclass

    def evaluate(self, X, y, model):
        '''Evaluate model.
        Input:
            X: numpy array of shape [n_samples, n_features].
            y: numpy array or python list of shape [n_samples]
            model: XGBoost model object.
        Output:
            report_dict: dict. Classification report.
            acc: float. Classification accuracy.
        '''
        predpro, predclass = self.predict(X, model)
        print(classification_report(y,predclass)) 
        report_dict = classification_report(y,predclass,output_dict=True)
        acc = accuracy_score(y,predclass)
        print('Acc: ',acc)
        return report_dict, acc

    def get_xgb_imp(self, model, feat_names): 
        '''Get feature importance.
        Input:
            model: XGBoost model object.
            feat_names: list of strings. Names of each feature. 
        Output:
            feats_imp: DataFrame.
        '''
        imp_vals = model.get_score() 
        feats_imp = pd.DataFrame(imp_vals,index=np.arange(2)).T 
        feats_imp.iloc[:,0]= feats_imp.index     
        feats_imp.columns=['feature','importance'] 
        feats_imp.sort_values('importance',inplace=True,ascending=False) 
        feats_imp.reset_index(drop=True,inplace=True) 
        return feats_imp

