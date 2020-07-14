# coding: utf-8

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances

class Metrics():
    def cosine_sim(self, X, Y=None, mutual=True):
        '''Calculate cosine similarity between two arrays.
        Input:
            X: array [n_samples_a, n_features]
            Y: array [n_samples_b, n_features], optional
            mutual: boolean (default=True). If mutual is False, then compute between each 
        sample in X and the corresponding sample in Y, and X.shape = Y.shape.  

        Output:
            res: array [n_samples_a, n_samples_b] if mutual=True, otherwise return list of         length: n_samples_a
        '''
        if mutual:
            res = cosine_similarity(X,Y)
        else:
            if X.shape == Y.shape:
                res = []
                for i in range(X.shape[0]):
                    tmp_X = X[i].reshape(1,-1)
                    tmp_Y = Y[i].reshape(1,-1)
                    tmp_res = cosine_similarity(tmp_X, tmp_Y)[0][0]
                    res.append(tmp_res)
            else:
                print('Error: shape of X and Y must be the same!')
                res = None
        return res
    
    def euclidean_dis(self, X, Y=None, mutual=True):
        '''Calculate euclidean distance between two arrays. 
        Input: 
            X: array [n_samples_a, n_features] 
            Y: array [n_samples_b, n_features], optional 
            mutual: boolean (default=True). If mutual is False, then compute between each  
        sample in X and the corresponding sample in Y, and X.shape = Y.shape.   

        Output: 
            res: array [n_samples_a, n_samples_b] if mutual=True, otherwise return list of         length: n_samples_a 
        '''
        if mutual:
            res = pairwise_distances(X,Y,metric='euclidean')
        else:
            if X.shape == Y.shape:
                res = []
                for i in range(X.shape[0]):
                    tmp_X = X[i].reshape(1,-1) 
                    tmp_Y = Y[i].reshape(1,-1)
                    tmp_res = pairwise_distances(tmp_X, tmp_Y, metric='euclidean')[0][0]
                    res.append(tmp_res)
            else: 
                print('Error: shape of X and Y must be the same!') 
                res = None 
        return res
