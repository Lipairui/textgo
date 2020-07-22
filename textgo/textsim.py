# coding: utf-8

import os
import sys
import logging 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)
import numpy as np
from sklearn.preprocessing import normalize

# import local modules
from .embeddings import Embeddings
from .preprocess import Preprocess
from .metrics import Metrics

class TextSim():
    def __init__(self, lang='zh', filter_words=[], method='word2vec', model_path=''):
        '''
        Input:
            lang: string. "zh" for Chinese or "en" for English.
            filter_words: list of strings. Words need to be filtered after tokenization.
            method: string. 
            model_path: string. Path of model like word2vec/glove/fasttext/bert. 
        '''
        self.tp = Preprocess(lang,filter_words)
        self.method = method
        self.emb = Embeddings()
        self.emb.load_model(method,model_path) # load embedding model
        self.metric = Metrics()
        self.search_index = None
        self.search_db = []

    def similarity(self, texts1, texts2=None, metric='cosine', mutual=True):
        '''Compute similarity score.
        Input:
            texts1: list of strings
            texts2: list of strings, optional
            metric: string. Metric used to calculate similarity (default='cosine')
            mutual: boolean (default=True). If mutual is False, then compute simlarity 
        between each string in texts1 and the corresponding string in texts2, and 
        len(texts1)=len(texts2). 

        Output:
            res: array [len(texts1), len(texts2)] if mutual=True, otherwise return list of
        length: len(texts1)
        '''
        if texts2 is not None:
            texts = texts1+texts2
            texts1_len = len(texts1)
        else:
            texts = texts1
        # Preprocess text
        logger.info('Preprocess text...')
        if self.method=='bert':
            ptexts = self.tp.clean(texts)
        else:
            ptexts = self.tp.preprocess(texts)
        # Get embeddings
        logger.info('Get embeddings...')
        embeddings = self.emb.get_embeddings(ptexts, method=self.method)
        if texts2 is not None:
            embeddings1 = embeddings[:texts1_len]
            embeddings2 = embeddings[texts1_len:]
        else:
            embeddings1 = embeddings
            embeddings2 = embeddings
        # Calculate similarity
        logger.info('Calculate similarity...')
        res = self.metric.cosine_sim(embeddings1, embeddings2, mutual)
        return res
    
    
    def distance(self, texts1, texts2=None, metric='euclidean', mutual=True):
        '''Compute distance score. 
        Input: 
            texts1: list of strings 
            texts2: list of strings, optional 
            metric: string. Metric used to calculate distance (default='cosine') 
            mutual: boolean (default=True). If mutual is False, then compute distance 
        between each string in texts1 and the corresponding string in texts2, and  
        len(texts1)=len(texts2).  
 
        Output: 
            res: array [len(texts1), len(texts2)] if mutual=True, otherwise return list of
        length: len(texts1) 
        '''
        if texts2 is not None:
            texts = texts1+texts2
            texts1_len = len(texts1)
        else:
            texts = texts1
        # Preprocess text
        logger.info('Preprocess text...')
        ptexts = self.tp.preprocess(texts)
        # Get embeddings
        logger.info('Get embeddings...')
        embeddings = self.emb.get_embeddings(ptexts, method=self.method)
        if texts2 is not None:
            embeddings1 = embeddings[:texts1_len]
            embeddings2 = embeddings[texts1_len:]
        else:
            embeddings1 = embeddings
            embeddings2 = embeddings
        # Calculate similarity
        logger.info('Calculate distance...')
        res = self.metric.euclidean_distance(embeddings1, embeddings2, mutual)
        return res

    def get_similar_res(self, texts1, texts2=None, metric='cosine', threshold=0.8, topn=3):
        '''Get most similar result based on threshold. If both texts1 and texts2 are 
        provided, then obtain each texts1 string's most similar text in texts2. Otherwise,
        obtain each texts1 string's most similar text in texts1 except itself.
        Input: 
            texts1: list of strings
            texts2: list of strings, optional
            metric: string. Metric used to calculate similarity (default='cosine')
            threshold: float. Return texts of which similarity>=threshold. 
            topn: int. Return topn most similar texts.

        Output:
            similar_result: list of list of tuples [[(text_index,text,similarity)]]
        '''
        
        sim_matrix = self.similarity(texts1, texts2, metric, mutual=True)
        similar_result = []
        for i in range(sim_matrix.shape[0]):
            sims = sim_matrix[i]
            if texts2 is not None:
                similar_res = list(zip(list(range(sim_matrix.shape[1])),texts2,sims))
                similar_res = sorted(similar_res,key=lambda x:x[-1],reverse=True)
            else:
                similar_res = list(zip(list(range(sim_matrix.shape[1])),texts1,sims)) 
                similar_res = sorted(similar_res,key=lambda x:x[-1],reverse=True)[1:]
            if topn is not None: # topn
                similar_res = similar_res[:topn]
            if threshold is not None: # sim >= threshold
                similar_res = [item for item in similar_res if item[-1]>=threshold]
            similar_result.append(similar_res)
        return similar_result

    def build_index(self, texts, metric='cosine'):
        '''Build search index based on faiss. If self.search_index = None, then build search index from scratch. Otherwise, add search index to history index.
        Input: 
            texts: list of strings, database of the search engine
            metric: default='cosine'
        Output:
            search_index object
        '''
        import faiss
        # Preprocess text  
        logger.info('Preprocess text...') 
        if self.method=='bert': 
            ptexts = self.tp.clean(texts)
        else: 
            ptexts = self.tp.preprocess(texts)  
        # Get embeddings  
        logger.info('Get embeddings...')  
        embeddings = self.emb.get_embeddings(ptexts, method=self.method)
        if metric=='cosine':
            embeddings = normalize(embeddings).astype(np.float32)
        else:
            embeddings = embeddings.astype(np.float32)
        dim = embeddings.shape[1]
        # Build search index
        if self.search_index is None:
            logger.info('Build search index...')
            if metric=='cosine':
                self.search_index = faiss.IndexFlatIP(dim)
            elif metric=='euclidean':
                self.search_index = faiss.IndexFlatL2(dim)
        else:
            logger.info('Add search index...')
        self.search_index.add(embeddings)
        self.search_db.extend(texts)
        self.search_metric = metric
        return self.search_index
    
    def clear_index(self):
        self.search_index = None
        self.search_db = []

    def search(self, texts, threshold=0.8, topn=3):
        '''Search and find the most similar texts in the database based on threshold and 
        topn.
        Input:
            texts: list of strings, search querys
            threshold: float. Return texts of which similarity>=threshold.
            topn: int. Return topn most similar texts.
        Output:
            result: list of list of tuples [[(text_index,text,similarity)]]
        '''
        # Preprocess text   
        logger.info('Preprocess text...')   
        if self.method=='bert':  
            ptexts = self.tp.clean(texts)
        else:
            ptexts = self.tp.preprocess(texts)   
        # Get embeddings   
        logger.info('Get embeddings...')   
        embeddings = self.emb.get_embeddings(ptexts, method=self.method) 
        if self.search_metric=='cosine':
            embeddings = normalize(embeddings).astype(np.float32)
        else:
            embeddings = embeddings.astype(np.float32)
        # Search
        logger.info('Search...')
        result = []
        if topn is None:
            topn = self.search_index.ntotal
        D, I = self.search_index.search(embeddings,int(topn))
        if threshold is None:
            for i in range(len(texts)):
                similar_indexs = I[i]
                similar_texts = np.array(self.search_db)[similar_indexs]
                similar_scores = D[i]
                res = list(zip(similar_indexs, similar_texts, similar_scores))
                result.append(res)
        else:
            for i in range(len(texts)):
                res = []
                for j in range(I.shape[1]):
                    sim_index = I[i][j]
                    sim_score = D[i][j]
                    if self.search_metric=='cosine': 
                        if sim_score>=threshold: # 余弦相似度越高越相似
                            res.append((sim_index, self.search_db[sim_index], sim_score))
                    elif self.search_metric=='euclidean':
                        if sim_score<=threshold: # 欧式距离越小越相似
                            res.append((sim_index, self.search_db[sim_index], sim_score))
                result.append(res)
        return result

if __name__ == '__main__':
    texts1 = ["一项研究发现，在某些法国和荷兰的奶酪中存在不同程度的K2。"]
    texts2 = [ 
        "但其含量多少取决于奶酪品种、成熟时间、脂肪含量和奶酪的产地等。",
        "但通常来讲，高脂肪和陈年奶酪的K2含量较高", 
        "此外，维生素K还是一种脂溶性维生素，就是说当它与富含健康脂肪的食物!"]
    #ts = TextSim(method='word2vec',model_path='../w2v_models/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding')
    ts = TextSim(method='bert',model_path='../../bert-base-chinese')
    #res = ts.similarity(texts1, texts2)
    #print(res)
    #res = ts.get_similar_res(texts2, threshold=1)
    #print(res)
    ts.build_index(texts2, metric='cosine') # build index
    res = ts.search(texts1, threshold=0.7)
    print(ts.search_index.ntotal)
    print(res)
    ts.build_index(texts1, metric='cosine') # add index
    res = ts.search(texts2, threshold=0.7)
    print(ts.search_index.ntotal)
    print(res)
    ts.clear_index() # clear index
    print(ts.search_index)
