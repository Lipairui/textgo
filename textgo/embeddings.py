# coding: utf-8
import os
import sys
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
import logging 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)

class Embeddings():
    def __init__(self):
        self.model = None

    def load_model(self, method='word2vec', model_path=''):
        '''Load embedding model including word2vec/fasttext/glove/bert.
        Input: 
            method: string. Including "word2vec"/"fasttext"/"glove"/"bert".
            model_path: string. Path of model.
        Output:
            model: model object.
        '''
        self.method = method
        self.model_path = model_path
        if model_path == '':
            self.model = None
            return None
        logger.info('Load embedding model...')
        if method in ['word2vec','glove']:
            from gensim.models import KeyedVectors
            if model_path[-4:]=='.txt':
                self.model = KeyedVectors.load_word2vec_format(model_path,binary=False).wv
            elif model_path[-4:] =='.bin':
                self.model = KeyedVectors.load_word2vec_format(model_path,binary=True).wv
            else:
                self.model = KeyedVectors.load(model_path,mmap='r').wv
            return self.model
        elif method == 'fasttext':
            from gensim.models.wrappers import FastText
            self.model = FastText.load_fasttext_format(model_path).wv
            return self.model
        elif method == 'bert':
            from transformers import BertTokenizer, BertModel 
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            self.model = BertModel.from_pretrained(model_path)
            return self.model
        else:
            self.model = None
            return None

    def bow(self, corpus, ngram_range=(1,1), min_df=1):
        '''Get BOW (bag of words) embeddings.
        Input:
            corpus: list of preprocessed strings. 
            ngram_range: tuple. (min_ngram, max_ngram) means min_ngram<=ngram<=max_ngram
            min_df: int. Mininum frequencey of a word. 
        Output:
            embeddings: array of shape [n_sample, dim]
        '''
        vectorizer = CountVectorizer(ngram_range=ngram_range, min_df=min_df, token_pattern=r'\b\w{1,}\b')
        # The default token_pattern r'\b\w+\b' tokenizes the string by extracting words of at least 2 letters, which is not suitable for Chinese
        X = vectorizer.fit_transform(corpus)
        #print(vectorizer.get_feature_names())
        embeddings = X.toarray()
        return embeddings
    
    def tfidf(self, corpus, ngram_range=(1,1), min_df=1):
        '''Get TFIDF embeddings. 
        Input: 
            corpus: list of preprocessed strings.  
            ngram_range: tuple. (min_ngram, max_ngram) means min_ngram<=ngram<=max_ngram 
            min_df: int. Mininum frequencey of a word.  
        Output: 
            embeddings: array of shape [n_sample, dim] 
        '''
        transformer = TfidfTransformer(use_idf=True, smooth_idf=True)
        counts = self.bow(corpus, ngram_range, min_df)
        X = transformer.fit_transform(counts)
        embeddings = X.toarray()
        return embeddings
    
    def lda(self, corpus, ngram_range=(1,1), min_df=1, dim=5, random_state=0):
        '''Get LDA embeddings. 
        Input: 
            corpus: list of preprocessed strings.  
            ngram_range: tuple. (min_ngram, max_ngram) means min_ngram<=ngram<=max_ngram 
            min_df: int. Mininum frequencey of a word.  
            dim: int. Dimention of embedding.
            random_state: int. 
        Output: 
            embeddings: array of shape [n_sample, dim] 
        '''
        transformer=LatentDirichletAllocation(n_components=dim, random_state=random_state)
        # transform corpus to bow format
        counts = self.bow(corpus, ngram_range, min_df)
        # get lda embeddings
        embeddings = transformer.fit_transform(counts)
        return embeddings
    
    def lsa(self, corpus, ngram_range=(1,1), min_df=1, n_iter=5, dim=5, base_embeddings='tfidf'):
        '''Get LSA embeddings.  
        Input:  
            corpus: list of preprocessed strings.   
            ngram_range: tuple. (min_ngram, max_ngram) means min_ngram<=ngram<=max_ngram  
            min_df: int. Mininum frequencey of a word.  
            n_iter: int. Number of iterations.
            dim: int. Dimention of embedding. 
            base_embeddings: string. "tfidf" or "bow"
        Output:  
            embeddings: array of shape [n_sample, dim]  
        '''
        # get base embeddings 
        if base_embeddings=='tfidf': 
            X = self.tfidf(corpus, ngram_range, min_df) 
        else: 
            X = self.bow(corpus, ngram_range, min_df)
        # get LSA embeddings
        transformer = TruncatedSVD(n_components=dim,algorithm='randomized',n_iter=n_iter) 
        embeddings = transformer.fit_transform(X)
        return embeddings

    def pca(self, corpus, ngram_range=(1,1), min_df=1, dim=5, base_embeddings='tfidf'):
        '''Get PCA embeddings.  
        Input:   
            corpus: list of preprocessed strings.    
            ngram_range: tuple. (min_ngram, max_ngram) means min_ngram<=ngram<=max_ngram   
            min_df: int. Mininum frequencey of a word.   
            dim: int. Dimention of embedding.  
            base_embeddings: string. "tfidf" or "bow" 
        Output:   
            embeddings: array of shape [n_sample, dim]   
        '''
        # get base embeddings
        if base_embeddings=='tfidf':
            X = self.tfidf(corpus, ngram_range, min_df)
        else:
            X = self.bow(corpus, ngram_range, min_df)
        # get PCA embeddings
        transformer = PCA(n_components=dim, svd_solver='auto')
        embeddings = transformer.fit_transform(X)
        return embeddings

    def word2vec(self, corpus, method='word2vec', model_path=''):
        '''Get Word2Vec embeddings.   
        Input:    
            corpus: list of preprocessed strings.   
            method: string. "word2vec"/"glove"/"fasttext"
            model_path: string. Path of model.   
        Output:    
            embeddings: array of shape [n_sample, dim]    
        '''
        # load model
        if self.model is None and model_path!='':
            self.load_model(method, model_path)
        embeddings = [] 
        # drop tokens which not in vocab
        for text in corpus:
            tokens = text.split(' ')
            tokens = [token for token in tokens if token in self.model.vocab]
            #logger.info(', '.join(tokens))
            if len(tokens)==0:
                embedding = self.model['unk'].tolist()
            else:
                embedding = np.mean(self.model[tokens],axis=0).tolist()
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        return embeddings
   
    def bert(self, corpus, model_path='', mode='cls'):
        '''Get BERT embeddings.   
        Input:    
            corpus: list of preprocessed strings.   
            model_path: string. Path of model.
            mode: string. "cls"/"mean". "cls" mode: get the embedding of the first 
            token of a sentence; "mean" mode: get the average embedding of all tokens of 
            a sentence except for the first [CLS] and the last [SEP] tokens.   
        Output:    
            embeddings: array of shape [n_sample, dim]    
        '''
        import torch
        # load model
        if self.model is None and model_path!='':
            self.load_model('bert',model_path)
            
        print(corpus)
        embeddings = []
        for text in corpus:
            # tokenize and encode
            input_ids = torch.tensor(self.tokenizer.encode(text)).unsqueeze(0) # Batch size 1
            # get embedding
            outputs = self.model(input_ids)
            embedding = outputs[0].detach().numpy()  # The last hidden-state is the first element of the output tuple
            if mode=='cls':
                embedding = embedding[0][0]
            elif mode=='mean':
                embedding = np.mean(embedding[0],axis=0)
            embeddings.append(embedding)
        embeddings = np.array(embeddings)
        return embeddings

    def get_embeddings(self, corpus, ngram_range=(1,1), min_df=1, dim=5, method='tfidf', model_path=''):
        '''Get embeddings according to params.
        Input:
            corpus: list of preprocessed strings.     
            ngram_range: tuple. (min_ngram, max_ngram) means min_ngram<=ngram<=max_ngram   
            min_df: int. Mininum frequencey of a word. 
            dim: int. Dimention of embedding.  
            method: string. Including "bow"/"tfidf"/"lda"/"lsa"/"pca"/"word2vec"/"glove"/
            "fasttext"/"bert"
        Output:
            embeddings: array of shape [n_sample, dim] 
        '''
        self.method = method 
        if self.method == 'bow': 
            return self.bow(corpus, ngram_range=ngram_range, min_df=min_df) 
        elif self.method == 'tfidf': 
            return self.tfidf(corpus, ngram_range=ngram_range, min_df=min_df) 
        elif self.method == 'lda': 
            return self.lda(corpus, ngram_range=ngram_range, min_df=min_df, dim=dim) 
        elif self.method == 'lsa': 
            return self.lsa(corpus, ngram_range=ngram_range, min_df=min_df, dim=dim) 
        elif self.method == 'pca': 
            return self.pca(corpus, ngram_range=ngram_range, min_df=min_df, dim=dim) 
        elif self.method in ['word2vec','glove','fasttext']: 
            return self.word2vec(corpus, method=method, model_path=model_path)
        elif self.method == 'bert':
            return self.bert(corpus, model_path=model_path)

if __name__ == '__main__': 
    corpus_en = ['This is the first document.',
         'This is the second second document.',
         'And the third one!',
         'Is this the first document?']
    corpus_zh = ["一项研究发现，在某些法国和荷兰的奶酪中存在不同程度的K2。",
        "但其含量多少取决于奶酪品种、成熟时间、脂肪含量和奶酪的产地等。",
        "但通常来讲，高脂肪和陈年奶酪的K2含量较高",
        "此外，维生素K还是一种脂溶性维生素，就是说当它与富含健康脂肪的食物!"]
    # text preprocess
    from preprocess import Preprocess
    tp_en = Preprocess('en')
    tp_zh = Preprocess('zh')
    corpus_en = tp_en.preprocess(corpus_en)
    print(corpus_en)
    corpus_zh = tp_zh.preprocess(corpus_zh)
    print(corpus_zh)
    '''
    # bow
    emb = Embeddings()
    bow_en = emb.bow(corpus_en)
    print(bow_en)
    bow_zh = emb.bow(corpus_zh)
    print(bow_zh)

    # tfidf
    emb = Embeddings() 
    tfidf_en = emb.tfidf(corpus_en)
    print(tfidf_en)
    tfidf_zh = emb.tfidf(corpus_zh)
    print(tfidf_zh)
    
    # lda
    emb = Embeddings()
    lda_en = emb.lda(corpus_en,dim=2)
    print(lda_en)
    lda_zh = emb.lda(corpus_zh,dim=2)
    print(lda_zh)
    
    # pca
    emb = Embeddings()
    pca_en = emb.pca(corpus_en,dim=2,base_embeddings='bow')
    print(pca_en)
    pca_zh = emb.pca(corpus_zh,dim=2)
    print(pca_zh)

    # lsa
    emb = Embeddings() 
    lsa_en = emb.lsa(corpus_en,dim=2,base_embeddings='bow') 
    print(lsa_en) 
    lsa_zh = emb.lsa(corpus_zh,dim=2) 
    print(lsa_zh)

    # word2vec
    emb = Embeddings() 
    emb.load_model(method='word2vec',model_path="../w2v_models/Tencent_AILab_ChineseEmbedding/Tencent_AILab_ChineseEmbedding") 
    word2vec_zh = emb.word2vec(corpus_zh) 
    print(word2vec_zh)
    
    # glove
    # transform glove model to word2vec format
    from utils import transformGlove
    source_model_path = os.path.join(os.path.dirname(__file__),"../w2v_models/GloVe/en/glove.42B.300d.txt")
    target_model = os.path.join(os.path.dirname(__file__),"../w2v_models/GloVe/en/glove.42B.300d.w2v")
    #transformGlove(source_model_path,target_model,binary=True)
    target_model_path = target_model+'.bin'
    # get embeddings
    emb = Embeddings()
    emb.load_model(method='word2vec',model_path=target_model_path)
    glove_en = emb.word2vec(corpus_en)
    print(glove_en)
    
    # fasttext
    model_path = os.path.join(os.path.dirname(__file__),"../w2v_models/FastText/en/cc.en.300.bin")
    
    emb = Embeddings()
    emb.load_model(method='fasttext', model_path=model_path)
    ft_en = emb.word2vec(corpus_en)
    print(ft_en.shape)
    '''

    emb = Embeddings()
    model_path = os.path.join(os.path.dirname(__file__),"../../bert-base-chinese")
    emb.load_model(method='bert', model_path=model_path)
    bert_zh = emb.bert(corpus_zh)
    print(bert_zh.shape)

