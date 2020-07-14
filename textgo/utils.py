# coding: utf-8

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def transformGlove(source_model_path, output_model_name, binary=True):
    '''Transform GloVe model text format into word2vec text/binary format 
    Input: 
        :source_model_path: GloVe model path (.txt)
        :output_model_name: name of Word2Vec model format
        :binary: whether save binary format model or not (.bin)  
    '''
    # GloVe vectors loading function into temporary file
    glove2word2vec(source_model_path, output_model_name+'.txt')
    if binary:
        model = KeyedVectors.load_word2vec_format(output_model_name+'.txt')
        model.save_word2vec_format(output_model_name+'.bin', binary=True)
    
