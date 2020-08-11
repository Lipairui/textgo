# coding: UTF-8
import os
import sys
import torch
import random
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta
import ast
import configparser

# import local modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
sys.path.append(os.path.join(BASE_DIR,'../'))
from preprocess import Preprocess
from embeddings import Embeddings

UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
 
def literalConf(conf): 
    new_conf = {} 
    for key in conf: 
        new_conf[key] = ast.literal_eval(conf[key]) 
    return new_conf

def load_config(config_path, section_name):
    config = configparser.ConfigParser() 
    config.optionxform = str # key not lowercase 
    config.read(config_path)
    conf = config._sections[section_name]
    conf = literalConf(conf)
    return conf

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

def get_device(): 
    # If there's a GPU available... 
    if torch.cuda.is_available():     
        # Tell PyTorch to use the GPU.     
        device = torch.device("cuda") 
 
        print('There are %d GPU(s) available.' %torch.cuda.device_count()) 
 
        print('We will use the GPU:%s' %torch.cuda.get_device_name(0)) 
    else: 
        print('No GPU available, using the CPU instead.') 
        device = torch.device("cpu") 
    return device

def build_vocab(X, tokenizer, vocab_path='', binary=True, max_vocab_size=10000, min_freq=1):
    '''
    Build vocab dict from corpus.
    '''
    vocab_dic = {}
    for content in X:    
        for word in tokenizer(content):
            vocab_dic[word] = vocab_dic.get(word, 0) + 1
    vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_vocab_size]
    vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    if vocab_path!='':
        save_vocab(vocab_dic, vocab_path, binary) 
    print("Vocab size: %i"%len(vocab_dic))
    return vocab_dic

def build_vocab_from_txt(txt_path, vocab_path=''):
    '''
    Build vocab dict from text file, where each line contain one word.
    '''
    vocab_dic = {}
    with open(txt_path) as fin:
        words = fin.read().strip().split('\n')
        index = 0
        for word in words:
            if (word!=UNK) and (word!=PAD) and word[:2]!='##':
                vocab_dic[word] = index
                index += 1
    vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    if vocab_path!='':
        save_vocab(vocab_dic, vocab_path) 
    print("Vocab size: %i"%len(vocab_dic)) 
    return vocab_dic

def save_vocab(vocab_dict, save_path, binary=True):
    if binary: # save as binary file
        pkl.dump(vocab_dict, open(save_path, 'wb')) 
    else: # save as txt file
        vocab_items = sorted(vocab_dict.items(),key=lambda x:int(x[1]))
        with open(save_path,'w') as f:
            for word, index in vocab_items:
                f.write(word)
                f.write('\n')    

class Tokenizer():
    def __init__(self, word_level=False, preprocess=True, lang='zh'):
        self.tp = Preprocess(lang=lang)
        self.word_level = word_level
        self.preprocess = preprocess
        self.lang = lang

    def tokenize_str(self, x):
        if self.preprocess:
            if self.word_level:
                x = self.tp.preprocess([x])[0]
            else:
                x = self.tp.clean([x],drop_space=True)[0]
        if self.word_level:
            tokens =  x.split(' ')
        else:
            tokens = [t for t in x]
        return tokens
            
    def __call__(self, X):
        if type(X) is str:
            return self.tokenize_str(X)
        else:
            tokens_list = []
            for x in X:
                tokens_list.append(self.tokenize_str(x))
            return tokens_list

def load_vocab(vocab_path, binary=True):
    if binary:
        vocab = pkl.load(open(vocab_path, 'rb'))
        print("Vocab size: %i"%len(vocab))
    else:
        vocab = build_vocab_from_txt(vocab_path)
    return vocab

def load_dataset(vocab, X, y, word_level=False, preprocess=True, lang='zh', max_len=32):
    tokenizer = Tokenizer(word_level, preprocess, lang)
    contents = []
    for i in range(len(X)):
        content = X[i]
        if y is not None:
            label = y[i]
        words_line = []
        tokens = tokenizer(content)
        seq_len = len(tokens)
        if max_len:
            if len(tokens) < max_len:
                tokens.extend([PAD] * (max_len - len(tokens)))
            else:
                tokens = tokens[:max_len]
                seq_len = max_len
        # word to id
        for word in tokens:
            words_line.append(vocab.get(word, vocab.get(UNK)))
        if y is not None:
            contents.append((words_line, int(label), seq_len)) # [([...], 0, 20), ([...], 1, 30), ...]
        else:
            contents.append((words_line, seq_len)) # [([...], 20), ([...], 30), ...]
    return contents 

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        # pad前的长度(超过max_len的设为max_len)
        seq_len = torch.LongTensor([_[-1] for _ in datas]).to(self.device)
        if len(datas[0])>=3:
            y = torch.LongTensor([_[1] for _ in datas]).to(self.device)
            return (x, seq_len), y
        else:
            return (x, seq_len)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def get_data_iterator(args, vocab, X, y=None):
    dataset = load_dataset(vocab, X, y, args['word_level'], args['preprocess'], args['lang'], args['max_len'])
    iter = DatasetIterater(dataset, args['batch_size'], args['device'])
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def get_pretrained_embeddings(path, vocab, method='word2vec'):
    emb = Embeddings()
    model = emb.load_model(method=method, model_path=path)
    embed_size = model.vector_size
    embeddings = np.zeros((len(vocab),embed_size))
    oov_count = 0
    for word in vocab:
        word_index = vocab[word]
        if word in model.vocab:
            embeddings[word_index] = model[word]
        else:
            oov_count += 1
    print('OOV count: %i'%oov_count)
    return embeddings.astype('float32')

if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    for i, line in enumerate(f.readlines()):
        # if i == 0:  # 若第一行是标题，则跳过
        #     continue
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
