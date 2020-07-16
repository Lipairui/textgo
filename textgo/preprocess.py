# coding: utf-8

import re
import os, sys
import jieba
import logging 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)

class Preprocess():
    def __init__(self, lang='zh', filter_words=[], stopwords_path=''):
        '''
        Input:
            lang: string. "zh" for Chinese or "en" for English.
            filter_words: list of strings. Words need to be filtered after tokenization.
            stopwords_path: string. Path of stopwords file.  
        '''
        self.lang = lang
        if stopwords_path=='':
            if lang == 'en':
                stopwords_path = os.path.join(os.path.dirname(__file__),"data/stopwords_en.txt")
            elif lang == "zh":
                stopwords_path = os.path.join(os.path.dirname(__file__),"data/stopwords_zh.txt")
        
        self.stopwords = open(stopwords_path).read().strip().split('\n')
        self.stopwords.extend(filter_words) 
        self.stopwords.append(' ')
        self.stopwords = set(self.stopwords)

    def clean(self, texts):
        '''Clean text, including dropping html tags, url, extra space and punctuation 
        as well as string lower.
        Input:
            text: list of strings.
        Output:
            text: preprocessed string.
        '''
        ptexts = []
        for text in texts:
            # drop html tags 
            text = re.sub('<[^>]*>|&quot|&nbsp','',text)
            # drop url
            url_regrex = u'((http|https)\\:\\/\\/)?[a-zA-Z0-9\\.\\/\\?\\:@\\-_=#]+\\.([a-zA-Z]){2,6}([a-zA-Z0-9\\.\\&\\/\\?\\:@\\-_=#])*'
            text = re.sub(url_regrex,'',text)
            # only keep Chinese/English/space
            text = re.sub(u"[^\u4E00-\u9FFF^a-z^A-Z^\s]", " ",text) 
            # drop space at the start and in the end
            text = re.sub(u"\s$|^\s","",text)
            # replace more than 2 space with 1 space
            text = re.sub(u"[\s]{2,}"," ",text).strip()
            # lower string
            text = text.lower()
            ptexts.append(text)
        return ptexts

    def tokenize(self, texts, drop_stopwords=True):
        '''Tokenize string.
        Input:
            text: list of strings.
            drop_stopwords: boolean. Whether drop stopwords or not.
        Output:
            tokens_list: list of list of tokens.
        '''
        tokens_list = []
        if self.lang == 'en':
            for text in texts:
                tokens = text.split(' ')
                if drop_stopwords:
                    tokens = [token for token in tokens if token not in self.stopwords]
                tokens_list.append(tokens)
        elif self.lang == "zh":
            for text in texts:
                tokens = list(jieba.cut(text))
                if drop_stopwords:
                    tokens = [token for token in tokens if token not in self.stopwords]
                tokens_list.append(tokens)
        return tokens_list

    def preprocess(self, texts, sep=' ', drop_stopwords=True):
        '''Text preprocess for English/Chinese, including clean text, tokenize and remove 
        stopwords.
        Input:
            texts: list of text strings
            sep: string used to join words after tokenization
            drop_stopwords: boolean. Whether drop stopwords or not.
        Output:
            list of preprocessed text strings (tokens join by sep)
        '''
        # clean text
        ptexts = self.clean(texts)
        # tokenize
        tokens_list = self.tokenize(ptexts, drop_stopwords)
        # join by sep
        result = [sep.join(tokens) for tokens in tokens_list]
        return result


if __name__ == '__main__':
    tp = Preprocess(lang='zh')
    texts = [u" hello world¥……*& <block> d</block> 100源 你好! \t\n",u"停用词stopwords就是句子没什么必要的单词，去掉他们以后对理解整个句子的语义没有影响"]
    ptexts = tp.preprocess(texts)
    print(ptexts)
        
        
