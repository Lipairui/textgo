# coding: utf-8

import re
import os, sys
import jieba
import jieba.posseg as pseg
import logging 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT) 
logger = logging.getLogger(__file__)

class Preprocess():
    def __init__(self, lang='zh', filter_words=[], stopwords_path='', userdict_path='', filter_pos=[]):
        '''
        Input:
            lang: string. "zh" for Chinese or "en" for English.
            filter_words: list of strings. Words need to be filtered after tokenization.
            stopwords_path: string. Path of stopwords file.  
            userdict_path: string. Path of userdict file.
            filter_pos: list of POS strings. POS need to be filtered during tokenization.
        '''
        self.lang = lang
        if stopwords_path=='':
            if lang == 'en':
                stopwords_path = os.path.join(os.path.dirname(__file__),"data/stopwords_en.txt")
            elif lang == "zh":
                stopwords_path = os.path.join(os.path.dirname(__file__),"data/stopwords_zh.txt")
                if userdict_path != '':
                    jieba.load_userdict(userdict_path)
                    
        self.stopwords = open(stopwords_path).read().strip().split('\n')
        self.stopwords.extend(filter_words) 
        self.stopwords.append(' ')
        if sys.version_info[0] == 2: # python2
            self.stopwords = [word.decode('utf-8') for word in self.stopwords] # for python2
        self.stopwords = set(self.stopwords)
        self.filter_pos_num = len(filter_pos)
        self.filter_pos = set(filter_pos)

    def clean(self, texts, drop_space=False):
        '''Clean text, including dropping html tags, url, extra space and punctuation 
        as well as string lower.
        Input:
            text: list of strings.
        Output:
            text: preprocessed string.
        '''
        ptexts = []
        for text in texts:
            # drop \n
            text = re.sub('\n','',text)
            # drop html tags 
            text = re.sub('<[^>]*>|&quot|&nbsp','',text)
            # drop url
            url_regrex = u'((http|https)\\:\\/\\/)?[a-zA-Z0-9\\.\\/\\?\\:@\\-_=#]+\\.([a-zA-Z]){2,6}([a-zA-Z0-9\\.\\&\\/\\?\\:@\\-_=#])*'
            text = re.sub(url_regrex,'',text)
            # only keep Chinese/English/space/numbers/decimal
            text = re.sub(u"[^\u4E00-\u9FFF^a-z^A-Z^\s^0-9^\d+(\.\d+)?]", " ",text) 
            if drop_space:
                # drop any space
                text = re.sub(u"[\s]{1,}","",text).strip()
            else:
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
                if self.filter_pos_num==0:
                    tokens = list(jieba.cut(text))
                else:
                    tokens = []
                    for word, flag in pseg.cut(text):
                        if flag not in self.filter_pos:
                            tokens.append(word)
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
        
        
