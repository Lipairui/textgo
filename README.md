# TextGo
TextGo is a python package to help you work with text data conveniently and efficiently. It's a powerful NLP tool, which provides various apis including text preprocessing, representation, similarity calculation, text search and classification. Besides, it supports both English and Chinese language.

## Highlights
* Provide various text representation algorithms including BOW, TF-IDF, LDA, LSA, PCA, Word2Vec/GloVe/FastText, BERT...
* Support fast text search based on Faiss
* Support various text classification algorithms including FastText, XGBoost, BERT

## Installing
Install and update using pip:      
`pip install textgo`

## Getting Started
**1. Text preprocessing**
   
* Clean
```
from textgo import Preprocess
# Chinese
tp1 = Preprocess(lang='zh')
texts1 = ["<text>自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。<\text>", "??文本预处理~其实很简单！"]
ptexts1 = [tp1.clean(text) for text in texts1]
print(ptexts1)
```
```['自然语言处理是计算机科学领域与人工智能领域中的一个重要方向', '文本预处理其实很简单']```

```
# English
tp2 = Preprocess(lang='en')
texts2 = ["<text>Natural Language Processing, usually shortened as NLP, is a branch of artificial intelligence that deals with the interaction between computers and humans using the natural language<\text>"]
ptexts2 = [tp2.clean(text) for text in texts2]
print(ptexts2)
```
```['natural language processing usually shortened as nlp is a branch of artificial intelligence that deals with the interaction between computers and humans using the natural language']```

* Tokenize
```
# Chinese
tokens1 = [tp1.tokenize(ptext) for ptext in ptexts1]
print(tokens1)
```
```[['自然语言', '处理', '是', '计算机科学', '领域', '与', '人工智能', '领域', '中', '的', '一个', '重要', '方向'], ['文本', '预处理', '其实', '很', '简单']]```

```
# English
tokens2 = [tp2.tokenize(ptext) for ptext in ptexts2]
print(tokens2)
```
```[['natural', 'language', 'processing', 'usually', 'shortened', 'as', 'nlp', 'is', 'a', 'branch', 'of', 'artificial', 'intelligence', 'that', 'deals', 'with', 'the', 'interaction', 'between', 'computers', 'and', 'humans', 'using', 'the', 'natural', 'language']]```

* Preprocess (Clean + Tokenize + Remove stopwords)
```
# Chinese
ptexts1 = tp1.preprocess(texts1)
print(ptexts1)
```
```['自然语言 处理 计算机科学 领域 人工智能 领域 中 重要 方向', '文本 预处理 其实 很 简单']```

```
# English
ptexts2 = tp2.preprocess(texts2)
print(ptexts2)
```
```['natural language processing usually shortened nlp branch artificial intelligence deals interaction computers humans using natural language']```

**2. Text representation**
```
from textgo import Embeddings
petxts = ['自然语言 处理 计算机科学 领域 人工智能 领域 中 重要 方向', '文本 预处理 其实 很 简单']
emb = Embeddings()
# BOW
bow_emb = emb.bow(ptexts)
# TF-IDF
tfidf_emb = emb.tfidf(ptexts)
# LDA
lda_emb = emb.lda(ptexts, dim=2)
# LSA
lsa_emb = emb.lsa(petxts, dim=2)
# PCA
pca_emb = emb.pca(ptexts, dim=2)
# Word2Vec
w2v_emb = emb.word2vec(ptexts, method='word2vec', model_path='model/word2vec.bin')
# GloVe
glove_emb = emb.word2vec(ptexts, method='glove', model_path='model/glove.bin')
# FastText
ft_emb = emb.word2vec(ptexts, method='fasttext', model_path='model/fasttext.bin')

```

**3. Similarity calculation**

**4. Text search**   

**5. Text classification**
  
