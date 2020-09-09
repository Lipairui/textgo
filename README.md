# TextGo

*TextGo* is a python package to help you work with text data conveniently and efficiently. It's a powerful NLP tool, which provides various apis including text preprocessing, representation, similarity calculation, text search and classification. Besides, it supports both English and Chinese language.

## Highlights
* Support both English and Chinese languages in text preprocessing
* Provide various text representation algorithms including BOW, TF-IDF, LDA, LSA, PCA, Word2Vec/GloVe/FastText, BERT...
* Support fast text search based on [Faiss](https://github.com/facebookresearch/faiss)
* Support various text classification algorithms including FastText, XGBoost, BERT

## Installing
Install and update using pip:      
`pip install textgo`

Note: successfully tested on python3.     
Tips: the fasttext package needs to be installed manually as follows:

```
git clone https://github.com/facebookresearch/fastText.git
cd fastText-master
make
pip install .
```

## Getting Started
### 1. Text preprocessing
   
**Clean text**

```
from textgo import Preprocess
# Chinese
tp1 = Preprocess(lang='zh')
texts1 = ["<text>自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。<\text>", "??文本预处理~其实很简单！"]
ptexts1 = tp1.clean(texts1)
print(ptexts1)
```

Output: `['自然语言处理是计算机科学领域与人工智能领域中的一个重要方向', '文本预处理其实很简单']`
  
```
# English
tp2 = Preprocess(lang='en')
texts2 = ["<text>Natural Language Processing, usually shortened as NLP, is a branch of artificial intelligence that deals with the interaction between computers and humans using the natural language<\text>"]
ptexts2 = tp2.clean(texts2)
print(ptexts2)
```
Output: `['natural language processing usually shortened as nlp is a branch of artificial intelligence that deals with the interaction between computers and humans using the natural language']`

**Tokenize and drop stopwords**
```
# Chinese
tokens1 = tp1.tokenize(ptexts1)
print(tokens1)
```
Output: `[['自然语言', '处理', '计算机科学', '领域', '人工智能', '领域', '中', '重要', '方向'], ['文本', '预处理', '其实', '很', '简单']]`

```
# English
tokens2 = tp2.tokenize(ptexts2)
print(tokens2)
```
Output: `[['natural', 'language', 'processing', 'usually', 'shortened', 'nlp', 'branch', 'artificial', 'intelligence', 'deals', 'interaction', 'computers', 'humans', 'using', 'natural', 'language']]`

**Preprocess (Clean + Tokenize + Remove stopwords + Join words)**
```
# Chinese
ptexts1 = tp1.preprocess(texts1)
print(ptexts1)
```
Output: `['自然语言 处理 计算机科学 领域 人工智能 领域 中 重要 方向', '文本 预处理 其实 很 简单']`

```
# English
ptexts2 = tp2.preprocess(texts2)
print(ptexts2)
```
Output: `['natural language processing usually shortened nlp branch artificial intelligence deals interaction computers humans using natural language']`

### 2. Text representation
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

# BERT
bert_emb = emb.bert(ptexts, model_path='model/bert-base-chinese')

```
Tips: For methods like Word2Vec and BERT, you can load the model first and then get embeddings to avoid loading model repeatedly. Take BERT For example:
```
emb.load_model(method="bert", model_path='model/bert-base-chinese')
bert_emb1 = emb.bert(ptexts1)
bert_emb2 = emb.bert(ptexts2)
```

### 3. Similarity calculation

Support calculating similarity/distance between texts based on text representation mentioned above. For example, we can use bert sentence embeddings to compute cosine similarity between two sentences one by one.
```
from textgo import TextSim
texts1 = ["她的笑渐渐变少了。","最近天气晴朗适合出去玩！"]
texts2 = ["她变得越来越不开心了。","近来总是风雨交加没法外出！"]

ts = TextSim(lang='zh', method='bert', model_path='model/bert-base-chinese')
sim = ts.similarity(texts1, texts2, mutual=False)
print(sim)
```   

Output: `[0.9143135, 0.7350756]`

Besides, we can also calculate similarity between each sentences among two datasets by setting mutual=True.
```
sim = ts.similarity(texts1, texts2, mutual=True)
print(sim)
```

Output: `
array([[0.9143138 , 0.772496  ],
       [0.704296  , 0.73507595]], dtype=float32)
`
       
### 4. Text search
It also supports searching query text in a large text database based on cosine similarity or euclidean distance. It provides two kinds of implementation: the normal one which is suitable for small dataset and the optimized one which is based on Faiss and suitable for large dataset.
```
from textgo import TextSim
# query texts
texts1 = ["A soccer game with multiple males playing."]
# database
texts2 = ["Some men are playing a sport.", "A man is driving down a lonely road.", "A happy woman in a fairy costume holds an umbrella."]
ts = TextSim(lang='en', method='word2vec', model_path='model/word2vec.bin')
```

**Normal search**
```
res = ts.get_similar_res(texts1, texts2, metric='cosine', threshold=0.5, topn=2)
print(res)
```
Output: `[[(0, 'Some men are playing a sport.', 0.828474), (1, 'A man is driving down a lonely road.', 0.60927737)]]`

**Fast search**
```
ts.build_index(texts2, metric='cosine')
res = ts.search(texts1, threshold=0.5, topn=2)
print(res)
```
Output: `[[(0, 'Some men are playing a sport.', 0.828474), (1, 'A man is driving down a lonely road.', 0.60927737)]]`

### 5. Text classification
Train a text classifier just in several lines. Models supported: FastText, TextCNN, TextRNN, TextRCNN, TextRCNN_Att, Bert, XLNet.
```
# Prepare data
X = [text1, text2, ... textn]
y = [label1, label2, ... labeln]

from textgo import Classifier

# load config
config_path = "./config.ini"  # Include all model parameters
model_name = "Bert" # Supported models: FastText, TextCNN, TextRNN, TextRCNN, TextRCNN_Att, Bert, XLNet
args = load_config(config_path, model_name) 
args['model_name'] = model_name 
args['save_path'] = "output/%s"%model_name

# train 
clf = Classifier(args) 
clf.train(X_train, y_train, evaluate_test=False) # If evaluate_test=True, then it will split 10% for test dataset and evaluate on test dataset. 

# predict
predclass = clf.predict(X_train) 
```

## LICENSE
TextGo is MIT-licensed.
