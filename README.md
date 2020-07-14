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
* Preprocess (Clean + Tokenize + Remove stopwords)

**2. Text representation**

**3. Similarity calculation**

**4. Text search**   

**5. Text classification**
  
