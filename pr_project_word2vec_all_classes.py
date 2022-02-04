#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk.corpus
import os
import nltk
import re
import numpy as np
import pandas as pd
import time


# In[2]:


from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.metrics.pairwise import cosine_similarity

import heapq
from copy import deepcopy


# In[3]:


datasetPath = './dataset/dataset_large/'

X_train_raw = []
y_train = []
X_test_raw = []
y_test = []

count = 0

for files in os.listdir(datasetPath):
    for f in os.listdir(datasetPath+files):
        if (count % 10000 == 0):
            print(count)
        count += 1
        d = open(datasetPath+files+"/"+f, "r")
        data = str(d.read())
        label = re.search("[\d]*_([\d]*)",f).group(1)
        if files == "train":
            #print(f)
            X_train_raw.append(data)
            y_train.append(label)
        if files == "test":
            #print(f)
            X_test_raw.append(data)
            y_test.append(label)


# In[4]:


from nltk.corpus import stopwords
tokens_to_ignore = stopwords.words('english')
tokens_to_ignore.append('br')

def clear_text(X):
    all_sentenses = []
    for i in range(len(X)):
        if(i < 50000):
            if(i % 10000 == 0):  
                print(i)
            x = X[i]
            processed_article = x.lower()
            processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
            processed_article = re.sub(r'\s+', ' ', processed_article)
            
            # Preparing the dataset
            all_sentences = nltk.sent_tokenize(processed_article)
            all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

            # Removing Stop Words
            for i in range(len(all_words)):
                all_words[i] = [w for w in all_words[i] if w not in tokens_to_ignore]
            all_sentenses.append(all_words[0])
    return all_sentenses

new_X = clear_text(X_train_raw)
new_X_test = clear_text(X_test_raw)


# In[5]:


new_X_shuffle, y_train_shuffle = shuffle(new_X, y_train)
new_X_test_shuffle, y_test_shuffle = shuffle(new_X_test, y_test)


# In[6]:


def update_lables(y, size=8):
    y_ = deepcopy(y)
    target_names = []
    for i in range(len(y)):
        if(size == 2):
            if(int(y[i]) <= 5):
                y_[i] = 0
            else:
                y_[i] = 1
        elif(size == 4):
            if(int(y[i]) == 1 or int(y[i]) == 2):
                y_[i] = 0
            elif(int(y[i]) == 3 or int(y[i]) == 4):
                y_[i] = 1
            elif(int(y[i]) == 6 or int(y[i]) == 7):
                y_[i] = 2
            else:
                y_[i] = 3
        elif(size == 8):
            if(int(y[i]) == 1):
                y_[i] = 0
            elif(int(y[i]) == 2):
                y_[i] = 1
            elif(int(y[i]) == 3):
                y_[i] = 2
            elif(int(y[i]) == 4):
                y_[i] = 3
            elif(int(y[i]) == 7):
                y_[i] = 4
            elif(int(y[i]) == 8):
                y_[i] = 5
            elif(int(y[i]) == 9):
                y_[i] = 6
            else:
                y_[i] = 7
    
    labels = np.unique(y_)
    for label in labels:
        target_names.append('Rating '+str(label))
    return y_, target_names


# In[7]:


import gensim
model = gensim.models.Word2Vec(new_X, size=100)
w2v = dict(zip(model.wv.index2word, model.wv.vectors))


# In[8]:


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.items())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# In[9]:


knn_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("KNN", KNeighborsClassifier(n_neighbors=19))])
mnbayes_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("Normalise between 0 and 1", MinMaxScaler()),
    ("Naive Bayes", MultinomialNB())])
svm_w2v = Pipeline([
    ("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
    ("SVM", svm.SVC(gamma='auto'))])


# In[10]:


classes_to_try = [8, 4, 2]


# In[11]:


show_full_report = False # set to True to see full report for all classifications
for i in range(len(classes_to_try)):
    print(f"\n\n---------- Results for {classes_to_try[i]} Classes ----------\n")
    y_train_1, _ = update_lables(y_train_shuffle, classes_to_try[i])
    y_test_1, target_names = update_lables(y_test_shuffle, classes_to_try[i])
    
    # KNN
    knn_w2v.fit(new_X_shuffle, y_train_1)
    knn_res = knn_w2v.predict(new_X_test_shuffle)
    knn_accuracy = accuracy_score(y_test_1, knn_res)
    print(f"Test dataset accuracy for KNN is {knn_accuracy}")
    if(show_full_report):
        print(classification_report(y_test_1, knn_res, target_names=target_names, zero_division=True))
    
    # Multinomial Naive Bayes
    mnbayes_w2v.fit(new_X_shuffle, y_train_1)
    nb_res = mnbayes_w2v.predict(new_X_test_shuffle)
    nb_accuracy = accuracy_score(y_test_1, nb_res)
    print(f"Test dataset accuracy for Multinomial Naive Bayes is {nb_accuracy}")
    if(show_full_report):
        print(classification_report(y_test_1, nb_res, target_names=target_names, zero_division=True))
    
    # SVM SVC
    svm_w2v.fit(new_X_shuffle, y_train_1)
    svm_res = svm_w2v.predict(new_X_test_shuffle)
    svm_accuracy = accuracy_score(y_test_1, svm_res)
    print(f"Test dataset accuracy for SVM is {svm_accuracy}")
    if(show_full_report):
        print(classification_report(y_test_1, svm_res, target_names=target_names, zero_division=True))


# In[ ]:




