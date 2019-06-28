
# coding: utf-8

import os
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import *
import numpy as np


sklearn_tfidf = TfidfVectorizer()


def remove_stopwords(doc):
    l_stopwords = set(stopwords.words("english"))
    words = word_tokenize(doc)
    words = [w.lower() for w in words] #case normalisation
    words = [w for w in words if not w in l_stopwords] # removing stop words
    words = [w for w in words if w.isalpha()] #removing non alphabets
    return words


 
def remove_stop(example_sent):
    stemmer = PorterStemmer()
    word_tokens = remove_stopwords(example_sent)
    singles = [stemmer.stem(plural) for plural in word_tokens]
    return singles

all_docs = []
doc_ids = []
for root, dirs, files in os.walk("20news-bydate-train/alt.atheism"):
    for name in files:
        f=open(os.path.join(root, name))
        all_docs+=remove_stop(f.read())
        f.close()
        doc_ids.append(name)

sklearn_representation = sklearn_tfidf.fit_transform(all_docs).toarray()

def get_vectors(list_sen):
    return sklearn_tfidf.fit_transform(list_sen).toarray()

def get_vector_for(doc_id, dataset):
    return dataset[doc_id]



def find_min_ind(point, kmeans):
    min_ind = 0
    min_dist = np.linalg.norm(point - kmeans[0])
    for i in range(1, len(kmeans)):
        dist = np.linalg.norm(point - kmeans[i])
        if dist < min_dist:
            min_dist = dist
            min_ind = i
    return min_ind




def find_centroid(vector):
    sum = 0
    sum += np.sum(vector, axis = 0)
    sum /= len(vector)
    return sum




def single_loop(dataset, kmeans,k):
    bin = [[] for i in range(len(kmeans))]
    for j,point in enumerate(dataset):
        min_ind = find_min_ind(point, kmeans)
        bin[min_ind].append(j)
#         print(min_ind, " RAg")
    kmeans_new = []
    for i in bin:
        new_vector = []
        for j in i:
            new_vector.append(get_vector_for(j, dataset))
        kmeans_new.append(find_centroid(new_vector))    
    # print(np.sum(np.sum(np.array(kmeans_new) - np.array(kmeans), axis = 1)))
    return kmeans_new, bin


def kmeans_algo(k, iterations,dataset):
    kmeans = dataset[:k]
#     kmeans = random.sample(list(dataset), k)
    print(kmeans)
    for i in range(iterations):
        print("Iteration:"+str(i))
        kmeans, bin = single_loop(dataset, kmeans,k)
    print(kmeans)
    return kmeans, bin

print("Enter Number of clusters:")
k = int(input())
print("Enter Number of iterations:")
iterations = int(input())
kmeans, bin = kmeans_algo(k, iterations,sklearn_representation)

