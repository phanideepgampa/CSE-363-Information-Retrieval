import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pprint import pprint
from sklearn.feature_extraction.text import CountVectorizer
import collections

def process_text(text,stop_words):
    words = word_tokenize(text)
    words = [w.lower() for w in words] #case normalisation
    words = [w for w in words if not w in stop_words] # removing stop words
    words = [w for w in words if w.isalpha()] #removing non alphabets
    stemmed = [porter.stem(w) for w in words] # stemming using porter stemmer
    return stemmed


data ="20news-bydate-train"
directories = os.listdir(data)
corpus =[]
stop_words= set(stopwords.words("english"))
porter = PorterStemmer()
dire=directories[0]
vectorizer = CountVectorizer()
doc_ids = os.listdir(data+"/"+dire+"/")
for sd in doc_ids:
    f = open(data+"/"+dire+"/"+sd)
    temp = f.read()
    f.close()
    corpus.append(collections.Counter(process_text(temp,stop_words))) #individual document probabilities
    
documents= sum(corpus,collections.Counter()) # whole collection probabilities
"""
Gives the top 10 documents with decreasing probabilities according to the formula
P(d|q) proportional to product of[((1− λ)P(t|Mc) + λP(t|Md))] for each t in q 
with λ=1/2
"""
def rank_documents(query):
    query=process_text(query,stop_words)
    result = []
    index=0
    for d in corpus:
        temp=1
        for q in query:
            temp=temp*((d[q]/len(d))+(documents[q]/len(documents)))
        result.append((temp,doc_ids[index]))
        index+=1
    result.sort(reverse=True)
    return result[:10]

"""
Call rank_documents on any query as given below
2 examples are given by taking some sentences from the documents directly
"""

print(rank_documents("AMERICAN ATHEIST PRESS")) #answer: docid= 49960
print(rank_documents("hatred, disrespect, cowardness, and dishonesty.")) # answer: docid= 54200
