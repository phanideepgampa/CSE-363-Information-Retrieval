import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from pprint import pprint
data ="20news-bydate-train"
directories = os.listdir(data)
corpus =[]
stop_words= set(stopwords.words("english"))
porter = PorterStemmer()
dire=directories[12]
for sd in os.listdir(data+"/"+dire+"/"):
        f = open(data+"/"+dire+"/"+sd)
        temp = f.read()
        f.close()
        words = word_tokenize(temp)
        words = [w.lower() for w in words] #case normalisation
        words = [w for w in words if not w in stop_words] # removing stop words
        words = [w for w in words if w.isalpha()] #removing non alphabets
        stemmed = [porter.stem(w) for w in words] # stemming using porter stemmer
        corpus.append(stemmed)
        

index = {}
term_freq = {}
for d in range(0,len(corpus)):
        temp_freq = {}
        for term in corpus[d]:
                if term not in temp_freq:
                        temp_freq[term]=1
                else:
                        temp_freq[term]+=1
        for k in temp_freq:
                if k not in index:
                        index[k]=[]
                        term_freq[k]=0
                index[k].append((d,temp_freq[k]))
                term_freq[k]+=temp_freq[k]
keys=index.keys()
index2={}
for k in keys:
        index2[(k,term_freq[k])]= index[k]

with open('index12.txt', 'wt') as out:
    pprint(index2, stream=out)

