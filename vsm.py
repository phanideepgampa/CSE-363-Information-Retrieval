"""It asks you to enter a search query, and then returns all documents
matching the query, in decreasing order of cosine similarity,
according to the vector space model.

/////////////python vsm.py///////////// in the command line
"""

from collections import defaultdict
import os
import math
import sys
from functools import reduce

data_dir ="20news-bydate-train/alt.atheism/"
files = os.listdir(data_dir)

document_filenames = {}

for i in range(0,len(files)):
        document_filenames[i]= data_dir+files[i]

# The size of the corpus
N = len(document_filenames)

# dictionary: a set to contain all terms (i.e., words) in the document
# corpus.
dictionary = set()

postings = defaultdict(dict)

document_frequency = defaultdict(int)

length = defaultdict(float)

# The list of characters (mostly, punctuation) we want to strip out of
# terms in the document.
characters = " .,!#$%^&*();:\n\t\\\"?!{}[]<>"

def main():
    initialize_terms_and_postings()
    initialize_document_frequencies()
    initialize_lengths()
    while True:
        do_search()

def initialize_terms_and_postings():
    global dictionary, postings
    for id in document_filenames:
        f = open(document_filenames[id],'r')
        document = f.read()
        f.close()
        terms = tokenize(document)
        unique_terms = set(terms)
        dictionary = dictionary.union(unique_terms)
        for term in unique_terms:
            postings[term][id] = terms.count(term) # the value is the
                                                   # frequency of the
                                                   # term in the
                                                   # document

def tokenize(document):
    terms = document.lower().split()
    return [term.strip(characters) for term in terms]

def initialize_document_frequencies():
    global document_frequency
    for term in dictionary:
        document_frequency[term] = len(postings[term])

def initialize_lengths():
    """Computes the length for each document."""
    global length
    for id in document_filenames:
        l = 0
        for term in dictionary:
            l += imp(term,id)**2
        length[id] = math.sqrt(l)

def imp(term,id):
    """Returns the importance of term in document id.  If the term
    isn't in the document, then return 0."""
    if id in postings[term]:
        return postings[term][id]*inverse_document_frequency(term)
    else:
        return 0.0

def inverse_document_frequency(term):
    """Returns the inverse document frequency of term.  Note that if
    term isn't in the dictionary then it returns 0, by convention."""
    if term in dictionary:
        return math.log(N/document_frequency[term],2)
    else:
        return 0.0

def do_search():
    """Asks the user what they would like to search for, and returns a
    list of relevant documents, in decreasing order of cosine
    similarity."""
    query = tokenize(input("Search query >> "))
    if query == []:
        sys.exit()
    # find document ids containing all query terms.  Works by
    # intersecting the posting lists for all query terms.
    relevant_document_ids = intersection(
            [set(postings[term].keys()) for term in query])
    if not relevant_document_ids:
        print("No documents matched all query terms.")
    else:
        scores = sorted([(id,similarity(query,id))
                         for id in relevant_document_ids],
                        key=lambda x: x[1],
                        reverse=True)
        print("Score: filename")
        for (id,score) in scores:
            print(str(score)+": "+document_filenames[id])

def intersection(sets):
    return reduce(set.intersection, [s for s in sets])

def similarity(query,id):
    similarity = 0.0
    for term in query:
        if term in dictionary:
            similarity += inverse_document_frequency(term)*imp(term,id)
    similarity = similarity / length[id]
    return similarity

if __name__ == "__main__":
    main()
