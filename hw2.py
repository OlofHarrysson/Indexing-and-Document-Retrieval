from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import sys


def pause():
    programPause = input("Press the <ENTER> key to continue...")


def read_files(dir_path):
    data = []
    for subdir, dirs, files in os.walk(dir_path):
        for file in files:
            if file.startswith('.'): #Skip hidden files
                continue
            path = os.path.join(subdir, file)
            text = read_file(path)
            data.append(read_file(path))

    return data

def read_file(path): # Splits at blank space
    file = open(path, 'r')
    return file.read().split(' ')

def create_bool_rep(documents, query):
    bool_rep = []
    for doc in documents:
        query_vec = []
        for q in query:
            if q in doc:
                query_vec.append(1)
            else:
                query_vec.append(0)
        bool_rep.append(query_vec)

    return bool_rep


def create_tf(documents, query):
    td_rep = []
    for doc in documents:
        query_vec = []
        for q in query:
            nbr_q = doc.count(q)
            query_vec.append(nbr_q)

        sum_vec = sum(query_vec)
        if sum_vec != 0:
            query_vec = [float(i)/sum_vec for i in query_vec]
        td_rep.append(query_vec)

    return td_rep


def create_tfidf(documents, query):
    tfidf_rep = []

    for q in query:



    return tfidf_rep




################## START #######################

doc_path = 'cranfield/d'
documents = read_files(doc_path)

query_path = 'cranfield/q'
queries = read_files(query_path)

# bool_rep = create_bool_rep(documents, queries[0]) # One query, many documents
# tf_rep = create_tf(documents, queries[0])
tfidf_rep = create_tfidf(documents, queries[0])

# tfidf_vectorizer = TfidfVectorizer()
# tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
# print(tfidf_matrix)
sys.exit(1)


# print(documents[0])
# print(queries[0])
# sys.exit(1)


# prepare corpus
corpus = []
for d in range(1400):
    f = open("cranfield/d/"+str(d+1)+".txt")
    corpus.append(f.read())
# add query to corpus
for q in [1]:
    f = open("cranfield/q/"+str(q)+".txt")
    corpus.append(f.read())

# init vectorizer
tfidf_vectorizer = TfidfVectorizer()

# prepare matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

# compute similarity between query and all docs (tf-idf) and get top 10 relevant
sim = np.array(cosine_similarity(tfidf_matrix[len(corpus)-1], tfidf_matrix[0:(len(corpus)-1)])[0])
topRelevant = sim.argsort()[-10:][::-1]+1
print(topRelevant)




