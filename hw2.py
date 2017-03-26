from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os
import sys
import math
from scipy.spatial import distance


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

    sum_doc = len(documents)
    query_count = []
    for q in query:
        nbr_q_in_docs = sum(doc.count(q) for doc in documents)
        query_count.append(nbr_q_in_docs)

    log_query_score = []
    for q in query_count:
        if q == 0:
            log_query_score.append(0)
        else:
            log_query_score.append(math.log(q/sum_doc))


    tf = create_tf(documents, query)

    tfidf_score = []
    for doc_s in tf: # For every score in tf, multiply it with the corresponding log_q_s
        new_doc_s = []
        for i, q_s in enumerate(doc_s):
            new_doc_s.append(q_s * log_query_score[i])

        tfidf_score.append(new_doc_s)

    return tfidf_rep

def comp_euc_dist(v1, v2):
    return distance.euclidean(v1, v2)

################## START #######################

doc_path = 'cranfield/d'
documents = read_files(doc_path)

query_path = 'cranfield/q'
queries = read_files(query_path)

bool_reps = []
for q in queries:
    bool_reps.append(create_bool_rep(documents, q))

tf_reps = []
for q in queries:
    tf_reps.append(create_tf(documents, q))

tfidf_reps = []
for q in queries:
    tfidf_reps.append(create_tfidf(documents, q))

bool_rep = bool_reps[0]
euc_distances = []
for i, d_outer in enumerate(bool_rep):
    for d_inner in bool_rep:
        euc_distances.append(comp_euc_dist(bool_rep[i], d_inner))
