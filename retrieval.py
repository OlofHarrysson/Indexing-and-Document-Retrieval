
# coding: utf-8

# In[80]:

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial.distance import euclidean
from sklearn.metrics import precision_score, recall_score, f1_score
import os


# In[10]:

# prepare corpus
corpus = []
for d in range(1400):
    f = open("C:\\Users\\Rita\\Documents\\cranfild\\"+"./d/"+str(d+1)+".txt")
    corpus.append(f.read())


# In[24]:

# prepare queries
queries=[]

path='C:\\Users\\Rita\\Documents\\cranfild\\q\\'
os.chdir(path)
dir_work = os.listdir(path)
a = len (dir_work)

for q in range(0, a):
    f = open(path+str(q+1)+".txt")
    queries.append(f.read())


# In[ ]:

#
#preprocessing of documents
#


# In[60]:

#tokenization of documents
doc_tokens=[nltk.word_tokenize(sent) for sent in corpus]

#delete punct
nopunc_doc=[]
for token in doc_tokens:
    temp=[]
    for t in token:
        if t not in punctuation:
            temp.append(t)
    nopunc_doc.append(temp)

# delete stop-words
st_words_doc=[]
stops = stopwords.words('english')

for token in nopunc_doc:
    temp=[]
    for t in token:
        if t not in stops:
            temp.append(t)
    st_words_doc.append(temp)    

#stemming
stemmer = PorterStemmer()
stems_d=[]

for token in st_words_doc:
    temp1=[]
    for t in token:
        temp1.append(stemmer.stem(t))
    stems_d.append(temp1)


# In[ ]:

#
#preprocessing queries
#


# In[35]:

#tokenization of queries
q_tokens=[nltk.word_tokenize(sent) for sent in queries]

#delete punct & stops
nopunc_st_q=[]
for token in q_tokens:
    temp=[]
    for t in token:
        if (t not in punctuation) & (t not in stops):
            temp.append(t)
    nopunc_st_q.append(temp)

#stemming
stems_q=[]

for token in nopunc_st_q:
    temp1=[]
    for t in token:
        temp1.append(stemmer.stem(t))
    stems_q.append(temp1)


# In[ ]:

#
#preprocessing of corpus
#


# In[57]:

#tokenization
corp_t=[nltk.word_tokenize(sent) for sent in corpus]

corpus_all=[]
for token in corp_t:
    for t in token:
        if t not in corpus_all:
            corpus_all.append(t)

#delete punct
corp_nopunc_st=[]
for token in corpus_all:
    if (token not in punctuation) & (token not in stops):
        corp_nopunc_st.append(token) 

#stemming
stems_corp=[stemmer.stem(t) for t in corp_nopunc_st]


# In[63]:

#binarization + term freq of documents
binary_doc=np.zeros([len(stems_d), len(stems_corp)])
freq_doc=np.zeros([len(stems_d), len(stems_corp)])

for i in range(0, len(stems_d)):
    for j in range(0, len(stems_d[i])):
        if stems_d[i][j] in stems_corp:
            idx=stems_corp.index(stems_d[i][j])
            binary_doc[i, idx]=1
            freq_doc[i, idx]+=1
    if len(stems_d[i])==0:
        print('aaa')
    else:
        freq_doc[i, :]= freq_doc[i, :]/len(stems_d[i])


# In[64]:

#binarization + term freq of queries
binary_q=np.zeros([len(stems_q), len(stems_corp)])
freq_q=np.zeros([len(stems_q), len(stems_corp)])

for i in range(0, len(stems_q)):
    for j in range(0, len(stems_q[i])):
        if stems_q[i][j] in stems_corp:
            idx=stems_corp.index(stems_q[i][j])
            binary_q[i, idx]=1
            freq_q[i, idx]+=1
    if len(stems_q[i])==0:
        print('aaa')
    else:   
        freq_q[i, :]= freq_q[i, :]/len(stems_q[i])


# In[70]:

#TF-IDF
tfidf_vectorizer = TfidfVectorizer()
 
# prepare matrix docs+queries
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
tfidf_matrix_q = tfidf_vectorizer.transform(queries)


# In[71]:

#
# preparing ground truth
#


# In[85]:

#read files
ground_truth=[]

path='C:\\Users\\Rita\\Documents\\cranfild\\r\\'
os.chdir(path)
dir_work = os.listdir(path)
a = len (dir_work)

for r in range(0, a):
    f = open(path+str(r+1)+".txt")
    temp=f.read()
    st=temp.split('\n')
    stt=[]
    for s in st:
        if s!='':
            stt.append(int(s))
    ground_truth.append(stt[:10])


# In[79]:

#distance measure

def distance_measure(docs, sample):
    cos_d=np.array(cosine_similarity(sample, docs), [0])
    top_cos = cos_d.argsort()[-10:][::-1]+1
    euc_d = np.array(euclidean(sample, docs), [0])
    top_eu = euc_d.argsort()[-10:][::-1]+1
    return top_cos, top_eu


# In[86]:

#metrics measure

def metrics_measure(predicted, ground_truth):
    if len(predicted)==len(ground_truth):
        pres=precision_score(ground_truth, predicted)
        rec=recall_score(ground_truth, predicted)
        f1=f1_score(ground_truth, predicted)
    else:
        a=min(len(predicted), len(ground_truth))
        pres=precision_score(ground_truth[:a], predicted)
        rec=recall_score(ground_truth[:a], predicted)
        f1=f1_score(ground_truth[:a], predicted)
    return pres, rec, f1                     


# In[ ]:

#running calculating relevance scores and metrics, 0 - cos, 1 - euc
#0 - precision, 1 - recall, 2 - f1
#for binary

scores_matrix_b=np.zeros([len(queries), 2])
metrics_matrix_b=np.zeros([len(queries), 3])

for i in range(0, len(queries)):
    scores_matrix_b[i, 0], scores_matrix_b[i, 1] = distance_measure(, sample)


# In[8]:

# init vectorizer    
tfidf_vectorizer = TfidfVectorizer()
 
# prepare matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

 
# compute similarity between query and all docs (tf-idf) and get top 10 relevant
sim = np.array(cosine_similarity(tfidf_matrix[len(corpus)-1], tfidf_matrix[0:(len(corpus)-1)])[0])
topRelevant = sim.argsort()[-10:][::-1]+1
#print(topRelevant)


# In[ ]:



