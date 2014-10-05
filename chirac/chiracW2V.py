# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 19:16:18 2014

@author: charles
"""

import codecs
import re
import string
from tools import *
import gensim
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


fname = "data/chiracLearn"

nblignes = compteLignes(fname)
print "nblignes = %d"%nblignes

alltxts = []
labs = np.ones(nblignes)
s=codecs.open(fname, 'r','utf-8') # pour régler le codage
splitters = set(string.punctuation)
cpt = 0

for i in range(nblignes):
    txt = s.readline()
    
    #print txt
    txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
    out = txt.translate(string.maketrans("",""), string.punctuation)
    alltxts.append(out)
    
    cpt += 1
print "loaded %d quotes and labels" % cpt 

alltxtsmul = alltxts

model = gensim.models.Word2Vec(alltxtsmul, size=200, window=5, min_count=0, workers=4,sample=0)

#reduced_data = PCA(n_components=2).fit_transform(model.syn1)


km = KMeans(n_clusters=3, init='k-means++', n_init=3, max_iter=5000, tol=0.000001, precompute_distances=False, verbose=0, random_state=None, copy_x=True, n_jobs=1)
res = km.fit_predict(model.syn1)


len0 = len(np.where(res == 0)[0])
len1 = len(np.where(res == 1)[0])
len2 = len(np.where(res == 2)[0])

maximum = max(len0,len1,len2)

words = np.array(model.index2word)

if maximum == len0:
    featureW = np.concatenate((words[np.where(res == 1)],words[np.where(res == 2)]))
elif maximum == len1:
    featureW = np.concatenate((words[np.where(res == 0)],words[np.where(res == 2)]))
else:
    featureW = np.concatenate((words[np.where(res == 0)],words[np.where(res == 1)]))

print featureW.shape


#reduced_data = PCA(n_components=2).fit_transform(model.syn1)

#plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)



# -*- coding: utf-8 -*-

import codecs
import re
from tools import *
from gensim import corpora
from gensim import matutils as mtutils
from sklearn import linear_model as lm
from sklearn import cross_validation as cv
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm



###########LOAD & FIT Learn

fname = "data/chiracLearn"

nblignes = compteLignes(fname)
print "nblignes = %d"%nblignes

alltxts = []
labs = []
s=codecs.open(fname, 'r','utf-8') # pour régler le codage

cpt = 0
cptC=0
cptM=0
for i in range(nblignes):
    txt = s.readline()
    #print txt

    lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
    txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)

    #assert(lab == "C" or lab == "M")

    if lab.count('M') >0:
        labs.append(-1)
        alltxts.append(txt)
        cptM+=1
    else:
        #if len(txt)>250:
        cptC+=1
        labs.append(0)
        alltxts.append(txt)

    cpt += 1
     #if cpt %1000 ==0:
       # print cpt

labs = np.array(labs)    
stoplist = set()
#stoplist.add()

## DICO
splitters = u';|,|\*|\.| |\'|'

dictionary = corpora.Dictionary(re.split(splitters, doc.lower()) for doc in alltxts)

print len(dictionary)
upper = len(alltxts)*2
lower = 2

stop_ids = [dictionary.token2id[stopword] for stopword in stoplist  if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq < lower or docfreq > upper]
feature_ids = [dictionary.token2id[feat] for feat in featureW  if feat in dictionary.token2id]
dictionary.filter_tokens(good_ids=feature_ids) # remove stop words and words that appear only once
dictionary.compactify() # remove gaps in id sequence after words that were removed

print "dic-len"
print len(dictionary)

## PROJ

texts = [[word for word in re.split(splitters, document) if word not in stoplist]  for document in alltxts]
corpus = [dictionary.doc2bow(text) for text in texts]

## exemple de doc
# corpus[0]
# avec les mots
print [dictionary[i] for i,tmp in corpus[0]]

vecteurs = mtutils.corpus2csc(corpus, num_terms=len(dictionary), num_docs=nblignes)
labels = labs

vecteurs = vecteurs.T
##Perce: 0.84 mean
classifier = lm.Perceptron(penalty=None, alpha=0.00001, fit_intercept=True, n_iter=2000, shuffle=True, verbose=0, eta0=1.0, n_jobs=7, class_weight=None, warm_start=False)
classifier.fit(vecteurs,labels)

#MultinomialNaiveBayes : 0.88 
#classifier = MultinomialNB()
#classifier.fit(vecteurs,labels)

#SVM
#classifier = svm.SVC(class_weight="auto",max_iter=10000)
#classifier.fit(vecteurs,labels)

scores = cv.cross_val_score(classifier, vecteurs,labels, cv=5)

print scores.mean()

####Load & Test
fname = "data/chiracTest"

nblignesTest = compteLignes(fname)
print "nblignes = %d"%nblignes

alltxts = []

s=codecs.open(fname, 'r','utf-8') # pour régler le codage

cpt = 0
for i in range(nblignesTest):
    txt = s.readline()
    txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
    alltxts.append(txt)

    cpt += 1
    #if cpt %1000 ==0:
        # print cpt


texts = [[word for word in re.split(splitters, document.lower()) if word not in stoplist]  for document in alltxts]
corpusTest = [dictionary.doc2bow(text) for text in texts]


vecteursTest = mtutils.corpus2csc(corpusTest, num_terms=len(dictionary), num_docs=nblignesTest)

vecteursTest = vecteursTest.T

predicted = classifier.predict(vecteursTest)


#print to file
f = open("res.txt","w")

for label in predicted:
    if label == -1:
        f.write("M\n")
    else:
        f.write("C\n")

f.close()











