# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 11:20:09 2014

@author: charles
"""
import codecs
import re
import string
from tools import *
from gensim import corpora
from gensim import matutils as mtutils
from sklearn import linear_model as lm
from sklearn import cross_validation as cv
from sklearn.naive_bayes import MultinomialNB
from sklearn import feature_selection as feat
from sklearn import svm
###########LOAD & FIT Learn

fname = "data/chiracLearn"
regexP = ur"[_,;.:'\"]+"
nblignes = compteLignes(fname)
print "nblignes = %d"%nblignes
alltxts = []
labs = []
s=codecs.open(fname, 'r','utf-8') # pour régler le codage
cpt=0
cptC=0
cptM=0

for i in range(nblignes):
    txt = s.readline()
    lab = re.sub(r"<[0-9]*:[0-9]*:(.)>.*","\\1",txt)
    txt = re.sub(r"<[0-9]*:[0-9]*:.>(.*)","\\1",txt)
    if lab.count('M') >0:
        labs.append(-1)
        alltxts.append(txt)
        cptM+=1
    else:
        cptC+=1
        labs.append(0)
        alltxts.append(txt)
    cpt += 1

print "read %d lines andf labels - %d%% of Chirac, %d%% of Mitterand" % (cpt,(cptC+0.0)/cpt*100, (cptM+0.0)/cpt*100)
labs = np.array(labs)

processed = []

for text in alltxts:
    txt = re.sub(regexP, "", text)
    processed.append(txt.strip())
    

dictionary = corpora.Dictionary([text.split() for text in processed])

print "taille du dictionnaire: %d" % len(dictionary)

stoplist = set()

#stopfile = open("stopwords","r")
#for line in stopfile:
#    stoplist.add(line.strip())

upper = (len(alltxts)/3)
print "upperbound = %d" % upper
lower = 5
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist  if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq < lower or docfreq > upper]
dictionary.filter_tokens(once_ids + stop_ids) # remove stop words and words that appear only once
dictionary.compactify() # remove gaps in id sequence after words that were removed
print "taille du dictionnaire après diminution: %d" % len(dictionary)


## PROJECTION
texts = [[word for word in doc.split(" ") ]for doc in processed]
corpus = [dictionary.doc2bow(text) for text in texts]

vecteurs = mtutils.corpus2csc(corpus, num_terms=len(dictionary), num_docs=nblignes)
labels = labs
vecteurs = vecteurs.T

res = feat.chi2(vecteurs, labels)

ids = [index for index in np.where(res == min(res[0]))[1]]
dictionary.filter_tokens(ids) # remove stop words and words that appear only once
dictionary.compactify() # remove gaps in id sequence after words that were removed
print "taille du dictionnaire après Chi2 diminution: %d" % len(dictionary)

vecteurs = mtutils.corpus2csc(corpus, num_terms=len(dictionary), num_docs=nblignes)
labels = labs
vecteurs = vecteurs.T

## PROJECTION
texts = [[word for word in doc.split(" ") ]for doc in processed]
corpus = [dictionary.doc2bow(text) for text in texts]
vecteurs = mtutils.corpus2csc(corpus, num_terms=len(dictionary), num_docs=nblignes)
vecteurs = vecteurs.T

##Perce: 0.84 mean
#classifier = lm.Perceptron(penalty=None, alpha=0.00001, fit_intercept=True, n_iter=2000, shuffle=True, verbose=0, eta0=1.0, n_jobs=7, class_weight='auto', warm_start=False)
#classifier.fit(vecteurs,labels)

#MultinomialNaiveBayes : 0.88 
prior = np.array([0.25,0.75])
classifier = MultinomialNB(fit_prior=True,class_prior=prior, alpha=0.56)
classifier.fit(vecteurs,labels)

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


processed2 = []

for text in alltxts:
    txt = re.sub(regexP, "", text)
    processed2.append(txt.strip())

texts = [[word for word in doc.split(" ") ]for doc in processed2]
corpusTest = [dictionary.doc2bow(text) for text in texts]


vecteursTest = mtutils.corpus2csc(corpusTest, num_terms=len(dictionary), num_docs=nblignesTest)

vecteursTest = vecteursTest.T

predicted = classifier.predict(vecteursTest)

lissage = True
if lissage:
    window = 3
    for i in xrange(0,len(predicted)):
       somme = 0
       for j in xrange(i-window,i+window):
           if j<0 or j>=len(predicted):
               continue
           if predicted[j] == -1:
               somme+=predicted[j]
           else:
               somme+=1
              
       if somme < 0:
           predicted[i] = -1
       elif somme > 0:
            predicted[i] = 0
       else:
           continue
       
    for i in xrange(0,len(predicted)):
         if i == 0 or i == len(predicted)-1:
             continue
         
         if predicted[i-1] ==  predicted[i+1] and predicted[i] != predicted[i-1]:
             predicted[i] = predicted[i-1]
       


#print to file
f = open("res.txt","w")

for label in predicted:
    if label == -1:
        f.write("M\n")
    else:
        f.write("C\n")

f.close()


