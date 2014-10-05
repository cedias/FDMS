# -*- coding: utf-8 -*-
"""
Created on Sun Oct  5 08:46:56 2014

@author: charles
"""

def preprocess(review):
    regexP = ur"[,;.:\(\)\"]+" #regex of signs to take off
    review = re.sub(regexP, "", review)
    review = review.strip();#remove useless whitespace
    return review
    

positiveFile = open("pos.txt","r")
negativeFile = open("neg.txt","r")
testFile = open("testSentiment.txt","r")

cptLinePos = 0
cptLineNeg = 0
cptLineTest = 0
positive = []
negative = []
test = []

for pos in positiveFile:
    cptLinePos+=1
    positive.append(preprocess(pos))

for neg in negativeFile:
    cptLineNeg+=1
    negative.append(preprocess(neg))
    
for tes in testFile:
    cptLineTest+=1
    test.append(preprocess(tes)) 
    
print "Training set contains %d positive samples (%d%%) and %d negative samples (%d%%) - Test has %d samples" % (cptLinePos,cptLinePos/(0.0+cptLinePos+cptLineNeg)*100,cptLineNeg,cptLineNeg/(0.0+cptLinePos+cptLineNeg)*100, cptLineTest)
reviews = positive + negative
labels = np.concatenate((np.ones(len(positive)),-1*np.ones(len(negative))))


dictionary = corpora.Dictionary([text.split() for text in reviews])

print "taille du dictionnaire: %d" % len(dictionary)

stoplist = set()

stopfile = open("stopwords","r")
for line in stopfile:
    stoplist.add(line.strip())

upper = (len(alltxts)/3)*10000
print "upperbound = %d" % upper
lower = 0
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist  if stopword in dictionary.token2id]
once_ids = [tokenid for tokenid, docfreq in dictionary.dfs.iteritems() if docfreq < lower or docfreq > upper]
dictionary.filter_tokens(stop_ids) # remove stop words and words that appear only once
dictionary.compactify() # remove gaps in id sequence after words that were removed
print "taille du dictionnaire après diminution: %d" % len(dictionary)


## PROJECTION
texts = [[word for word in doc.split(" ") ]for doc in reviews]
corpus = [dictionary.doc2bow(text) for text in texts]

vecteurs = mtutils.corpus2csc(corpus, num_terms=len(dictionary), num_docs=len(labels))
vecteurs = vecteurs.T

res = feat.chi2(vecteurs, labels)

ids = [index for index in np.where(res == min(res[0]))[1]]
dictionary.filter_tokens(ids) # remove stop words and words that appear only once
dictionary.compactify() # remove gaps in id sequence after words that were removed
print "taille du dictionnaire après Chi2 diminution: %d" % len(dictionary)

vecteurs = mtutils.corpus2csc(corpus, num_terms=len(dictionary), num_docs=len(labels))
vecteurs = vecteurs.T

## PROJECTION
texts = [[word for word in doc.split(" ") ]for doc in reviews]
corpus = [dictionary.doc2bow(text) for text in texts]
vecteurs = mtutils.corpus2csc(corpus, num_terms=len(dictionary), num_docs=len(labels))
vecteurs  = vecteurs.T

##Perce: 0.84 mean
#classifier = lm.Perceptron(penalty=None, alpha=0.00001, fit_intercept=True, n_iter=2000, shuffle=True, verbose=0, eta0=1.0, n_jobs=7, class_weight='auto', warm_start=False)
#classifier.fit(vecteurs,labels)

#MultinomialNaiveBayes : 0.88 
#prior = np.array([0.25,0.75])
classifier = MultinomialNB()
classifier.fit(vecteurs,labels)

#SVM
#classifier = svm.SVC(class_weight="auto",max_iter=10000)
#classifier.fit(vecteurs,labels)

#scores = cv.cross_val_score(classifier, vecteurs,labels, cv=5)

#print scores.mean()

####Load & Test


texts = [[word for word in doc.split(" ") ]for doc in test]
corpusTest = [dictionary.doc2bow(text) for text in texts]

vecteursTest = mtutils.corpus2csc(corpusTest, num_terms=len(dictionary), num_docs=nblignesTest)

vecteursTest = vecteursTest.T

predicted = classifier.predict(vecteursTest)


#print to file
f = open("res.txt","w")

for label in predicted:
    if label == -1:
        f.write("C\n")
    else:
        f.write("M\n")

f.close()