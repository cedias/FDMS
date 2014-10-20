# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 16:19:04 2014

@author: 3060501
"""

#user id | item id | rating | timestamp.

import numpy as np
import pandas as pd


def biais_hist(uid,iid,filename="recoBias.pdf"):
    plt.figure()
    plt.subplot(211)
    plt.hist(uid,100)
    plt.title('Utilisateur')
    plt.subplot(212)
    plt.hist(iid,100)
    plt.title('Item')
    plt.savefig(filename)

def meanBias():
    meanErrU=0
    meanErrI=0
    
    for i in xrange(1,6):
        sumErrU = 0
        sumErrI = 0
        fbase = open("ml-100k/u"+str(i)+".base")
        ftest = open("ml-100k/u"+str(i)+".test")
        
        dataBase = np.loadtxt(fbase, dtype=int) #All the data's
        dataTest = np.loadtxt(ftest, dtype=int) #All the data's
        
        frame = pd.DataFrame(dataBase,columns=['uid', 'iid','rating','timestamp'],dtype="int")
        
        uid_rating = frame[['uid','rating']]
        iid_rating = frame[['iid','rating']]
       
        uid_biais = uid_rating.groupby('uid').mean().to_dict()["rating"]
        iid_biais = iid_rating.groupby('iid').mean().to_dict()["rating"]
        
        for row in xrange(0,len(dataTest)):
            uid = dataTest[row][0]
            iid = dataTest[row][1]
            rating = dataTest[row][2]
            try:
                scoringI = iid_biais[iid]
                scoringU = uid_biais[uid]
            except:
                scoringI = 0
                scoringU = 0
                
            errU = (scoringU-rating)**2
            errI = (scoringI-rating)**2  
            sumErrU += errU
            sumErrI += errI
            
        #biais_hist(uid_biais.values(),iid_biais.values())
        
        sumErrU/=len(dataTest)
        sumErrI/=len(dataTest)
        meanErrU += sumErrU
        meanErrI += sumErrI
        print "Set numero %d - MSE_User: %f - MSE_Item: %f" %(i,sumErrU,sumErrI)
        
    print "Moyenne CV 5-FOLD MSE_User: %f - MSE_Item: %f" %(meanErrU/5,meanErrI/5)
        

         
        #uid_biais[uid_biais.index == 3]


def similarityUsers():
   
    
    for i in xrange(1,6):
        fbase = open("ml-100k/u"+str(i)+".base")
        ftest = open("ml-100k/u"+str(i)+".test")
        
                
        
        dataBase = np.loadtxt(fbase, dtype=int) #All the data's
        dataTest = np.loadtxt(ftest, dtype=int) #All the data's  
        frame = pd.DataFrame(dataBase,columns=['uid', 'iid','rating','timestamp'],dtype="int")  
        uid_rating = frame[['uid','rating']]
        iid_rating = frame[['iid','rating']]
        rating = frame[['uid','iid','rating']]
        rating.sort(['uid'])
        
        uid_biais = uid_rating.groupby('uid').mean().to_dict()["rating"]
        iid_biais = iid_rating.groupby('iid').mean().to_dict()["rating"]
        userDiff = np.zeros(max(iid_biais.keys()))
                        
        
        for ind,row in rating.iterrows():
            uid = row["uid"]
            iid = row["iid"]
            rating = row["rating"]
            userDiff[uid] += rating - uid_biais[uid]
            
        
                    
        
        print "%d/5" % i 
            
            
            
            
            
for i in xrange(1,6):
    fbase = open("ml-100k/u"+str(i)+".base")
    ftest = open("ml-100k/u"+str(i)+".test")
    
    
    dataBase = np.loadtxt(fbase, dtype=int) #All the data's
    dataTest = np.loadtxt(ftest, dtype=int) #All the data's  
    frame = pd.DataFrame(dataBase,columns=['uid', 'iid','rating','timestamp'],dtype="int")  
    uid_rating = frame[['uid','rating']]
    iid_rating = frame[['iid','rating']]
    rating = frame[['uid','iid','rating']]
    rating.sort(['uid'])
    
    uid_biais = uid_rating.groupby('uid').mean().to_dict()["rating"]
    iid_biais = iid_rating.groupby('iid').mean().to_dict()["rating"]
    userDiff = np.zeros(max(iid_biais.keys()))
                    
    
    for ind,row in rating.iterrows():
        uid = row["uid"]
        iid = row["iid"]
        rating = row["rating"]
        userDiff[uid] += rating - uid_biais[uid]
        
    
                
    
    print "%d/5" % i 
        

