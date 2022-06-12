#!pip install matplotlib
#!pip install statsmodels
#!pip install pydataset
#!pip install  sklearn
from pickle import TRUE
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd
import seaborn as sns
import math
import statsmodels.api as sm                     
import random
import warnings
from sklearn.model_selection import train_test_split
from sklearn import datasets
from numpy import genfromtxt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import minmax_scale
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.preprocessing import minmax_scale
plt.style.use("seaborn")
pd.options.mode.chained_assignment = None  # default='warn'
plt.style.use("seaborn")
np.random.seed(42)

warnings.filterwarnings('ignore')









def accuracy(gene1,x,bb,y):
    
    start = 1
    end = 10
    width = end - start
    model = (bb - bb.min())/(bb.max() - bb.min()) * width + start
    for i in range(len(gene1)) :
        model[x[i]]=gene1[i]*model[x[i]]
        
    knn_model = KNeighborsRegressor(n_neighbors=3)
    knn_model.fit(model, y)
    train_preds = knn_model.predict(model)
    train_preds=pd.DataFrame(train_preds, columns = ['knnPreds'])
#     print(type(kl))
    mse = mean_squared_error(y, train_preds)
    rmse = sqrt(mse)
    mytable=model.join(y)
    mytable=mytable.join(train_preds)
#     return mytable 
    
    count=0
    alltrueGuesses=0
    for index, row in mytable.iterrows():
        count+=1
        c1=approx(row['knnPreds'])      #predicted output
        c2=approx(row['y'])      #real output
        if(c1==c2 ):
            alltrueGuesses+=1

 #         fsData['x1'][index]=50
# temp=  (truepos/allpos)*100  
    temp2=(alltrueGuesses/count)*100
    return temp2

def approx(x):
    c=x %1
    if c>=0.5 :
        x=int(x)
        x=x+1
    else:
        x=int(x)
        
    return x
    
def genx2(n):
    s=[]
    for i in range(n):
        temp=i+1
        s.append("x"+str(temp))
    return s
def populateFE(generationSize,m):
    z=[]
    for i in range(1):
        temp=[]
        for j in range(m):
            f=random.random()
            f = math.ceil(f * 100) 
            f=f%2
            c=random.random()
            if f==1:
                c = math.ceil(c * 100) /10
            else:
                c = math.ceil(c * 100) /100 
            
            temp.append(c)
        z.append(temp)
        
    return z

# print(populate(2,10))

def fitnessFE(gene1):
    
 print(gene1)
 return int(input())


def crossoverFE(gene1,gene2,p):
    res=[]
    for i in range(len(gene1)):
        if(i<p):
            res.append(gene1[i])
        else:
            res.append(gene2[i])
    return res
    

    
def mutateFE(gene,p):
    res=[]
    for i in range(len(gene)):
        if(i<p):
            res.append(gene[i])
        else:
            f=random.random()
            f = math.ceil(f * 100) 
            f=f%2
            c=random.random()
            if f==1:
                c = math.ceil(c * 100) /10
            else:
                c = math.ceil(c * 100) /100 
            res.append(gene[i]+c)
    return res

def selectFE(pop):
    res=[]
    rmsvalues=[]
    test=[]
    for i in range(len(pop)) :
#         here we are ........................................................
        rmsvalues.append(fitnessFE(pop[i]))
    if(TRUE):
        for i in range(1):
            idxmin=-1
            minval=-1000
            for j in range(len(rmsvalues)):
                if rmsvalues[j]>=minval:
                    idxmin=j
                    minval=rmsvalues[j]
            res.append(pop[idxmin])
            test.append(rmsvalues[idxmin])
            rmsvalues[idxmin]=-10000
        
         
    return res
    

def GeneticAlgorithmFE(generationSize,numberGenerations,elite):
    
    m=1
    pop=populateFE(generationSize,m)
    history=[]
    genehistory=[]
    for i in range(numberGenerations):
        parents=selectFE(pop)
        rms=fitnessFE(parents[0])#edit adj r2        
        history.append(rms)
        genehistory.append(parents[0])
        print('Generation ',i, 'LRValue ', rms)
      
        child=[]
        for t in parents:
            child.append(t)
    
              
        rest=generationSize-elite 
        rpop=populateFE(rest,m)
        for q in rpop:
            child.append(q)           
        pop=child
    if(True):
        bestgeneidx=-1
        bestgenerms=1000
        for i in range(len(history)):
            if(history[i]<=bestgenerms):
                bestgeneidx=i
                bestgenerms=history[i]        
    else:
        
        bestgeneidx=-1
        bestgenerms=1000
        for i in range(len(history)):
            if(history[i]<=bestgenerms):
                bestgeneidx=i
                bestgenerms=history[i]
    
    print('Best generation value : ',bestgenerms)
    plt.plot(range(len(history)), history, color="skyblue")
    plt.xlabel("Generation Number ")
    plt.ylabel("Accuracy")
    plt.show()
    return genehistory[bestgeneidx]
GeneticAlgorithmFE(2,10,0.5)