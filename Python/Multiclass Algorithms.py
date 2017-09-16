import pandas as pd
import numpy as np
from pandas import DataFrame, read_csv
import random as rd

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

import datetime


file = 'balance scale.csv'
data = DataFrame()
data = read_csv(file, header=None, index_col=None, delimiter=',')



label = data[0]



list = rd.sample(range(0, 625), 187)
sortedList = sorted(list)



testData = data.iloc[sortedList,:]
testLabel = data[0][sortedList]



data[0][sortedList] = 'None'



trainData = data.loc[~data.index.isin(sortedList)]
trainLabel = data[0][data[0] != 'None']


testData = testData.drop([0], 1)
trainData = trainData.drop([0], 1)



testLabel = testLabel.apply(int)
trainLabel = trainLabel.apply(int)



def TrainModelWithOnevsRest(algoritm, trainData, testData, trainLabel):
    prediction = OneVsRestClassifier(algoritm).fit(trainData, trainLabel).predict(testData)
    return prediction

def TrainModelWithOnevsOne(algoritm, trainData, testData, trainLabel):
    prediction = OneVsOneClassifier(algoritm).fit(trainData, trainLabel).predict(testData)
    return prediction

def CalculateAcc(testLabel, prediction):
    acc = accuracy_score(testLabel, prediction)
    return acc

def PrintLog(text):
    print  str(datetime.datetime.today()) + " " + str(text)

def RemoveString(txt):
    start = txt.find( '(' )
    end = txt.find( ')' )
    if start != -1 and end != -1:
        result = txt[start+1:end]
        txt = txt.replace(result, '')
    return txt.replace('(', '').replace(')', '')



algoritms = [LinearSVC(random_state=0), svm.SVC(), GaussianNB(), tree.DecisionTreeClassifier(), RandomForestClassifier(), 
             BernoulliNB(), MultinomialNB()]

PrintLog("****** This Results are OnevsRestClassifier ******")
for algoritm in algoritms:
    pred = TrainModelWithOnevsRest(algoritm, trainData, testData, trainLabel)
    acc = CalculateAcc(testLabel, pred)
        
    algoritmName = RemoveString(str(algoritm))
    PrintLog(str(algoritmName) + " " +  "--" + " " + "Accuracy : " + str(acc))



PrintLog("****** This Results are OnevsOneClassifier ******")
for algoritm in algoritms:
    pred = TrainModelWithOnevsOne(algoritm, trainData, testData, trainLabel)
    acc = CalculateAcc(testLabel, pred)
        
    algoritmName = RemoveString(str(algoritm))
    PrintLog(str(algoritmName) + " " +  "--" + " " + "Accuracy : " + str(acc))






