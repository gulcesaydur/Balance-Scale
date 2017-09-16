from pandas import DataFrame, read_csv

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

import numpy as np
import datetime

class BalanceData:

    def GetFile(filepath):
        dataFrame = DataFrame()
        dataFrame = read_csv(filepath, header=None, index_col=None, delimiter=',')
        data = dataFrame.dropna()
        return data
    
    def SplitData(data, splitCount):
        splitData = np.split(data, [splitCount], axis=0)
        data1 = splitData[0]
        data2 = splitData[1]
        return data1, data2
    
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
    
    data = GetFile("Balance Scale.csv")
    label = GetFile("Balance Scale Label.csv")
    
    trainData, testData = SplitData(data, 500)
    trainLabel, testLabel = SplitData(label, 500)
    
    algoritms = [LinearSVC(random_state=0), svm.SVC(), GaussianNB(), tree.DecisionTreeClassifier(), RandomForestClassifier(), BernoulliNB(), MultinomialNB()]
    
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
        