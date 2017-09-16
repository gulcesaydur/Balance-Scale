# coding: utf-8

from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

from Read import ReadFile as r
from DataManipulation import DataManipulation as d

from sklearn.metrics import accuracy_score

import datetime

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

filePath = "C:\Users\Lenovo\Desktop\Balance Scale\\balance scale.csv"
data = r.readCSVFile(filePath)
label = data[0]

sortedList = d.createRandList(625, 187)

missingTrainData, missingTestData, missingTrainLabel = d.createTestandTrain(sortedList,0,1,data)

missingPred = TrainModelWithOnevsRest(LinearSVC(random_state=0), missingTrainData, missingTestData, missingTrainLabel)

d.fillData(missingPred, data, sortedList, 0)

trainData, trainLabel, testData, testLabel = d.createTestandTrain(sortedList,0,0,data,0.7)

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