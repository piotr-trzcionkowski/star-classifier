import pandas as pd
import numpy as np
from pandas import DataFrame
from itertools import combinations
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier #
from sklearn import metrics 
import fileinput
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression #
from sklearn.tree import DecisionTreeClassifier #
from sklearn.neighbors import KNeighborsClassifier #
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #
from sklearn.naive_bayes import  BernoulliNB #
from sklearn.ensemble import RandomForestClassifier #
from sklearn.svm import SVC #
from sklearn.ensemble import AdaBoostClassifier #

### 1st part: loading data

dtr = pd.read_csv("training.csv") #training
X = dtr.iloc[:, :-1].values
y = dtr.iloc[:, -1].values

inpo = pd.read_csv("ulens_candidate.csv") #data to be predicted
dflast=inpo.iloc[:, inpo.columns != "name"] #input without name column
print(dflast)
nm = pd.Series(inpo['name'])
print(nm)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(len(X_train))
#print(X_train)
#print(y_train)
print(len(X_test))
#print(X_test)
#print(y_test)


### 2nd part: comparing classifiers by their accuracy

rfc = RandomForestClassifier(n_estimators=500, random_state=42)
rfc.fit(X_train, y_train)
y_predrfc = rfc.predict(X_test)
rfc_sc = metrics.accuracy_score(y_test, y_predrfc)
print("RFC model accuracy:", rfc_sc)
    

    
lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)
lr.fit(X_train, y_train)
y_predlr = lr.predict(X_test)
lr_sc = metrics.accuracy_score(y_test, y_predlr)
print("LR model accuracy:", lr_sc)
    



knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn.fit(X_train, y_train)
y_predknn = knn.predict(X_test)
knn_sc = metrics.accuracy_score(y_test, y_predknn)
print("KNeighbors accuracy:", knn_sc)
    
    


dtc = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
dtc.fit(X_train, y_train)
y_preddtc = dtc.predict(X_test)
dtc_sc = metrics.accuracy_score(y_test, y_preddtc)
print("DecisionTree Classifier accuracy:", dtc_sc)
    
    
    
brn = BernoulliNB()
brn.fit(X_train, y_train)
y_predbrn = brn.predict(X_test)
brn_sc = metrics.accuracy_score(y_test, y_predbrn)
print("Bernoulli accuracy:", brn_sc)
    
 
    
svc = SVC(gamma='auto',probability=True)
svc.fit(X_train, y_train)
y_predsvc = svc.predict(X_test)
svc_sc = metrics.accuracy_score(y_test, y_predsvc)
print("SVC accuracy:", svc_sc)
    

    
ada = AdaBoostClassifier()
ada.fit(X_train, y_train)
y_predada = ada.predict(X_test)
ada_sc = metrics.accuracy_score(y_test, y_predada)
print("Ada Boost Classifier accuracy:", ada_sc)
    

    
mak1 = {rfc_sc:"rfc", lr_sc:"lr", knn_sc:"knn", dtc_sc:"dtc", brn_sc:"brn", svc_sc:"svc", ada_sc:"ada"}    
mak = mak1.get(max(mak1))


### 3rd part: predicting candidate

pred_cols = list(dflast.columns.values)
print("--------------")
if mak == 'rfc':
        pred = pd.Series(rfc.predict(dflast[pred_cols]))
        prob0 = pd.Series(rfc.predict_proba(dflast[pred_cols])[:, 0]) # probability of B
        prob1 = pd.Series(rfc.predict_proba(dflast[pred_cols])[:, 1]) # probability of E
        prob2 = pd.Series(rfc.predict_proba(dflast[pred_cols])[:, 2]) # probability of S
        prob3 = pd.Series(rfc.predict_proba(dflast[pred_cols])[:, 3]) # probability of Y
        classifer = pd.Series(['RFC'])
        result = pd.concat([nm, pred, prob0, prob1, prob2, prob3, classifer], axis=1, sort=False)
        result = result.round(decimals=4)
        result.columns = ['name', 'class', 'B_prob', 'E_prob', 'S_prob', 'Y_prob', 'classifier']
        print(result)
        print("==================================================")
        print(" ")
        result.to_csv('results.csv', mode='a', header=["name", "best guess", "B", "E", "S", "Y", "classifier"], index=False)
        print("RFC classifier")
    
        
        cmrfc = confusion_matrix(y_test, y_predrfc)
        print(cmrfc)
        accuracy_score(y_test, y_predrfc)
