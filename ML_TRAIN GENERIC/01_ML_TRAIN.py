# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:05:09 2019

https://www.kaggle.com/dejavu23/titanic-survival-seaborn-and-ensembles


https://www.kaggle.com/dansbecker/shap-values
https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values
https://www.kaggle.com/soutothales/exercise-advanced-uses-of-shap-values

https://www.kaggle.com/nirajvermafcb/comparing-various-ml-models-roc-curve-comparison

    https://machinelearningmastery.com/implementing-stacking-scratch-python/
"""

import pandas as pd
import numpy as np

import seaborn as sns

path = "C:/Users/Rafael/Desktop/DATASET'S/TITANIC/"
file = "train.csv"

df = pd.read_csv(path+file)

print(df.columns)
df = df.drop(['PassengerId'],axis =1)

#SPLIT IN SIMPLE DF FOR TEST MODELS!

df2 = df[['Survived', 'Pclass', 'Sex', 'SibSp','Parch', 'Fare']]

df2['Sex'].replace({'male':1 , 'female':0},inplace = True)


##########
# DAY 0
##########


#SPLIT

Y = df2['Survived']
X = df2.drop('Survived',axis =1)


"""
from sklearn.model_selection import train_test_split

Xtrain,Xtest , Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4,random_state = 999)

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgbm

from sklearn.ensemble import VotingClassifier

lgbm_model = lgbm.LGBMClassifier(learning_rate = 0.01,n_estimators = 300)
lgbm_model.fit(Xtrain,Ytrain)


xgb_model = XGBClassifier(learning_rate = 0.08,gamma = 2)
xgb_model.fit(Xtrain,Ytrain)

ada_model = AdaBoostClassifier(learning_rate = 0.1,n_estimators = 400)
ada_model.fit(Xtrain,Ytrain)

nb_model = GaussianNB()
nb_model.fit(Xtrain,Ytrain)

svm_model = SVC(kernel = 'linear' , gamma = 0.8)
svm_model.fit(Xtrain,Ytrain)

knn_model = KNeighborsClassifier(n_neighbors = 6)
knn_model.fit(Xtrain,Ytrain)

rfc_model = RandomForestClassifier(n_estimators = 300)
rfc_model.fit(Xtrain,Ytrain)


vtg_model = VotingClassifier(estimators = [('lgbm',lgbm_model),('xgb',xgb_model),('ada',ada_model),
                                           ('nb',nb_model),('svm',svm_model),('knn',knn_model),
                                           ('rfc',rfc_model)]  ,  voting = 'hard')

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold_use = KFold(n_splits = 5,random_state = 999)

#LISTS
std_metric =[]
mean_metric =[]
accuracy = []

model_name = ['knn','nb','svm','rfc','ada','xgb','lgbm','voting']
model_work = [knn_model,nb_model,svm_model,rfc_model,ada_model,xgb_model,lgbm_model,vtg_model]

for model in model_work:
    model = model
    cv_result = cross_val_score(model , X,Y,cv = kfold_use , scoring = 'accuracy')
    cv_result = cv_result
    
    std_metric.append(cv_result.std())
    mean_metric.append(cv_result.mean())
    accuracy.append(cv_result)
    
box_model = pd.DataFrame({'models' : model_name , 'STD':std_metric ,'Mean':mean_metric})

sns.boxplot(x = model_name , y = accuracy , palette = 'inferno')
"""

##########
# DAY 1
##########

"""

from sklearn.model_selection import train_test_split

Xtrain,Xtest , Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4)

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

nb_model = GaussianNB()
svm_model = SVC(kernel='linear')
knn_model = KNeighborsClassifier(n_neighbors = 4)
rfc_model = RandomForestClassifier(n_estimators = 300)


from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgbm

ada_model = AdaBoostClassifier(learning_rate = 0.01)
xgb_model = XGBClassifier(learning_rate = 0.01)
lgbm_model = lgbm.LGBMClassifier(learning_rate = 0.01)



#BLEND MODELS (VOTING OR STACKING)

#VotingClassifier(estimators = [('name',model_a),('name_k',model_k)],voting = 'Hard')
from sklearn.ensemble import VotingClassifier

vtg_model = VotingClassifier(estimators = [ ('nb',nb_model), ('svm',svm_model)  , ('knn',knn_model), ('rfc',rfc_model), ('ada',ada_model), ('xgb',xgb_model), ('lgbm',lgbm_model)      ]
,voting = 'hard')

vtg_model.fit(Xtrain,Ytrain)
Ypred_vtg = vtg_model.predict(Xtest)

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold_use = KFold(n_splits = 5 , random_state = 999)

std_metric = []
mean_metric = []
accuracy = []

model_name = ['nb','svm','knn','rfc','ada','xgb','lgbm','vtg']
models_work = [nb_model,svm_model,knn_model,rfc_model,ada_model,xgb_model,lgbm_model,vtg_model]

for model in models_work:
    model = model
    cv_result = cross_val_score(model , X,Y,cv = kfold_use , scoring = 'accuracy')
    cv_result = cv_result
    
    std_metric.append(cv_result.std())
    mean_metric.append(cv_result.mean())
    accuracy.append(cv_result)

box_models = pd.DataFrame({'Name':model_name ,'Mean':mean_metric ,'Std': std_metric})

sns.boxplot(x = model_name , y = accuracy , palette = 'inferno')

"""

##########
# DAY 2
##########
from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4 , random_state = 999)

from sklearn.preprocessing import StandardScaler

sds = StandardScaler()
sds.fit(Xtrain)
Xtrain = sds.transform(Xtrain)
Xtest = sds.transform(Xtest)

#CLASSIC MODELS : KNN / SVM / RFC / TREE / NB  <1>
#ENSEMBLE MODELS : XGBOOST/ADABOOST / LGBM / VOTING <2>
#MORE LAYERS : VOTING / STACKING <3>

#<1>
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier()
knn_model.fit(Xtrain,Ytrain)
Yknn_pred = knn_model.predict(Xtest)

from sklearn.svm import SVC
svm_model = SVC(kernel = "linear")
svm_model.fit(Xtrain,Ytrain)
Ysvm_pred = svm_model.predict(Xtest)

from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(n_estimators = 300)
rfc_model.fit(Xtrain,Ytrain)
Yrfc_pred = rfc_model.predict(Xtest)

from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(Xtrain,Ytrain)
Ynb_pred = nb_model.predict(Xtest)

"""
from sklearn.tree import TreeClassifier
tree_model = TreeClassifier()
tree_model.fit(Xtrain,Ytrain)
Ytree_pred = tree_model.predict(Xtest)
"""
#<2>

from sklearn.ensemble import AdaBoostClassifier
ada_model = AdaBoostClassifier(learning_rate = 0.01)
ada_model.fit(Xtrain,Ytrain)
Yada_pred = ada_model.predict(Xtest)


from xgboost import XGBClassifier
xgb_model = XGBClassifier(learning_rate = 0.01)
xgb_model.fit(Xtrain,Ytrain)
Yxgb_pred = xgb_model.predict(Xtest)

import lightgbm as lgbm
lgbm_model = lgbm.LGBMClassifier(learning_rate = 0.01)
lgbm_model.fit(Xtrain,Ytrain)
Ylgbm_pred = lgbm_model.predict(Xtest)


from sklearn.ensemble import VotingClassifier
vtg_model = VotingClassifier(estimators = [("knn_model",knn_model),("nb_model",nb_model),   ("svm_model",svm_model),   ("rfc_model",rfc_model),   ("ada_model",ada_model),   ("lgbm_model",lgbm_model),   ("xgb_model",xgb_model)   ],voting = "hard")

#TUNNING GRID_SEARCH


#TEST MODEL : CONFUSION MATRIX AND REPORT CLASSIFICATION
#COMPARSION AND TEST STRONG : CROSS VALIDATION

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

kfold_use = KFold(n_splits = 5 , random_state = 999)

std_metric = []
mean_metric = []
accuracy = []

models_name = ["knn","nb","svm","rfc","ada","lgbm","xgb","vtg"]
models_work = [knn_model,nb_model,svm_model,rfc_model,ada_model,lgbm_model,xgb_model,vtg_model]

for model in models_work:
    model = model
    cv_result = cross_val_score(model ,X,Y,cv = kfold_use,scoring = 'accuracy')
    cv_result = cv_result
    
    std_metric.append(cv_result.std())
    mean_metric.append(cv_result.mean())
    accuracy.append(cv_result)
    
box_model = pd.DataFrame({'mean':mean_metric ,'std': std_metric,'name': models_name})
sns.boxplot(x = models_name , y = accuracy , palette = 'inferno')





