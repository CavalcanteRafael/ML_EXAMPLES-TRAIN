# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 23:50:22 2019

@author: Rafael
"""

import pandas as pd
import numpy as np

import seaborn as sns

path = "C:/Users/Rafael/Desktop/DATASET'S/IRIS/"
file = "Iris.csv"

df = pd.read_csv(path + file)


df.drop('Id',axis = 1,inplace =True)

df['Species'] = df['Species'].astype('category')
df['Species'] = df['Species'].cat.codes

"""
df['Species'].replace({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2},inplace=True)
"""

#Prepare data (SPLIT DATA | Normalization) ; KFOLD ; SVM + GRIDSEARCH
Y = df['Species']
X = df.drop('Species',axis = 1)

##################
####REPETIÇÃO 1 ##
##################

"""
from sklearn.model_selection import train_test_split
Xtrain,Xtest ,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4)


from sklearn.preprocessing import StandardScaler
scl = StandardScaler()
scl.fit(Xtrain)
Xtrain = scl.transform(Xtrain)
Xtest = scl.transform(Xtest)

#Tunning with GridSearch (SVM EXAMPLE)
from sklearn.svm import SVC
        #svm : C ; gamma ; Kernel
svm = SVC()

from sklearn.model_selection import GridSearchCV
C = [0.1,0.3,0.4,0.6,0.8,1]
gamma = [0.1,0.3,0.4,0.6,0.8,1]
kernel = ['linear','rbf']
hype = {'kernel':kernel,'gamma':gamma,'C':C}

gs = GridSearchCV(estimator = svm, param_grid =hype,verbose = True)
gs.fit(Xtrain,Ytrain)

print('The best score is: ' ,gs.best_score_, '\n')
print(gs.best_estimator_)
"""
##################
####REPETIÇÃO 2 ##
##################
"""
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4,random_state = 999)

from sklearn.preprocessing import StandardScaler
sds = StandardScaler()
sds.fit(Xtrain)
Xtrain = sds.transform(Xtrain)
Xtest = sds.transform(Xtest)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state = 999)

from sklearn.model_selection import GridSearchCV

n_estimators = [100,200,300,400]
criterion = ["gini","entropy"]
hyper = {"n_estimators":n_estimators , "criterion":criterion}

gs = GridSearchCV(estimator = rfc, param_grid = hyper, verbose=True)

gs.fit(Xtrain,Ytrain)

print(gs.best_score_)
print(gs.best_estimator_)
"""
##################
####REPETIÇÃO 3 ##
##################
"""

from sklearn.model_selection import train_test_split

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4)

from sklearn.preprocessing import StandardScaler
scl = StandardScaler()
scl.fit(Xtest)

Xtrain = scl.transform(Xtrain)
Xtest = scl.transform(Xtest)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB

svm_model = SVC()
rfc_model = RandomForestClassifier()
lr_model = LogisticRegression()


from sklearn.model_selection import GridSearchCV

#SVM PARAM <C;GAMMA;KERNEL
C = [0.1,0.3,0.45,0.5,0.6,0.75,0.8,0.93,1]
gamma = [0.1,0.3,0.45,0.5,0.6,0.75,0.8,0.93,1]
kernel = ['rbf','linear']
    #compile param
hyper = {'C':C,'gamma':gamma,'kernel':kernel}

# GridSearchCV(estimator = model , param_grid = params)
gs1 = GridSearchCV(estimator = svm_model , param_grid = hyper , verbose = True)
gs1.fit(Xtrain,Ytrain)

print(gs1.best_score_)
print(gs1.best_estimator_)


print('\n')
#RFC PARAM <n_estimators ; criterion

n_estimators = [100,150,200,300]
hyper = {'n_estimators' : n_estimators}

gs2 = GridSearchCV(estimator = rfc_model , param_grid = hyper,verbose = True )
gs2.fit(Xtrain,Ytrain)


print(gs2.best_score_)
print(gs2.best_estimator_)

"""
##################
####REPETIÇÃO 4 ##
##################


"""
from sklearn.model_selection import train_test_split

Xtrain , Xtest , Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4)

from sklearn.preprocessing import StandardScaler

sds = StandardScaler()
sds.fit(Xtest)
Xtrain = sds.transform(Xtrain)
Xtest = sds.transform(Xtest)

from sklearn.ensemble import RandomForestClassifier

rfc_model = RandomForestClassifier()

n_estimators = [100,200,300,450,600]

hyper = {'n_estimators' : n_estimators}

from sklearn.model_selection import GridSearchCV
gs4 = GridSearchCV(estimator = rfc_model , param_grid = hyper)
gs4.fit(Xtrain , Ytrain)

print('The best score is:', gs4.best_score_ , '\n')
print('The awesome param is:',gs4.best_estimator_ )
        ##CROSS VALIDATION (KFOLD)
#STEPS : IMPORT ;USE ; CREATE METRIC LIST ; CREATE MODEL LIST ; CREATE FOR ; VISUALIZATION

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

#KFold(n_splits = K , random_state = k)
kfold_use = KFold(n_splits = 10 , random_state = 999)

#3 LISTS MEAN ; STD ; ACCURACY
mean_metric = []
std_metric = []
accuracy = []

#MODELS (LIST NAME AND MODELS)
classifier_names = ['Random Forest'] # YOU CAN PUT OTHERS
model_work = [RandomForestClassifier(n_estimators = 200)]  #YOU CAN PUT MORE MODELS!

#CREATE FOR

for model in model_work:
    model = model
    cv_result = cross_val_score(model, X, Y, cv = kfold_use, scoring = 'accuracy')
    cv_result = cv_result
    
    std_metric.append(cv_result.std())
    mean_metric.append(cv_result.mean())
    accuracy.append(cv_result)

box_models = pd.DataFrame({'Classifier':classifier_names , 'Mean': mean_metric , 'Std':std_metric })
print(box_models)

sns.boxplot(x = classifier_names , y = accuracy , palette = 'inferno')
"""

##################
####REPETIÇÃO 5 ##
##################
"""


from sklearn.model_selection import train_test_split

Xtrain,Xtest ,Ytrain,Ytest = train_test_split(X,Y,train_size = 0.4,random_state = 999)

from sklearn.preprocessing import StandardScaler

scl = StandardScaler()
scl.fit(Xtest)

Xtest = scl.transform(Xtest)
Xtrain = scl.transform(Xtrain)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.


        ##model for tunning
svm_model = SVC()
svm_model.fit(Xtrain,Ytrain)
Ypred1 = svm_model.predict(Xtest)


#Tunning
from sklearn.model_selection import GridSearchCV
    #Set param
gamma = [1,0.8,0.75,0.6,0.5,0.4,0.35,0.2,0.1]
C = [1,0.8,0.75,0.6,0.5,0.4,0.35,0.2,0.1]
kernel = ['linear','rbf']
hyper = {'gamma':gamma,'C':C,'kernel':kernel}
    #start and fit
gs5 = GridSearchCV(estimator = svm_model,param_grid = hyper)
gs5.fit(Xtrain,Ytrain)
    #show
print(gs5.best_score_ , '\n')
print(gs5.best_estimator_ )

    
    #MODELS
rfc_model = RandomForestClassifier(n_estimators = 200)
rfc_model.fit(Xtrain,Ytrain)
Ypred2 = rfc_model.predict(Xtest)

nb_model = GaussianNB()
nb_model.fit(Xtrain,Ytrain)
Ypred3 = nb_model.predict(Xtest)


    ##KFOLD
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

kfold_use = KFold(n_splits = 5,random_state = 999)

    #START LISTS (METRICS ; MODELS)
std_metric = []
mean_metric = []
accuracy = []

model_name = ['svm','rfc','nb']

    #PUT MODELS
models = [SVC(kernel = 'linear') ,RandomForestClassifier(n_estimators = 200) , GaussianNB()]

    #RUN KFOLD

for model in models_work:
    model = model
    cv_result = cross_val_score(model, X, Y, cv = kfold_use, scoring = 'accuracy')
    cv_result = cv_result
    
    std_metric.append(cv_result.std())
    mean_metric.append(cv_result.mean())
    accuracy.append(cv_result)
    

box_models = pd.DataFrame({'Model':model_name , 'mean':mean_metric ,'std':std_metric})
sns.boxplot(x = model_name , y=accuracy,palette = 'viridis')

"""

##################  train test & normalization
####REPETIÇÃO 6 ##  models ; gridsearch
##################  kfold

"""

from sklearn.model_selection import train_test_split

Xtrain , Xtest, Ytrain , Ytest = train_test_split(X,Y,test_size = 0.4 , random_state = 999)

from sklearn.preprocessing import StandardScaler

sds = StandardScaler()
sds.fit(Xtest)

Xtrain = sds.transform(Xtrain)
Xtest = sds.transform(Xtest)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

rfc_model = RandomForestClassifier(n_estimators = 200)
svm_model = SVC()
bayes_model = GaussianNB()
knn_model = KNeighborsClassifier()

# KFOLDS <IMPORT ; LISTS (METRIC & NAME) ; MODELS ; RUN <FOR>

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

kfold_use = KFold(n_splits = 5 , random_state = 999)

std_metric = []
mean_metric = []
accuracy = []

model_name = ['rfc','svm','nb','knn']

model_works = [rfc_model ,svm_model ,bayes_model,knn_model]

for model in model_works:
    model = model
    cv_result = cross_val_score(model, X, Y, cv = kfold_use, scoring = 'accuracy')
    cv_result = cv_result
    
    std_metric.append(cv_result.std())
    mean_metric.append(cv_result.mean())
    accuracy.append(cv_result)
    
box_model = pd.DataFrame({'model': model_name ,'mean':mean_metric , 'std':std_metric})
sns.boxplot(x = model_name , y = accuracy,palette = 'inferno')

"""
##################  train test & normalization
####REPETIÇÃO 7 ##  models ; gridsearch
##################  kfold

"""
from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4 , random_state = 999)

from sklearn.preprocessing import StandardScaler

sds = StandardScaler()
sds.fit(Xtrain)

Xtest = sds.transform(Xtest)
Xtrain = sds.transform(Xtrain)

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

svm_model = SVC()
rfc_model = RandomForestClassifier()
nb_model = GaussianNB()
knn_model = KNeighborsClassifier()

#GridSearchCV(estimator = models* , param_grid = hyper*)
from sklearn.model_selection import GridSearchCV
gamma = [1,0.8,0.7,0.65,0.5,0.4,0.3,0.2,0.1]
C = [1,0.8,0.7,0.65,0.5,0.4,0.3,0.2,0.1]
kernels = ['linear','rbf']

hyper ={'gamma':gamma , 'C':C , 'kernel':kernels}

gs7 = GridSearchCV(estimator = svm_model , param_grid = hyper)
gs7.fit(Xtrain,Ytrain)

print(gs7.best_score_)
print(gs7.best_estimator_)

#KFOLD <import and set , lists , models , runner

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

kfold_use = KFold(n_splits = 5 , random_state = 999)

std_metric = []
mean_metric = []
accuracy = []

model_name = ['svm','rfc','nb','knn']
model_work = [svm_model ,rfc_model , nb_model , knn_model]

for model in model_work:
    model = model
    cv_result = cross_val_score(model , X,Y,cv = kfold_use ,scoring = 'accuracy')
    cv_result = cv_result
    
    std_metric.append(cv_result.std())
    mean_metric.append(cv_result.mean())
    accuracy.append(cv_result)
    
box_model = pd.DataFrame({'Models':model_name , 'Mean':mean_metric , 'Std':std_metric})

sns.boxplot(x = model_name , y = accuracy,palette = 'viridis')

"""

##################  train test & normalization
####REPETIÇÃO 8 ##  models ; gridsearch
##################  kfold

"""

from sklearn.model_selection import train_test_split

Xtrain,Xtest ,Ytrain , Ytest = train_test_split(X,Y,test_size = 0.4 , random_state = 999)

from sklearn.preprocessing import StandardScaler

sds = StandardScaler()
sds.fit(Xtest)

Xtest = sds.transform(Xtest)
Xtrain = sds.transform(Xtrain)

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

rfc_model = RandomForestClassifier(n_estimators = 200)
nb_model = GaussianNB()
svm_model = SVC()
knn_model = KNeighborsClassifier()

from sklearn.model_selection import GridSearchCV

gamma = [1,0.8,0.7,0.65,0.5,0.4,0.3,0.2,0.1]
C = [1,0.8,0.7,0.65,0.5,0.4,0.3,0.2,0.1]
kernels = ['linear','rbf']

hyper = {'gamma':gamma ,'C': C ,'kernel':kernels}

gs8 = GridSearchCV(estimator = svm_model ,param_grid = hyper)
gs8.fit(Xtrain,Ytrain)

print(gs8.best_score_)
print(gs8.best_estimator_)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

kfold_use = KFold(n_splits = 5 , random_state = 999)

#lists

mean_metric = []
std_metric = []
accuracy = []

model_use = ['rfc','nb','svm','knn']
model_work = [rfc_model,nb_model,svm_model,knn_model]


for model in model_work:
    model = model
    cv_result = cross_val_score(model,X,Y,cv = kfold_use,scoring = 'accuracy')
    cv_result = cv_result
    
    mean_metric.append(cv_result.mean())
    std_metric.append(cv_result.std())
    accuracy.append(cv_result)
    
box_models = pd.DataFrame({'Models':model_use , 'std':std_metric , 'mean':mean_metric})
sns.boxplot(x = model_use , y = accuracy , palette = 'inferno')
"""

##################  train test & normalization
####REPETIÇÃO 9 ##  models ; gridsearch
##################  kfold
"""

from sklearn.model_selection import train_test_split

Xtrain,Xtest ,Ytrain ,Ytest = train_test_split(X,Y,test_size =0.4 , random_state = 999)

from sklearn.preprocessing import StandardScaler

sds = StandardScaler()
sds.fit(Xtest)

Xtest = sds.transform(Xtest)
Xtrain = sds.transform(Xtrain)

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

nb_model = GaussianNB()
rfc_model = RandomForestClassifier(n_estimators = 200)
svm_model = SVC()
knn_model = KNeighborsClassifier()

nb_model.fit(Xtrain,Ytrain)
Ypred1 = nb_model.predict(Xtest)

rfc_model.fit(Xtrain,Ytrain)
Ypred2 = rfc_model.predict(Xtest)

svm_model.fit(Xtrain,Ytrain)
Ypred3 = svm_model.predict(Xtest)

knn_model.fit(Xtrain,Ytrain)
Ypred4 = knn_model.predict(Xtest)

from sklearn.model_selection import GridSearchCV


C  = [0.2,0.4,0.8,0.9]
gamma = [0.2,0.4,0.8,0.9]
kernels = ['linear','rbf']

hyper = {'C':C,'gamma':gamma , 'kernel':kernels}

gs10 = GridSearchCV(estimator = svm_model , param_grid = hyper)
gs10.fit(Xtrain,Ytrain)

print(gs10.best_score_)
print(gs10.best_estimator_)

#KFOLDS <LIST's(metrics&models) , Models , Runner and comparsion

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

kfold_use = KFold(n_splits = 5,random_state = 999)

mean_metric = []
std_metric = []
accuracy = []

models_name = ['NB','RFC','SVM','KNN']
models_work = [nb_model,rfc_model,svm_model,knn_model]

for model in models_work:
    model = model
    cv_result = cross_val_score(model,X,Y,cv = kfold_use,scoring = 'accuracy')
    cv_result = cv_result
    
    mean_metric.append(cv_result.mean())
    std_metric.append(cv_result.std())
    accuracy.append(cv_result)
    
box_models = pd.DataFrame({'Models':models_name , 'STD':std_metric , 'MEAN':mean_metric})

sns.boxplot(x = models_name , y = accuracy , palette = 'viridis')
"""


##################  train test & normalization
####REPETIÇÃO 10##  models ; gridsearch
##################  kfold


"""
#SPLIT
from sklearn.model_selection import train_test_split
Xtrain,Xtest, Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4 , random_state = 999)

#NORMALIZATION
from sklearn.preprocessing import StandardScaler

sds = StandardScaler()
sds.fit(Xtrain)

Xtrain = sds.transform(Xtrain)
Xtest = sds.transform(Xtest)

#MODELS
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

rfc_model = RandomForestClassifier(n_estimators = 200)

svm_model = SVC()


nb_model = GaussianNB()

knn_model = KNeighborsClassifier()


ada_model = AdaBoostClassifier(n_estimators = 50 , learning_rate = 0.1)

xgb_model = XGBClassifier(gamma = 2 , max_depth = 7 , learning_rate = 0.1)

lgbm_model = lgb.LGBMClassifier(n_estimators = 200,max_depth = 10 , learning_rate = 0.01 , num_leaves = 100)


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

kfold_use = KFold(n_splits = 5,random_state = 999)

#LIST <LIST NAMEMODEL / METRICS ; MODEL'S WORK

std_metric = []
mean_metric = []
accuracy = []

name_model = ['rfc','svm','nb','knn','ada','xgb','lgbm']
model_work =[rfc_model ,svm_model,nb_model,knn_model,ada_model ,xgb_model,lgbm_model]


#FOR

for model in model_work:
    model = model
    cv_result = cross_val_score(model,X,Y,cv =kfold_use , scoring = 'accuracy')
    cv_result = cv_result
    
    std_metric.append(cv_result.std())
    mean_metric.append(cv_result.mean())
    accuracy.append(cv_result)
    
box_models = pd.DataFrame({'models':name_model ,'std':std_metric ,'mean':mean_metric})

sns.boxplot(x = name_model , y = accuracy,palette = 'inferno')
"""
##################  train test & normalization
####REPETIÇÃO 11##  models ; gridsearch
##################  kfold
"""

from sklearn.model_selection import train_test_split

Xtrain,Xtest , Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4 , random_state = 888)

from sklearn.preprocessing import StandardScaler

sds = StandardScaler()
sds.fit(Xtrain)

Xtest = sds.transform(Xtest)
Xtrain = sds.transform(Xtrain)


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier

import lightgbm as lgb


#MODELS
nb_model = GaussianNB()
svm_model = SVC()
knn_model = KNeighborsClassifier()
rfc_model = RandomForestClassifier(n_estimators = 200)

xgb_model = XGBClassifier(learning_rate = 0.001,gamma = 3)
ada_model = AdaBoostClassifier(learning_rate = 0.001,n_estimators = 300)

lgbm_model = lgb.LGBMClassifier(learning_rate = 0.001,n_estimators = 900)




#CROSS VAL
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

kfold_use = KFold(n_splits = 5,random_state = 999)

#LISTS (NAME_MODEL AND METRICS ; MODELS)

std_metric = []
mean_metric = []
accuracy = []

name_model = ['nb','svm','knn','rfc','xgb','ada','lgbm',]
model_work = [nb_model , svm_model , knn_model , rfc_model , xgb_model , ada_model , lgbm_model]

for model in model_work:
    model = model
    cv_result = cross_val_score(model , X,Y,cv = kfold_use,scoring = 'accuracy')
    cv_result = cv_result
    
    std_metric.append(cv_result.std())
    mean_metric.append(cv_result.mean())
    accuracy.append(cv_result)
    
box_model = pd.DataFrame({'models':name_model,'std':std_metric,'mean' : mean_metric})
sns.boxplot(x = name_model , y = accuracy , palette = 'viridis')

"""

##################  train test & normalization
####REPETIÇÃO 11##  models ; gridsearch
##################  kfold
from sklearn.model_selection import train_test_split

Xtrain,Xtest, Ytrain,Ytest = train_test_split(X,Y,test_size = 0.4 , random_state = 999)

from sklearn.preprocessing import StandardScaler

sds = StandardScaler()
sds.fit(Xtrain)

Xtrain = sds.transform(Xtrain)
Xtest = sds.transform(Xtest)

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
    



##################  train test & normalization
####REPETIÇÃO 12##  models ; gridsearch
##################  kfold