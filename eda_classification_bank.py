import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

dataset=pd.read_csv('bank.csv')

# New dataset with only categorical attributes
categorical_dataset=dataset.select_dtypes(include=[object])

# Label encoder
le=preprocessing.LabelEncoder()
categorical_dataset=categorical_dataset.apply(le.fit_transform)


# Features matrix

# Get integer columns
b=dataset[['balance','duration']]
b.index.name='index'

categorical_dataset.index.name='index'
relevant_dataset=pd.merge(categorical_dataset,b,on='index')

# Check correlations to find relevant features

relevant_dataset_corr=relevant_dataset.corr()

X=relevant_dataset.drop(columns=['deposit'])
sc=StandardScaler()
X=sc.fit_transform(X)

# Output values
Y=categorical_dataset['deposit'].values


# One Hot Encoder
enc=preprocessing.OneHotEncoder()
enc.fit(X)
X_onehotencoded=enc.transform(X).toarray()


# Split train and test

X_train,X_test,Y_train,Y_test=train_test_split(X_onehotencoded,Y,test_size=0.2)

# PCA

pca=PCA(n_components=4)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)


# Build and train the model
#classifier=LogisticRegression()
#classifier=RandomForestClassifier(max_depth=2,random_state=0)
#classifier=GaussianNB()
#classifier=KNeighborsClassifier(3)
#classifier=SVC()

# Hyperparameters tuning / Decission Tree
#parameters={'max_depth':[2,4,5,7,8,10]}
#grid_search=GridSearchCV(DecisionTreeClassifier(),parameters,cv=3,return_train_score=True)
#grid_search.fit(X_train,Y_train)
#max_depth=grid_search.best_params_['max_depth']

# Hyperparameters tuning / Logistic Regression
parameters={'penalty':['l1','l2'],'C':[0.1,0.4,0.8,1,2,5]}
grid_search=GridSearchCV(LogisticRegression(solver='liblinear'),parameters,cv=3,return_train_score=True)
grid_search.fit(X_train,Y_train)
penalty=grid_search.best_params_['penalty']
C=grid_search.best_params_['C']


#Build the model
#classifier=DecisionTreeClassifier(max_depth=max_depth)
classifier=LogisticRegression(penalty=penalty,C=C,solver='liblinear')

classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)

# Check accuracy (best classification metric)
cm=confusion_matrix(Y_test,Y_pred)

target_names=['deposit subscribed','not subscribed']

print (pca.explained_variance_ratio_)
print (classification_report(Y_test,Y_pred,target_names=target_names))
print (cm)


