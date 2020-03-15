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

dataset=pd.read_csv('adult.data.csv')

# New dataset with only categorical attributes
categorical_dataset=dataset.select_dtypes(include=[object])

# Label encoder
le=preprocessing.LabelEncoder()
categorical_dataset=categorical_dataset.apply(le.fit_transform)


# Features matrix

# Get integer columns
b=dataset[['age','hours-per-week']]
b.index.name='index'

categorical_dataset.index.name='index'
relevant_dataset=pd.merge(categorical_dataset,b,on='index')

# Check correlations

relevant_dataset_corr=relevant_dataset.corr()


X=relevant_dataset.drop(columns=['salary'])
sc=StandardScaler()
X=sc.fit_transform(X)

# Output values
Y=categorical_dataset['salary'].values


# One Hot Encoder
enc=preprocessing.OneHotEncoder()
enc.fit(X)
X_onehotencoded=enc.transform(X).toarray()


# Split train and test

X_train,X_test,Y_train,Y_test=train_test_split(X_onehotencoded,Y,test_size=0.2)

# Get the PCs

pca=PCA(n_components=6)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)


# Build and train the model
#classifier=LogisticRegression()
#classifier=RandomForestClassifier(max_depth=2,random_state=0)
#classifier=DecisionTreeClassifier(max_depth=5)
classifier=GaussianNB()
#classifier=KNeighborsClassifier(3)
#classifier=SVC()
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)

# Check accuracy
cm=confusion_matrix(Y_test,Y_pred)

target_names=['<=50K','>50K']

print (pca.explained_variance_ratio_)
print (classification_report(Y_test,Y_pred,target_names=target_names))
print (cm)


