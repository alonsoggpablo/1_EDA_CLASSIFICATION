import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

dataset=pd.read_csv('adult.data.csv')

# New dataset with only categorical attributes
categorical_dataset=dataset.select_dtypes(include=[object])

# Label encoder
le=preprocessing.LabelEncoder()
categorical_dataset=categorical_dataset.apply(le.fit_transform)


# Attributes matrix

# Get integer columns
b=dataset[['age','hours-per-week']]
b.index.name='index'

categorical_dataset.index.name='index'
relevant_dataset=pd.merge(categorical_dataset,b,on='index')
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

# Build and train the model
classifier=LogisticRegression()
classifier.fit(X_train,Y_train)
Y_pred=classifier.predict(X_test)

# Check accuracy
cm=confusion_matrix(Y_test,Y_pred)

target_names=['<=50K','>50K']
print (classification_report(Y_test,Y_pred,target_names=target_names))

print (cm)


