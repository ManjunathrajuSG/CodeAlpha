import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data=sns.load_dataset("titanic")
# print(data.head())
# print(len(data))
# print(data.info())
# print(data['survived'].value_counts())
# sns.countplot(x=data['survived'],hue=data['pclass'])
# plt.show()
# sns.countplot(x=data['survived'],hue=data['sex'])
# plt.show()
# sns.countplot(x=data['survived'],hue=data['embarked'])
# plt.show()
# print(data.isnull().sum())

# print(data.columns)
cols=['fare', 'class', 'who', 'adult_male', 'deck', 'embark_town','alive', 'alone']
data_new=data.drop(cols,axis=1)
# print(data_new.head())
# print(data_new.head())
# print(data_new.isnull().sum())

mean_age=data_new['age'].mean()
# print(mean_age)
mean_age=np.round(mean_age,2)
# print(mean_age)
data_new['age']=data_new['age'].fillna(mean_age)
# print(data_new.isnull().sum())

data_new=data_new.dropna()
# print(data_new.isnull().sum())
print(data_new.info())
# print(data_new.head())

###converting string to numeric
from sklearn.preprocessing import  LabelEncoder
enc=LabelEncoder()
data_new['sex']=enc.fit_transform(data_new['sex'])
data_new['embarked']=enc.fit_transform(data_new['embarked'])
# print(data_new.head())

#features-------X
x=np.array(data_new.iloc[:,1:])
print(x.shape) #alraedy in 2D

y=np.array(data_new.iloc[:,0])
print(y.shape)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,train_size=0.8,random_state=3)
# print(pd.DataFrame(y).value_counts())
# print(pd.DataFrame(ytrain).value_counts())


from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3, p=2)
model.fit(xtrain,ytrain)


ypred=model.predict(xtest)
# print(ypred)
# print(ytest)

# print(ytest[1]==ypred[1])
count=0
for i in range(len(ytest)):
    if ypred[i]==ytest[i]:
        count=count+1
print(count)
# # print(len(ytest))
# # print(count/

from sklearn.metrics import accuracy_score
a=accuracy_score(ytest,ypred)
# print(a)


import joblib
joblib.dump(model,"titanicManju.pkl")

mymodel=joblib.load("C:/Users/manju/PycharmProjects/pythonProject/3rdSem-MachineLearning/titanicManju.pkl")
print(data_new.head())
print(mymodel.predict([[1,0,20,2,1,1]]))