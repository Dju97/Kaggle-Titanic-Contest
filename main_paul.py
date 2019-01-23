
#All the import needed to use the sklearn function
from numpy import *
from sklearn import cross_validation
import csv as csv
from classify import classify
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


####################### Load and treat data  #######################

################# 1 load data
X=pd.read_csv('train.csv',delimiter=',',index_col=False)
Xtest=pd.read_csv('test.csv',delimiter=',',index_col=False)

# Conservation of X test fill to have the passenger ID to create the predicted_class.csv
Xtestcons=pd.read_csv('test.csv',delimiter=',',index_col=False)


# Creation of the class dataframe
Y=X['Survived']
X.drop('Survived',1,inplace=True)

################ 2 Data preprocessing 

####### 2.1 Feature Sex


#Transform Sex feature into a digital one
X['Sex'].replace('female', 0,inplace=True)
X['Sex'].replace('male', 1,inplace=True)

Xtest['Sex'].replace('female', 0,inplace=True)
Xtest['Sex'].replace('male', 1,inplace=True)

#Fill the blank with the rounded mean value of the feature
X['Sex'].fillna(round(X['Sex'].mean()),inplace=True)
Xtest['Sex'].fillna(round(Xtest['Sex'].mean()),inplace=True)

####### 2.2 Feature SibSP

#Fill the blank with 0
X['SibSp'].fillna(0,inplace=True)
Xtest['SibSp'].fillna(0,inplace=True)


####### 2.3 Feature Parch

#Fill the blank with 0
X['Parch'].fillna(0,inplace=True)
Xtest['Parch'].fillna(0,inplace=True)

####### 2.4 Feature Age

#Fill the blank with the feature mean
X['Age'].fillna(X['Age'].mean(),inplace=True)
Xtest['Age'].fillna(X['Age'].mean(),inplace=True)


####### 2.4 Feature Fare

#Fill the blank with the feature mean
X['Fare'].fillna(X['Fare'].mean(),inplace=True)
Xtest['Fare'].fillna(X['Fare'].mean(),inplace=True)
####### 2.4 Feature Embarked

#Transform the feature into a digital one
X.replace({'Embarked' : { 'C' : 0, 'S' : 1, 'Q' : 2 }},inplace=True)
Xtest.replace({'Embarked' : { 'C' : 0, 'S' : 1, 'Q' : 2 }},inplace=True)

#Fill the feature with the rounded feature mean
X['Embarked'].fillna(round(X['Embarked'].mean()),inplace=True)
Xtest['Embarked'].fillna(round(X['Embarked'].mean()),inplace=True)

################ 3 Creation of new feature

####### 3.1 Cabnum : Had he/she a cabin num or not ?

#Fill the blanks with T which is not a floor on the boat
X['Cabin'].fillna('T',inplace=True)
Xtest['Cabin'].fillna('T',inplace=True)


#Creation of a new feature : have they a cabin or not ? 
X['Cabnum'] = np.where(X['Cabin']==0,0,1)
Xtest['Cabnum'] = np.where(Xtest['Cabin']==0,0,1)


####### 3.2 IsAlone : Had he/she SibSp or Parch ?

X['IsAlone'] = np.where((X['SibSp']==0) & (X['Parch']==0),0,1)
Xtest['IsAlone'] = np.where((Xtest['SibSp'])==0 & (Xtest['Parch']==0),0,1)

####### 3.2 Function or job title
Title=X['Name']
Title=pd.DataFrame(Title,columns=['Title'])
X = pd.concat([X, Title], axis=1)


X.loc[X['Name'].str.contains('Rev'),'Title'] = 1
X.loc[X['Name'].str.contains('Don'),'Title'] = 1
X.loc[X['Name'].str.contains('Capt'),'Title'] = 1
X.loc[X['Name'].str.contains('Sir'),'Title'] = 1
X.loc[X['Name'].str.contains('Dr'),'Title'] = 3
X.loc[X['Name'].str.contains('Col'),'Title'] = 1
X.loc[X['Name'].str.contains('Lady'),'Title'] = 2
X.loc[X['Name'].str.contains('The Countess'),'Title'] = 2
X.loc[X['Name'].str.contains('Dona'),'Title'] = 2
X.loc[X['Name'].str.contains('Miss'),'Title'] = 2
X.loc[X['Name'].str.contains('Mrs.'),'Title'] = 2
X.loc[X['Name'].str.contains('Master'),'Title'] = 1
X.loc[X['Name'].str.contains('Mr.'),'Title'] = 0
X['Title'].fillna(0,inplace=True)

Title2=Xtest['Name']
Title2=pd.DataFrame(Title,columns=['Title'])
Xtest = pd.concat([Xtest, Title2], axis=1)


Xtest.loc[Xtest['Name'].str.contains('Rev'),'Title'] = 1
Xtest.loc[Xtest['Name'].str.contains('Don'),'Title'] = 1
Xtest.loc[Xtest['Name'].str.contains('Capt'),'Title'] = 1
Xtest.loc[Xtest['Name'].str.contains('Sir'),'Title'] = 1
Xtest.loc[Xtest['Name'].str.contains('Dr'),'Title'] = 3
Xtest.loc[Xtest['Name'].str.contains('Col'),'Title'] = 1
Xtest.loc[Xtest['Name'].str.contains('Lady'),'Title'] = 2
Xtest.loc[Xtest['Name'].str.contains('The Countess'),'Title'] =2
Xtest.loc[Xtest['Name'].str.contains('Dona'),'Title'] = 2
Xtest.loc[Xtest['Name'].str.contains('Miss'),'Title'] = 2
Xtest.loc[Xtest['Name'].str.contains('Mrs.'),'Title'] = 2
Xtest.loc[Xtest['Name'].str.contains('Master'),'Title'] = 1
Xtest.loc[Xtest['Name'].str.contains('Mr.'),'Title'] = 0
Xtest['Title'].fillna(0,inplace=True)


####### 3.2 Cabin situation

Cabinlevel=X['Name']
Cabinlevel=pd.DataFrame(Cabinlevel,columns=['Cabinlevel'])
X = pd.concat([X, Cabinlevel], axis=1)



X.loc[X['Cabin'].str.contains('G'),'Cabinlevel'] = 0
X.loc[X['Cabin'].str.contains('F'),'Cabinlevel'] = 0
X.loc[X['Cabin'].str.contains('E'),'Cabinlevel'] = 0
X.loc[X['Cabin'].str.contains('D'),'Cabinlevel'] = 1
X.loc[X['Cabin'].str.contains('C'),'Cabinlevel'] = 1
X.loc[X['Cabin'].str.contains('B'),'Cabinlevel'] = 2
X.loc[X['Cabin'].str.contains('A'),'Cabinlevel'] = 2
X['Cabinlevel'].fillna(2,inplace=True)


Cabinlevel2=Xtest['Name']
Cabinlevel2=pd.DataFrame(Cabinlevel2,columns=['Cabinlevel'])
Xtest = pd.concat([Xtest, Cabinlevel2], axis=1)


Xtest.loc[Xtest['Cabin'].str.contains('G'),'Cabinlevel'] = 0
Xtest.loc[Xtest['Cabin'].str.contains('F'),'Cabinlevel'] = 0
Xtest.loc[Xtest['Cabin'].str.contains('E'),'Cabinlevel'] = 0
Xtest.loc[Xtest['Cabin'].str.contains('D'),'Cabinlevel'] = 1
Xtest.loc[Xtest['Cabin'].str.contains('C'),'Cabinlevel'] = 1
Xtest.loc[Xtest['Cabin'].str.contains('B'),'Cabinlevel'] = 2
Xtest.loc[Xtest['Cabin'].str.contains('A'),'Cabinlevel'] = 2
Xtest['Cabinlevel'].fillna(2,inplace=True)



################ 4 Delation of all the feature unusefull


X.drop('Cabin',1,inplace=True)
X.drop('Firstname',1,inplace=True)
X.drop('Name',1,inplace=True)
X.drop('Ticket',1,inplace=True)
X.drop('PassengerId',1,inplace=True)


Xtest.drop('Cabin',1,inplace=True)
Xtest.drop('Firstname',1,inplace=True)
Xtest.drop('Name',1,inplace=True)
Xtest.drop('Ticket',1,inplace=True)
Xtest.drop('PassengerId',1,inplace=True)





### Test 1
"""
#LDA
#sklearn_lda = LDA(n_components=7)
#X_lda_sklearn = sklearn_lda.fit_transform(X, Y)


#PCA
pca = PCA(n_components=2)
pca.fit(X,Y)
Xtest_pca_sklearn=pca.transform(Xtest)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_pca_sklearn, Y)
Y_predict=clf.predict(Xtest_pca_sklearn)

Y_predict=np.transpose(Y_predict)
Y_predict=pd.DataFrame(Y_predict,columns=['Survived'])

Xpass=pd.DataFrame(Xtest['PassengerId'],columns=['PassengerId'])

result = pd.concat([Xpass, Y_predict], axis=1)
result.to_csv('result.csv',index = False)

#score=0,55
"""

### Test 2 
"""

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
Y_predict=clf.predict(Xtest)

Y_predict=np.transpose(Y_predict)
Y_predict=pd.DataFrame(Y_predict,columns=['Survived'])

Xpass=pd.DataFrame(Xtest['PassengerId'],columns=['PassengerId'])

result = pd.concat([Xpass, Y_predict], axis=1)
result.to_csv('result.csv',index = False)
"""
#score=0,75



###Test 3 
"""
sklearn_lda = LDA(n_components=7)
sklearn_lda = sklearn_lda.fit(X, Y)

X_sklearn_lda=sklearn_lda.transform(X)
Xtest_sklearn_lda=sklearn_lda.transform(Xtest)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_sklearn_lda, Y)
Y_predict=clf.predict(Xtest_sklearn_lda)

Y_predict=np.transpose(Y_predict)
Y_predict=pd.DataFrame(Y_predict,columns=['Survived'])

Xpass=pd.DataFrame(Xtest['PassengerId'],columns=['PassengerId'])

result = pd.concat([Xpass, Y_predict], axis=1)
result.to_csv('result.csv',index = False)

#score=0,69
"""


### Test 4
"""
Done with only tree decision and cabin feature used
"""

### Test 5 


######Treatement LDA 
sklearn_lda = LDA()

## Fit the LDA
sklearn_lda = sklearn_lda.fit(X, Y)

## Transform the data
X_sklearn_lda=sklearn_lda.transform(X)
Xtest_sklearn_lda=sklearn_lda.transform(Xtest)


##### Classification Random forest
clf = RandomForestClassifier(n_estimators=500,max_features=0.7,max_depth=2,
                             random_state=0)

#### Make cross validation before fiting the classifier
scores = cross_val_score(clf, X_sklearn_lda, Y, cv=5)

print ("The score of the cross validation is :", scores.mean() )

## Fit the classifier
clf.fit(X_sklearn_lda, Y)


## Predict the class of X test
Y_predict=clf.predict(Xtest_sklearn_lda)

## Retreat the data in order to present understandable data to the kaggle submission
Y_predict=np.transpose(Y_predict)
Y_predict=pd.DataFrame(Y_predict,columns=['Survived'])
Xpass=pd.DataFrame(Xtestcons['PassengerId'],columns=['PassengerId'])
result = pd.concat([Xpass, Y_predict], axis=1)
result.to_csv('result.csv',index = False)

# Premier score avant retraitement final score=0,775

### Test 6
"""
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier


clf = AdaBoostClassifier(n_estimators=100)
sklearn_lda = LDA()
sklearn_lda = sklearn_lda.fit(X, Y)

X_sklearn_lda=sklearn_lda.transform(X)
Xtest_sklearn_lda=sklearn_lda.transform(Xtest)


clf.fit(X_sklearn_lda, Y)

Y_predict=clf.predict(Xtest_sklearn_lda)

Y_predict=np.transpose(Y_predict)
Y_predict=pd.DataFrame(Y_predict,columns=['Survived'])

Xpass=pd.DataFrame(Xtestcons['PassengerId'],columns=['PassengerId'])

result = pd.concat([Xpass, Y_predict], axis=1)
result.to_csv('result.csv',index = False)

#score=0,775
"""
### Test 6 on rajoute à ce moment le Isalone

"""
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier


clf = AdaBoostClassifier(n_estimators=100)
sklearn_lda = LDA()
sklearn_lda = sklearn_lda.fit(X, Y)

X_sklearn_lda=sklearn_lda.transform(X)
Xtest_sklearn_lda=sklearn_lda.transform(Xtest)


clf.fit(X_sklearn_lda, Y)

Y_predict=clf.predict(Xtest_sklearn_lda)

Y_predict=np.transpose(Y_predict)
Y_predict=pd.DataFrame(Y_predict,columns=['Survived'])

Xpass=pd.DataFrame(Xtestcons['PassengerId'],columns=['PassengerId'])

result = pd.concat([Xpass, Y_predict], axis=1)
result.to_csv('result.csv',index = False)

#score=0,760

"""

