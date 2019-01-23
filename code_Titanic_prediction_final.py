import pandas as pd
import numpy as np
from sklearn import cross_validation
import csv as csv
from classify import classify
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from numpy import nan
import re
import random
from sklearn.svm import SVC
from sklearn.model_selection import KFold

#This function is used to find which status has a passenger, knowing what's his designation   
def find_status(name):
    if 'Dr.' in name:
        return 1
    if 'Mr.' in name:
        return 2
    if 'Mrs.' in name or 'Mme' in name:
        return 3
    if 'Miss' in name or 'Mlle' in name:
        return 4
    if 'Master' in name:
        return 5
    if 'Rev' in name:
        return 6
    if 'Don' in name:
        return 7
    if 'Capt' in name:
        return 8
    if 'Sir' in name:
        return 9
    if 'Col.' in name:
        return 10
    if 'Lady' in name:
        return 11
    if 'Countess' in name:
        return 12
    if 'Major' in name:
        return 13
    if 'Jonkheer' in name:
        return 14
    else:
        return 15
    
#Used to find the last Name of passengers   
def find_name(name):
    return name.split(',')[0]

#Group tickets together by pairs
def group_tickets(ticket_number):
    tickets = ticket_number.split(' ')
    if(len(tickets)>1):
        try:
            tickets[len(tickets)-1] = int(tickets[len(tickets)-1])//2*2
        except:
            return nan
        ticket_final = tickets[0]+' '+str(tickets[1])
    else:
        try:
            tickets[0] = int(tickets[0])//2 *2
        except:
            return nan
        ticket_final = tickets[0]
    return ticket_final

#This function creates a dataframe with cleaned features and features engineered.
def read_treat_file(csv):
    df = pd.read_csv(csv,index_col='PassengerId')
    #Sex equals 1 if it is a Male or 0 if not
    df['Sex'] = df['Sex'].apply(lambda x:int(x=='male'))
    # Fill nan values
    df['Age'] = df['Age'].replace(nan,np.mean(df['Age']))
    df['Fare'] = df['Fare'].replace(nan,np.mean(df['Fare']))
    #Find status
    df['Status'] = df['Name'].apply(find_status)
    df['Cabin'] = df['Cabin'].fillna(0)
    df['Cabin'] = df['Cabin'].apply(lambda x:0 if x==0 else 1)
    #Find Last Name
    df['Last Name'] = df['Name'].apply(find_name)
    #Create dummies variables
    df = pd.get_dummies(df,columns=['Embarked','Status'])
    #Create variables to show relationships
    df['isChild'] = np.logical_and(df['Parch']>0,df['Age'] <= 18)
    df['isParent'] = np.logical_and(df['Parch']>0,df['Age'] > 18)
    df['isSibling'] = np.logical_or(np.logical_and(df['SibSp']>0,np.logical_and(df['Age'] <= 18, df['Sex']==1)),np.logical_and(df['SibSp']>0,df['Status_4']))
    df['isSpouse'] = np.logical_or(np.logical_and(df['SibSp']>0,np.logical_and(df['Age'] > 18, df['Sex']==1)),np.logical_and(df['SibSp']>0,df['Status_4'].apply(lambda x:0 if x==1 else 1)))
    df['Travel alone'] = np.logical_and(df['SibSp']==0,df['Parch']==0).apply(int)
    df['Individual Fare'] = df['Fare']/(df['SibSp'] + df['Parch'] + 1)
    df['Ticket grouped'] = df['Ticket'].apply(group_tickets)
    for i in range(1,16):
        if 'Status_'+str(i) not in df.columns:
            df['Status_'+str(i)] = [0 for k in range(len(df.index))]
    return df

#Create the train dataframe
df = read_treat_file('train.csv')

#These are functions to separe dataset into a train one and a target one

#These are the base inputs, which can change depending on what works the best for the model
inputs = ['isChild','isSpouse','Sex','Age','SibSp','Parch','Pclass','Embarked_C','Embarked_Q','Embarked_S','Fare','Individual Fare','Travel alone','Cabin']
#These are the features that can find bounds between passengers
liste_inputs_parents = ['Travel alone','Ticket grouped','Fare','Individual Fare','Ticket','Sex','Age','isParent','isChild','isSpouse','isSibling','SibSp','Last Name','Parch','Pclass','Cabin','Embarked_C','Embarked_Q','Embarked_S']

for i in range(1,16):
    liste_inputs.append('Status_' + str(i))
    liste_inputs_parents.append('Status_' + str(i))

#Separe the dataframe into features and target
def separe(df,inputs = liste_inputs):
    df_predictor = df[inputs]
    df_target = df['Survived']
    df_predictor.dropna()
    return df_predictor,df_target

#Normalized the datas while separing
def separe_normalized(df):
    df_predictor = df[liste_inputs]
    for i in df_predictor.columns:
        df_predictor[i] = (df_predictor[i]-np.mean(df_predictor[i]))/np.var(df_predictor[i])
    df_target = df['Survived']
    df_predictor.dropna()
    return df_predictor,df_target

def separe_for_parents(df):
    df_predictor = df[liste_inputs_parents]
    df_target = df['Survived']
    df_predictor.dropna()
    return df_predictor,df_target

#This function return a cross validated score for a classifier, parameters are : the train dataframe,
#the list of features used to train this classifier, the classifier, and a boolean depending on whether we use lda or not
def cross_validate_classifier(df,inputs,clf,bool_lda):
    df_predictor,df_target = separe(df,inputs)
    lda = LinearDiscriminantAnalysis()
    if bool_lda:
        X_lda = lda.fit(df_predictor,df_target).transform(df_predictor)
    else:
        X_lda = df_predictor
    clf.fit(X_lda,df_target)
    cv_results = cross_val_score(clf,df_predictor,df_target,cv = 8)
    results = cv_results.mean()
    print(results)
    print(np.var(cv_results))

#This function cross validated score for a classifier with in parallell the manual classifier that shed light on bounds between passengers
def cross_validate_manual_classifier(df,inputs,clf,bool_lda):
    df_predictor,df_target = separe_for_parents(df.dropna())
    lda = LinearDiscriminantAnalysis()
    kf = KFold(n_splits=6,shuffle = False)
    results = []
    results_corrected = []
    lda = LinearDiscriminantAnalysis()
    for train_index,test_index in kf.split(df_predictor):
        X_train, X_test = df_predictor.iloc[train_index], df_predictor.iloc[test_index]
        y_train, y_test = df_target.iloc[train_index], df_target.iloc[test_index]
        if bool_lda:
            X_train_lda = lda.fit(X_train[inputs],y_train).transform(X_train[inputs])
            X_test_lda = lda.transform(X_test[inputs])
        else:
            X_train_lda = X_train[inputs]
            X_test_lda = X_test[inputs]
        
        clf.fit(X_train_lda,y_train)
        result1 = clf.predict(X_test_lda)
        result1 = pd.DataFrame({'Survived':result1},index=X_test.index)
        results.append(clf.score(X_test_lda,y_test))

        result2 = predict_parents(pd.concat([X_train,y_train],axis=1),X_test)
        result1.loc[result2.index,:] = result2.values.reshape((result2.values.shape[0],1))
        results_corrected.append(sum(result1['Survived'] == y_test)/result1.shape[0])

    print(np.mean(results))
    print(np.mean(results_corrected))

#This is the manual classifier, it needs the train set end the test set
def predict_parents(train,test):
parent_list = []
for val,row in train.iterrows():
    #If the child is dead, parents have died
    if row['Survived']==0 and row['isChild']:
        parent = test[test['Ticket']==row['Ticket']]
        for val1,row1 in parent.iterrows():
            dict_1 = {'PassengerId':val1,'Survived':0}
            parent_list.append(dict_1)
    #If the parent survived, children have survived
    if row['isParent'] and row['Survived'] == 1:
        children = test[np.logical_and(test['Ticket']==row['Ticket'],test['Age']<16)]
        for val1,row1 in children.iterrows():
            dict_1 = {'PassengerId':val1,'Survived':1}
            parent_list.append(dict_1)
    #If the husband survived, the wife survived
    if row['Survived']==1 and row['isSpouse'] and row['Sex']==1:
        spouse = test[np.logical_and(np.logical_and(test['Ticket']==row['Ticket'],test['isSpouse']),test['Sex']==0)]
        for val1,row1 in spouse.iterrows():
            dict_1 = {'PassengerId':val1,'Survived':1}
            parent_list.append(dict_1)
    #if the wife is dead, the husband also
    if row['Survived']==0 and row['isSpouse'] and row['Sex']==0:
        spouse = test[np.logical_and(np.logical_and(test['Ticket']==row['Ticket'],test['isSpouse']),test['Sex']==1)]
        for val1,row1 in spouse.iterrows():
            dict_1 = {'PassengerId':val1,'Survived':0}
            parent_list.append(dict_1)
#We predict whether a child is dead or not depending on his brother and sister
for val,row in test.iterrows():
    if row['isSibling']:
        children = train[np.logical_and(train['isSibling'],train['Ticket'] == row['Ticket'])]
        if(children.size > 0):
            sum_survived = sum(children['Survived'])
            if sum_survived/children.shape[0] > 0.5:
                dict_1 = {'PassengerId':val,'Survived':1}
            else:
                dict_1 = {'PassengerId':val,'Survived':0}
df_result = pd.DataFrame(parent_list).groupby('PassengerId').agg(np.mean)
return np.round(df_result['Survived']).apply(int)

#This is a function allowing us to generate a csv containing the prediction
def final_output(df,df_final,inputs,clf,bool_lda):
    lda = LinearDiscriminantAnalysis()
    df_final_predictor = df_final[inputs]
    df_predictor,df_target = separe(df,inputs = inputs)
    if bool_lda:
        lda = LinearDiscriminantAnalysis()
        X_lda = lda.fit(df_predictor,df_target).transform(df_predictor)
        df_final_lda = lda.transform(df_final_predictor)
    else:
        X_lda = df_predictor
        df_final_lda = df_final_predictor
    clf.fit(X_lda,df_target)
    y_result = clf.predict(df_final_lda)
    result_parents = predict_parents(df,df_final)
    df_resultat = pd.DataFrame({'PassengerId':df_final.index,'Survived':y_result})
    df_resultat.index = df_resultat['PassengerId']
    df_resultat.loc[result_parents.index.values,'Survived'] = result_parents 
    df_resultat.to_csv('resultat.csv', sep=',',index=False)

#######################Testing Random Forest without LDA################################################
#inputs = ['isChild','isSpouse','Sex','Age','SibSp','Parch','Pclass','Embarked_C','Embarked_Q','Embarked_S','Fare','Individual Fare','Travel alone','Cabin']
#for i in range(1,16):
    #inputs.append('Status_' + str(i))
#clf = RandomForestClassifier(n_estimators = 500,random_state=42,bootstrap = True,max_depth=10,min_samples_split=15,min_samples_leaf=4)
#cross_validate_classifier(df,inputs,clf,False)
#cross_validate_manual_classifier(df,inputs,clf,False)
#final_output(df,read_treat_file('test.csv'),inputs,clf,False)

#######################"Testing Logistic Regression #######################################################"
#inputs = ['isSpouse','Sex','Age','SibSp','Parch','Pclass','Embarked_C','Embarked_Q','Embarked_S','Fare','Individual Fare','Travel alone','Cabin']
#for i in range(1,16):
#    inputs.append('Status_' + str(i))
#clf = LogisticRegression(solver='liblinear',penalty='l2')
#cross_validate_classifier(df,inputs,clf,True)
#cross_validate_manual_classifier(df,inputs,clf,True)
#final_output(df,read_treat_file('test.csv'),inputs,clf,True)

###################### Our code with SVM that gave her our best submission ###################################
inputs = ['isSpouse','Sex','Age','SibSp','Parch','Pclass','Embarked_C','Embarked_Q','Embarked_S','Travel alone','Cabin']
for i in range(1,16):
    inputs.append('Status_' + str(i))
clf=SVC(kernel='rbf',C=100,gamma='auto',shrinking=False,probability = False,tol=0.01,max_iter=-1)
cross_validate_classifier(df,inputs,clf,True)
cross_validate_manual_classifier(df,inputs,clf,True)
final_output(df,read_treat_file('test.csv'),inputs,clf,True)

#GAVE A SCORE OF 81.339 ON KAGGLE