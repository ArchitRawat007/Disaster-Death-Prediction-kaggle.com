import pandas as pd
import numpy as np 

#importing the required datasets into dataframes
trainset = pd.read_csv('train.csv',index_col = 0 )
testset = pd.read_csv('test.csv',index_col = 0)
result = pd.read_csv('gender_submission.csv',index_col = 0)

#Examining the training dataset
trainset.head()

#Preparing the training dataset for machine learning by filling missing values, preprocessing etc 
X_train = trainset.drop(['Name','Survived','Ticket','Cabin'],axis = 1)
mean = X_train['Age'].mean()
X_train['Age'] = X_train['Age'].fillna(mean).astype(int)
X_train['Fare'] = X_train['Fare'].fillna(0).astype(int)

Y_train = trainset['Survived']

X_train = X_train.join(pd.DataFrame(pd.get_dummies(X_train['Pclass'])))
X_train= X_train.drop('Pclass',axis = 1,)

X_train = X_train.join(pd.DataFrame(pd.get_dummies(X_train['Embarked'])))
X_train= X_train.drop('Embarked',axis = 1,)

X_train = X_train.join(pd.DataFrame(pd.get_dummies(X_train['Sex'])))
X_train= X_train.drop('Sex',axis = 1,)
print (X_train)

# Import the StandardScaler
from sklearn.preprocessing import StandardScaler

# Scale the features and set the values to a new variable
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

#Preparing the training dataset for machine learning by filling missing values, preprocessing etc

X_test = testset.drop(['Name','Ticket','Cabin'],axis = 1)
avg = X_test['Age'].mean()
X_test['Age'] = X_test['Age'].fillna(avg).astype(int)
X_test['Fare'] = X_test['Fare'].fillna(0).astype(int)

X_test = X_test.join(pd.DataFrame(pd.get_dummies(X_test['Pclass'])))
X_test= X_test.drop('Pclass',axis = 1,)

X_test = X_test.join(pd.DataFrame(pd.get_dummies(X_test['Embarked'])))
X_test= X_test.drop('Embarked',axis = 1,)

X_test = X_test.join(pd.DataFrame(pd.get_dummies(X_test['Sex'])))
X_test= X_test.drop('Sex',axis = 1,)
print (X_test)

# Import the StandardScaler
from sklearn.preprocessing import StandardScaler

# Scale the features and set the values to a new variable
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

Y_test = result

#Applying Adaptive Boosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
dt = DecisionTreeClassifier(max_depth=1,random_state=10)
adb = AdaBoostClassifier(base_estimator=dt,n_estimators=100)
adb.fit(X_train,Y_train)
Y_pred = adb.predict(X_test)

#Setting a K-Fold CV so that hyperparameters of our model could be optimized afterwards
from sklearn.model_selection import KFold, cross_val_score

# Set up our K-fold cross-validation
kf = KFold(20,random_state=10)

#Generating the report matrix for Adaptive Boosting thus reflecting f1-score, recall, precision etc.
from sklearn.metrics import classification_report
rpt = classification_report(Y_test,Y_pred)
print(rpt)

#Trying XGBoost in case it can provide better results
import xgboost as xgb
xg = xgb.XGBClassifier(Objective = 'binary:logistic',n_estimators=10,seed = 123)
xg.fit(X_train,Y_train)
Y_pred_xg = xg.predict(X_test)

#Generating the report for XGBoost
report = classification_report(Y_test,Y_pred_xg)
print(report)

#Chose XGBoost as it gave a f1-score of 0.97 
#A new  output dataframe in desired format
tests = pd.read_csv('test.csv')
Pas = tests['PassengerId'].values
df = pd.DataFrame({'PassengerId':Pas, 'Survived':Y_pred_xg})
cols = df.columns.tolist()

df = df[cols]
df = df.set_index(['PassengerId'])
print(df.head())

#Exporting the dataframe to csv file
df.to_csv('output.csv')

