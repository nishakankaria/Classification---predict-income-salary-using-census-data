'''Q1.3 Ignore any instance with missing value(s) and use Scikit-learn to build a decision tree
for classifying an individual to one of the <= 50K and > 50K categories. Compute the error
rate of the resulting tree.'''

import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

# Ignore warnings
import warnings
warnings.filterwarnings('ignore');

#---------------------Read the data------------------

D = pd.read_csv('Adult.csv')
D = D.drop(['fnlwgt'], axis = 1) 

#Ignore any instance with missing value(s)

df=D
df = df.dropna() 


#-------------------------Data Analyzing-------------------

#taking a look at the data first and visualising to understand it better

rcParams['figure.figsize'] = 20, 12
df[['age','education-num', 'capitalgain', 'capitalloss', 'hoursperweek']].hist()

#--bucketing the age into separate bins and plotting a bar graph for Age against Class to see the co-relation between these columns 
df['age'] = pd.cut(df['age'], bins = [0, 1.5, 3.3, 4], labels = ['Young', 'Adult', 'Old'])
sns.countplot(x = 'age', hue = 'class', data = df)
df['age'].fillna('Young', inplace=True)


#--bucketing the hoursperweek into separate bins and plotting a bar graph for Hours Per Week against Class to see the co-relation between these columns 
df['hoursperweek'] = pd.cut(df['hoursperweek'],  bins = [0, 2,3,4], labels = ['Lesser Hours', 'Normal Hours', 'Extra Hours'])
df['hoursperweek'].fillna('Lesser Hours', inplace=True)


#--checking if there is any relation between Education and education-num.
education = df['education'].unique()
for edu in education:
    print("education:  {}, ; education-num: {}"
          .format(edu, df[df['education'] == edu]['education-num'].unique()))
    
#Note: It can be infered that education-num and education are giving similar information, hence it's better to delete the education-num attribute
#Feature removal: education-num
df = df.drop(['education-num'], axis = 1) 


#-------For capital gain and capital loss: Defining a value of 0 as 'No' and 1 as 'Yes'

df.astype({'capitalloss': 'object'}).dtypes
df.astype({'capitalgain': 'object'}).dtypes
df.loc[df['capitalloss']!=0,'capitalloss'] = 'Yes'
df.loc[df['capitalloss']!='Yes','capitalloss'] = 'No'
df.loc[df['capitalgain']!=0,'capitalgain'] = 'Yes'
df.loc[df['capitalgain']!='Yes','capitalgain'] = 'No'



#--plotting a bar graph for Education against Class to see the co-relation between these columns 
fig = plt.figure(figsize = (9,6))
ax=sns.countplot(df['education'], hue=df['class'])
ax.set_title('Education vs Income')
plt.xlabel("Education",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('')


#combining all information from 10th to 12th into one class, HS-grad. 
hs_grad = ['HS-grad','11th','10th','9th','12th']
df['education'].replace(to_replace = hs_grad,value = 'HS-grad',inplace = True)

#combining all information from 1st to 8th into one class,elementary. 
elementary = ['1st-4th','5th-6th','7th-8th']
df['education'].replace(to_replace = elementary,value = 'elementary_school',inplace = True)   


#--plotting a bar graph for Marital status against Class to see the co-relation between these columns 
fig = plt.figure(figsize = (9,6))
ax=sns.countplot(df['marital-status'], hue=df['class'])
ax.set_title('Marital Status vs Income')
plt.xlabel("Marital Status",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('')

#Combining Married-civ-spouse,Married-spouse-absent,Married-AF-spouse information under category 'Married'
married= ['Married-spouse-absent','Married-civ-spouse','Married-AF-spouse']
df['marital-status'].replace(to_replace = married ,value = 'Married',inplace = True)

#Combining Divorced, separated again comes under category 'separated'.
separated = ['Separated','Divorced']
df['marital-status'].replace(to_replace = separated,value = 'Separated',inplace = True)


#plotting a bar graph for Workclass against Class to see the co-relation between these columns 
fig = plt.figure(figsize = (9,6))
ax=sns.countplot(df['workclass'], hue=df['class'])
ax.set_title('Workclass vs Income')
plt.xlabel("workclass",fontsize = 14);
plt.xticks(rotation=90)
plt.ylabel('')


#Combining Self-emp-not-inc, Self-emp-inc information under category self employed
self_employed = ['Self-emp-not-inc','Self-emp-inc']
df['workclass'].replace(to_replace = self_employed ,value = 'Self_employed',inplace = True)


#Combining Local-gov,State-gov,Federal-gov information under category goverment emloyees
govt_employees = ['Local-gov','State-gov','Federal-gov']
df['workclass'].replace(to_replace = govt_employees,value = 'Govt_employees',inplace = True)



#Extracting all Independent variables
X = df.iloc[:,:-1].values
#Dependant variable
y = df.iloc[:,-1].values


#Encode the independent variable (categorical) using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
X= ohe.fit_transform(X)


#Encode the dependant variable using LabelEncoder

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y= le.fit_transform(y)

#Splitting the dataset into the training set and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.2,random_state = 0)

#Training the model
from sklearn import tree
dectree = tree.DecisionTreeClassifier()
DTmodel = dectree.fit(X_train,y_train)
y_pred = DTmodel.predict(X_test)


#Computing the accuracy score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred) 
print('Q1.3:- Accuracy Score of the resulting tree :', accuracy)


#Computing the error rate of the resulting tree
error_rate = 1 - accuracy
print('Q1.3:- Error rate of the resulting tree:- ', error_rate)


'''-----------------Q1.4-------------------'''
#storing the original dataset information into a new dataframe
D_new = D

#deleting education-num attribute because as per the analysis done for Q1.3,it was found that education and education-num are same. 
D_new = D_new.drop(['education-num'], axis = 1) 

#constructing a smaller data set D' from the original data set, containig all instances with at least one missing value,
D_hat = D_new[D_new.isnull().any(axis=1)]

#an equal number of randomly selected instances without missing values.
D_without_nan = D_new.dropna()
D_without_nan = D_without_nan.sample(n=len(D_hat),replace=True)
D_hat = D_hat.append(D_without_nan, ignore_index=True)

D_hat_1 = D_hat
D_hat_1 = D_hat.replace(np.nan, 'missing', regex=True)

D_hat_2 = D_hat
D_hat_2 = D_hat_2.fillna(D_hat_2.mode().iloc[0])

#Extracting all Independent variables of D'1
X1 = D_hat_1.iloc[:,:-1].values
#Dependant variable of D'1
y1 = D_hat_1.iloc[:,-1].values

#Encode D'1 independent variables
ohe1 = OneHotEncoder()
X1= ohe1.fit_transform(X1)

#Encode D'1 dependant variables
le1 = LabelEncoder()
y1= le1.fit_transform(y1)

#Extracting all Independent variables of D'2
X2 = D_hat_1.iloc[:,:-1].values

#Dependant variable of D'2
y2 = D_hat_1.iloc[:,-1].values

#Encode D'2 independent variables
ohe2 = OneHotEncoder()
X2= ohe2.fit_transform(X2)

#Encode D'2 dependant variables
le2 = LabelEncoder()
y2= le2.fit_transform(y2)


#Extracting sample data from original dataset for testing.
testdata = D_new.sample(n = 40000, random_state=0) 

#Handling missing data
td1 = testdata
td1 = td1.replace(np.nan, 'missing', regex=True)

#Handling missing data
td2 = testdata
td2 = td2.fillna(td2.mode().iloc[0])

td1_X = td1.iloc[:,:-1].values
td1_y = td1.iloc[:,-1].values

td2_X = td1.iloc[:,:-1].values
td2_y = td1.iloc[:,-1].values

#Encode

from sklearn.preprocessing import OneHotEncoder
ohe_td1 = OneHotEncoder()
td1_X= ohe_td1.fit_transform(td1_X)

from sklearn.preprocessing import LabelEncoder
le_td1 = LabelEncoder()
td1_y= le_td1.fit_transform(td1_y)

#Encode

from sklearn.preprocessing import OneHotEncoder
ohe_td2 = OneHotEncoder()
td2_X= ohe_td2.fit_transform(td2_X)

from sklearn.preprocessing import LabelEncoder
le_td2 = LabelEncoder()
td2_y= le_td2.fit_transform(td2_y)


###
#Training D'1 decision tree
detree_1 = tree.DecisionTreeClassifier()
dtmodel_1 = detree_1.fit(X1,y1)
y_pred_1 = dtmodel_1.predict(td1_X)

#Computing Accuracy Score of  D'1 decision tree
accuracy_d1 = accuracy_score(td1_y, y_pred_1) 
print("Q1.4:- Accuracy Score of D'1 decision tree :", accuracy_d1)

#Computing error rate of  D'1 decision tree
error_rate_d1 = 1 - accuracy_d1
print("Q1.4:- Error rate of D'1 decision tree :", error_rate_d1)


####
#Training D'2 decision tree
detree_2 = tree.DecisionTreeClassifier()
dtmodel_2 = detree_2.fit(X2,y2)
y_pred_2 = dtmodel_2.predict(td2_X)

#Computing Accuracy Score of  D'2 decision tree
accuracy_d2 = accuracy_score(td2_y, y_pred_2) 
print("Q1.4:- Accuracy Score of D'2 decision tree :",accuracy_d2)

#Computing error rate of  D'2 decision tree
error_rate_d2 = 1 - accuracy_d2
print("Q1.4:- Error rate of D'2 decision tree :", error_rate_d2)
