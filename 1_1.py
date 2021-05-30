''' Question 1.1: Create a table in the report stating the following information about the adult data
set: (i) number of instances, (ii) number of missing values, (iii) fraction of missing values over
all attribute values, (iv) number of instances with missing values and (v) fraction of instances
with missing values over all instances. '''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Adult.csv')
dataset = dataset.drop(['fnlwgt'], axis = 1) 

#(i) number of instances
print('Number of instances: ', len(dataset))


#(ii) number of missing values
print('Number of missing values: ', dataset.isna().sum().sum())


# (iii) fraction of missing values over all attribute values.
from fractions import Fraction 
print ('Fraction of missing values over all attribute values: ', Fraction(dataset.isna().sum().sum(), dataset.count(0).sum()))

#(iv) number of instances with missing values
print ('Number of instances with missing values: ', sum([True for idx,row in dataset.iterrows() if any(row.isnull())]))


#(v) fraction of instances with missing values over all instances.
from fractions import Fraction 
print ('Fraction of instances with missing values over all instances: ', Fraction(sum([True for idx,row in dataset.iterrows() if any(row.isnull())]), len(dataset)  ))



####################################################################################################

''' Question 1.2: Convert all 13 attributes into nominal using a Scikit-learn LabelEncoder. 
    Then, print the set of all possible discrete values for each attribute. '''


#Extracting data for 13 attributes
X=dataset.iloc[:,:-1]

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

#Converting all 13 attributes into nominal using a Scikit-learn LabelEncoder. 
encoder_dict = defaultdict(LabelEncoder)
encoded_label = X.apply(lambda x: encoder_dict[x.name].fit_transform(x.astype(str)))

#Decoding back to original values to print the discrete values 
inverse_transform_lambda = lambda x: encoder_dict[x.name].inverse_transform(x)
decoded_label = encoded_label.apply(inverse_transform_lambda)

#The below code will print the set of all possible discrete values for each attribute
c=0
for i in list(decoded_label.columns):
    c=c+1
    print('Attribute ' , c, ':-',i) 
    print(decoded_label[i].unique())
    print('\n')
        
