> 1_1.py file contains code for 1.1 and 1.2 questions of the coursework.
> 1_2.py file contains code for 1.3 and 1.4 questions of the coursework.

> 2.py file contains code for 2.1 and 2.2 questions of the coursework.


1_1.py source code:-
> Computing (i) number of instances, (ii) number of missing values, (iii) fraction of missing values over
all attribute values, (iv) number of instances with missing values and (v) fraction of instances
with missing values over all instances. The details of the same is mentioned in the Report pdf file as well.
> Extracting data for 13 attributes
> Converting all 13 attributes into nominal using a Scikit-learn LabelEncoder.
> Decoding back to original values to print the set of all possible discrete values for each attribute


1_2.py source code:-
>Reading the adult.csv
> Ignoring any instance with missing value(s)
> Taking a look at the data and drawing visualisation to analsye better.
> bucketing the age into separate bins and plotting a bar graph for Age against Class to see the co-relation between these columns 
> bucketing the hoursperweek into separate bins and plotting a bar graph for Hours Per Week against Class to see the co-relation between these columns 
> checking if there is any relation between Education and Education-num.
> It was found that education-num and education are giving similar information, therefore deleting the education-num attribute
> For capital gain and capital loss: Defining a value of 0 as 'No' and 1 as 'Yes'
> plotting a bar graph for each attribute against Class/Income to see the co-relation between these columns 
> For education: combining all information from 10th to 12th into one class, HS-grad. 
> For education: combining all information from 1st to 8th into one class,elementary. 
> Combining Married-civ-spouse,Married-spouse-absent,Married-AF-spouse information under category 'Married'
> Combining Divorced, separated again comes under category 'separated'.
> Combining Self-emp-not-inc, Self-emp-inc information under category self employed
> Combining Local-gov,State-gov,Federal-gov information under category goverment emloyees
> Encoding independent variable using OneHotEncoder and dependant variable using LabelEncoder
> Training the model and computing accuracy score and error rate
> constructing a smaller data set D' from the original data set D, containing (i) all instances with at least one missing value and (ii) an equal number of randomly selected instances without missing values.
> Replacing NaN with a value 'missing' for D'1
> Replacing NaN with the most frequent value for D'2
> Compyting error rate for D'1 and D'2


2.py source code
> Reading wholesale_customers.csv data
> Computing mean and range for each attribute. The details of the same is mentioned in the Report pdf file as well.
>  No. of cluster = 3 and constructing a scatterplot for each pair of attributes using Pyplot.  The details of the same is mentioned in the Report pdf file as well.