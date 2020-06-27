from first import complete
from check import df
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
#print(df)
#print(complete)
df.drop_duplicates(inplace=True)
complete.drop_duplicates(inplace=True)
#dataset=pd.concat([complete,df],axis=1)
dataset=pd.merge(df,complete,how='left',on='Remarks')

#print(dataset.head)

for c in dataset.columns:
    lbl=LabelEncoder()
    if dataset[c].dtype=='object' and c in ['Question.1','Question.2','Question.3','Question.4','Question.5']:
        dataset[c]=lbl.fit_transform(dataset[c])

print(len(dataset.index))

train,test=train_test_split(dataset[['CGPA:','Question.1','Question.2','Question.3','Question.4','Question.5',
                                     'Score']],test_size=0.2)
print(len(train.index))
print(len(test.index))

train_x=train[['CGPA:','Question.1','Question.2','Question.3','Question.4','Question.5']]
train_y=train[['Score']]

test_x=test[['CGPA:','Question.1','Question.2','Question.3','Question.4','Question.5']]
test_y=test[['Score']]
print("Polynomial Regression")
poly=PolynomialFeatures(degree=2)
train_x_polynomial=poly.fit_transform(train_x)
test_x_polynomial=poly.fit_transform(test_x)

polynomial_regressor=LinearRegression()
polynomial_regressor.fit(train_x_polynomial,train_y)

predictions_polynomial=polynomial_regressor.predict(test_x_polynomial)

print("coefficients - ",polynomial_regressor.coef_)
print("mean squared eror :",mean_squared_error(predictions_polynomial,test_y))

print("Linear Regression")
linear_regressor=LinearRegression()
linear_regressor.fit(train_x,train_y)

predictions=linear_regressor.predict(test_x)

print("coefficients - ",linear_regressor.coef_)
print("mean squared eror :",mean_squared_error(predictions,test_y))

newdata=pd.DataFrame({"input":[1]})

x=float(input("enter cgpa"))
newdata=pd.merge(newdata,pd.DataFrame({"CGPA:":[x],"input":[1]}),how='left',on='input')
x=int(input("enter q1"))
newdata=pd.merge(newdata,pd.DataFrame({"Question.1":[x],"input":[1]}),how='left',on='input')

x=int(input("enter q2"))
newdata=pd.merge(newdata,pd.DataFrame({"Question.2":[x],"input":[1]}),how='left',on='input')
x=int(input("enter q3"))
newdata=pd.merge(newdata,pd.DataFrame({"Question.3":[x],"input":[1]}),how='left',on='input')
x=int(input("enter q4"))
newdata=pd.merge(newdata,pd.DataFrame({"Question.4":[x],"input":[1]}),how='left',on='input')
x=int(input("enter q5"))
newdata=pd.merge(newdata,pd.DataFrame({"Question.5":[x],"input":[1]}),how='left',on='input')
newdata=newdata.drop(['input'],axis=1)
print(newdata)

newdata_prediction=linear_regressor.predict(newdata)
newdata_prediction_polynomial=polynomial_regressor.predict(poly.fit_transform(newdata))
print("linear_regressor.prediction: ",newdata_prediction)
print("polynomial_regressor.prediction: ",newdata_prediction_polynomial)
