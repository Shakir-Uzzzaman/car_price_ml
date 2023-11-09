import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import numpy as np


car_data=pd.read_csv(r'F:\machine learning\car_price\car data.csv')

c_type=car_data.dtypes
print(c_type)

car_data.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

car_data.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)


car_data.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

print(car_data)
x=car_data.drop(['Car_Name','Selling_Price'],axis=1) 
#axis=1 means collum and axis=0 means row,so it must be include
y=car_data['Selling_Price']
print(x)

x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.1,random_state=2)
print(x_test.shape)

lin_reg_model=LinearRegression()
lin_reg_model.fit(x_train,y_train)


training_data_prediction =lin_reg_model.predict(x_train)
error_score = metrics.r2_score(y_train, training_data_prediction)
print("R squared Error : ", error_score)




test_data_prediction =lin_reg_model.predict(x_test)
error_score = metrics.r2_score(y_test, test_data_prediction)
print("R squared Error : ", error_score)









