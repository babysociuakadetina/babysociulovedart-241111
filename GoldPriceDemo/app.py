import streamlit as st
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error, mean_absolute_percentage_error

df = pd.read_csv('Data sample_GoldUP.csv',encoding="UTF-8")
original_df = df.copy(deep=True)
df['Date'] = pd.to_datetime(df['Date'])

df['Day'] = df['Date'].dt.day
#df['dayofweek'] = df['date'].dt.dayofweek  # Lấy thứ trong tuần (0=Monday, 6=Sunday)
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

df.drop(['Date'],axis=1, inplace=True)
rn,cn= original_df.shape
# Loại bỏ duplicate rows (nếu có)
df.drop_duplicates(inplace=True)
# Tách biến là target và features là các biến độc lập
target = 'Gold_Price'
features = [i for i in df.columns if i not in target]
#display(features)
# Checking number of unique rows in each feature
nu = df[features].nunique().sort_values()

df1 = df.copy()
for i in features:
    Q1 = df1[i].quantile(0.25)
    Q3 = df1[i].quantile(0.75)
    IQR = Q3-Q1
    df1 = df1[df1[i] <= (Q3+(1.5*IQR))]
    df1 = df1[df1[i] >= (Q1-(1.5*IQR))]
    df1 = df1.reset_index(drop= True)
lst = []
for i in df1.columns.values:
    lst.append(i.replace(' ','_'))
#print(lst)

df.columns = lst

X = df1.drop([target], axis =1)
#print(X)
Y = df1[target]
Train_X, Test_X, Train_Y, Test_Y = train_test_split(X,Y, train_size = 0.8, test_size=0.2, random_state = 100)
Train_X.reset_index(drop= True, inplace= True)
#Feature scaling (Standardization)
std = StandardScaler()

Train_X_std = std.fit_transform(Train_X)
Train_X_std = pd.DataFrame(Train_X_std, columns = X.columns )
#display(Train_X.head())

Test_X_std = std.transform(Test_X)
Test_X_std = pd.DataFrame(Test_X_std, columns = X.columns)
#display(Test_X.head())
LR = LinearRegression()
# Huấn luyện mô hình MLR
sklearn_model = LR.fit(Train_X_std, Train_Y)
predict1 = sklearn_model.predict(Train_X_std)
# display(predict1)
from sklearn.metrics import root_mean_squared_error

predict2 = LR.predict(Test_X_std)
Initial_train_RMSE = np.sqrt(mean_squared_error(Train_Y, predict1))
Initial_test_RMSE = np.sqrt(mean_squared_error(Test_Y, predict2))
# print('Initial train RMSE:', Initial_train_RMSE)
# print('Initial test RMSE:', Initial_test_RMSE)
MLR = LinearRegression().fit(Train_X_std, Train_Y)
predict1 = MLR.predict(Train_X_std)
predict2 = MLR.predict(Test_X_std)
22# Hàm để dự đoán giá vàng dựa trên giá trị của 9 biến độc lập
def predict_gold_price(variables):
    if len(variables) != 9:
        raise ValueError("Value")
    y_pred = MLR.intercept_ + sum(coef * var for coef, var in zip(MLR.coef_, variables))
    return y_pred
# Streamlit UI
# Streamlit UI
st.title(":trident: Gold Price Prediction :trident: ")
st.header("Enter values for the independent variables:")
variables = []
for col in X.columns:
    user_input = st.number_input(f"{col}:", value=0)
    variables.append(float(user_input))

if st.button("Predict"):
    predicted_gold_price = predict_gold_price(variables)
    st.write(predicted_gold_price)


