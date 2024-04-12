import pickle as p1
import pandas as pd
from sklearn import datasets, linear_model
data = pd.read_csv("optdigits.tra", sep=',', header=None)
data_X=data.iloc[:,0:64]
data_Y=data.iloc[:,64:65]
print(data_X)
print(data_Y.T)
print(data_X)
print(data_Y)
regr = linear_model.LinearRegression()
preditor_linear_model=regr.fit(data_X, data_Y)
preditor_Pickle = open('../numbers.pkl', 'wb')
print("number_predictor")
p1.dump(preditor_linear_model, preditor_Pickle)