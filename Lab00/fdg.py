import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yz as yz
from sklearn import linear_model
XX =  pd.read_csv("../Lab00/casas1.csv")
X = np.array(XX, ndmin=2) #matriz n\times2
y = X[: , 1:].T
y = y[0]
X = X[: , :1]
print(y)
print(X)
regr = linear_model.LinearRegression()
z = regr.fit(X, y)
yz = regr.predict(X)
yz = yz.round(3)
plt.plot(X, yz, color="blue", linewidth = 3)
plt.show()
