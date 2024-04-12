import pickle as p1
import pandas as pd
data = pd.read_csv("optdigits.tra", sep=',', header=None)
data_X=data.iloc[:,0:64]
data_Y=data.iloc[:,64:65]
print(data_X)
print(data_Y.T)
loaded_model = p1.load(open('../numbers.pkl', 'rb'))
print("Coefficients: \n", loaded_model.coef_)
y_pred = loaded_model.predict(data_X)
z_pred = y_pred-data_Y
print("number_predictor \n")
print("y_pred", y_pred, "\n")
right=0
wrong=0
total=0
for x in z_pred:
    z=int(x)
    total=total+1
    if z==0:
        right=right+1
    else:
        wrong=wrong+1
print("accuraccy1= ",right/total,"accuraccy2= ",wrong/total, "\n")
print("z_pred", z_pred)