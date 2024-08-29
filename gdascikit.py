import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dfx=pd.read_csv('./linearX.csv')
dfy=pd.read_csv('./linearY.csv')

dfx=dfx.values
dfy=dfy.values
x= dfx.reshape((-1,1))
y= dfy.reshape((-1,1    ))

x=(x-x.mean())/x.std()


plt.scatter(x,y)
plt.show()

from sklearn.linear_model import LinearRegression
# Create the model
model = LinearRegression()
# Fit the model
model.fit(x, y)
#predict
output = model.predict(x)
bias = model.intercept_
coeff = model.coef_
print(bias, coeff)

model.score(x, y)

plt.scatter(x, y)
plt.plot(x, output, color='red')
plt.legend()
plt.show()
