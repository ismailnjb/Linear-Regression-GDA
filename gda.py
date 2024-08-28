import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



dfx=pd.read_csv('./linearX.csv')
dfy=pd.read_csv('./linearY.csv')

dfx=dfx.values
dfy=dfy.values
x= dfx.reshape((-1,))
y= dfy.reshape((-1,))

x=(x-x.mean())/x.std()


plt.scatter(x,y)
plt.show()