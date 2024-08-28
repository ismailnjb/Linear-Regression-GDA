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

#gradient descent algorithm
def hypothesis(x,theta):
    return theta[0]+theta[1]*x

def error(x,y,theta):
    m=x.shape[0]
    error=0
    for i in range(m):
        hx=hypothesis(x[i],theta)
        error+=(hx-y[i])**2
    return error

#update rule
def gradient(x,y,theta):
    grad=np.zeros((2,))
    m=x.shape[0]
    for i in range(m):
        hx=hypothesis(x[i],theta)
        grad[0]+=(hx-y[i])
        grad[1]+=(hx-y[i])*x[i]
    return grad

#Algorithm
def gradientDescent(x,y,learning_rate=0.001):
    theta=np.zeros((2,))
    itr=0
    max_itr=100
    error_list=[]
    theta_list=[]
    while(itr<=max_itr):
        grad=gradient(x,y,theta)
        e=error(x,y,theta)
        error_list.append(e)
        theta_list.append((theta[0],theta[1]))
        theta[0]=theta[0]-learning_rate*grad[0]
        theta[1]=theta[1]-learning_rate*grad[1]
        itr+=1
    return theta,error_list,theta_list

final_theta,error_list,theta_list=gradientDescent(x,y)
plt.plot(error_list)
plt.show()

print(final_theta)

xtest=np.linspace(-2,6,10)
print(xtest)    

plt.scatter(x,y,label='Training Data')
plt.plot(xtest,hypothesis(xtest,final_theta),color='orange',label="Prediction")
plt.legend()    
plt.show()
        



