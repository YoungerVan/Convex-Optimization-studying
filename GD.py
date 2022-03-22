import pandas as pd
import numpy as np
from math import exp
import matplotlib.pyplot as plt
from sklearn import preprocessing

dataset=pd.read_table(r'C:\Users\hp\Desktop\机器学习\wdbc.data',sep=',')
dataset.loc[dataset.shape[0]]=dataset.columns#将被作为列名的数据添加到样本集中
feature_number_int=range(1,31)
feature_number=[]
for i in range(len(feature_number_int)):
    feature_number.append("Feature"+str(feature_number_int[i]) )#新列名
dataset.columns=['ID','Diagnosis']+feature_number
standard_dataset = preprocessing.scale(dataset.iloc[:,2:])
dataset.iloc[:,2:33]=standard_dataset
dataset.iloc[dataset['Diagnosis']=='B',1]=1
dataset.iloc[dataset['Diagnosis']=='M',1]=0
dataset['changshu'] =1.0
label=dataset.iloc[:,1]
feature=dataset.iloc[:,2:]
A=np.mat(feature).T
y=np.mat(label).reshape(569,1)
beta=np.mat([[0.1]*31]).T
T=1000
s=0.01
def get_mu(A,beta):
    mu=[]
    for i in range(A.shape[1]):
        inner_product=-A[:,i].T*beta
        if inner_product<-20:
            inner_product=-20
        elif inner_product>20:
            inner_product=20
        m=1/(1+exp(inner_product))
        mu.append(m)
    return np.mat(mu).T
#plt.figure()
for i in range(T):
    mu=get_mu(A,beta)
    minus=mu-y
    nabla=A*minus
    beta=beta-s*nabla
    print(i)
    #plt.scatter(i,get_fvalue(A,y,beta))
#plt.show()
print(beta)