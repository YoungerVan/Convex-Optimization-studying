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
beta=np.mat([[-0.1]*31]).T
def get_diag(A,beta):
    s_mu,mu_list=[],[]
    for i in range(A.shape[1]):
        inner_product=float(-A[:,i].T*beta)
        if inner_product<-10:
            inner_product=-10
        elif inner_product>10:
            inner_product=10
        mu=1/(1+exp(inner_product))
        mu_list.append(mu)
        s=mu*(1-mu)
        s_mu.append(s)
    S=np.diag(s_mu)
    return np.mat(S),np.mat(mu_list).T
plt.figure()
for i in range(500):
    S,mu=get_diag(A,beta)
    H_0=(A*S*A.T).astype(float)
    H=H_0.I
    minus=mu-y
    nabla=A*minus
    beta=beta-H*nabla
    plt.scatter(i,np.linalg.norm(np.array(nabla).astype(float)))
    if np.linalg.norm(np.array(nabla).astype(float), ord=2)<= 0.3:
         break
plt.show()
print(beta)