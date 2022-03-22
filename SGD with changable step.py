import numpy as np
from  math import exp,log
import random
import sklearn.datasets
import matplotlib.pyplot as plt
A, b= sklearn.datasets.load_svmlight_file(r"C:\Users\hp\Desktop\a9a.txt")
x=np.array([0.1]*A.shape[1]).T
lam=0.0001
step=0.01
row = A.shape[0]
T=4000
def get_loss(A, x, lam, b):
    loss=0
    for i in range(A.shape[0]):
        a_i=A[i,:]
        b_i=b[i]
        inner_product = a_i.dot(x)
        exp_inner_product = exp(-b_i * inner_product)
        x_norm = np.linalg.norm(x, ord=2)
        loss += log(1 + exp_inner_product) + lam * x_norm
    return loss
def SGD_fixed(A,x,lam,step,b,row):
    i = random.randint(0, row-1)
    a_i=A[i,:]
    b_i=b[i]
    inner_product=a_i.dot(x)
    exp_inner_product=exp(-b_i*inner_product)
    exp_lest=(inner_product*exp_inner_product)/(1+exp_inner_product)
    grad=exp_lest*a_i+2*lam*x
    x=x-step*grad
    return x,i
plt.figure()
for j in range(T):
    x,c=SGD_fixed(A,x,lam,1/(j+1),b,row)
    if j<200:
        loss = get_loss(A, x, lam,b)
        plt.scatter(j, loss)
plt.show()
print(x)

