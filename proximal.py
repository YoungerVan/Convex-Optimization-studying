import numpy as np
np.random.seed(2021) # set a constant seed to get samerandom matrixs
A = np.random.rand(500, 100)
x = np.zeros([100, 1])
x[:5, 0] += np.array([i+1 for i in range(5)]) # x_denotes expected x
b = np.matmul(A, x) + np.random.randn(500, 1) * 0.1 #add a noise to b
lam = 0.1 # try some different values in {0.1, 1, 10}
def grad(x,A,b):
    AT=A.T
    Ax=np.matmul(A,x)
    Axb=Ax-b
    grad1=np.matmul(AT,Axb)
    return grad1
def proximal(z,lam,r):
    x_zero=np.zeros([100,1])
    for i in range(100):
        if z[i]>r*lam:
            x_zero[i]=z[i]-r*lam
        elif z[i]<-r*lam:
            x_zero[i]=z[i]+r*lam
        else:
            x_zero[i]=0
    return x_zero
def get_eig(A):
    AT=A.T
    ATA=np.matmul(AT,A)
    u,v=np.linalg.eig(ATA)
    return 1/max(u)
def proximal_gradec(x,A,b,k):
    x0=x
    if k==0:
        return x0
    else:
        eig=get_eig(A)
        zt=x-eig*grad(x,A,b)
        x0=proximal(zt,lam,eig)
        k=k-1
        return proximal_gradec(x0,A,b,k)

test=proximal_gradec(x,A,b,101)
print(test)