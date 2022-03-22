import numpy as np
np.random.seed(2021)
A = np.random.rand(500, 100)
x = np.zeros([100, 1])
x[:5, 0] += np.array([i+1 for i in range(5)]) # x_denotes expected x
b = np.matmul(A, x) + np.random.randn(500, 1)*0.1 #add a noise to b
lam = 0.1
rou=0.05
T=10
z=np.zeros([100, 1])
u=np.zeros([100, 1])
def x_update(A, x, b, z, u,rou):
    I=np.eye(x.shape[0])
    p=rou*I
    B=np.dot(A.T,A)+rou*p
    c=np.dot(A.T,b)+rou*z+rou*u
    x=np.dot(np.linalg.inv(B),c)
    return x
def z_update(x, u, lam):
    for i in range(z.shape[0]):
        condition1 = x[i,0] + u[i,0] - (lam / rou)
        condition2 = x[i,0] + u[i,0] + (lam / rou)
        if condition1 > 0:
            z[i,0] = condition1
        elif condition2 < 0:
            z[i,0] = condition2
        else:
            z[i,0] = 0
    return z
def u_update(x, z):
    u=x+z
    return u
for i in range(T):
    x=x_update(A, x, b, z, u,rou)
    z = z_update(x, u, lam)
    u = u_update(x, z)
print(x)
