import numpy as np
import math
import random
import sklearn.datasets
import matplotlib.pyplot as plt
A, b = sklearn.datasets.load_svmlight_file(r"C:\Users\hp\Desktop\a9a.txt")
x = np.array([0.1] * A.shape[1]).T
lam = 0.0001
eta = 0.01
def get_loss(A, x, lam, b):
    loss=0
    for i in range(A.shape[0]):
        a_i=A[i,:]
        b_i=b[i]
        inner_product = a_i.dot(x)
        exp_inner_product = math.exp(-b_i * inner_product)
        x_norm = np.linalg.norm(x, ord=2)
        loss += math.log(1 + exp_inner_product) + lam * x_norm
    return loss
def get_z(A, x, lam, b):
    z = np.array([float(0)] * A.shape[1]).reshape(123,1)
    for i in range(A.shape[0]):
        a_i = A[i, :]
        b_i = b[i]
        inner_product = a_i.dot(x.reshape(123,1))
        exp_inner_product = math.exp(-b_i * inner_product)
        exp_lest = float((inner_product * exp_inner_product) / (1 + exp_inner_product))
        grad = exp_lest * a_i.T + 2 * lam * x.reshape(123,1)
        z += grad/A.shape[0]
    return z
def SVRG(A, x_t, x_tut, lam, eta, b, z):
    i = random.randint(0, A.shape[0])
    a_i = A[i, :]
    b_i = b[i]
    inner1=a_i.dot(x_t.reshape(123,1))
    inner2 = a_i.dot( x_tut.reshape(123, 1))
    if inner1<-100:
        inner1=0
    if inner2<-100:
        inner2=0
    print(inner1,inner2)
    xt_grad = (-1 * b_i * math.exp(b_i * inner1)) / (1 + math.exp(b_i * inner1))*a_i.T + 2 * lam * x_t.reshape(123,1)
    tut_grad = (-1 * b_i * math.exp(b_i * inner2)) / (1 + math.exp(b_i * inner2))*a_i.T + 2 * lam * x_tut.reshape(123,1)
    x_t=x_t.reshape(123,1)
    x_t = x_t - eta * (xt_grad - tut_grad + z)
    return x_t, i


for T in range(10):
    x_tut = x
    z = get_z(A, x, lam, b)
    for j in range(100):
        x, index = SVRG(A, x, x_tut, lam, eta, b, z)
print(x)
