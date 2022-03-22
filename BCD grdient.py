import numpy as np
np.random.seed(2021)
A = np.random.rand(500, 100)
x = np.zeros([100, 1])
x[:5, 0] += np.array([i+1 for i in range(5)]) # x_denotes expected x
b = np.matmul(A, x) + np.random.randn(500, 1)*0.1 #add a noise to b
lam = 0.1
z=np.zeros([100, 1])
u=np.zeros([100, 1])
def get_grad(A, b, i, lam, x):
    h = b[i, 0]
    b_t = b - np.mat(A) * np.mat(x)
    b_t[i, 0] = h
    a_i = np.mat(A[:, i])
    inner_product = a_i* b_t
    a_production = a_i * a_i.T
    condition1 = inner_product - lam
    condition2 = inner_product + lam
    if condition1 > 0:
        grad = condition1 / a_production
    elif condition2 < 0:
        grad = condition2 / a_production
    else:
        grad = 0
    return grad


def BCD(x, A, b, lam, T):
    for i in range(T):
        for j in range(len(x)):
            x_update = get_grad(A, b, j, lam, x)
            x[i, 0] = x_update
    return x


test1 = BCD(x, A, b, lam, 20)
test2 = BCD(x, A, b, lam, 25)
print(test1)
