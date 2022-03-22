import numpy as np
from scipy import linalg
from numpy import mat
A=np.array([[1,1,1,0],[2,0.5,0,1]])
c=np.array([-5,-1,0,0])
b=np.array([[5],[8]])
vec1=np.array([[1,],[1],[1],[1]])
lambda_t=np.array([[1],[1],[1],[1]])
nu_t=np.array([[1],[1]])
x=np.array([[1],[1],[1],[1]])
mu=1
max_it=50#最大迭代次数
solver_t=np.vstack([x,nu_t,lambda_t])
def get_Fvalue(A,c,b,x,lambda_t,nu_t,mu):
    vec1 = np.array([[1], [1], [1], [1]])
    lambda_mat= mat(np.diag(list(lambda_t.reshape(1, 4)[0])))
    x_mat = mat(np.diag(list(x.reshape(1, 4)[0])))
    F1=mat(A).T*mat(nu_t)+mat(c).T-mat(lambda_t)
    F2=mat(A)*mat(x)-mat(b)
    F3=x_mat*lambda_mat*vec1-mu*vec1
    Fvalue=np.vstack([F1,F2,F3])
    return Fvalue
for i in range(max_it):
    Q1 = np.diag([0, 0, 0, 0])
    Q2 = A.T
    Q3 = np.diag([-1, -1, -1, -1])
    Q4 = A
    Q5 = np.diag([0, 0])
    Q6 = np.hstack([Q5, Q5])
    Q7 = np.diag(list(np.array(solver_t[0:4].T)[0, :]))
    Q8 = np.vstack([Q5, Q5])
    Q9 = np.diag(list(np.array(solver_t[6:10].T)[0, :]))
    QQ1 = np.hstack([Q1, Q2, Q3])
    QQ2 = np.hstack([Q4, Q5, Q6])
    QQ3 = np.hstack([Q7, Q8, Q9])
    Jacobi = np.vstack([QQ1, QQ2, QQ3])
    Fvalue=get_Fvalue(A,c,b,solver_t[0:4],solver_t[6:10],solver_t[4:6],mu)
    direction=linalg.solve(Jacobi,-Fvalue)
    inner=float(mat(solver_t[0:4].T)*mat(solver_t[6:10]))
    alpha=(0.1*inner)/(i+1)
    print(alpha)
    solver_t=solver_t+alpha*direction
    print(solver_t[0:4])
