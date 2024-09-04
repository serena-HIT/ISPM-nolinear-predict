import numpy as np
import scipy.linalg
from cvxopt import solvers
from cvxopt import matrix
import matplotlib.pyplot as plt

A = np.array([[1, 1], [-1, 2]])
n = A.shape[0]

B = np.array([[1, 1],[1, 2]])
p = B.shape[1]

Q = np.array([[1, 0],[0, 1]])
F = np.array([[1, 0],[0, 1]])
R = np.array([[1, 0],[0, 0.1]])

k_steps = 100

X_k = np.zeros((n, k_steps))

X_k[:,0] = [10, -10]

U_k = np.zeros((p, k_steps))

N = 5

def cal_matrices(A,B,Q,R,F,N):

    n = A.shape[0]
    p = B.shape[1]

    M = np.vstack((np.eye((n)), np.zeros((N*n,n))))
    C = np.zeros(((N+1)*n,N*p))
    tmp = np.eye(n)

    for i in range(N):
        rows = i * n + n
        C[rows:rows+n,:] = np.hstack((np.dot(tmp, B), C[rows-n:rows, 0:(N-1)*p]))
        tmp = np.dot(A, tmp)
        M[rows:rows+n,:] = tmp

    Q_bar_be = np.kron(np.eye(N), Q)
    Q_bar = scipy.linalg.block_diag(Q_bar_be, F)
    R_bar = np.kron(np.eye(N), R)

    G = np.matmul(np.matmul(M.transpose(),Q_bar),M)
    E = np.matmul(np.matmul(C.transpose(),Q_bar),M)
    H = np.matmul(np.matmul(C.transpose(),Q_bar),C) + R_bar

    return H, E

def Prediction(M,T,N = 1):

    sol = solvers.qp(M,T)
    U_thk = np.array(sol["x"])
    #print(U_thk.shape)
    u_k = U_thk[0:N, :]

    return u_k

M, C = cal_matrices(A,B,Q,R,F,N)
M = matrix(M)
'''
for k in range(1,k_steps):
    x_kshort = X_k[:, k - 1].reshape(2, 1)
    u_kshort = U_k[:, k - 1].reshape(2, 1)
    #print(x_kshort.shape)
    #print(u_kshort.shape)
    T = np.dot(C,x_kshort)
    T = matrix(T)
    for i in range(2):
        U_k[i:,k-1] = Prediction(M,T)[i,0]

    X_knew = np.dot(A,x_kshort) + np.dot(B,u_kshort)
    #print(X_knew.shape)
    for j in range(2):
        X_k[j:,k] = X_knew[j,0]
        #print(X_k[j:,k].shape)

#print(X_k)
'''
