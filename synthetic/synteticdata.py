import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import array as arr
import scipy.sparse
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
import timeit
import math
import random
import scipy.io
from scipy.linalg import svd
matrixSize=10
mx_n=50
alpha=0.002
lambdanormalized=1
mx_m=mx_n
Q=np.zeros([mx_n,mx_m,matrixSize,matrixSize])
for i in range(0,mx_n):
    for j in range(0,mx_m):
        A = np.absolute(np.random.rand(matrixSize, matrixSize))
        B = np.dot(A, A.transpose())
        Q[i,j,:,:]=B
#######################################
def relu(X):
    return np.maximum(0, X)
#######################################
def func_val(S, X):
    # \sum _{i}\max_{j\in S}x_i^TQ_{i,j}x_j
    f=0
    sumnorm=0
    for i in range(0, mx_m):
        sumnorm=sumnorm+np.linalg.norm(X[i])**2
    if sumnorm==0:
        sumnorm=0.0000000001
    for i in range(mx_n):
        maxi=0
        for j in S:
            if np.dot(X[i],np.dot(Q[i,j],X[j]))>maxi:
                maxi=np.dot(X[i],np.dot(Q[i,j],X[j]))

        f=f+maxi
    return f+(lambdanormalized/sumnorm)
#######################################
#######################################

#  comment: replacment greedy algorithm
def rep_greedy(k, P, set):
    "rep greedy algorithm"
    greedyindex = arr.array('i', {})
    min_marginal = 0
    min_index = 0
    greedyindex = set
    sz2=mx_n
    if len(set) > 0:
        for i in range(0, len(set)):
            min_marginal = 0
            new_set = np.delete(set, i)

            a = func_val(new_set, P)

            if a > min_marginal:
                min_marginal = a
                min_index = i

        greedyindex= np.delete(greedyindex, min_index)

    max_marginal = 0
    max_index = 0
    max_marginal = 0

    for i in range(0, sz2):
        baux=np.append(greedyindex,[i])
        a = func_val(baux, P)

        if a > max_marginal:
            max_marginal = a
            del max_index
            max_index = i

    ################################################################
    greedyindex = np.append(greedyindex,[max_index])

    # print('func_val(repg)', func_val(greedyindex, P))
    return greedyindex


#######################################
def lazygreedy(k, P):
    "greedy algorithm"
    greedyindex = arr.array('i', {})
    optfind = 0
    sz2=mx_n
    rho = np.zeros(sz2)
    max_ind = 0

    # % lazy greedy
    # % optfind=1 if we find the element which maximizes the marginal f(.|S^k)
    optfind = 0
    for i in range(0, sz2):
        s = arr.array('i', [i])
        rho[s] = func_val(greedyindex+s, P)

    for i in range(0, k-1):
        if max(rho) <= 0:
            greedyindex.append(0)
        else:
            optfind = 0
            while optfind == 0:
                # find max  marginal
                max_ind = 0
                max_ind = np.argmax(rho)
                # recompute its marginal value
                s = arr.array('i', [max_ind])
                if i == 0:
                    rho[max_ind] = func_val(greedyindex+s, P)
                else:
                    rho[max_ind] = func_val(greedyindex+s, P)-func_val(greedyindex, P)
                if max_ind == np.argmax(rho):
                    greedyindex.append(max_ind)
                    rho[max_ind] = -1
                    optfind = 1  # if it is bigger it is the answer


    return greedyindex
#######################################
def Projto(x,sz,matrixSize):
    for i in range(0,sz):
        x[i]=relu(x[i])
        a=np.linalg.norm(x[i])
        if a==0:
            x[i]=np.ones(matrixSize)/10000000000
        if a>0:
            if a>1.0000000001:
                x[i]=x[i]/a

    return x
#######################################
def omegamaxfacilcompute(S, X):
    omegamax=np.zeros([mx_n, mx_n], dtype=int)

    for i in range(mx_n):
        maxi=0
        maxindex=0
        for j in S:
            if np.dot(X[i],np.dot(Q[i,j],X[j]))>maxi:
                maxi=np.dot(X[i],np.dot(Q[i,j],X[j]))
                maxindex=j
            omegamax[i,maxindex]=1
    return omegamax

#######################################
def gradientgreedy(k, T, Q):
    x_avg=np.zeros([mx_m,matrixSize])
    x=np.ones([T+1,mx_m,matrixSize])
    val=np.zeros(T)
    omegamaxfacil=np.zeros([mx_m,mx_m])
    for t in range(0, T):
        sumnorm=0
        for i in range(0, mx_m):
            sumnorm=sumnorm+np.linalg.norm(x[t, i])**2

        for i in range(0, mx_m):
            omega=0
            for k in range(0,mx_m):
                if omegamaxfacil[k,i]>0:
                    omega=omega+omegamaxfacil[k,i]*np.dot(Q[k,i,:,:],x[t,k])
                if omegamaxfacil[i,k]>0:
                    omega=omega+omegamaxfacil[i,k]*np.dot(Q[k,i,:,:],x[t,k])
            #gradient  descent
            omega=omega-lambdanormalized*x[t, i]/(sumnorm**2)
            x[t+1, i] = x[t, i] -(omega)*alpha/(t** 0.5+1)
        x[t+1] = Projto(x[t+1], mx_n,matrixSize)

        s = lazygreedy(k, x[t+1])
        val[t]=func_val(s, x[t+1])
        omegamaxfacil=omegamaxfacilcompute(s, x[t])
        x_avg=t*x_avg+x[t+1]
    x_avg=x_avg/(t+1)
    s_avg = lazygreedy(k, x_avg)
    print("func_val(",t,")", val[t])
    return x_avg ,val
#######################################
def gradientrepgreedy(k, T, Q):
    x_avg=np.zeros([mx_m,matrixSize])
    x=np.ones([T+1,mx_m,matrixSize])
    val=np.zeros(T)
    omegamaxfacil=np.zeros([mx_m,mx_m])
    for t in range(0, T):
        sumnorm=0
        for i in range(0, mx_m):
            sumnorm=sumnorm+np.linalg.norm(x[t, i])**2

        for i in range(0, mx_m):
            omega=0
            for k in range(0,mx_m):
                if omegamaxfacil[k,i]>0:
                    omega=omega+omegamaxfacil[k,i]*np.dot(Q[k,i,:,:],x[t,k])
                if omegamaxfacil[i,k]>0:
                    omega=omega+omegamaxfacil[i,k]*np.dot(Q[k,i,:,:],x[t,k])
            #gradient  descent
            omega=omega-lambdanormalized*x[t, i]/(sumnorm**2)
            x[t+1, i] = x[t, i] -(omega)*alpha/(t** 0.5+1)

        x[t+1] = Projto(x[t+1], mx_n,matrixSize)
        if t>0:
            si=s
            s = rep_greedy(k, x[t+1],si)
        else:
            s = lazygreedy(k, x[t+1])
        val[t]=func_val(s, x[t+1])
        #print("func_val(",t,")", val[t])
        omegamaxfacil=omegamaxfacilcompute(s, x[t])
        x_avg=t*x_avg+x[t+1]
    x_avg=x_avg/(t+1)
    s_avg = lazygreedy(k, x_avg)

    return x_avg ,val
#######################################
def extragradientgreedy(k, T, Q):
    x_avg=np.zeros([mx_m,matrixSize])
    x1=np.ones([T+1,mx_m,matrixSize])
    x2=np.ones([T+1,mx_m,matrixSize])
    val=np.zeros(T)

    omegamaxfacil=np.zeros([mx_m,mx_m])
    for t in range(0, T):
        ########
        #phase1#
        ########
        sumnorm=0
        for i in range(0, mx_m):
            sumnorm=sumnorm+np.linalg.norm(x2[t, i])**2

        for i in range(0, mx_m):
            omega=0
            for k in range(0,mx_m):
                if omegamaxfacil[k,i]>0:
                    omega=omega+omegamaxfacil[k,i]*np.dot(Q[k,i,:,:],x2[t,k])
                if omegamaxfacil[i,k]>0:
                    omega=omega+omegamaxfacil[i,k]*np.dot(Q[k,i,:,:],x2[t,k])
            #gradient  descent
            omega=omega-lambdanormalized*x2[t, i]/(sumnorm**2)
            x1[t+1, i] = x2[t, i] -(omega)*alpha/(t** 0.5+1)
        x1[t+1] = Projto(x1[t+1], mx_n,matrixSize)

        s1 = lazygreedy(k, x1[t+1])
        val[t]=func_val(s1, x1[t+1])
        #print("func_val(",t,")", val[t])
        omegamaxfacil=omegamaxfacilcompute(s1, x1[t])
        ########
        #phase2#
        ########

        sumnorm=0
        for i in range(0, mx_m):
            sumnorm=sumnorm+np.linalg.norm(x1[t, i])**2

        for i in range(0, mx_m):
            omega=0
            for k in range(0,mx_m):
                if omegamaxfacil[k,i]>0:
                    omega=omega+omegamaxfacil[k,i]*np.dot(Q[k,i,:,:],x1[t,k])
                if omegamaxfacil[i,k]>0:
                    omega=omega+omegamaxfacil[i,k]*np.dot(Q[k,i,:,:],x1[t,k])
                #gradient  descent
            omega=omega-lambdanormalized*x1[t, i]/(sumnorm**2)
            x2[t+1, i] = x2[t, i] -(omega)*alpha/(t** 0.5+1)
        x2[t+1] = Projto(x2[t+1], mx_n,matrixSize)

        s2 = lazygreedy(k, x2[t+1])
        omegamaxfacil=omegamaxfacilcompute(s2, x2[t])

    x_avg=x_avg/(t+1)
    s_avg = lazygreedy(k, x_avg)
    return x_avg ,val
#######################################
def extragradientrepgreedy(k, T, Q):
    x_avg=np.zeros([mx_m,matrixSize])
    x1=np.ones([T+1,mx_m,matrixSize])
    x2=np.ones([T+1,mx_m,matrixSize])
    val=np.zeros(T)
    ########
    #phase1#
    ########
    omegamaxfacil=np.zeros([mx_m,mx_m])
    for t in range(0, T):
        sumnorm=0
        for i in range(0, mx_m):
            sumnorm=sumnorm+np.linalg.norm(x2[t, i])**2

        for i in range(0, mx_m):
            omega=0
            for k in range(0,mx_m):
                if omegamaxfacil[k,i]>0:
                    omega=omega+omegamaxfacil[k,i]*np.dot(Q[k,i,:,:],x2[t,k])
                if omegamaxfacil[i,k]>0:
                    omega=omega+omegamaxfacil[i,k]*np.dot(Q[k,i,:,:],x2[t,k])
            #gradient  descent
            omega=omega-lambdanormalized*x2[t, i]/(sumnorm**2)
            x1[t+1, i] = x2[t, i] -(omega)*alpha/(t** 0.5+1)
        x1[t+1] = Projto(x1[t+1], mx_n,matrixSize)

        if t>0:
            si=s1
            s1 = rep_greedy(k, x1[t+1],si)
        else:
            s1 = lazygreedy(k, x1[t+1])
        val[t]=func_val(s1, x1[t+1])
        #print("func_val(",t,")", val[t])
        omegamaxfacil=omegamaxfacilcompute(s1, x1[t])
        ########
        #phase2#
        ########

        sumnorm=0
        for i in range(0, mx_m):
            sumnorm=sumnorm+np.linalg.norm(x1[t, i])**2

        for i in range(0, mx_m):
            omega=0
            for k in range(0,mx_m):
                if omegamaxfacil[k,i]>0:
                    omega=omega+omegamaxfacil[k,i]*np.dot(Q[k,i,:,:],x1[t,k])
                if omegamaxfacil[i,k]>0:
                    omega=omega+omegamaxfacil[i,k]*np.dot(Q[k,i,:,:],x1[t,k])
                #gradient  descent
            omega=omega-lambdanormalized*x1[t, i]/(sumnorm**2)
            x2[t+1, i] = x2[t, i] -(omega)*alpha/(t** 0.5+1)
        x2[t+1] = Projto(x2[t+1], mx_n,matrixSize)

        if t>0:
            si=s2
            s2 = rep_greedy(k, x2[t+1],si)
        else:
            s2 = lazygreedy(k, x2[t+1])
        omegamaxfacil=omegamaxfacilcompute(s2, x2[t])

    x_avg=x_avg/(t+1)
    s_avg = lazygreedy(k, x_avg)

    return x_avg ,val
#######################################

xgg,valgg=gradientgreedy(11, 1000, Q)
print("valgg:",valgg)
rep_x,rep_val=gradientrepgreedy(11, 1000, Q)
print("rep_val:",rep_val)
extra_x,extra_val=extragradientgreedy(11, 1000, Q)
print("extra_val:",extra_val)
extra_rep_x,extra_rep_val=extragradientrepgreedy(11, 1000, Q)
print("extra_rep_val:",extra_rep_val)

plt.plot(rep_val)
plt.plot(extra_rep_val)
plt.plot(extra_val)
plt.plot(valgg)

plt.show()
