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

# import h5py
# f = h5py.File('somefile.mat', 'r')
# data = f.get('data/variable1')
# data = np.array(data)
########################################
#mat = scipy.io.loadmat('Summarydata.mat')
#matrixuserrate= pd.io.parsers.read_csv('matrixuserrate.csv')
file = open("matrixuserrate.csv")
matrixuserrate = np.loadtxt(file, delimiter=",")
print(np.shape(matrixuserrate))
[sz1, sz2] = np.shape(matrixuserrate)

M = matrixuserrate
omega = (M != 0)
#######################################
alpha=0.7
rk=50
#  comment: lazygreedy algorithm
#######################################

#  comment: replacment greedy algorithm
def rep_greedy(k, P, set):
    "rep greedy algorithm"
    greedyindex = arr.array('i', {})
    min_marginal = 0
    min_index = 0
    greedyindex = set

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
        #print('print(func_val)', func_val(greedyindex, P))

    return greedyindex


#######################################
def Projto(x, rk):
    U, s, VT = svd(x)

    # create m x n Sigma matrix
    Sigma = np.zeros([x.shape[0], x.shape[1]])
    # populate Sigma with n x n diagonal matrix
    Sigma[:x.shape[0], :x.shape[0]] = np.diag(s)
    # reconstruct matrix
    B=np.dot(U[:, :rk],np.dot(Sigma[:rk, :rk],VT[:rk, :]))

    return B
#######################################


def omegamaxfacilcompute(S, X):
    omegamax=np.zeros([200, 1995,18], dtype=int)
    for i in range(0,200):
        for g in range(0,18):
            s_aux=0
            jmax=S[0]
            for j in S:

                #if j in genre[g]:
                if X[i,j]>s_aux:
                    s_aux=X[i,j]
                    jmax=j
            omegamax[i,jmax,g]=1
    return omegamax
    #######################################

def func_val(S, X):
    # \sum _{i,j\in omega^c, j \in S}x_{i,j}^2+\sum_i \sum_{genre} w_{u,m}max_{j\in S\cap G_m }X_{i,j}
    fpart1=0
    fpart2=0
    for i in range(0,200):
        for j in S:
            if omega[i,j]==0:
                    fpart1=fpart1+(X[i,j]-5)**2

        for g in range(0,18):
            maxigenere=0
            for j in S:
                #if j in genre[g]:
                    if X[i,j]>maxigenere:
                        maxigenere=X[i,j]
            fpart2=fpart2+maxigenere
    return fpart2+fpart1
    #######################################
        # \sum _{i,j\in omega^c, j \in S}x_{i,j}^2+\sum_i \sum_{genre} w_{u,m}max_{j\in S\cap G_m }X_{i,j}

def func_val_complete(S, X):
    fpart1=0
    fpart2=0
    fpart3=0
    for i in range(0,200):

        for j in S:
            if omega[i,j]==0:
                fpart1=fpart1+((X[i,j]-5)**2)
    print("fpart1",fpart1)

    for i in range(0,200):
        for g in range(0,18):
            maxigenere=0
            for j in S:
                 #if j in genre[g]:
                if X[i,j]>maxigenere:
                    maxigenere=X[i,j]
            fpart2=fpart2+maxigenere
    print("fpart2",fpart2)
    for i in range(0,200):
        for j in range(0, 1995):
            if omega[i,j]==1:
                fpart3=fpart3+(X[i,j]-M[i,j])**2
    print("fpart3",fpart3)
    return (fpart2+fpart1+fpart3)/(200*1995)
#######################################
#  comment: extragradientgreedy  algorithm
#
def extragradientgreedy(k, T, omega):
    x_avg=np.zeros([200,1995])
    x1=np.zeros([T+1,200,1995])
    x2=np.zeros([T+1,200,1995])
    w=np.ones([200,18])
    val=np.zeros(T)
    omegamaxfacil=np.zeros([200,1995,18])
    omegaadv=np.zeros([200,1995])
    for t in range(0, T):
        for j in range(0, 1995):
            for i in range(0, 200):
                sumdevfacility = 0
                for m in range(0, 18):
                    sumdevfacility = sumdevfacility+w[i, m]*omegamaxfacil[i, j, m]
                x1[t+1, i, j] = x2[t, i, j] - (omega[i, j]*2*(x2[t, i, j]-M[i, j]))*alpha/(t** 0.5+1)-(omegaadv[i, j]*2*(x2[t, i, j]-5)+sumdevfacility)*alpha/(t** 0.5+0.5)

        x1[t+1] = Projto(x1[t+1], rk)
        s1 = lazygreedy(k, x1[t+1])
        print("shat[",t,"]=",s1)
        omegaadv=np.zeros([200,1995])
        for j in s1:
            for i in range(0, 200):
                #if j in sadv[t]:
                    if not omega[i, j] == 1:

                        omegaadv[i, j] = 1
        omegamaxfacil = omegamaxfacilcompute(s1, x1[t+1])
    ########
    #phase2#
    ########
        for j in range(0, 1995):
            for i in range(0, 200):
                sumdevfacility = 0
                for m in range(0, 18):
                    sumdevfacility = sumdevfacility+w[i, m]*omegamaxfacil[i, j, m]
                x2[t+1, i, j] = x2[t, i, j] - (omega[i, j]*2*(x1[t+1, i, j]-M[i, j]))*alpha/(t** 0.5+1)-(omegaadv[i, j]*2*(x1[t+1, i, j]-5)+sumdevfacility)*alpha/(t** 0.5+0.5)

        x2[t+1] = Projto(x2[t+1], rk)
        s2 = lazygreedy(k, x2[t+1])
        print("s[",t,"]=",s2)
        omegaadv=np.zeros([200,1995])
        for j in s2:
            for i in range(0, 200):
                #if j in sadv[t]:
                    if not omega[i, j] == 1:

                        omegaadv[i, j] = 1
        omegamaxfacil = omegamaxfacilcompute(s2, x2[t+1])


    #####################
    #####################
        x_avg=t*x_avg+x1[t+1]
        x_avg=x_avg/(t+1)
        s_avg = lazygreedy(k, x_avg)
        val[t]=func_val_complete(s1, x1[t+1])
        print("func_val(",t,")", val[t])
    return x_avg ,val

#######################################
#######################################
#  comment: extragradientrepgreedy  algorithm
#


def extragradientrepgreedy(k, T, omega):
    x_avg=np.zeros([200,1995])
    x1=np.zeros([T+1,200,1995])
    x2=np.zeros([T+1,200,1995])

    w=np.ones([200,18])
    val=np.zeros(T)
    omegamaxfacil=np.zeros([200,1995,18])
    omegaadv=np.zeros([200,1995])
    for t in range(0, T):
        for j in range(0, 1995):
            for i in range(0, 200):
                sumdevfacility = 0
                for m in range(0, 18):
                    sumdevfacility = sumdevfacility+w[i, m]*omegamaxfacil[i, j, m]
                x1[t+1, i, j] = x2[t, i, j] - (omega[i, j]*2*(x2[t, i, j]-M[i, j]))*alpha/(t** 0.5+1)-(omegaadv[i, j]*2*(x2[t, i, j]-5)+sumdevfacility)*alpha/(t** 0.5+0.5)

        x1[t+1] = Projto(x1[t+1], rk)
        if t>0:
            si=s1
            s1 = rep_greedy(k, x1[t+1],si)
        else:
            s1 = lazygreedy(k, x1[t+1])
        print("shat[",t,"]=",s1)
        omegaadv=np.zeros([200,1995])
        for j in s1:
            for i in range(0, 200):
                #if j in sadv[t]:
                    if not omega[i, j] == 1:

                        omegaadv[i, j] = 1
        omegamaxfacil = omegamaxfacilcompute(s1, x1[t+1])
    ########
    #phase2#
    ########
        for j in range(0, 1995):
            for i in range(0, 200):
                sumdevfacility = 0
                for m in range(0, 18):
                    sumdevfacility = sumdevfacility+w[i, m]*omegamaxfacil[i, j, m]
                x2[t+1, i, j] = x2[t, i, j] - (omega[i, j]*2*(x1[t+1, i, j]-M[i, j]))*alpha/(t** 0.5+1)-(omegaadv[i, j]*2*(x1[t+1, i, j]-5)+sumdevfacility)*alpha/(t** 0.5+0.5)

        x2[t+1] = Projto(x2[t+1], rk)
        if t>0:
            si=s2
            s2 = rep_greedy(k, x2[t+1],si)
        else:
            s2 = lazygreedy(k, x2[t+1])
        print("s[",t,"]=",s2)
        omegaadv=np.zeros([200,1995])
        for j in s2:
            for i in range(0, 200):
                #if j in sadv[t]:
                    if not omega[i, j] == 1:

                        omegaadv[i, j] = 1
        omegamaxfacil = omegamaxfacilcompute(s2, x2[t+1])


    #####################
    #####################
        x_avg=t*x_avg+x1[t+1]
        x_avg=x_avg/(t+1)
        s_avg = lazygreedy(k, x_avg)
        val[t]=func_val_complete(s1, x1[t+1])
        print("func_val(",t,")", val[t])
    return x_avg ,val

#######################################
    #######################################
    #  comment: gradientgreedy  algorithm
    # to do omegaadv omega  w


def gradientgreedy(k, T, omega):
    x_avg=np.zeros([200,1995])
    x=np.zeros([T+1,200,1995])
    w=np.ones([200,18])
    val=np.zeros(T)
    omegamaxfacil=np.zeros([200,1995,18])
    omegaadv=np.zeros([200,1995])
    for t in range(0, T):
        for j in range(0, 1995):
            for i in range(0, 200):
                sumdevfacility = 0
                for m in range(0, 18):
                    sumdevfacility = sumdevfacility+w[i, m]*omegamaxfacil[i, j, m]
                x[t+1, i, j] = x[t, i, j] - (omega[i, j]*2*(x[t, i, j]-M[i, j]))*alpha/(t** 0.5+1)-(omegaadv[i, j]*2*(x[t, i, j]-5)+sumdevfacility)*alpha/(t** 0.5+10)

        x[t+1] = Projto(x[t+1], rk)
        s = lazygreedy(k, x[t+1])
        print("s[",t,"]=",s)
        omegaadv=np.zeros([200,1995])
        for j in s:
            for i in range(0, 200):
                #if j in sadv[t]:
                    if not omega[i, j] == 1:

                        omegaadv[i, j] = 1
        omegamaxfacil = omegamaxfacilcompute(s, x[t])
        x_avg=t*x_avg+x[t+1]
        x_avg=x_avg/(t+1)
        s_avg = lazygreedy(k, x_avg)
        val[t]=func_val_complete(s, x[t+1])
        print("func_val(",t,")", val[t])
    return x_avg ,val

#######################################
def gradientrepgreedy(k, T, omega):
    x_avg=np.zeros([200,1995])
    x=np.zeros([T+1,200,1995])
    w=np.ones([200,18])
    val=np.zeros(T)
    omegamaxfacil=np.zeros([200,1995,18])
    omegaadv=np.zeros([200,1995])
    for t in range(0, T):
        for j in range(0, 1995):
            for i in range(0, 200):
                sumdevfacility = 0
                for m in range(0, 18):
                    sumdevfacility = sumdevfacility+w[i, m]*omegamaxfacil[i, j, m]
                x[t+1, i, j] = x[t, i, j] - (omega[i, j]*2*(x[t, i, j]-M[i, j]))*alpha/(t** 0.5+1)-(omegaadv[i, j]*2*(x[t, i, j]-5)+sumdevfacility)*alpha/(t** 0.5+10)

        x[t+1] = Projto(x[t+1], rk)
        if t>0:
            si=s
            s = rep_greedy(k, x[t+1],si)
        else:
            s = lazygreedy(k, x[t+1])
        print("s[",t,"]=",s)
        omegaadv=np.zeros([200,1995])
        for j in s:
            for i in range(0, 200):
                #if j in sadv[t]:
                    if not omega[i, j] == 1:

                        omegaadv[i, j] = 1
        omegamaxfacil = omegamaxfacilcompute(s, x[t])
        x_avg=t*x_avg+x[t+1]
        x_avg=x_avg/(t+1)
        s_avg = lazygreedy(k, x_avg)
        val[t]=func_val_complete(s, x[t+1])
        print("func_val(",t,")", val[t])
    return x_avg ,val

#######################################
def gradientdescent(k, T, omega):
    x_avg=np.zeros([200,1995])
    x=np.zeros([T+1,200,1995])
    w=np.ones([200,18])
    val=np.zeros(T)
    omegamaxfacil=np.zeros([200,1995,18])
    omegaadv=np.zeros([200,1995])
    for t in range(0, T):
        for j in range(0, 1995):
            for i in range(0, 200):

                x[t+1, i, j] = x[t, i, j] - (omega[i, j]*2*(x[t, i, j]-M[i, j]))*alpha/(t** 0.5+1)
        x[t+1] = Projto(x[t+1], rk)
        s = lazygreedy(k, x[t+1])

        print("s[",t,"]=",s)

        val[t]=func_val_complete(s, x[t+1])
        print("func_val_withoutadv(",t,")", val[t])

    return x[t] ,val

#######################################
#xadv,valadv=gradientrepgreedy(100, 100, omega)
xadvrep,valadvrep=extragradientrepgreedy(100, 100, omega)
print("valadvrep:",valadvrep)
xadv,valadv=extragradientgreedy(100, 100, omega)
print("valadv:",valadv)
#xadv,valadv=gradientgreedy(100, 100, omega)
#print("valadv:",valadv)
#xwithoutadv,valwithoutadv=gradientdescent(10, 30, omega)
#print("valwithoutadv:",valwithoutadv)
#print("norm_adv-nonadv",np.linalg.norm(xadv-xwithoutadv))
plt.plot(valadvrep)
plt.plot(valadv)
#plt.plot(valwithoutadv)
plt.show()
