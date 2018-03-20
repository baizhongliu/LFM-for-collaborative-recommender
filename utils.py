#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:18:20 2018

@author: baifrank
"""
import numpy as np
import math

##定义LFM矩阵分解函数//def function of matrix decomposition
def decompose_LFM_bias(mat_input,n_factor,n_iter=10,lr=0.01,lam=0.01,bias=False,shrinkage=0):
        
    assert type(mat_input) == np.ndarray
    if bias == True:    
        mu = mat_input[mat_input>0].mean()
    m, n = mat_input.shape
    u = np.random.rand(m,n_factor)
    v = np.random.randn(n_factor,n)
    
    if bias == True:    
        ##偏置初始化为0//initialize the bias to 0
        b_u = np.zeros(m)
        b_i = np.zeros(n)
        
    for t in range(n_iter):
        print("Iteration:"+str(t))
        for i in range(m):
            for j in range(n):
                ##只有不为0的矩阵元素才对参数的更新有影响//only the rating>0 influence the update of parameters
                if math.fabs(mat_input[i][j]) > 1e-4:
                    err = mat_input[i][j] - np.dot(u[i],v[:,j])
                    if bias == True:
                        err = mat_input[i][j] - (np.dot(u[i],v[:,j])+b_u[i]+b_i[j]+mu)
                        ##更新bias//update the bias
                        b_u[i] = b_u[i] + lr*(err - lam*b_u[i])
                        b_i[j] = b_i[j] + lr*(err - lam*b_i[j])
                    ##更新矩阵U、V里面的元素值//update the matrix U and V 
                    for r in range(n_factor):
                        gu = err * v[r][j] - lam * u[i][r]
                        gv = err * u[i][r] - lam * v[r][j]
                        u[i][r] += lr * gu
                        v[r][j] += lr * gv
                    ##将迭代步长在每次迭代的末尾缩小，提高迭代的稳定性//set the shrinkage
                    if shrinkage:
                        lr *= shrinkage
    if bias == True:
        return u,v,b_u,b_i,mu 
    else:
        return u,v


##定义计算误差的函数:给定原始矩阵与重构矩阵(小数要转化成整数)//def function to compute the error:mae and mse
def compute_error(mat_real,mat_pre):
        
    mat_pre_round = mat_pre
    mat_pre_round[mat_pre<0] = 0
    mat_pre_round[mat_pre>5] = 5
    mat_pre_round = np.round(mat_pre_round)
    ##只看有评分部分的误差//only the part of rating>0
    mat_dif = (mat_real-mat_pre_round)[mat_real>0]
    mae = sum(abs(mat_dif))/len(mat_dif)
    mse = sum(pow(mat_dif,2))/len(mat_dif)
    
    return mae,mse
  
    
    
    
    
