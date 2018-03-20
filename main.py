#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:18:28 2018

@author: baifrank
"""
import numpy as np
import pandas as pd

from LFM_decom import dataset
from LFM_decom import utils

if __name__ == "__main__":

    path = "/Users/baifrank/Desktop/recomm_output"
    R_train,R_test = dataset.load_data(path+'/ua_base_df_continuous.txt',path+'/ua_test_df_continuous.txt')
    
    #path_1m = "/Users/baifrank/Desktop/recomm_output/ml_1m_output"
    #R_train,R_test = dataset.load_data(path_1m+'/ua_base_df_continuous.txt',path_1m+'/ua.test.txt')
    R_train_ndarray = R_train.toarray()
    R_test_ndarray = R_test.toarray()


    ###############################################################################
    ##设置不同的迭代次数和隐因子个数，计算误差//set several iterations and number of latent factors
    ##无bias的情况//with out bias
    l_mae_train,l_mse_train,l_mae_test,l_mse_test = [],[],[],[]
    l_factor = [2,10,50,100,200]
    for k_factor in l_factor:
        b,c = utils.decompose_LFM_bias(R_train_ndarray,n_factor=k_factor,n_iter=200,lr=0.01,lam=0.01,bias=False,shrinkage=0)
        R_pre = np.dot(b,c)
        mae_train,mse_train = utils.compute_error(R_train_ndarray,R_pre)
        mae_test,mse_test = utils.compute_error(R_test_ndarray,R_pre)
        l_mae_train.append(mae_train)
        l_mse_train.append(mse_train)
        l_mae_test.append(mae_test)
        l_mse_test.append(mse_test)
        
    df_error = pd.DataFrame([l_mae_train,l_mse_train,l_mae_test,l_mse_test])
    df_error.index = ["mae_train","mse_train","mae_test","mse_test"]
    df_error.columns = l_factor
    print(df_error)

    ##有bias的情况//with bias 
    l_mae_train,l_mse_train,l_mae_test,l_mse_test = [],[],[],[]
    l_factor = [2,10,50,100,200]
    for k_factor in l_factor:
        U,V,B_u,B_i,Mu = utils.decompose_LFM_bias(R_train_ndarray,n_factor=k_factor,n_iter=200,lr=0.01,lam=0.01,bias=True,shrinkage=0)
        R_pre = np.dot(U,V) + B_u.reshape((len(B_u),1)) + B_i.reshape((1,len(B_i))) + Mu
        mae_train,mse_train = utils.compute_error(R_train_ndarray,R_pre)
        mae_test,mse_test = utils.compute_error(R_test_ndarray,R_pre)
        l_mae_train.append(mae_train)
        l_mse_train.append(mse_train)
        l_mae_test.append(mae_test)
        l_mse_test.append(mse_test)
        
    df_error = pd.DataFrame([l_mae_train,l_mse_train,l_mae_test,l_mse_test])
    df_error.index = ["mae_train","mse_train","mae_test","mse_test"]
    df_error.columns = l_factor
    print(df_error)
    
    ##作折线图，绘制不同factor数量下的误差//make line chart（x:number of factors,y:error-mae and mse）
    import matplotlib.pyplot as plt
    '''
    x = l_factor
    y_mae_train = l_mae_train
    y_mae_test = l_mae_test
    y_mse_train = l_mse_train
    y_mse_test = l_mse_test
    plt.plot(x, y_mse_train, marker='o', mec='r', mfc='w',label=u'train set')
    plt.plot(x, y_mse_test, marker='*', ms=10,label=u'test set')
    plt.legend()  # 让图例生效//for legend
    #plt.xticks(x, names, rotation=45)
    #plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"num_factor") #X轴标签//label of x-axis
    plt.ylabel("mse") #Y轴标签//label of y-axis
    plt.title("LFM(bias)'s mse") #标题//title
    plt.show()
    '''
    


