#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:18:03 2018

@author: baifrank
"""

import pandas as pd
import numpy as np
import scipy.sparse as sparse

##定义函数将原始的dataframe转化成稀疏矩阵
def df_to_mat(df_input,nb_users,nb_item):
    
    data = df_input.rating
    col = df_input.item_id
    row = df_input.user_id
    R = sparse.coo_matrix((data,(row-1,col-1)),shape=(nb_users,nb_item))
    
    return R


##导入数据
def load_data(path_train,path_test):
    
    ua_train = pd.read_table(path_train,sep=' ')
    ua_test = pd.read_table(path_test,sep=' ')

    uid_train = np.unique(ua_train['user_id'])
    iid_train = np.unique(ua_train['item_id'])
    iid_test = np.unique(ua_test['item_id'])
    iid_all = np.unique(np.append(iid_train,iid_test))
    
    nb_users,nb_item = len(uid_train),len(iid_all)
    R_train = df_to_mat(ua_train,nb_users,nb_item)
    R_test = df_to_mat(ua_test,nb_users,nb_item)
    
    return R_train,R_test
