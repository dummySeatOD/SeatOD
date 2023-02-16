import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from CoVariance import cal_cov
from utils import *
from metric import *

def selection(file_name, file_num, feature_num, note_A, note_B, case_num):
    '''
    Description: load the deep feature for calculate
    Input:
        file_name: name of the Deepfeature file
        file_num: total number of deep features extracted 
        feature_num: number of features in each deep feature
        case_num: the number of cases for selection
    Output:
        slist: the id of cases selection (shape [case_num])
    '''
    slist = []
    select = np.zeros(file_num)
    density_list = SEA_D_Se(file_name, file_num, feature_num, note_A, note_B)
    boundary_list = SEA_B_Se(file_name, file_num, feature_num, note_A, note_B)
    for i in range(file_num):
        select[i] =  boundary_list[i] - density_list[i]
    aa = np.argsort(-select)
    tt = np.zeros(file_num)
    for i in range(case_num):
        tt[aa[i]] = 1
    for i in range(case_num):
        if(tt[i] == 1):
            slist.append(i)
    return slist