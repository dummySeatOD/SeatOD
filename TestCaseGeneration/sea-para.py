import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import csv
import math

def SEA_D_Para(file_name, file_num, feature_num, note):
    '''
    Calculate the coefficients to be used for semantic enhancement
    Output:
    para is the coefficients to be used for semantic enhancement guided by SEA-D
    '''
    df = load_df(file_name, file_num, feature_num)
    label = load_label(file_name, file_num)
    C_A, C_B = cal_label(label, file_num, note)
    centroid = cal_centroid(df, file_num, feature_num, label, note)
    dist = dist_D(df, label, C_A, feature_num, centroid, note)
    para = np.zeros(feature_num)
    for i in range(feature_num):
        para[i] = dist[:,i].mean()
    max_ = para.max()
    min_ = para.min()
    for i in range(feature_num):
        para[i] = (para[i]-min_) / (max_ - min_) * 0.01  
    torch.save(para, 'sea')
    return para


def SEA_B_Para(file_name, file_num, feature_num, note):
    '''
    Calculate the coefficients to be used for semantic enhancement
    Output:
    para is the coefficients to be used for semantic enhancement guided by SEA-B
    '''
    df = load_df(file_name, file_num, feature_num)
    label = load_label(file_name, file_num)
    C_A, C_B = cal_label(label, file_num, note)
    centroid = cal_centroid(df, file_num, feature_num, label, note)
    dist = dist_B(df, label, C_A, feature_num, centroid, note)
    para = np.zeros(feature_num)
    for i in range(feature_num):
        para[i] = dist[:,i].mean()
    max_ = para.max()
    min_ = para.min()
    for i in range(feature_num):
        para[i] = (para[i]-min_) / (max_ - min_) * 0.01 
    torch.save(para, 'sea')
    return para