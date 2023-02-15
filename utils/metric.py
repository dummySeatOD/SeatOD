import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from CoVariance import cal_cov
from utils import *

def SEA_D(file_name, file_num, feature_num, note):
    '''
    Description: calculate the density metric SEA_D
    Input:
        file_name: name of the Deepfeature file
        file_num: total number of deep features extracted 
        feature_num: number of features in each deep feature
        note: markers for different classes (class A)
    Output:
        Density: the dendity metric SEA_D of class A  (value:[0,1])
    '''
    df = load_df(file_name, file_num, feature_num)
    label = load_label(file_name, file_num)
    C_A, C_B = cal_label(label, file_num, note)
    centroid = cal_centroid(df, file_num, feature_num, label, note)
    dist = dist_D(df, label, C_A, feature_num, centroid, note)
    cov = cal_cov(df)
    cov_dist = dist_cov_(dist)
    cov_dist_tol = cal_tol_(cov_dist)
    Density = 1 - (cov_dist_tol.mean() - cov_dist_tol.min()) / (cov_dist_tol.max() - cov_dist_tol.min())
    return Density
    
def SEA_B(file_name, file_num, feature_num, note_A, note_B):
    '''
    Description: calculate the boundary metric SEA_B
    Input:
        file_name: name of the Deepfeature file
        file_num: total number of deep features extracted 
        feature_num: number of features in each deep feature
        note_A: markers for different classes (class A)
        note_B: markers for different classes (class B)
    Output:
        Boundary: the Boundary metric SEA_B of class A to class B  value:[0,1]
    '''
    df = load_df(file_name, file_num, feature_num)
    label = load_label(file_name, file_num)
    C_A, C_B = cal_label(label, file_num, note)
    centroid_A = cal_centroid(df, file_num, feature_num, label, note_A)
    centroid_B = cal_centroid(df, file_num, feature_num, label, note_B)
    dist = dist_B(df, label, C_A, feature_num, centroid_B, note_A)
    dist_ = dist_B(df, label, C_B, feature_num, centroid_B, note_B)
    cov = cal_cov(df)
    cov_dist = dist_cov_(dist)
    cov_dist_ = dist_cov_(dist)
    cov_dist_tol = cal_tol_(cov_dist)
    cov_dist_tol_ = cal_tol_(cov_dist)
    t1 = (cov_dist_tol.mean() - cov_dist_tol.min()) / (cov_dist_tol.max() - cov_dist_tol.min())
    t2 = (cov_dist_tol_.mean() - cov_dist_tol_.min()) / (cov_dist_tol_.max() - cov_dist_tol_.min())
    Boundary = (t1 + t2) / 2
    return Boundary   
    
def SEA_S(file_name, file_name_, file_num, feature_num, note_A, note_B, inconsistent_para):
    '''
    Description: calculate the feature space offset under perturbution
    Input:
        file_name: name of the original Deepfeature file
        file_name_: name of the perturbed Deepfeature file
        file_num: total number of deep features extracted 
        feature_num: number of features in each deep feature
        note_A: markers for different classes (class A)
        note_B: markers for different classes (class B)
        inconsistent_para: Number of inconsistent actions
    Output:
        Stability: the Stability metric of model
    '''
    df = load_df(file_name, file_num, feature_num)
    df_ = load_df(file_name, file_num, feature_num)
    label = load_label(file_name, file_num)
    C_A, C_B = cal_label(label, file_num, note)
    centroid_A = cal_centroid(df, file_num, feature_num, label, note_A)
    centroid_A_ = cal_centroid(df_, file_num, feature_num, label, note_A)
    t = dist(centroid_A,centroid_A_) * inconsistent_para
    Stability = 1 - (t.mean() - t.min()) / (t.max() - t.min())
    return Stability

def SEA_D_Se(file_name, file_num, feature_num, note_A, note_B):
    '''
    Description: calculate the density list for selection
    Input:
        file_name: name of the Deepfeature file
        file_num: total number of deep features extracted 
        feature_num: number of features in each deep feature
        note: markers for different classes (class A)
    Output:
        density_list: the dendity list of evary test case  (shape [file_num,1])
    '''
    density_list = np.zeros(file_num)
    df = load_df(file_name, file_num, feature_num)
    label = load_label(file_name, file_num)
    C_A, C_B = cal_label(label, file_num, note)
    centroid_A = cal_centroid(df, file_num, feature_num, label, note_A)
    centroid_B = cal_centroid(df, file_num, feature_num, label, note_B)
    dist = dist(df, label, file_num, feature_num, centroid_A, centroid_B, note_A)
    for i in range(file_num):
        density_list[i] = dist[i].mean()
    # cov = cal_cov(df)
    # cov_dist = dist_cov_(dist)
    # cov_dist_tol = cal_tol_(cov_dist)
    # Density = 1 - (cov_dist_tol.mean() - cov_dist_tol.min()) / (cov_dist_tol.max() - cov_dist_tol.min())
    return density_list

def SEA_B_Se(file_num, feature_num, centroid_A, centroid_B, note_A, note_B):
    '''
    Description: calculate the boundary list for selection
    Input:
        file_name: name of the Deepfeature file
        file_num: total number of deep features extracted 
        feature_num: number of features in each deep feature
        note_A: markers for different classes (class A)
        note_B: markers for different classes (class B)
    Output:
        Boundary: the Boundary metric SEA_B of class A to class B  value:[0,1]
    '''
    boundary_list = np.zeros(file_num)
    df = load_df(file_name, file_num, feature_num)
    label = load_label(file_name, file_num)
    C_A, C_B = cal_label(label, file_num, note)
    centroid_A = cal_centroid(df, file_num, feature_num, label, note_A)
    centroid_B = cal_centroid(df, file_num, feature_num, label, note_B)
    # dist = dist_B(df, label, C_A, feature_num, centroid_B, note_A)
    # dist_ = dist_B(df, label, C_B, feature_num, centroid_B, note_B)
    dist = dist(df, label, file_num, feature_num, centroid_A, centroid_B, note_B)
    for i in range(file_num):
        boundary_list[i] = dist[i].mean()
    return boundary_list  