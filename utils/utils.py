import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from CoVariance import cal_cov


def load_df(file_name, file_num, feature_num):
    '''
    Description: load the deep feature for calculate
    Input:
        file_name: name of the Deepfeature file
        file_num: total number of deep features extracted 
        feature_num: number of features in each deep feature
    Output:
        df: deep feature matrix (shape of df [file_num, feature_num])
    '''
    df = np.zeros((file_num, feature_num))
    for i in range(file_num):
        temp = torch.load(file_name + '/' + str(i).zfill(6))
        df_ = temp.numpy() 
        df[i] = df_
    return df


def load_label(file_name, file_num):
    '''
    Description: load the label corresponds to the deep feature for calculate
    Input:
        file_name: name of the label file
        file_num: total number of deep features/label extracted 
    Output:
        label: label matrix (corresponds to the deep feature)
    '''
    label = np.zeros(file_num) 
    for i in range(file_num):
        temp = torch.load(file_name + '/' + str(i).zfill(6))
        label_ = temp.numpy() 
        label[i] = label_
    return label


def cal_label(label, file_num, note):
    '''
    Description: calculate the number of deep feature in each label(As an example, two categories are included)
    Input:
        label: label matrix
        file_num: total number of deep features/label
        note: markers for different classes (class A)
    Output:
        C_A: the number of samples in class A
        C_B: the number of samples in class B
    '''
    cnt = 0
    for i in label:
        if i == note:
            cnt = cnt + 1
    C_A = cnt
    C_B = file_num - car_cnt
    return C_A, C_B

def cal_dist(a, b):
    '''
    Description: calculate the distance of vector a and b
    '''
    dist = math.sqrt((a - b) * (a - b))
    return dist

def cal_centroid(df, file_num, feature_num, label, note):
    '''
    Description: load the deep feature for calculate
    Input:
        df: deep feature matrx (shape of df [file_num, feature_num])
        file_num: total number of deep features extracted 
        feature_num: number of features in each deep feature
        label: label matrix
        note: markers for different classes (class C)
    Output:
        centroid: the centroid of class C's deep feature space (shape of centroid [feature_num])
    '''
    C_ = np.zeros((centroid, feature_num))
    centroid = np.zeros(feature_num)
    cnt = 0
    for i in range(file_num):
        if(label[i] == note):
            C_[cnt] = df[i]
            cnt = cnt + 1
    for i in range(feature_num):
        centroid[i] = C_[:, i].mean()
    return centroid

def show_pca(df, C_A, C_B, file_num, feature_num, label, note, centroid_A, centroid_B, title):
    '''
    Description: PCA deep feature visualization 
    '''
    df_more = np.zeros((file_num + 2, feature_num))
    for i in range(file_num):
        df_more[i] = df[i]
    df_more[file_num-2] = centroid_A
    df_more[file_num-1] = centroid_B
    
    pca = PCA(n_components=2)
    reduced_x = pca.fit_transform(df_more)
    print(reduced_x.shape)

    a_x, a_y = [], []
    b_x, b_y = [], []
    c_x, c_y = [], []
    d_x, d_y = [], []

    for i in range(1):
        for j in range(file_num):
            if label[j] == note:
                a_x.append(reduced_x[j][0])
                a_y.append(reduced_x[j][1])
            else:
                b_x.append(reduced_x[j][0])
                b_y.append(reduced_x[j][1])
                    
    c_x.append(reduced_x[file_num-2][0]) 
    c_y.append(reduced_x[file_num-2][1])
    d_x.append(reduced_x[file_num-1][0])
    d_y.append(reduced_x[file_num-1][1])
    
    plt.scatter(a_x, a_y, c='g', marker='.',label='class A')
    plt.scatter(b_x, b_y, c='y', marker='x',label='class B')   
    plt.scatter(c_x, c_y, c='b', marker='o',label='Centroid_A')
    plt.scatter(d_x, d_y, c='r', marker='o',label='Centroid_B')
    
    plt.title(title) 
    plt.legend()
    plt.show()
    print(pca.explained_variance_ratio_) 
    

def dist_(df, label, file_num, feature_num, centroid_A, centroid_B, note):
    '''
    Description: 
    calculate the density of class A and the boundary between class A and class B
    '''
    dist_D = np.zeros((file_num, feature_num))
    dist_B = np.zeros((file_num, feature_num))
    for i in range(file_num):
        if label[i] == note:
            for j in range(feature_num):
                dist_D[i,j] = cal_dist(df[i,j], centroid_A[j])
        else:
            for j in range(feature_num):
                dist_D[i,j] = cal_dist(df[i,j], centroid_B[j])
    return dist_D, dist_B

def dist(df, label, file_num, feature_num, centroid_A, centroid_B, note_A):
    '''
    Description: 
    calculate the vector dist in whole deep feature(density)
    '''
    dist = np.zeros((file_num, feature_num))  
    for i in range(file_num):
        if label[i] == note_A:
            for j in range(feature_num):
                dist[i,j] = cal_dist(df[i,j], centroid_A[j])
        else:
            for j in range(feature_num):
                dist[i,j] = cal_dist(df[i,j], centroid_B[j])
    return dist

def dist_D(df, label, C, feature_num, centroid, note):
    '''
    Description: 
    calculate the density of class
    
    Output:
        dist_D: Density matrix (shape of dist_D [C, feature_num])
    '''
    dist_D_ = np.zeros((file_num, feature_num))
    dist_D = np.zeros((C, feature_num))
    for i in range(file_num):
        if label[i] == note:
            for j in range(feature_num):
                dist_D_[i,j] = cal_dist(df[i,j], centroid[j])
    cnt = 0
    for i in range(file_num):
        if label[i] == note:
            dist_D[cnt] =  dist_D_[i]
            cnt  = cnt + 1
    return dist_D

def dist_B(df, label, C_A, feature_num, centroid_B, note_A):
    '''
    Description: 
    calculate the Boundary of class A to class B
    
    Output:
        dist_B: Boundary matrix (shape of dist_B [C_A, feature_num])
    '''
    dist_B_ = np.zeros((file_num, feature_num))
    dist_B = np.zeros((C_A, feature_num))
    for i in range(file_num):
        if label[i] == note_A:
            for j in range(feature_num):
                dist_B_[i,j] = cal_dist(df[i,j], centroid_B[j])
    cnt = 0
    for i in range(file_num):
        if label[i] == note_A:
            dist_B[cnt] =  dist_B_[i]
            cnt  = cnt + 1
    return dist_B


def dist_cov(df, file_num, feature_num):
    '''
    Description: calculate the weighted distances
    '''
    cov = cal_cov(df)
    cov_dist = np.zeros((file_num, feature_num))
    for i in range(file_num):
        for j in range(feature_num):
            cov_dist[i,j] = dist_[i,j] * cov[j]
    return  cov_dist

def dist_cov_(dist):
    '''
    Description: calculate the weighted distances
    '''
    cov_dist = np.zeros((dist.shape[0], dist.shape[1]))
    for i in range(dist.shape[0]):
        for j in range(dist.shape[1]):
            cov_dist[i,j] = dist[i,j] * cov[j]
    return  cov_dist

def dist_cov_tol(cov_dist, file_num):
    cov_dist_tol_ = np.zeros(file_num)
    for i in range(file_num):
        cov_dist_tol[i] = cov_dist[i].sum()
    return cov_dist_tol

def cal_tol_(cov_dist):
    cov_dist_tol_ = np.zeros(cov_dist.shape[0])
    for i in range(cov_dist.shape[0]):
        cov_dist_tol[i] = cov_dist[i].sum()
    return cov_dist_tol





