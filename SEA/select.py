import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import csv
import math

file_num = 100
feature_num = 2048
note = 1 # object
df_name = 'df/'
sc_name = 'sc/'
select_num = 20

def label_cnt(label, file_num, note):
    cnt = 0
    for i in label:
        if i == note:
            cnt = cnt + 1
    car_cnt = cnt
    back_cnt = file_num - car_cnt
    return car_cnt, back_cnt

def cal_mean(data, car_cnt, back_cnt, file_num, feature_num, label, note):
    car = np.zeros((car_cnt, feature_num))
    back = np.zeros((back_cnt, feature_num))
    c = 0
    b = 0
    for i in range(file_num):
        if(label[i] == note):
            car[c] = data[i]
            c = c + 1
        else:
            back[b] = data[i]
            b = b + 1
    mean_car = np.zeros(feature_num)
    mean_back = np.zeros(feature_num)
    for i in range(feature_num):
        mean_car[i] = car[:, i].mean()
        mean_back[i] = back[:, i].mean()
    return mean_car, mean_back

def cal_dist(a, b):
    dist = math.sqrt((a - b) * (a - b))
    return dist

def dist(data, label, file_num, feature_num, mean_car, mean_back, note=1):
    dist_D = np.zeros((file_num, feature_num))
    dist_B = np.zeros((file_num, feature_num))
    for i in range(file_num):
        if label[i] == note: #car
            for j in range(feature_num):
                dist_D[i,j] = cal_dist(data[i,j], mean_car[j])
                dist_B[i,j] = cal_dist(data[i,j], mean_back[j])
        else:
            for j in range(feature_num):
                dist_D[i,j] = cal_dist(data[i,j], mean_back[j])
                dist_B[i,j] = cal_dist(data[i,j], mean_car[j])
    return dist_D, dist_B


data = np.zeros((file_num, feature_num))
for i in range(file_num):
    num = str(i).zfill(6)
    aa = torch.load(df_name + num , map_location=torch.device('cpu')).data
     # y = aa.numpy() 
    y = aa
    data[i] = y
label = np.zeros(file_num)
for i in range(file_num):
    num = str(i).zfill(6)
    aa = torch.load(sc_name + num , map_location=torch.device('cpu'))
    label[i] = aa
car_cnt, back_cnt = label_cnt(label, file_num, note)
# print(car_cnt, back_cnt)

mean_car, mean_back = cal_mean(data, car_cnt, back_cnt, file_num, feature_num, label, note)
# print(mean_car, mean_back)

dist_D, dist_B = dist(data, label, file_num, feature_num, mean_car, mean_back)
# print(dist_D, dist_B)

ttD = np.zeros(file_num)
ttB = np.zeros(file_num)
for i in range(file_num):
    ttD[i] = dist_D[i].mean()
    ttB[i] = dist_B[i].mean()
select = np.zeros(file_num)
for i in range(file_num):
    select[i] = ttB[i] - ttD[i]
aa = np.argsort(-select)
tt = np.zeros(file_num)
for i in range(select_num):
    tt[aa[i]] = 1
ff = open('select.txt',mode='w')
for i in range(file_num):
    if(tt[i] == 1):
        ff.write(str(i).zfill(6) + '\n')
print('Got the selection list in select.txt!')
ff.close()