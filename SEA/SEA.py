import torch
import numpy as np
import math

file_num = 100
feature_num = 2048
note = 1 # object
df_name = 'df/'
sc_name = 'sc/'

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

'''
calculate the SEA-D
'''
ttD = np.zeros(file_num)
ttB = np.zeros(file_num)
for i in range(file_num):
    ttD[i] = dist_D[i].mean()
    ttB[i] = dist_B[i].mean()
car_D = np.zeros(car_cnt)
back_D = np.zeros(back_cnt)
cnt = 0
cnt_ = 0
for i in range(file_num):
    if(label[i] == 1):
        car_D[cnt] =  ttD[i]
        cnt = cnt + 1
    else:
        back_D[cnt_] =  ttD[i]
        cnt_ = cnt_ + 1
# cnt, cnt_
SEA_D = 1 - (car_D.mean() - car_D.min()) / (car_D.max() - car_D.min())
print('SEA_D:', SEA_D)

'''
calculate the SEA-B
'''
car_B = np.zeros(car_cnt)
back_B = np.zeros(back_cnt)
cnt = 0
cnt_ = 0
for i in range(file_num):
    if(label[i] == 1):
        car_B[cnt] =  ttB[i]
        cnt = cnt + 1
    else:
        back_B[cnt_] =  ttB[i]
        cnt_ = cnt_ + 1
# print(cnt, cnt_)
SEA_B = ((car_B.mean() - car_B.min())/(car_B.max() - car_B.min()))
print('SEA_B:',SEA_B)


'''
calculate the SEA-S
'''
d1 = torch.load('df-test')
d2 = torch.load('df-g-test')
sc = torch.load('sc-test')
car_cnt, back_cnt = label_cnt(sc, file_num, note)
mean_car1, _ = cal_mean(d1, car_cnt, back_cnt, file_num, feature_num, sc, note)
mean_car2, _ = cal_mean(d2, car_cnt, back_cnt, file_num, feature_num, sc, note)
tol = np.zeros(feature_num)
for i in range(feature_num):
    tol[i] = np.abs(mean_car1[i] - mean_car2[i])
SEA_S = 1 - ((tol.mean() - tol.min())/(tol.max() - tol.min()))
print('SEA_S:', SEA_S)