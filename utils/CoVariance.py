import torch
import torch.nn as nn
import torchvision
import os
import numpy as np



class EstimatorCV():
    def __init__(self, feature_num, class_num):
        super(EstimatorCV, self).__init__()
        self.class_num = class_num
        self.CoVariance = torch.zeros(class_num, feature_num)
        self.Ave = torch.zeros(class_num, feature_num)
        self.Amount = torch.zeros(class_num)

    def update_CV(self, features, labels):
        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C)
        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - \
                   ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(
            sum_weight_CV + self.Amount.view(C, 1).expand(C, A)
        )

        weight_CV[weight_CV != weight_CV] = 0

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.Ave - ave_CxA).pow(2))

        self.CoVariance = (self.CoVariance.mul(1 - weight_CV) + var_temp
                           .mul(weight_CV)).detach() + additional_CV.detach()

        self.Ave = (self.Ave.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)).detach()

        self.Amount += onehot.sum(0)
        
        
def cal_cov(df):
    '''
    Description: calculate the CoVariance vector of deep featrure matrix
    
    Output:
        CoVariance: cov vector
    '''
    features_map = df
    feature_num =  df.shape[1]
    criterion_isda = ISDALoss(feature_num, 1)
    CoVariance = torch.zeros(1, feature_num)
    Ave = torch.zeros(1, feature_num)
    Amount = torch.zeros(1)
    lambda_0 = 7.5
    N = features_map.size(0)
    C = 1
    A = features_map.size(1)
    NxCxFeatures = features_map.view(
            N, 1, A
        ).expand(
            N, C, A
        )

    labels = torch.from_numpy(np.zeros(df.shape[0], dtype=np.int64))
    onehot = torch.zeros(N, C)
    onehot.scatter_(1, labels.view(-1, 1), 1)
    NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
    features_by_sort = NxCxFeatures.mul(NxCxA_onehot)
    Amount_CxA = NxCxA_onehot.sum(0)
    Amount_CxA[Amount_CxA == 0] = 1

    ave_CxA = features_by_sort.sum(0) / Amount_CxA
    var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

    var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

    sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

    weight_CV = sum_weight_CV.div(
        sum_weight_CV + Amount.view(C, 1).expand(C, A)
    )

    weight_CV[weight_CV != weight_CV] = 0

    additional_CV = weight_CV.mul(1 - weight_CV).mul((Ave - ave_CxA).pow(2))

    CoVariance = (CoVariance.mul(1 - weight_CV) + var_temp.mul(weight_CV)).detach() + additional_CV.detach()
    
    return CoVariance

