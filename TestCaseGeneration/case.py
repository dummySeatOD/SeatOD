import time, argparse, datetime
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import os
import numpy as np
import random
from PIL import Image
import resnet
import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from pytorch_pretrained_biggan import save_as_images

import legacy
'''
This is a code document for generating test samples that we need.
1. a GAN generation model trained by the kitti dataset: here, styleGAN2 is chosen, and the pre-training file is test.pkl
2. an image for semantic enhancement: 000001.png
3. a parameter file for semantic enhancement: sea
'''

# styleGAN2 test
parser = argparse.ArgumentParser(description='test case generation')
parser.add_argument('--print_freq', '-p', default=1, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--epoch1', default=5, type=int, metavar='E1', help='step1 epoch')
parser.add_argument('--epoch2', default=5, type=int, metavar='E2', help='step2 epoch')
parser.add_argument('--schedule1', nargs='+', default=[3000,6000,9000], type=int, metavar='SC1', help='step1 schedule')
parser.add_argument('--schedule2', nargs='+', default=[4000,6000], type=int, metavar='SC2', help='step2 schedule')

parser.add_argument('--lr1', default=100, type=float, metavar='LR1', help='LR1')
parser.add_argument('--lr2', default=0.001, type=float, metavar='LR2', help='LR2')
parser.add_argument('--truncation', default=1, type=float, metavar='TRUNC',
                    help='truncation for BigGAN noise vector, default=0.4')
parser.add_argument('--noise_seed', default=None, type=int, metavar='NS', help='Seed for noise vector')

parser.add_argument('--aug_num', default=1, type=int, metavar='AN', help='ISDA aug number')
parser.add_argument('--aug_alpha', default=0.2, type=float, metavar='AA', help='For feature augmentation control')
parser.add_argument('--eta', default=5e-3, type=float, metavar='ETA', help='For step1 loss ratio')
parser.add_argument('--loss_component', default='r', type=str, help='1, 2, 3, 4, r')


parser.add_argument('--recon_dir', default='./', type=str, metavar='RD', help='For noise vector')
parser.add_argument('--img_dir', default='000001.png', type=str, metavar='ID', help='target image')
parser.add_argument('--name', default='test-case-generation', type=str, help='name of experiment')
parser.add_argument('--job_id', default=8, type=int, help='shell id')
parser.add_argument('--size', default=512, type=int, help='image size')

parser.add_argument('--data_url', default='./', type=str, help='root to train data')
parser.add_argument('--train_url', default='./', type=str, help='for huawei cloud')
args = parser.parse_args()
print(args)



class_num = 1
feature_num = 2048

img_dir = args.img_dir
class_id = 0
img_id = 0
target = 0
seed = 265
date = datetime.date.today()
localtime = time.localtime(time.time())
job_id = str(args.job_id)
print('job_id:', job_id)
'''
basic_path = args.train_url \
            + job_id + '_' + str(date.month) + '_' + str(date.day) + '_' + str(localtime.tm_hour) + '_' + str(localtime.tm_min)\
            +'_twosteps_' + str(class_id) + '_' + str(img_id)\
            +'_epoch2_' + str(args.epoch2)\
            +'_lr_' + str(args.lr2) \
            +'_alpha_' + str(args.aug_alpha)
if args.name:
    basic_path += '_' + str(args.name)

isExists = os.path.exists(basic_path)
if not isExists:
    os.makedirs(basic_path)
print('basic_path:', basic_path)
'''
loss_file = 'loss_epoch.txt'

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def _image_restore(normalized_image):
    size = normalized_image.size()[-1]
    mean_mask = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).expand(1, 3, size, size).cuda()
    std_mask = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).expand(1, 3, size, size).cuda()
    return normalized_image.mul(std_mask) + mean_mask

def _image_norm(ini_image):
    size = ini_image.size()[-1]
    mean_mask = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).expand(1, 3, size, size).cuda()
    std_mask = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).expand(1, 3, size, size).cuda()
    return (ini_image - mean_mask).div(std_mask)

def F_inverse(model, netG, input, class_vector, features_ini):
    G = netG
    truncation = args.truncation
    class_idx = 'None'
    device='cuda'
    # Labels.
    label = torch.zeros([1, netG.c_dim], device='cuda')
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')           
    print(netG.z_dim)
    noise_vector = torch.from_numpy(np.random.RandomState(seed).randn(1, netG.z_dim)).to(device)
    noise_vector = noise_vector.cuda()
    noise_vector.requires_grad = True
    print('Initial noise_vector:', noise_vector.size())

    # noise_vector_batch = noise_vector.expand(args.aug_num, 128)
    noise_vector_batch = noise_vector
    noise_vector_batch = torch.nn.Parameter(noise_vector_batch.cuda())
    noise_vector_batch.requires_grad = True
    class_vector_batch = class_vector.expand(args.aug_num, 1).cuda()
    
    
    # mse_loss = torch.nn.MSELoss(reduction='sum')
    # opt1 = optim.Adam([{'params': noise_vector}], lr=args.lr1, weight_decay=1e-4)
    
    feature_num = 2048
    feature_origin_batch = features_ini.expand(args.aug_num, feature_num).float().cuda()
    feature_objective_batch = feature_origin_batch
    tt = torch.load('sea')
    aug = torch.tensor(tt).cuda()
    print("====> Start Augmentating")
    for i in range(args.aug_num):
        # aug_np = np.random.multivariate_normal([0 for ij in range(feature_num)], args.aug_alpha * CV)
        # aug = torch.Tensor(aug_np).float().cuda()
        # print("aug[{0}]:".format(i), aug.size())
        print("feature_origin_batch[i].size(): ", feature_origin_batch[i].size())
        feature_objective_batch[i] = (feature_origin_batch[i] + aug).detach()
        # feature_objective_batch[i] = feature_origin_batch[i].detach()
    print("====> End Augmentating")
    
    mse_loss = torch.nn.MSELoss(reduction='sum')
    opt2 = optim.SGD([{'params': noise_vector_batch}], lr=args.lr2, momentum=0.9, weight_decay=1e-4, nesterov=True)

    for epoch in range(args.epoch2):
        if epoch in args.schedule2:
            for paras in opt2.param_groups:
                paras['lr'] /= 10
                print("lr:", paras['lr'])

        n_mean = noise_vector_batch.mean(axis=1).unsqueeze(1).expand(noise_vector_batch.size(0), noise_vector_batch.size(1))
        n_std = noise_vector_batch.std(axis=1).unsqueeze(1).expand(noise_vector_batch.size(0), noise_vector_batch.size(1))
        noise_vector_normalized_batch = (noise_vector_batch - n_mean) / n_std
        fake_img_batch = netG(noise_vector_normalized_batch, class_vector_batch, truncation)

        if epoch % args.print_freq == 0:
            for i in range(fake_img_batch.size(0)):
                save_as_images(fake_img_batch[i].unsqueeze(0).detach().cpu(),
                               '{0}/step2_epoch_{1}_img_{2}'.format('test-case', epoch // args.print_freq, i))
            # print("noise_vector_batch:", noise_vector_batch)

        fake_img_224 = F.interpolate(fake_img_batch, size=(224, 224), mode='bilinear', align_corners=True)
        fake_img_224.require_grad = True

        _fake_img_224 = fake_img_224.view(fake_img_224.size(0), -1)
        f_mean = _fake_img_224.mean(axis=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(fake_img_224.size(0), 3, 224, 224)
        f_std = _fake_img_224.std(axis=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(fake_img_224.size(0), 3, 224, 224)
        fake_img_norm = (fake_img_224 - f_mean) / f_std
        fake_img_norm = fake_img_norm.cuda()
        fake_img_norm.require_grad = True

        _, feature_fake_img_batch = model(fake_img_norm, isda=True)
        loss_b = mse_loss(feature_fake_img_batch, feature_objective_batch)
        opt2.zero_grad()
        loss_b.backward(retain_graph=True)
        opt2.step()

        if epoch % 1 == 0:
            # print('Step2: Epoch: %d  loss_b: %.5f' % (epoch, loss_b.data.item()))
            fd = open(loss_file, 'a+')
            string = ('Step2: Epoch: {0}\t'
                     'loss_b {1}\t'.format(epoch, loss_b.data.item()))
            print(string)
            fd.write(string + '\n')
            fd.close()

def train(input, target, netG, model):
    onehot_class_vector = torch.zeros(1, 1)
    onehot_class_vector[0][0] = 1
    onehot_class_vector = onehot_class_vector.cuda()

    # netG.train()
    model.train()
    model = model.cuda()
    netG = netG.cuda()

    print('====> Test Image Norm & Restore')
    input = input.cuda()
    print('input:', input.size())

    _, feature_ini = model(input, isda=True)

    print('====> Start Training')
    F_inverse(model, netG, input, onehot_class_vector, feature_ini)


def main():
    acc_epoch = []
    loss_epoch = []

    img_dir = args.train_url + args.img_dir
    raw_img = Image.open(img_dir)
    input = trans(raw_img)
    input = input.view(1, input.size(0), input.size(1), input.size(2))
    # input = torch.unsqueeze(input, 0)
    # print('input:', input.size())

    print('=========> Load models from train_url')
    checkpoint_dir = os.path.join(args.train_url, 'resnet50-19c8e357.pth')
    print('checkpoint_dir:', checkpoint_dir)
    checkpoint = torch.load(checkpoint_dir)
    model = resnet.resnet50()
    model.load_state_dict(checkpoint)
    
    # img = generate_images()
    # print(img.shape)
    
    network_pkl = 'test1600.pkl'
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        netG = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    train(input, target, netG, model)

if __name__ == '__main__':
    main()
    # generate_images()
    print('lxx1123')

