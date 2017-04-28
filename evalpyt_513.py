import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
sys.path.insert(0,'/data1/ravikiran/SketchObjPartSegmentation/src/caffe-switch/caffe/python')
import caffe
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import deeplab_resnet 
from collections import OrderedDict
import os
from os import walk
import matplotlib.pyplot as plt
import torch.nn as nn
def get_iou(pred,gt):
    if pred.shape!= gt.shape:
        print 'pred shape',pred.shape, 'gt shape', gt.shape
    assert(pred.shape == gt.shape)    
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

    max_label = 20 
    count = np.zeros((max_label+1,))
    for j in range(max_label+1):
        x = np.where(pred==j)
        p_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        x = np.where(gt==j)
        GT_idx_j = set(zip(x[0].tolist(),x[1].tolist()))
        #pdb.set_trace()     
        n_jj = set.intersection(p_idx_j,GT_idx_j)
        u_jj = set.union(p_idx_j,GT_idx_j)
    
        
        if len(GT_idx_j)!=0:
            count[j] = float(len(n_jj))/float(len(u_jj))

    result_class = count
    Aiou = np.sum(result_class[:])/float(len(np.unique(gt)))
    
    return Aiou



gpu0 = 0 
im_path = 'data/img'
model = getattr(deeplab_resnet,'Res_Deeplab')()
model.eval()
counter = 0
model.cuda(gpu0)
prefix = 'r0_scenes_A4_'
gt_path = 'data/gt'
img_list = open('data/list/val.txt').readlines()

for iter in range(13,20):
    saved_state_dict = torch.load('snapshots/'+prefix+str(iter)+'000.pth')
    if counter==0:
	print prefix
    counter+=1
    model.load_state_dict(saved_state_dict)

    pytorch_list = [];
    for i in img_list:
        img = np.zeros((513,513,3));
        #img[:,:,0] = 104.008*img[:,:,0]
        #img[:,:,1] = 116.669*img[:,:,1]
        #img[:,:,2] = 122.675*img[:,:,2]

        img_temp = cv2.imread(im_path+'/'+i[:-1]+'.jpg').astype(float)
        #img_temp = cv2.resize(img,(321,321)).astype(float)
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img[:img_temp.shape[0],:img_temp.shape[1],:] = img_temp
        gt = cv2.imread(gt_path+'/'+i[:-1]+'.png',0)
        gt[gt==255] = 0
       # gt = cv2.resize(gt,(321,321) , interpolation = cv2.INTER_NEAREST)

        output = model(Variable(torch.from_numpy(img[np.newaxis, :].transpose(0,3,1,2)).float(),volatile = True).cuda(gpu0))
        interp = nn.UpsamplingBilinear2d(size=(513, 513))
        output = interp(output[3]).cpu().data[0].numpy()
        output = output[:,:img_temp.shape[0],:img_temp.shape[1]]
        #output_temp = output.transpose(1,2,0)
        #output_temp = np.argmax(output_temp,axis = 2)
        #plt.imshow(img)
        #plt.show()
        
        output = output.transpose(1,2,0)
        output = np.argmax(output,axis = 2)
    #    plt.subplot(3, 1, 1)
    #    plt.imshow(img_temp2)
    #    plt.subplot(3, 1, 2)
    #    plt.imshow(gt)
    #    plt.subplot(3, 1, 3)
    #    plt.imshow(output)
           
    #    plt.show()
        iou_pytorch = get_iou(output,gt)       
    #    print iou_pytorch 
        pytorch_list.append(iou_pytorch)

    print 'pytorch',iter, np.sum(np.asarray(pytorch_list))/len(pytorch_list)
