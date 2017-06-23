import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
sys.path.insert(0,'/data1/ravikiran/SketchObjPartSegmentation/src/caffe-switch/caffe/python')
import caffe
#import deeplab_resnet 
import os
from os import walk
import matplotlib.pyplot as plt
import torch.nn as nn



max_label = 20 
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)


def get_iou(pred,gt):
    if pred.shape!= gt.shape:
        print 'pred shape',pred.shape, 'gt shape', gt.shape
    assert(pred.shape == gt.shape)    
    gt = gt.astype(np.float32)
    pred = pred.astype(np.float32)

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
#    print result_class    
    Aiou = np.sum(result_class[:])/float(len(np.unique(gt)))
    
    return Aiou



gpu0 = 0 
im_path = 'data/img'
#model = getattr(deeplab_resnet,'Res_Deeplab')()
#model.eval()
caffe.set_mode_cpu()
caffe.set_device(0)
prototxt = "data/test_513.prototxt"
caffemodel = "data/train2_iter_20000.caffemodel"
net = caffe.Net(prototxt,caffemodel,  caffe.TEST)


counter = 0
prefix = 'r0_scenes_A1_'
gt_path = 'data/gt'
img_list = open('data/list/val.txt').readlines()
c=0

for iter in range(15,16):
    hist = np.zeros((max_label+1, max_label+1))
    caffe_list = [];
    #saved_state_dict = torch.load('snapshots/'+prefix+str(iter)+'000.pth')
    if counter==0:
	print prefix
    counter+=1

    pytorch_list = [];
    for i in img_list:
        print c," of ",len(img_list)
        c+=1
        img = np.zeros((513,513,3));
#        img[:,:,0] = 104.008*img[:,:,0]
#        img[:,:,1] = 116.669*img[:,:,1]
#        img[:,:,2] = 122.675*img[:,:,2]

        img_temp2 = cv2.imread(im_path+'/'+i[:-1]+'.jpg')

        img_temp = cv2.imread(im_path+'/'+i[:-1]+'.jpg').astype(float)
        #img_temp = cv2.resize(img,(321,321)).astype(float)
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img[:img_temp.shape[0],:img_temp.shape[1],:] = img_temp
        gt = cv2.imread(gt_path+'/'+i[:-1]+'.png',0)
        gt[gt==255] = 0
       # gt = cv2.resize(gt,(321,321) , interpolation = cv2.INTER_NEAREST)

#        plt.subplot(3, 1, 1)
#        plt.imshow(img_temp2)
#        plt.subplot(3, 1, 2)
#        plt.imshow(gt)
#        plt.subplot(3, 1, 3)
#        plt.imshow(output)
#        plt.show()

        net.blobs['data'].data[0] = img.transpose((2, 0, 1))
        assert net.blobs['data'].data[0].shape == (3, 513, 513)
        net.forward()
#        out_caffe = interp(Variable(torch.from_numpy(net.blobs['fc_fusion'].data)).cuda(gpu0)).cpu().data.numpy()[0]
        out_caffe = np.argmax(net.blobs['fc1_interp'].data[0][:,:img_temp.shape[0],:img_temp.shape[1]].transpose(1,2,0),axis = 2)
        iou_caffe = get_iou(out_caffe,gt)      
        caffe_list.append(iou_caffe)
        hist += fast_hist(gt.flatten(),out_caffe.flatten(),max_label+1)
    miou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print 'pytorch',iter, np.sum(np.asarray(caffe_list))/len(caffe_list),"miou = ",np.sum(miou)/len(miou)
