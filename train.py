import torch
import torch.nn as nn
import numpy as np
import pickle
import deeplab_resnet 
import cv2
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import sys
import matplotlib.pyplot as plt
from tqdm import *
import random
cudnn.enabled = False
gpu0 = 0 

def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def find_med_frequency(img_list,max_):
    gt_path = '/data1/ravikiran/pytorch-deeplab-resnet/data/gt/'
    dict_store = {}
    for i in range(max_):
        dict_store[i] = []
    for i,piece in tqdm(enumerate(img_list)):
        gt = cv2.imread(gt_path+piece+'.png')[:,:,0]
        gt[gt==255] = 0
        #unique, counts = np.unique(gt,return_counts=True)
        #dict_c = dict(zip(unique,counts))
        for i in range(21):
            dict_store[i].append(np.count_nonzero(gt == i))
    global_stats_median = np.zeros((21,))
    global_stats_sum = np.zeros((21,))
    global_stats_presence = np.zeros((21,))
    for i in range(21):
        global_stats_median[i] = np.median(dict_store[i])
        global_stats_sum[i] = np.sum(dict_store[i])
        global_stats_presence[i] = np.count_nonzero(dict_store[i]) #changed to new 
    return global_stats_median,global_stats_sum,global_stats_presence


def read_file(path_to_file):
    with open(path_to_file) as f:
        img_list = []
        for line in f:
            img_list.append(line[:-1])
    return img_list

def chunker(seq, size):
 return (seq[pos:pos+size] for pos in xrange(0,len(seq), size))

def resize_label_batch(label, size):
    label_resized = np.zeros((size,size,1,label.shape[3]))
    interp = nn.UpsamplingBilinear2d(size=(size, size))
    labelVar = Variable(torch.from_numpy(label.transpose(3, 2, 0, 1)))
    label_resized[:, :, :, :] = interp(labelVar).data.numpy().transpose(2, 3, 1, 0)

    return label_resized

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def scale_im(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims).astype(float)

def scale_gt(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims,interpolation = cv2.INTER_NEAREST).astype(float)
   
def get_data_from_chunk_v2(chunk):
    gt_path = '/data1/ravikiran/pytorch-deeplab-resnet/data/gt/'
    img_path = '/data1/ravikiran/pytorch-deeplab-resnet/data/img/'
    scale = random.uniform(0.5, 1.3)
    dim = int(scale*321)
    images = np.zeros((dim,dim,3,len(chunk)))
    gt = np.zeros((dim,dim,1,len(chunk)))
    for i,piece in enumerate(chunk):
        flip_p = random.uniform(0, 1)
        img_temp = cv2.imread(img_path+piece+'.jpg').astype(float)
        img_temp = cv2.resize(img_temp,(321,321)).astype(float)
        img_temp = scale_im(img_temp,scale)
        img_temp[:,:,0] = img_temp[:,:,0] - 104.008
        img_temp[:,:,1] = img_temp[:,:,1] - 116.669
        img_temp[:,:,2] = img_temp[:,:,2] - 122.675
        img_temp = flip(img_temp,flip_p)
        images[:,:,:,i] = img_temp

        gt_temp = cv2.imread(gt_path+piece+'.png')[:,:,0]
        gt_temp[gt_temp == 255] = 0
        gt_temp = cv2.resize(gt_temp,(321,321) , interpolation = cv2.INTER_NEAREST)
        gt_temp = scale_gt(gt_temp,scale)
        gt_temp = flip(gt_temp,flip_p)
        gt[:,:,0,i] = gt_temp
        a = outS(321*scale)#41
        b = outS(321*0.5*scale)#21
    labels = [resize_label_batch(gt,i) for i in [a,a,b,a]]
#    gt = gt.transpose((3,2,0,1))
#   image shape H,W,3,batch -> batch,3,H,W
    images = images.transpose((3,2,0,1))
#    gt.cuda(gpu0)
    images = torch.from_numpy(images).float()
#    images.cuda(gpu0)
    return images, labels



def loss_calc(criterion, out, label,gpu0):
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    #plt.imshow(label[:,:,0,0])
    #plt.show()
    label = label[:,:,0,:].transpose(2,0,1)
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    m = nn.LogSoftmax()
    criterion = nn.NLLLoss2d()
    out = m(out)
    
    return criterion(out,label)



def loss_calc_seg(criterion, out, label,gpu0,seg_weights):
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    #plt.imshow(label[:,:,0,0])
    #plt.show()
    label = label[:,:,0,:].transpose(2,0,1)
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    m = nn.LogSoftmax()
    criterion = nn.NLLLoss2d(torch.from_numpy(seg_weights).float().cuda(gpu0))
    out = m(out)
    
    return criterion(out,label)





def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def get_1x_lr_params(model):
#    all_params = model.parameters()
#    b = []
#    b.append(model.Scale1.layer5.parameters())
#    b.append(model.Scale2.layer5.parameters())
#    b.append(model.Scale3.layer5.parameters())

#    for i in all_params:
#        if (i not in b[0]) and (i not in b[1]) and (i not in b[2]):
#            yield i
    b = []

    b.append(model.Scale1.conv1.parameters())
    b.append(model.Scale1.bn1.parameters())
    b.append(model.Scale1.layer1.parameters())
    b.append(model.Scale1.layer2.parameters())
    b.append(model.Scale1.layer3.parameters())
    b.append(model.Scale1.layer4.parameters())

    b.append(model.Scale2.conv1.parameters())
    b.append(model.Scale2.bn1.parameters())
    b.append(model.Scale2.layer1.parameters())
    b.append(model.Scale2.layer2.parameters())
    b.append(model.Scale2.layer3.parameters())
    b.append(model.Scale2.layer4.parameters())

    b.append(model.Scale3.conv1.parameters())
    b.append(model.Scale3.bn1.parameters())
    b.append(model.Scale3.layer1.parameters())
    b.append(model.Scale3.layer2.parameters())
    b.append(model.Scale3.layer3.parameters())
    b.append(model.Scale3.layer4.parameters())

    for i in range(len(b)):
        for j in b[i]:
            yield j

def get_1x_lr_params_NOscale(model):
#    all_params = model.parameters()
#    b = []
#    b.append(model.Scale1.layer5.parameters())
#    b.append(model.Scale2.layer5.parameters())
#    b.append(model.Scale3.layer5.parameters())

#    for i in all_params:
#        if (i not in b[0]) and (i not in b[1]) and (i not in b[2]):
#            yield i
    b = []

    b.append(model.Scale1.conv1)
    b.append(model.Scale1.bn1)
    b.append(model.Scale1.layer1)
    b.append(model.Scale1.layer2)
    b.append(model.Scale1.layer3)
    b.append(model.Scale1.layer4)

    b.append(model.Scale2.conv1)
    b.append(model.Scale2.bn1)
    b.append(model.Scale2.layer1)
    b.append(model.Scale2.layer2)
    b.append(model.Scale2.layer3)
    b.append(model.Scale2.layer4)

    b.append(model.Scale3.conv1)
    b.append(model.Scale3.bn1)
    b.append(model.Scale3.layer1)
    b.append(model.Scale3.layer2)
    b.append(model.Scale3.layer3)
    b.append(model.Scale3.layer4)

    for i in range(len(b)):
        for j in b[i].modules():
            jj = 0
            #print j
            for k in j.parameters():
                jj+=1
                #print type(k), k.requires_grad,'completed_round 1'
                if k.requires_grad:
                    yield k
            #print jj

def get_10x_lr_params(model):
    b = []
    b.append(model.Scale1.layer5.parameters())
    b.append(model.Scale2.layer5.parameters())
    b.append(model.Scale3.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i




model = getattr(deeplab_resnet,'Res_Deeplab')()
saved_state_dict = torch.load('MS_DeepLab_resnet_pretrained_COCO_init.pth')
model.load_state_dict(saved_state_dict)

max_iter = 20000
batch_size = 1
train_epocs = 10
model.float()
model.eval()

img_list = read_file('/data1/ravikiran/pytorch-deeplab-resnet/data/list/train_aug.txt')
#global_stats_median,global_stats_sum,global_stats_presence = find_med_frequency(img_list,21)
#freq_c = global_stats_sum/global_stats_presence
#seg_weights = np.median(freq_c)/freq_c
#print seg_weights

data_list = []
for i in range(train_epocs):
    np.random.shuffle(img_list)
    data_list.extend(img_list)

model.cuda(gpu0)
criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
#optimizer = optim.SGD(model.parameters(), lr=0.00025, momentum=0.9)
optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': 0.00025 }, {'params': get_10x_lr_params(model), 'lr': 0.0025} ], lr = 0.00025, momentum = 0.9,weight_decay = 0.0005)
#optimizer = optim.SGD([{'params': get_10x_lr_params(model), 'lr': 0.00025 }], lr = 0.00025, momentum = 0.9)

optimizer.zero_grad()

for iter,chunk in enumerate(chunker(data_list, batch_size)):
    images, label = get_data_from_chunk_v2(chunk)

    images = Variable(images).cuda(gpu0)


    out = model(images)
    loss = loss_calc(criterion, out[0], label[0],gpu0)
    iter_size = 10
    for i in range(len(out)-1):
        loss = loss + loss_calc(criterion, out[i+1],label[i+1],gpu0)
    loss = loss/iter_size 
    loss.backward()

    if iter %1 == 0:
        print 'iter = ',iter, 'of',max_iter,'completed, loss = ', iter_size*(loss.data.cpu().numpy())

    if iter % iter_size  == 0:
        optimizer.step()
        lr_ = lr_poly(0.00025,iter,max_iter,0.9)
        print '(poly lr policy) learning rate',lr_
#       optimizer = optim.SGD(model.parameters(),lr = updated_lr,  momentum = 0.9)
        optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_ }, {'params': get_10x_lr_params(model), 'lr': 10*lr_} ], lr = lr_, momentum = 0.9,weight_decay = 0.0005)
        optimizer.zero_grad()

    if iter % 1000 == 0 and iter!=0:
        print 'taking snapshot ...'
        torch.save(model.state_dict(),'snapshots/r0_scenes_A4_'+str(iter)+'.pth')
    if iter ==max_iter:
        break


execfile('evalpyt_513.py')
