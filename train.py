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
from docopt import docopt

docstr = """Train ResNet-DeepLab on VOC12 (scenes) in pytorch using MSCOCO pretrained initialization 

Usage: 
    train.py [options]

Options:
    -h, --help                  Print this message
    --segnetLoss                Weigh each class differently
    --GTpath=<str>              Ground truth path prefix [default: data/gt/]
    --IMpath=<str>              Sketch images path prefix [default: data/img/]
    --LISTpath=<str>            Input image number list file [default: data/list/train_aug.txt]
    --lr=<float>                Learning Rate [default: 0.00025]
    -i, --iterSize=<int>        Num iters to accumulate gradients over [default: 10]
    --wtDecay=<float>          Weight decay during training [default: 0.0005]
    --gpu0=<int>                GPU number [default: 0]
    --maxIter=<int>             Maximum number of iterations [default: 20000]
"""

#    -b, --batchSize=<int>       num sample per batch [default: 1] currently only batch size of 1 is implemented, arbitrary batch size to be implemented soon
args = docopt(docstr, version='v0.1')
print(args)

cudnn.enabled = False
gpu0 = int(args['--gpu0'])


def outS(i):
    i = int(i)
    i = (i+1)/2
    i = int(np.ceil((i+1)/2.0))
    i = (i+1)/2
    return i

def find_med_frequency(img_list,max_):
    gt_path = args['--GTpath'] 
    dict_store = {}
    for i in range(max_):
        dict_store[i] = []
    for i,piece in tqdm(enumerate(img_list)):
        gt = cv2.imread(gt_path+piece+'.png')[:,:,0]
        gt[gt==255] = 0
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
    gt_path =  args['--GTpath']
    img_path = args['--IMpath']

    scale = random.uniform(0.5, 1.3)
    dim = int(scale*321)
    images = np.zeros((dim,dim,3,len(chunk)))
    gt = np.zeros((dim,dim,1,len(chunk)))
    for i,piece in enumerate(chunk):
        flip_p = random.uniform(0, 1)
        print img_path+piece+'.jpg'
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
    images = images.transpose((3,2,0,1))
    images = torch.from_numpy(images).float()
    return images, labels



def loss_calc(out, label,gpu0):
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label[:,:,0,:].transpose(2,0,1)
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    m = nn.LogSoftmax()
    criterion = nn.NLLLoss2d()
    out = m(out)
    
    return criterion(out,label)



def loss_calc_seg(out, label,gpu0,seg_weights):
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = label[:,:,0,:].transpose(2,0,1)
    label = torch.from_numpy(label).long()
    label = Variable(label).cuda(gpu0)
    m = nn.LogSoftmax()
    criterion = nn.NLLLoss2d(torch.from_numpy(seg_weights).float().cuda(gpu0))
    out = m(out)
    
    return criterion(out,label)





def lr_poly(base_lr, iter,max_iter,power):
    return base_lr*((1-float(iter)/max_iter)**(power))


def get_1x_lr_params_NOscale(model):
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
            for k in j.parameters():
                jj+=1
                if k.requires_grad:
                    yield k

def get_10x_lr_params(model):
    b = []
    b.append(model.Scale1.layer5.parameters())
    b.append(model.Scale2.layer5.parameters())
    b.append(model.Scale3.layer5.parameters())

    for j in range(len(b)):
        for i in b[j]:
            yield i

if not os.path.exists('data/snapshots'):
    os.makedirs('data/snapshots')


model = getattr(deeplab_resnet,'Res_Deeplab')()
saved_state_dict = torch.load('data/MS_DeepLab_resnet_pretrained_COCO_init.pth')
model.load_state_dict(saved_state_dict)

max_iter = int(args['--maxIter']) 
batch_size = 1
weight_decay = float(args['--wtDecay'])
base_lr = float(args['--lr'])

model.float()
model.eval()

img_list = read_file(args['--LISTpath'])
#global_stats_median,global_stats_sum,global_stats_presence = find_med_frequency(img_list,21)
#freq_c = global_stats_sum/global_stats_presence
#seg_weights = np.median(freq_c)/freq_c
#print seg_weights

data_list = []
for i in range(10):  # make list for 10 epocs, though we will only use the first max_iter*batch_size entries of this list
    np.random.shuffle(img_list)
    data_list.extend(img_list)

model.cuda(gpu0)
criterion = nn.CrossEntropyLoss() # use a Classification Cross-Entropy loss
optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': base_lr }, {'params': get_10x_lr_params(model), 'lr': 10*base_lr} ], lr = base_lr, momentum = 0.9,weight_decay = weight_decay)

optimizer.zero_grad()
data_gen = chunker(data_list, batch_size)

for iter in range(max_iter+1):
    chunk = data_gen.next()

    images, label = get_data_from_chunk_v2(chunk)
    images = Variable(images).cuda(gpu0)

    out = model(images)
    loss = loss_calc(out[0], label[0],gpu0)
    iter_size = int(args['--iterSize']) 
    for i in range(len(out)-1):
        loss = loss + loss_calc(out[i+1],label[i+1],gpu0)
    loss = loss/iter_size 
    loss.backward()

    if iter %1 == 0:
        print 'iter = ',iter, 'of',max_iter,'completed, loss = ', iter_size*(loss.data.cpu().numpy())

    if iter % iter_size  == 0:
        optimizer.step()
        lr_ = lr_poly(base_lr,iter,max_iter,0.9)
        print '(poly lr policy) learning rate',lr_
        optimizer = optim.SGD([{'params': get_1x_lr_params_NOscale(model), 'lr': lr_ }, {'params': get_10x_lr_params(model), 'lr': 10*lr_} ], lr = lr_, momentum = 0.9,weight_decay = weight_decay)
        optimizer.zero_grad()

    if iter % 1000 == 0 and iter!=0:
        print 'taking snapshot ...'
        torch.save(model.state_dict(),'data/snapshots/VOC12_scenes_'+str(iter)+'.pth')


