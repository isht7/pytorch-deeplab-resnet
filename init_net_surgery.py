import numpy as np
import os
os.environ["GLOG_minloglevel"] = "2"
import sys
sys.path.insert(0,'/data1/ravikiran/SketchObjPartSegmentation/src/caffe-switch/caffe/python')
import re
import caffe
import numpy as np
import skimage.io
import torch
import cv2
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import deeplab_resnet
from collections import OrderedDict

class CaffeParamProvider():
    def __init__(self, caffe_net):
        self.caffe_net = caffe_net

    def conv_kernel(self, name):
        k = self.caffe_net.params[name][0].data
        return k

    def conv_biases(self, name):
        k = self.caffe_net.params[name][1].data
        return k

    def bn_gamma(self, name):
        return self.caffe_net.params[name][0].data

    def bn_beta(self, name):
        return self.caffe_net.params[name][1].data

    def bn_mean(self, name):
        return (self.caffe_net.params[name][0].data/self.caffe_net.params[name][2].data)

    def bn_variance(self, name):
        return (self.caffe_net.params[name][1].data/self.caffe_net.params[name][2].data)

    def fc_weights(self, name):
        w = self.caffe_net.params[name][0].data
        #w = w.transpose((1, 0))
        return w


    def fc_biases(self, name):
        b = self.caffe_net.params[name][1].data
        return b


def preprocess(out):
    """Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    #out = np.copy(img) * 255.0
    out = out[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    out[0] -= 104.008
    out[1] -= 116.669
    out[2] -= 122.675
    return out


def assert_almost_equal(caffe_tensor, th_tensor):
    t = th_tensor[0]
    c = caffe_tensor[0]

    if t.shape != c.shape:
        print "t.shape", t.shape
        print "c.shape", c.shape

    d = np.linalg.norm(t - c)
    print "d", d
    assert d < 500

def dist_(caffe_tensor, th_tensor):
    t = th_tensor[0]
    c = caffe_tensor[0]

    if t.shape != c.shape:
        print "t.shape", t.shape
        print "c.shape", c.shape

    d = np.linalg.norm(t - c)
    print "d", d



# returns image of shape [321, 321, 3]
# [height, width, depth]
def load_image(path, size=321):
    img = cv2.imread(path)
    resized_img = cv2.resize(img,(size,size)).astype(float)
    return resized_img


def load_caffe(img_p, ):
    caffe.set_mode_cpu()
    #caffe.set_device(0)

    prototxt = "data/test_Scale1.prototxt" 
    caffemodel = "data/init.caffemodel" 
    net = caffe.Net(prototxt,caffemodel,  caffe.TEST)

    net.blobs['data'].data[0] = img_p.transpose((2, 0, 1))
    assert net.blobs['data'].data[0].shape == (3, 321, 321)
    net.forward()

    return net



def parse_pth_varnames(p, pth_varname, num_layers):
    # this function has been modified to fix #4

    post = ''
    EXP = 'voc12'
    #    Scale3.layer5.conv2d_list.2.bias
    if ('weight' in pth_varname and 'conv2d_list' in pth_varname):
#        #print ('res%d%s_branch%d%s'+post) % x
        if len(post)!=0:
            post = post[1:]+'_' 
        y = (EXP,int(pth_varname[25]))
        return p.conv_kernel(('fc1_%s_'+post+ 'c%d') % y)

    if ('bias' in pth_varname and 'conv2d_list' in pth_varname):
#        #print ('res%d%s_branch%d%s'+post) % x
         if len(post)!=0:
            post = post[1:]+'_'
         y = (EXP,int(pth_varname[25]))
         return p.conv_biases(('fc1_%s_'+post+'c%d') % y)


    if pth_varname == 'Scale.conv1.weight':
        return p.conv_kernel('conv1'+post)

    elif pth_varname == 'Scale.bn1.weight':
        return p.bn_gamma('scale_conv1'+post)

    elif pth_varname == 'Scale.bn1.bias':
        return p.bn_beta('scale_conv1'+post)

    elif pth_varname == 'Scale.bn1.running_mean':
        return p.bn_mean('bn_conv1'+post)

    elif pth_varname == 'Scale.bn1.running_var':
        return p.bn_variance('bn_conv1'+post)

    elif pth_varname == 'fc.weight':
        return p.fc_weights('fc1000')

    elif pth_varname == 'fc.bias':
        return p.fc_biases('fc1000')

    re1 = 'Scale.layer(\d+).(\d+).(downsample|conv1|bn1|conv2|bn2|conv3|bn3)' #changed to handle issue #4
    m = re.search(re1, pth_varname)

    def letter(i):
        return chr(ord('a') + i - 1)

    scale_num = int(m.group(1)) + 1

    block_num = int(m.group(2)) + 1



    if scale_num == 2:
        # scale 2 uses block letters
        block_str = letter(block_num)
    elif scale_num == 3 or scale_num == 4:
        # scale 3 uses numbered blocks 
        # scale 4 uses numbered blocks
        if num_layers == 50:
            block_str = letter(block_num)
        else:
            if block_num == 1:
                block_str = 'a'
            else:
                block_str = 'b%d' % (block_num - 1)
    elif scale_num == 5:
        # scale 5 uses block letters
        block_str = letter(block_num)
    else:
        raise ValueError("unexpected scale_num %d" % scale_num)

    branch = m.group(3)
    if branch == "downsample":
        branch_num = 1
        conv_letter = ''
    else:
        branch_num = 2
        conv_letter = letter(int(branch[-1]))

    x = (scale_num, block_str, branch_num, conv_letter)

    if ('weight' in pth_varname and 'conv' in pth_varname) or 'downsample.0.weight' in pth_varname:
        return p.conv_kernel(('res%d%s_branch%d%s'+post) % x)

    if ('weight' in pth_varname and 'bn' in pth_varname) or 'downsample.1.weight' in pth_varname:
        return p.bn_gamma(('scale%d%s_branch%d%s'+post) % x)

    if ('bias' in pth_varname and 'bn' in pth_varname) or 'downsample.1.bias' in pth_varname:
        return p.bn_beta(('scale%d%s_branch%d%s'+post) % x)

    if ('running_mean' in pth_varname and 'bn' in pth_varname) or 'downsample.1.running_mean' in pth_varname:
        return p.bn_mean(('bn%d%s_branch%d%s'+post) % x)

    if ('running_var' in pth_varname and 'bn' in pth_varname) or 'downsample.1.running_var' in pth_varname:
        return p.bn_variance(('bn%d%s_branch%d%s'+post) % x)

    raise ValueError('unhandled var ' + pth_varname)

def convert(img_p, layers):
    caffe_model = load_caffe(img_p)

    param_provider = CaffeParamProvider(caffe_model)
    model = getattr(deeplab_resnet,'Res_Deeplab')()
    old_dict = model.state_dict()
    new_state_dict = OrderedDict()
    keys = model.state_dict().keys()
    for var_name in keys[:]:
        data = parse_pth_varnames(param_provider, var_name, layers)
        new_state_dict[var_name] = torch.from_numpy(data).float()
    
    model.load_state_dict(new_state_dict)

    
    o = []
    def hook(module, input, output):
        #print module
        o.append(input[0].data.numpy())
    
    model.Scale.conv1.register_forward_hook(hook)   #0, data
    model.Scale.bn1.register_forward_hook(hook)     #1 conv1 out
    model.Scale.relu.register_forward_hook(hook)  #2 batch norm out
    model.Scale.maxpool.register_forward_hook(hook)    #3 bn1, relu out
    model.Scale.layer1._modules['0'].conv1.register_forward_hook(hook)   #4, pool1 out 
    model.Scale.layer1._modules['1'].conv1.register_forward_hook(hook) #5, res2a out
    model.Scale.layer5.conv2d_list._modules['0'].register_forward_hook(hook) #6, res5c out

    model.eval()
    output = model(Variable(torch.from_numpy(img_p[np.newaxis, :].transpose(0,3,1,2)).float(),volatile=True))  
    

    dist_(caffe_model.blobs['data'].data,o[0])
    dist_(caffe_model.blobs['conv1'].data,o[3])
    dist_(caffe_model.blobs['pool1'].data,o[4])
    dist_(caffe_model.blobs['res2a'].data,o[5])
    dist_(caffe_model.blobs['res5c'].data,o[6])
    dist_(caffe_model.blobs['fc1_voc12'].data,output[3].data.numpy())

    print 'input image shape',img_p[np.newaxis, :].transpose(0,3,1,2).shape
    print 'output shapes -'
    for a in output:
	print 	a.data.numpy().shape

    torch.save(model.state_dict(),'data/MS_DeepLab_resnet_pretrained_COCO_init.pth')


def main():
    img = load_image("data/cat.jpg")
    img_p = preprocess(img)

    print "CONVERTING Multi-scale DeepLab_resnet COCO init"
    convert(img_p, layers = 101)


if __name__ == '__main__':
    main()





