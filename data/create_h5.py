"""
This file is used to create the hdf5 file (cat.h5) which is used by convert_deeplab_resnet.py and init_net_surgery.py
"""
import cv2
import h5py, os
import numpy as np
def preprocess(out):
    #"""Changes RGB [0,1] valued image to BGR [0,255] with mean subtracted."""
    #out = np.copy(img) * 255.0
    out = out[:, :, [2, 1, 0]]  # swap channel from RGB to BGR
    out[0] -= 104.008
    out[1] -= 116.669
    out[2] -= 122.675
    return out

def load_image(path, size=321):
    img = cv2.imread(path)
    resized_img = cv2.resize(img,(321,321)).astype(float)
    return resized_img


img = load_image("cat.jpg")
print img.shape
img = preprocess(img)
print img.shape

gts = np.zeros((1, 1, 321, 321)) #garbage value as this is irrelavent for conversion to .pth files


comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
with h5py.File('cat.h5', 'w') as f:
    f.create_dataset('data', data=img[np.newaxis,:].transpose(0,3,1,2), **comp_kwargs)
    f.create_dataset('label', data=gts, **comp_kwargs)
