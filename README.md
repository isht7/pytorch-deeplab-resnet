# pytorch-deeplab-resnet
DeepLab resnet model in pytorch

# Usage
### Replicating caffe performance
The repository contains model definition of deeplab-resnet in pytorch. To use this code, download the deeplab-resnet caffemodel (train\_iter\_20000.caffemodel) pretrained on VOC into the data folder. After that, run
```
python convert_deeplab_resnet.py
```
to generate the pytorch model file(.pth)
### Training 
Step 1: Convert init.caffemodel to a .pth file. init.caffemodel contains MSCOCO trained weights. We use these weights as initilization for all but the final layer of our model. For the last layer, we use random gaussian with a standard deviation of 0.01
To convert init.caffemodel to a .pth file, run
```
python init_net_surgery.py
```
Step 2: Now that we have our initialization, we can train deeplab-resnet by running,
```
python train.py
```
By default, snapshots are saved in every 1000 iterations.
The following features have been implemented in this repository -
* The iter\_size parameter of caffe has been implemented, effectively increasing the batch\_size to batch\_size times iter\_size
* Random flipping and random scaling of input has been used as data augmentation
### Evaluation
Evaluation of the saved models can be done by running
```
python evalpyt_513.py
```
## Acknowledgement
A part of the code has been borrowed from [https://github.com/ry/tensorflow-resnet](https://github.com/ry/tensorflow-resnet)
