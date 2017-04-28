# pytorch-deeplab-resnet
[DeepLab resnet](https://arxiv.org/abs/1606.00915) model in pytorch
* The architecture of deepLab-ResNet has been replicated exactly as it is from the caffe implementation.
# Usage
### Replicating caffe performance
The repository is an implementation of deeplab-resnet in pytorch. To convert the caffemodel released by [authors](https://arxiv.org/abs/1606.00915), download the deeplab-resnet caffemodel (`train_iter_20000.caffemodel`) pretrained on VOC into the data folder. After that, run
```
python convert_deeplab_resnet.py
```
to generate the corresponding pytorch model file (.pth). If you want to train your model in pytorch, move to the next section.
### Training 
Step 1: Convert `init.caffemodel` to a .pth file, 'init.caffemodel' contains MS COCO trained weights. We use these weights as initilization for all but the final layer of our model. For the last layer, we use random gaussian with a standard deviation of 0.01 as the initialization.
To convert `init.caffemodel` to a .pth file, run
```
python init_net_surgery.py
```
Step 2: Now that we have our initialization, we can train deeplab-resnet by running,
```
python train.py
```
By default, snapshots are saved in every 1000 iterations in the  data/snapshots.
The following features have been implemented in this repository -
* Training regime is the same as that of the caffe implementation - SGD with momentum is used, along with the `poly` lr decay policy. A weight decay has been used. The last layer has `10` times the learning of other layers.  
* The iter\_size parameter of caffe has been implemented, effectively increasing the batch\_size to batch\_size times iter\_size
* Random flipping and random scaling of input has been used as data augmentation.
* The boundary label (255 in ground truth labels) has not been ignored in the loss function in the current version, instead it has been merged with the background. The ignore\_label caffe parameter would be implemented in the future versions. Post processing using CRF has not been implemented. 
### Evaluation
Evaluation of the saved models can be done by running
```
python evalpyt.py
```
### Results
When trained on VOC augmented training set (with 10582 images) using MS COCO pretrained initialization in pytorch, we get a validation accuray of 78.49% (validation set has 1449 images, [authors](https://arxiv.org/abs/1606.00915) report validation performance of 77.69% with their caffe implementation)
## Acknowledgement
A part of the code has been borrowed from [https://github.com/ry/tensorflow-resnet](https://github.com/ry/tensorflow-resnet)
