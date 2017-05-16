# pytorch-deeplab-resnet
[DeepLab resnet](https://arxiv.org/abs/1606.00915) model implementation in pytorch. 

The architecture of deepLab-ResNet has been replicated exactly as it is from the caffe implementation. This architecture calculates losses on input images over multiple scales ( 1x, 0.75x, 0.5x ). Losses are calculated individually over these 3 scales. In addition to these 3 losses, one more loss is calculated after merging the output score maps on the 3 scales. These 4 losses are added to calculate the total loss.

# Usage
### Replicating caffe performance
To convert the caffemodel released by [authors](https://arxiv.org/abs/1606.00915), download the deeplab-resnet caffemodel (`train_iter_20000.caffemodel`) pretrained on VOC into the data folder. After that, run
```
python convert_deeplab_resnet.py
```
to generate the corresponding pytorch model file (.pth). The generated .pth snapshot file can be used to get the same test performace as offered by using the caffemodel in caffe. If you do not want to generate the .pth file yourself, you can download it [here](https://drive.google.com/file/d/0BxhUwxvLPO7Tb210MFB5WmQ1elE/view?usp=sharing).

To run `convert_deeplab_resnet.py`, [deeplab v2 caffe](https://bitbucket.org/aquariusjay/deeplab-public-ver2) and pytorch (python 2.7) are required.

If you want to train your model in pytorch, move to the next section.
### Training 
Step 1: Convert `init.caffemodel` to a .pth file: `init.caffemodel` contains MS COCO trained weights. We use these weights as initilization for all but the final layer of our model. For the last layer, we use random gaussian with a standard deviation of 0.01 as the initialization.
To convert `init.caffemodel` to a .pth file, run (or download the converted .pth [here](https://drive.google.com/file/d/0BxhUwxvLPO7TMVNWWWhPUzNFVU0/view?usp=sharing))
```
python init_net_surgery.py
```
To run `init_net_surgery .py`, [deeplab v2 caffe](https://bitbucket.org/aquariusjay/deeplab-public-ver2) and pytorch (python 2.7) are required.

Step 2: Now that we have our initialization, we can train deeplab-resnet by running,
```
python train.py
```
To get a description of each command-line arguments, run
```
python train.py -h
```
To run `train.py`, pytorch (python 2.7) is required.


By default, snapshots are saved in every 1000 iterations in the  data/snapshots.
The following features have been implemented in this repository -
* Training regime is the same as that of the caffe implementation - SGD with momentum is used, along with the `poly` lr decay policy. A weight decay has been used. The last layer has `10` times the learning rate of other layers.  
* The iter\_size parameter of caffe has been implemented, effectively increasing the batch\_size to batch\_size times iter\_size
* Random flipping and random scaling of input has been used as data augmentation.
* The boundary label (255 in ground truth labels) has not been ignored in the loss function in the current version, instead it has been merged with the background. The ignore\_label caffe parameter would be implemented in the future versions. Post processing using CRF has not been implemented.
* Batchnorm parameters are kept fixed during training. Also, caffe setting `use_global_stats = True` is reproduced during training. Running mean and variance is not calculated during.

When run on a Nvidia Titan X GPU, `train.py` occupies about 11.9 GB of memory. 

### Evaluation
Evaluation of the saved models can be done by running
```
python evalpyt.py
```
To get a description of each command-line arguments, run
```
python evalpyt.py -h
```
### Results
When trained on VOC augmented training set (with 10582 images) using MS COCO pretrained initialization in pytorch, we get a validation performance of 78.46% (mean IOU over all classes, validation set has 1449 images, [authors](https://arxiv.org/abs/1606.00915) report validation performance of 77.69% with their caffe implementation).
You can download the corresponding .pth file [here](https://drive.google.com/file/d/0BxhUwxvLPO7TSktPZFpSRzJDems/view?usp=sharing)

To replicate this performance, run 
```
train.py --lr 0.00025 --wtDecay 0.0005 --maxIter 20000 --GTpath <train gt images path here> --IMpath <train images path here> --LISTpath data/list/train_aug.txt
```
## Acknowledgement
This work was done during my time at [Video Analytics Lab](http://val.serc.iisc.ernet.in/valweb/). A big thanks to them for their GPUs.
 
A part of the code has been borrowed from [https://github.com/ry/tensorflow-resnet](https://github.com/ry/tensorflow-resnet)
