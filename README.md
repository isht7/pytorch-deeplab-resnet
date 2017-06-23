# pytorch-deeplab-resnet
[DeepLab resnet](https://arxiv.org/abs/1606.00915) model implementation in pytorch. 

The architecture of deepLab-ResNet has been replicated exactly as it is from the caffe implementation. This architecture calculates losses on input images over multiple scales ( 1x, 0.75x, 0.5x ). Losses are calculated individually over these 3 scales. In addition to these 3 losses, one more loss is calculated after merging the output score maps on the 3 scales. These 4 losses are added to calculate the total loss.
The older version is available [here](https://github.com/isht7/pytorch-deeplab-resnet/tree/independent_wts)
## Updates

**24 June 2017**

Now, weights over the 3 scales ( 1x, 0.75x, 0.5x ) are shared as in the caffe implementation. Previously, each of the 3 scales had seperate weights. Results are almost same after making this change (more in the results section). However, the size of the trained .pth model has reduced significantly. Memory occupied on GPU(11.9 GB) and time taken (~3.5 hours) during training are same as before. Links to corresponding .pth files have been updated.

# Usage
### Replicating caffe performance
To convert the caffemodel released by [authors](https://arxiv.org/abs/1606.00915), download the deeplab-resnet caffemodel (`train_iter_20000.caffemodel`) pretrained on VOC into the data folder. After that, run
```
python convert_deeplab_resnet.py
```
to generate the corresponding pytorch model file (.pth). The generated .pth snapshot file can be used to get the same test performace as offered by using the caffemodel in caffe. If you do not want to generate the .pth file yourself, you can download it [here](https://drive.google.com/open?id=0BxhUwxvLPO7TeXFNQ3YzcGI4Rjg).

To run `convert_deeplab_resnet.py`, [deeplab v2 caffe](https://bitbucket.org/aquariusjay/deeplab-public-ver2) and pytorch (python 2.7) are required.

If you want to train your model in pytorch, move to the next section.
### Training 
Step 1: Convert `init.caffemodel` to a .pth file: `init.caffemodel` contains MS COCO trained weights. We use these weights as initilization for all but the final layer of our model. For the last layer, we use random gaussian with a standard deviation of 0.01 as the initialization.
To convert `init.caffemodel` to a .pth file, run (or download the converted .pth [here](https://drive.google.com/open?id=0BxhUwxvLPO7TVFJQU1dwbXhHdEk))
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
* Random flipping and random scaling of input has been used as data augmentation. The caffe implementation uses 4 fixed scales (0.5,0.75,1,1.25,1.5) while in the pytorch implementation, for each iteration scale is randomly picked in the range - [0.5,1.3].
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
When trained on VOC augmented training set (with 10582 images) using MS COCO pretrained initialization in pytorch, we get a validation performance of 78.48% (Mean IOU is calculated for each image and these values are averaged together. This way of calculating mean IOU is different than the one used by [authors](https://arxiv.org/abs/1606.00915). The method used by [authors](https://arxiv.org/abs/1606.00915) is presently in [development](https://github.com/isht7/pytorch-deeplab-resnet/tree/development) branch [here](https://github.com/isht7/pytorch-deeplab-resnet/blob/development/evalpyt2.py) and will be moved here in some time.)
You can download the corresponding .pth file [here](https://drive.google.com/open?id=0BxhUwxvLPO7TT0Y5UndZckIwMVE)

To replicate this performance, run 
```
train.py --lr 0.00025 --wtDecay 0.0005 --maxIter 20000 --GTpath <train gt images path here> --IMpath <train images path here> --LISTpath data/list/train_aug.txt
```
## Acknowledgement
This work was done during my time at [Video Analytics Lab](http://val.serc.iisc.ernet.in/valweb/). A big thanks to them for their GPUs.
 
A part of the code has been borrowed from [https://github.com/ry/tensorflow-resnet](https://github.com/ry/tensorflow-resnet)
