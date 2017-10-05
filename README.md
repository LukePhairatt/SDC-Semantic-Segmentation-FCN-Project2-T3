# **CarND-Semantic-Segmentation Project**
Self-Driving Car Engineer Nanodegree Program

[//]: # (Image References)
[image0]: ./result/bin1.png "background"
[image1]: ./result/bin2.png "road"
[image2]: ./result/um_000003.png "result1"
[image3]: ./result/um_000043.png "result2"
[image4]: ./result/um_000062.png "result3"
[video]: ./result/out_video_raw.mp4 "video"

### Introduction  
In this project, the pixels of a road in images are labels using a Fully Convolutional Network (FCN). The data contains the train/test image of RGB.  
The ground truth images are combined to a set of two binary images (2 classes) segmented to the road and background for the training and prediction in this project. All roads (both lanes) will be detected. The model has been built from pre-trained VGG 16 network in which the 3rd,and 4th pooling layer are connected (skip connections) to the final fully connected layer 7. The architecture and dimenstion of the network are presented below.    
![bg][image0]  
Not road (background)  
![rd][image1]  
Road  

Figure 1:  2 binary classes (road/background)

### FCN Architecture  
The pre-trained VGG 16 model has these dimensionalities (wxhxd) on the input image of shape (160, 576)   
VGG16 Max-pooling layer 3: 20x72x256  
VGG16 Max-pooling layer 4: 10x36x512  
VGG16 layer 7: 5x18x4096   


Building the FCN
```sh

(5x18x4096)                              (5x18x2)                    (10x36x2)     (10x36x2)                      (20x72x2)    (20x72x2)                 (160x576x2)
vgg_layer_7 -->[1x1 conv,2 classes]--> layer_7_1x1 -->[upscale x2]--> decode_1 -+--> decode_2 -->[upscale x2]--> decode_3-+--> decode_4 -->[upscale x8] --> output
                                                                                |                                         |
(10x36x512)                             (10x36x2)      [add skip connection]    |                                         |
vgg_layer_4 -->[1x1 conv,2 classes]--> layer_4_1x1  ----------------------------+                                         |
                                                                                                                          |
(20x72x256)                             (20x72x2)                [add skip connection]                                    |
vgg_layer_3 -->[1x1 conv,2 classes]--> layer_3_1x1  ----------------------------------------------------------------------+

```



### Training Parameters
The result of the image segmentation has been completed with the following paramaters:  
Optimizer: Adam  
Learning rate: 1e-4  
Dropout: 0.7  
Epochs: 10  
Batch sizes: 1  
Number of classes: 2  

The augmentation includes image: flip, brightness adjustment, and adding random shadow. However, it is optional. The result on the tests images was good without using it.  

 
### Run
Run the following command to run the project:

```sh
python main.py
```

### Results
![result][image2]
![result][image3]
![result][image4]
![video][video]



##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)

##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.


