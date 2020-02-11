# single-pixel-depth-imaging
A pyTorch implementation of SPDI network, with support for training, inference and evaluation.

## Environments

        1.Ubuntu 16.04 x86_64 
        2.Python 3.6 
        3.LaptopPC (Intel Core i7-8750H) 
        4.Pytorch 0.4 


## Installation

### Download pretrained weights
    
    The weights of each network are stored in the weights file, where： 
    for CONV1: CNN1.pt  
    for CONV2: CNN2.pt
    for CONV3: CNN3.pt 
    
    All the weights can be downloaded as following website.

### Download datasets 
    
  Complete datasets:  <br>
  https://pan.baidu.com/s/1M6O54pHen4va2dXEkXwXLA
  
  code: yhz8
  
  
  Background image: STL10 datasets:  <br>
  https://cs.stanford.edu/~acoates/stl10/ 
  
## Network structure
An end-to-end deep neural network structure is designed, which is shown as follows. For depth reconstruction, it contains the self-encoding network and parallel residual network respectively used for reconstruction of two-dimensional scene and depth information. Specifically, the self-encoding network (CONV1) firstly uses the fully-connected layer to convert the one-dimensional measurement to a two-dimensional image, and then uses a three-dimensional convolution kernel to perform multiple convolution operations to extract deep features. <br>


The depth information reconstruction of the three-dimensional image is performed by connecting the parallel residual network (CONV2), and finally a grayscale image representing the height is obtained. The grayscale image is visualized in three dimensions to obtain the three-dimensional information of the object. <br>
    
 <div align=center><img width="650" height="350" src="http://github.com/bianlab/single-pixel-depth-imaging/raw/master/images/network.jpg"/></div> 
  
## Test 
To understand the complete process of SPDI network, run demo.py. 
    
    $ cd ~ 
    $ python3 demo.py 

This Demo gives the reconstruction of mountain peak, and background intensity is 1. 

        For each detailed information of network, see the corresponding file in network_py_file.
        each named CONV1.py, CONV2.py and CONV3.py.

## Results
Here are some figures of height and background reconstruction results from testsets.
    
 <div align=center><img width="550" height="450" src="https://github.com/bianlab/single-pixel-depth-imaging/raw/master/images/simulation.png"/></div>

## Train
Use 12000 datasets, 10 percent of datasets are used for validation, 10 percent of datasets are used for test. 

    $ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]

## Create
The process of create 1-D measurement of SPI measurement, run create.py. This code is for forming the 1-D measurement from 2-D pictures and the corresponding 0-1 pattern. 
