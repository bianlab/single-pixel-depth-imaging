# single-pixel-depth-imaging
A pyTorch implementation of SPDI network, with support for training, inference and evaluation.

## Environments

        1.Ubuntu 16.04 x86_64 <br>
        2.Python 3.6 <br>
        3.LaptopPC (Intel Core i7-8750H) <br>
        4.Pytorch 0.4 <br>


## Installation
Download pretrained weights
    
    The weights of each network are stored in the weights file, where： <br>
    for CONV1:net_SPI.pt  <br>
    for CONV2:net_depth.pt <br>
    for CONV3: net_bg.pt <br>

Download datasets 
    
    Complete datasets: <br>
    Https://www.baidunetdisc <br>
    Background image: STL10 datasets <br>
    https://cs.stanford.edu/~acoates/stl10/ <br>
  
## Network structure
    An end-to-end deep neural network structure is designed, which is shown as follows. For depth reconstruction, it contains the self-encoding network and parallel residual network respectively used for reconstruction of two-dimensional scene and depth information. Specifically, the self-encoding network (CONV1) firstly uses the fully-connected layer to convert the one-dimensional measurement to a two-dimensional image, and then uses a three-dimensional convolution kernel to perform multiple convolution operations to extract deep features. <br>
    
    The depth information reconstruction of the three-dimensional image is performed by connecting the parallel residual network (CONV2), and finally a grayscale image representing the height is obtained. The grayscale image is visualized in three dimensions to obtain the three-dimensional information of the object. <br>
    
    FIG1
  
## Test 
To understand the complete process of SPDI network, run demo.py. 
    
    $ python3 demo.py 
This Demo gives the reconstruction of mountain peak, and background intensity is 1. 

For each detailed information of network, see the corresponding file name, eg:CONV1.py.

## Results
Here are some figures of height and background reconstruction results from testsets.
    
    FIG2 
    
    FIG3

## Train
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
The process of create 1-D measurement of SPI measurement, run create.py.
