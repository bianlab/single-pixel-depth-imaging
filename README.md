# single-pixel-depth-imaging
A pyTorch implementation of SPDI network, with support for training, inference and evaluation.

## 1. Environments

        1.Ubuntu 16.04 x86_64 
        2.Python 3.6 
        3.LaptopPC (Intel Core i7-8750H) 
        4.Pytorch 0.4 


## 2. Installation

### [1] Download pretrained weights
    
    The weights of each network are stored in the weights file, where： 
    for CONV1: CNN1.pt  
    for CONV2: CNN2.pt
    for CONV3: CNN3.pt 
    
  All the weights can be downloaded in the following website.
    
  https://pan.baidu.com/s/1M6O54pHen4va2dXEkXwXLA
  
  code: yhz8
  

### [2] Download datasets 
The dataset contains 12000 3D models, including different hemispheres, pyramids and their random compositions, and multi-peak terrains. Each illumination pattern contains 64×64 pixels.
 
  Complete datasets:  <br>
  https://pan.baidu.com/s/1M6O54pHen4va2dXEkXwXLA
  
  code: yhz8
  
  Background image: STL10 datasets  <br>
  https://cs.stanford.edu/~acoates/stl10/ 
  
  
  
## 3. Network structure
An end-to-end convolutional neural network is built to  reconstruct the depth H and reﬂectance O information from the one-dimensional measurement sequence. <br>
    
 <div align=center><img height="190" width="600" src="http://github.com/bianlab/single-pixel-depth-imaging/raw/master/images/network.png"/></div> 
 
 The depth reconstruction subnet consists of two parts, including the selfencoding subnet (CONV1) and the parallel residual subnet (CONV2). The CONV1 contains a fully connected layer and three 3D convolution layers to extract target features. The convolution kernel size is (9×9×1), (1×1×64) and (5×5×32), respectively. The CONV2 consists of two parallel residual subnets, each containing a set of residual blocks and convolution blocks. The structure of the residual block is shown on the bottom. A concatenate layer is ﬁnally employed to connect the two subnets and output the reconstructed depth map. 
 
 The reﬂectance reconstruction subnet (CONV3) contains a fully connect layer, three 3D convolution layers, and a set of residual blocks.
  
## 4. Test 
To understand the complete process of SPDI network, run demo.py. 
    
    $ cd ~ 
    $ python3 demo.py 

This Demo gives the reconstruction of mountain peak, and background intensity is 1. 

     For each detailed information of network, see the corresponding file in network_py_file.
     each named CONV1.py, CONV2.py and CONV3.py.

## 5. Results
Exemplar reconstructed images and corresponding error maps of different targets, with the sampling ratio as follows.
    
 <div align=center><img  height="263" width="600" src="https://github.com/bianlab/single-pixel-depth-imaging/raw/master/images/simulation.png"/></div>

## 6. Train
  We took 80 % of the dataset for training, 10 % for model validation, and the rest 10 % for testing.  <br>
  
  During network training, the learning rate was set 0.0001, and the Batch_size was set 12000. We used the ReLU function for activation, and used the Adam solver for gradient descent optimization. 

    $ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]

## 7. Create datasets
Using the composite modulation, the measurement sequence contains both depth and reﬂectance information of the target. The developed composite modulation contains two components, including the random coding used to encode the 2D spatial information, and the sinusoidal coding applied for depth modulation.
        
    The process of create 1-D measurement of SPI measurement, run create.py. 
