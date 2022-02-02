# Installing CUDA computation toolkit for Tensorflow or PyTorch

## Requirements:
- CUDA compatible GPU (NVIDIA GPU's are compatible)
- Installation of tensorflow or pytorch with GPU support (in tensorflow this is standard from version 2.0 onwards)
- Admin rights to the target PC

## Installation

####1. Download CUDA computation toolkit:
   
download the files from https://developer.nvidia.com/cuda-downloads for the relevant system.
    
#### 2. Install the software:
    
This step might require disabling the virus scanner. 
- Run the cuda_<>.exe installer
- Extract the files to the preferred installation location
- The "Express Installetion" suffices for running Tensorflow with CUDA
- Install

#### 3. Download CUDNN toolbox:

Downloading the latest version requires creating a NVIDIA Developer Program account, 
a version of the dll's is included in this repository. 
The extracted cudnn download contains a directory called \cuda, similar to the one in this repository

#### 4. Copy dll files for CUDNN to Cuda installation:

Default installation location for CUDA is:

    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\<version>

Copy all the dll files from the \bin, \include, and \lib directories of 
the extracted \cuda directory to their respective counterparts in the CUDA installation

#### 5. Testing Tensorflow with cuda support:

To ensure cuda was installed correctly, run the following lines of code in a python console of choice:
- For Tensorflow the following lines should print "True":

        import tensorflow as tf
  
        physical_devices = tf.config.list_physical_devices('GPU')
        print(len(physical_devices) == 1)

- For Pytorch the following lines should print "True":

        import torch

        torch.cuda.is_available()

