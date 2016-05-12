<div align="center">
  <img src="https://www.tensorflow.org/images/tf_logo_transp.png"><br><br>
</div>
-----------------

**TensorFlow** is an open source software library for numerical computation using
data flow graphs.  Nodes in the graph represent mathematical operations, while
the graph edges represent the multidimensional data arrays (tensors) that flow
between them.  This flexible architecture lets you deploy computation to one
or more CPUs or GPUs in a desktop, server, or mobile device without rewriting
code.  TensorFlow was originally developed by researchers and engineers
working on the Google Brain team within Google's Machine Intelligence research
organization for the purposes of conducting machine learning and deep neural
networks research.  The system is general enough to be applicable in a wide
variety of other domains, as well.

## Building TensorFlow in Ubuntu 16.04 LTS

Install NVIDIA CUDA packages from Universe repository:

    $ sudo apt install nvidia-cuda-toolkit

Install NVIDIA cuDNN 4.0:

    $ sudo tar -C /usr/local -xzf /mnt/dl/mirror/cudnn-7.0-linux-x64-v4.0-prod.tgz

Check out the TensorFlow 0.8.0, modified to work on Ubuntu 16.04 LTS
and with older NVIDIA GPUs with compute capability 3.0:

    $ git checkout git@git.frostbite.com/hholst/ea-tensorflow.git

Create an Anaconda environment containing the build tools needed:
```
$ conda create -n ea-tensorflow python=3.5 swig numpy
Using Anaconda Cloud api site https://api.anaconda.org
Fetching package metadata: ....
Solving package specifications: .........

Package plan for installation in environment /home/hholst/anaconda3/envs/ea-tensorflow:

The following NEW packages will be INSTALLED:

    mkl:        11.3.1-0     
    numpy:      1.11.0-py35_0
    openssl:    1.0.2h-0     
    pcre:       8.31-0       
    pip:        8.1.1-py35_1 
    python:     3.5.1-0      
    readline:   6.2-2        
    setuptools: 20.7.0-py35_0
    sqlite:     3.9.2-0      
    swig:       3.0.8-1      
    tk:         8.5.18-0     
    wheel:      0.29.0-py35_0
    xz:         5.0.5-1      
    zlib:       1.2.8-0      

Proceed ([y]/n)? 

Linking packages ...
[      COMPLETE      ]|############################################################################| 100%
#
# To activate this environment, use:
# $ source activate ea-tensorflow
#
# To deactivate this environment, use:
# $ source deactivate
#
```

You also need to activate the new Anaconda environment:

    $ source activate ea-tensorflow

*Optional:* You can re-run the `./configure` script.
Observe that we override the compute capability
to include support for CUDA compute capability 3.0!

```
$ ./configure
Please specify the location of python. [Default is /home/hholst/anaconda3/envs/ea-tensorflow/bin/python]: 
Do you wish to build TensorFlow with GPU support? [Y/n] 
GPU support will be enabled for TensorFlow
Please specify which gcc nvcc should use as the host compiler. [Default is /usr/bin/gcc]: 
Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave empty to use system default]: 
Please specify the location where CUDA  toolkit is installed. Refer to README.md for more details. [Default is /usr]: 
Please specify the Cudnn version you want to use. [Leave empty to use system default]: 
Please specify the location where cuDNN  library is installed. Refer to README.md for more details. [Default is /usr/local/cuda]: 
Please specify a list of comma-separated Cuda compute capabilities you want to build with.
You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your build time and binary size.
[Default is: "3.5,5.2"]: 3.0,3.5,5.2
Setting up Cuda include
Setting up Cuda lib
Setting up Cuda bin
find: File system loop detected; ‘/usr/bin/X11’ is part of the same file system loop as ‘/usr/bin’.
find: File system loop detected; ‘/usr/bin/X11’ is part of the same file system loop as ‘/usr/bin’.
Configuration finished
```

### Compiling

Build TensorFlow

    $ bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer

Run test

    $ bazel-bin/tensorflow/cc/tutorials_example_trainer --use_gpu

Build pip package

    $ bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

Install pip package:

    $ pip install /tmp/tensorflow_pkg/tensorflow-0.8.0-py3-none-any.whl

### *Try your first TensorFlow program*

NOTE: Make sure you're not standing inside the `ea-tensorflow` git repository 
when you are starting python. Doing so might cause problems with CUDA runtime library.

```python
$ python
Python 3.5.1 |Continuum Analytics, Inc.| (default, Dec  7 2015, 11:16:01) 
[GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcublas.so locally
I tensorflow/stream_executor/dso_loader.cc:99] Couldn't open CUDA library libcudnn.so. LD_LIBRARY_PATH: 
I tensorflow/stream_executor/cuda/cuda_dnn.cc:1562] Unable to load cuDNN DSO
I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcufft.so locally
I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:105] successfully opened CUDA library libcurand.so locally
>>> sess = tf.InteractiveSession()
I tensorflow/core/common_runtime/gpu/gpu_init.cc:102] Found device 0 with properties: 
name: GeForce GTX 980
major: 5 minor: 2 memoryClockRate (GHz) 1.2155
pciBusID 0000:03:00.0
Total memory: 4.00GiB
Free memory: 3.24GiB
I tensorflow/core/common_runtime/gpu/gpu_init.cc:126] DMA: 0 
I tensorflow/core/common_runtime/gpu/gpu_init.cc:136] 0:   Y 
I tensorflow/core/common_runtime/gpu/gpu_device.cc:755] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 980, pci bus id: 0000:03:00.0)
>>> a = tf.constant(10)
>>> b = tf.constant(32)
>>> sess.run(a+b)
42
>>> 
```

##For more information

* [TensorFlow website](http://tensorflow.org)
* [TensorFlow whitepaper](http://download.tensorflow.org/paper/whitepaper2015.pdf)
* [TensorFlow MOOC on Udacity] (https://www.udacity.com/course/deep-learning--ud730)

The TensorFlow community has created amazing things with TensorFlow, please see the [resources section of tensorflow.org](https://www.tensorflow.org/versions/master/resources#community) for an incomplete list.
