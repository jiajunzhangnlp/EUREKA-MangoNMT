# EUREKA-MangoNMT
A C++ toolkit for neural machine translation for CPU

This toolkit extends the simple LSTM encoder-decoder (only unidirectional, no attention, no feed-input) (Stutskever et al., 2014) developed by Ashish Vaswani, in which the models can be trained with Maximum Likelihood Loss and Noise Contrastive Estimation (Gutmann and Hyvärinen, 2010). EUREKA-MangoNMT has several key new features: 1) it adds rich code comments for important classes and functions; 2) it implements the bidirectional LSTMs to encode the input sequence; 3) it implements the global attention mechanism which attends different source contexts when generating outputs in different positions; 4) it implements the feed input mechanism which links the attention output to the next decoder LSTM node; 5) it supports training data shuffle in different training epoch; 6) it implements a novel NMT framework: one sentence one model for neural machine translation. All the new features are implemented by Jiajun Zhang.

About the translation performance:
EUREKA-MangoNMT can obtain the similar translation performance to the Theano-based DL4MT.

About the training time:
It highly depends on the power of the CPU you use. In our experiments using 24 threads on Intel(R) Xeon(R) CPU E5-2690 @ 2.90GHz, it takes about one week to training the NMT model on about 700k LDC parallel sentence pairs. In this experiments, the vocabulary size is set 30,000 on both source and target sides, the input embedding and encoding hidden dimension are set 256, while the output embedding and decoding hidden dimension are set 512. It is slightly slower than Theano-based DL4MT using GPUs.


Tips for compile:

Prerequisites

Before compiling, you must have the following:

A C++ compiler and GNU make

Boost 1.47.0 or later http://www.boost.org

Eigen 3.1.x http://eigen.tuxfamily.org (you can use the Eigen we provide in 3rdparty, different Eigen version may cause runtime error. If you find runtime error, you should check whether Eigen works well in your system.)

Optional (best to have):

Intel MKL 11.x http://software.intel.com/en-us/intel-mkl Recommended for better performance.

Building

To compile, edit the Makefile to reflect the locations of the Boost and Eigen include directories.

If you want to use the Intel MKL library (recommended if you have it), uncomment the line MKL=/path/to/mkl editing it to point to the MKL root directory.

By default, multithreading using OpenMP is enabled. To turn it off, comment out the line OMP=1

To compile:

cd src

make

Notes:
It is tested on Ubuntu server. More threads lead to more efficient training.

More details about the attention-based NMT (with bidirectional LSTM encoding and feed-input), the relationship between codes and NMT model, see EUREKA-MangoNMT.pdf.

For examples on how to use code, please look at the tutorial/README file. 

If you have any question about attention model, feed input and other new features, please contact jiajunzhangwing@gmail.com.

Reference:

One Sentence One Model for Neural Machine Translation. Xiaoqing Li, Jiajun Zhang and Chengqing Zong. 2016. (Initial version is available here and Full version will be availabel in arxiv).

EUREKA: A toolkit for training LSTM Languge Models and LSTM Encoder Decoder Models with MLE and Noise Contrastive Estimation. Ashish Vaswani. 2016.
