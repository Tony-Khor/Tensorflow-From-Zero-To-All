# Tensorflow-From-Zero-To-All

This tutorial was designed for easily diving into TensorFlow, through examples. For readability, it includes both notebooks and source codes with explanation, for both TF v1 & v2.

It is suitable for beginners who want to find clear and concise examples about TensorFlow. Besides the traditional 'raw' TensorFlow implementations, you can also find the latest TensorFlow API practices (such as `layers`, `estimator`, `dataset`, ...).

## Tutorial index

#### 0 - Prerequisite
- [Introduction to Machine Learning]
- [Introduction to MNIST Dataset]

#### 1 - Introduction
- **Hello World** Very simple example to learn how to print "hello world" using TensorFlow 2.0+.
- **Basic Operations** A simple example that cover TensorFlow 2.0+ basic operations.

#### 2 - Basic Models
- **Linear Regression**  Implement a Linear Regression with TensorFlow 2.0+.
- **Logistic Regression** Implement a Logistic Regression with TensorFlow 2.0+.
- **Word2Vec (Word Embedding)** Build a Word Embedding Model (Word2Vec) from Wikipedia data, with TensorFlow 2.0+.
- **GBDT (Gradient Boosted Decision Trees)** Implement a Gradient Boosted Decision Trees with TensorFlow 2.0+ to predict house value using Boston Housing dataset.

#### 3 - Neural Networks
##### Supervised

- **Simple Neural Network**  Use TensorFlow 2.0 'layers' and 'model' API to build a simple neural network to classify MNIST digits dataset.
- **Simple Neural Network (low-level)** Raw implementation of a simple neural network to classify MNIST digits dataset.
- **Convolutional Neural Network**  Use TensorFlow 2.0+ 'layers' and 'model' API to build a convolutional neural network to classify MNIST digits dataset.
- **Convolutional Neural Network (low-level)** Raw implementation of a convolutional neural network to classify MNIST digits dataset.
- **Recurrent Neural Network (LSTM)** Build a recurrent neural network (LSTM) to classify MNIST digits dataset, using TensorFlow 2.0 'layers' and 'model' API.
- **Bi-directional Recurrent Neural Network (LSTM)** Build a bi-directional recurrent neural network (LSTM) to classify MNIST digits dataset, using TensorFlow 2.0+ 'layers' and 'model' API.
- **Dynamic Recurrent Neural Network (LSTM)** Build a recurrent neural network (LSTM) that performs dynamic calculation to classify sequences of variable length, using TensorFlow 2.0+ 'layers' and 'model' API.

##### Unsupervised
- **Auto-Encoder**  Build an auto-encoder to encode an image to a lower dimension and re-construct it.
- **DCGAN (Deep Convolutional Generative Adversarial Networks)** Build a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images from noise.

#### 4 - Utilities
- **Save and Restore a model** Save and Restore a model with TensorFlow 2.0+.
- **Build Custom Layers & Modules**  Learn how to build your own layers / modules and integrate them into TensorFlow 2.0+ Models.
- **Tensorboard** Track and visualize neural network computation graph, metrics, weights and more using TensorFlow 2.0+ tensorboard.

#### 5 - Data Management
- **Load and Parse data** Build efficient data pipeline with TensorFlow 2.0 (Numpy arrays, Images, CSV files, custom data, ...).
- **Build and Load TFRecords**  Convert data into TFRecords format, and load them with TensorFlow 2.0+.
- **Image Transformation (i.e. Image Augmentation)**  Apply various image augmentation techniques with TensorFlow 2.0+, to generate distorted images for training.

#### 6 - Hardware
- **Multi-GPU Training** Train a convolutional neural network with multiple GPUs on CIFAR-10 dataset.

## TensorFlow v1

The tutorial index for TF v1 is available here: [TensorFlow v1.15 Examples](tensorflow_v1). Or see below for a list of the examples.

## Dataset
Some examples require MNIST dataset for training and testing. Don't worry, this dataset will automatically be downloaded when running examples.
MNIST is a database of handwritten digits, for a quick description of that dataset, you can check [official website](http://yann.lecun.com/exdb/mnist/).

## Installation

To download all the examples, simply clone this repository:
```
git clone https://github.com/aymericdamien/TensorFlow-Examples
```

To run them, you also need the latest version of TensorFlow. To install it:
```
pip install tensorflow
```

or (with GPU support):
```
pip install tensorflow_gpu
```

For more details about TensorFlow installation, you can check [TensorFlow Installation Guide](https://www.tensorflow.org/install/)


## TensorFlow v1 Examples - Index

The tutorial index for TF v1 is available here: [TensorFlow v1.15 Examples](tensorflow_v1).

#### 0 - Prerequisite
- [Introduction to Machine Learning]
- [Introduction to MNIST Dataset]

#### 1 - Introduction
- **Hello World**  Very simple example to learn how to print "hello world" using TensorFlow.
- **Basic Operations** A simple example that cover TensorFlow basic operations.
- **TensorFlow Eager API basics**  Get started with TensorFlow's Eager API.

#### 2 - Basic Models
- **Linear Regression**  Implement a Linear Regression with TensorFlow.
- **Linear Regression (eager api)** Implement a Linear Regression using TensorFlow's Eager API.
- **Logistic Regression** Implement a Logistic Regression with TensorFlow.
- **Logistic Regression (eager api)** Implement a Logistic Regression using TensorFlow's Eager API.
- **Nearest Neighbor**  Implement Nearest Neighbor algorithm with TensorFlow.
- **K-Means** Build a K-Means classifier with TensorFlow.
- **Random Forest** Build a Random Forest classifier with TensorFlow.
- **Gradient Boosted Decision Tree (GBDT)** Build a Gradient Boosted Decision Tree (GBDT) with TensorFlow.
- **Word2Vec (Word Embedding)**  Build a Word Embedding Model (Word2Vec) from Wikipedia data, with TensorFlow.

#### 3 - Neural Networks
##### Supervised

- **Simple Neural Network** Build a simple neural network (a.k.a Multi-layer Perceptron) to classify MNIST digits dataset. Raw TensorFlow implementation.
- **Simple Neural Network (tf.layers/estimator api)**  Use TensorFlow 'layers' and 'estimator' API to build a simple neural network (a.k.a Multi-layer Perceptron) to classify MNIST digits dataset.
- **Simple Neural Network (eager api)**  Use TensorFlow Eager API to build a simple neural network (a.k.a Multi-layer Perceptron) to classify MNIST digits dataset.
- **Convolutional Neural Network**  Build a convolutional neural network to classify MNIST digits dataset. Raw TensorFlow implementation.
- **Convolutional Neural Network (tf.layers/estimator api)**  Use TensorFlow 'layers' and 'estimator' API to build a convolutional neural network to classify MNIST digits dataset.
- **Recurrent Neural Network (LSTM)** Build a recurrent neural network (LSTM) to classify MNIST digits dataset.
- **Bi-directional Recurrent Neural Network (LSTM)** Build a bi-directional recurrent neural network (LSTM) to classify MNIST digits dataset.
- **Dynamic Recurrent Neural Network (LSTM)** Build a recurrent neural network (LSTM) that performs dynamic calculation to classify sequences of different length.

##### Unsupervised
- **Auto-Encoder**  Build an auto-encoder to encode an image to a lower dimension and re-construct it.
- **Variational Auto-Encoder** Build a variational auto-encoder (VAE), to encode and generate images from noise.
- **GAN (Generative Adversarial Networks)**  Build a Generative Adversarial Network (GAN) to generate images from noise.
- **DCGAN (Deep Convolutional Generative Adversarial Networks)** Build a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images from noise.

#### 4 - Utilities
- **Save and Restore a model**  Save and Restore a model with TensorFlow.
- **Tensorboard - Graph and loss visualization**  Use Tensorboard to visualize the computation Graph and plot the loss.
- **Tensorboard - Advanced visualization**  Going deeper into Tensorboard; visualize the variables, gradients, and more...

#### 5 - Data Management
- **Build an image dataset**  Build your own images dataset with TensorFlow data queues, from image folders or a dataset file.
- **TensorFlow Dataset API**  Introducing TensorFlow Dataset API for optimizing the input data pipeline.
- **Load and Parse data**  Build efficient data pipeline (Numpy arrays, Images, CSV files, custom data, ...).
- **Build and Load TFRecords** Convert data into TFRecords format, and load them.
- **Image Transformation (i.e. Image Augmentation)** Apply various image augmentation techniques, to generate distorted images for training.

#### 6 - Multi GPU
- **Basic Operations on multi-GPU** A simple example to introduce multi-GPU in TensorFlow.
- **Train a Neural Network on multi-GPU** A clear and simple TensorFlow implementation to train a convolutional neural network on multiple GPUs.

## More Examples
The following examples are coming from [TFLearn](https://github.com/tflearn/tflearn), a library that provides a simplified interface for TensorFlow. You can have a look, there are many [examples](https://github.com/tflearn/tflearn/tree/master/examples) and [pre-built operations and layers](http://tflearn.org/doc_index/#api).

### Tutorials
- [TFLearn Quickstart](https://github.com/tflearn/tflearn/blob/master/tutorials/intro/quickstart.md). Learn the basics of TFLearn through a concrete machine learning task. Build and train a deep neural network classifier.

### Examples
- [TFLearn Examples](https://github.com/tflearn/tflearn/blob/master/examples). A large collection of examples using TFLearn.

