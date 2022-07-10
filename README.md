# QuadraLib

QuadraLib is an easy-to-use, Pytorch-based library for experimenting with multiple state-of-the-art Quadratic Deep Neuron Networks (QDNNs). It aims to offer a consistent programming environment for experimenting various QDNN models so that machine learning researchers can readily explore, compare, and analyze those QDNNs models. Most importantly, QuadraLib offers a convenient library for researchers to try new ideas related to QDNNs to solve problems beyond what have been done in existing literature, such as image classification, image generation, and object detection. 


<p align="center">
  <img src="https://github.com/zarekxu/QuadraLib/blob/main/Figures/architecture.PNG" alt="library architecture" width="1200">
  <br>
  <b>Figure 1</b>: QuadraLib Architecture
</p>


## Highlighted Features

### Multiple Types of Quadratic Neuron Supporting

Quadratic Deep Neural Network is a new but rapidly evovling research topic, which brrow many piror knowledge from the first-order DNNs. The current QDNNs work use different DL framework and distinct benchmarks, which is not easy to comprehensive evaluate. We summarized the state-of-the-art QDNN neuron and structure design, and implement them in QuadraLib as an individual .py [file](https://github.com/zarekxu/QuadraLib/blob/main/image_classification/models/quadratic_layer.py). 

<p align="center">
  <img src="https://github.com/zarekxu/QuadraLib/blob/main/Figures/neuron_type_summary.PNG" alt="neuron type" width="1200">
  <br>
  <b>Figure 2</b>: SOTA Quadratic Neuron Types
</p>

### Models and benchmarks for QDNN researchers


QuadraLib collects the current [QDNN papers](SOTA_Papers.md) and their key designs. It also provides a variety of pre-defined QDNN models in multiple application scenarios (e.g. image classification, object detection, image generation) that achieve the state-of-the-art accuracy performance. Currently, QuadraLib provides pre-trained VGGNet and ResNet models with CIFAR-10/CIFAR-100 in image classification and VGG-based SSD with VOC2007 in object detection.

Users can also easily create their own QDNN models by using quandratic layers provided by QuadraLib, via mannually building or auto-builder. More models and application tasks will be supported in the future.  

### QDNN Model Definition: Manual Design and Auto-builder

First-order DNN already has many popular and high-performance network structures such as ResNet and EfficientNet. Therefore, QDNN model structures can be built based on the existing first order DNNs. 
QudraLib provides two ways to build QDNN model structures: Manual Definition and Auto-builder. 

#### Manual Design
Users can manually configure the model structure file in [./models] folder. Here are some insights for QDNN model design during configuration file definition: 1) since quadratic neuron has higher capability, the depth of QDNN can be reduced, thereby decrease the model computation cost and also avoids the potential gradient vanishing or model degeneration issues discussed before; 2) since second-order term will generate extreme values, batch-normalization layer is significantly important for QDNN to regulate the output activation values; 3) QDNNs with small network structures don’t need activation functions (e.g. ReLU) due to the high capability of quadratic neuron. However, when QDNN depth increases, activation functions are important since they can prevent gradient vanishing. 

#### Auto-Builder:
QuadraLib also provide function to automatically build QDNN model structure based on a baseline first-order DNN. For an existing learning task, manually designing a QDNN model from scratch needs a lot of prior domain experience and can involve significant effort, such as detector backbones. Therefore, besides to manually construct QDNN from scratch, another effective approach for efficient QDNN construction is to leverage the existing first-order DNN model pools, which already include various sophisticated pre-defined first-order DNN structure for different learning tasks. Moreover, in order to identify the suitable QDNN structure based on a targeted DNN model, QuadraLib first repalce all the first-order layers with quadratic layers and train a QDNN model. Then it leverages zero-masking (widely used in filter pruning) to evaluate the importance score (RI) of each quadratic layer according to Equation: 
![equation](https://latex.codecogs.com/png.image?\dpi{80}&space;RI=\frac{P_{M}&space;P_{W}}{\delta_A}&space;)
where ![equation](https://latex.codecogs.com/png.image?\dpi{80}&space;P_{M}) and ![equation](https://latex.codecogs.com/png.image?\dpi{80}&space;P_{W}) represent the ratio of the target layer to the entire model in terms of parameter amount and computation workload. ![equation](https://latex.codecogs.com/png.image?\dpi{80}&space;\delta_{A}) represents the accuracy drop when removing the target layer.

The codes about auto-builder can be find in folder [./image_classification/auto_builder](). 

### Analysis Tools

QuadraLib provides several analysis tools to help users to analyze activation, gradients, and weights of their generated models. These tools are written as functions in [Jupyter Notebook files](https://github.com/zarekxu/QuadraLib/tree/main/Analysis_Tools) for easy usage and modification. 

<p align="center">
  <img src="https://github.com/zarekxu/QuadraLib/blob/main/Figures/activation_visulization.png" alt="visualization" width="600">
  <br>
  <b>Figure 3</b>: Activation Visualization Examples
</p>

### Hybrid Back-propagation for Efficient Training

In the most DNN libraries, gradient calculation during back-propagation is supported by Auto-Differentiation (AD) with reverse-mode, which means the partial derivatives will backwardly go through each layer from output to input. Due to more complicated structure, quadratic layer will introduce more intermediate parameters during back-propagation. Compared to AD, Symbolic Differentiation (SD) can derive the symbolic expression of each parameter's partial gradients before the back-propagation. Therefore, training process doesn't need to save every intermediate parameters on the memory.

QuadraLib proposes a hybrid back-propagation scheme to decrease the memory usage during QDNN training, which is the combination of AD and SD. For each quadratic layer, when obtaining input values and output values after forward process, we use SD to calculate the corresponding gradients of weight parameters and input during back-propagation. Therefore, only necessary intermediate parameters will be stored before the backward. 
On the other hand, since the gradients of other layers such as batch normalization are not easy to formulate in SD, we can still leverage AD to calculate their values. 

Currently, it supports hybrid back-propagation for quadratic fully connected layers (check [here](https://github.com/zarekxu/QuadraLib/blob/main/image_classification/models/quadratic_layer.py)), the quadratic convolutional layer version will be released soon. 




## Requirements

- Python (**>=3.5**)
- PyTorch (**>=1.4.0**)
- Other dependencies (pyyaml, easydict, tensorboard)

## Get Started 

Users can find the two application tasks in [./image_classification](https://github.com/zarekxu/QuadraLib/tree/main/image_classification) and [./object_detection](https://github.com/zarekxu/QuadraLib). The step-by-step training and evaluation instructions are included in README file in the two folders, separately.  More recognition tasks will be added in the future. 

## License

QuadraLib is an open-source library under the MIT license (MIT).

## Cite

If you use QuadraLib in a scientific publication, we would appreciate citations to the following paper:
```
@article{xu2022quadrali,
    title={QuadraLib: A Performant Quadratic Neural Network Library for Architecture Optimization and Design Exploration},
    author={Xu, Zirui and Xiong, Jinjun and Yu, Fuxun and Chen, Xiang},
    year={2022},
    journal={arXiv preprint arXiv:2204.01701}
}
```
