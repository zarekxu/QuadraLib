# QuadraLib

QuadraLib is an easy-to-use, Pytorch library for providing implementations of multiple state-of-the-art Quadratic Deep Neuron Networks (QDNNs). It aims to offer an identical playground for QDNNs so that machine learning researchers can readily compare and analyze a new idea in multiple learning tasks, such as image classification, image generation, and object detection. 


<p align="center">
  <img src="https://github.com/zarekxu/QuadraLib/blob/main/figures/architecture.PNG" alt="library architecture" width="1200">
  <br>
  <b>Figure</b>: QuadraLib Architecture
</p>


## Highlighted Features

### Multiple Types of Quadratic Neuron Supporting

QDNNs is a new but rapidly evovling research topic, which brrow many piror knowledge from the first-order DNNs. The current QDNNs work use different DL framework and distinct benchmarks, which is not easy to comprehensive evaluate. We summarized the state-of-the-art QDNN neuron and structure design, and include them in QuadraLib. 

### Models and benchmarks for QDNN researchers

QuadraLib collects current [QDNN papers](https://github.com/zarekxu/QuadraLib/edit/main/SOTA%20Papers/paper_list.md) and their key designs. It also provides many state-of-the-art [QDNN layers](https://docs.dgl.ai/api/python/nn.html) for users to build new model architectures.

### Analysis Tools

QuadraLib provides several analysis tools to help users to analyze activation, gradients, and weights of their generated models. These tools are written as functions in [Jupyter Notebook files] for easy usage and modification.


## Get Started

