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

### Models, modules and benchmarks for GNN researchers

The field of graph deep learning is still rapidly evolving and many research ideas emerge by standing on the shoulders of giants. To ease the process, DGL collects a rich set of [example implementations](https://github.com/dmlc/dgl/tree/master/examples) of popular GNN models of a wide range of topics. Researchers can [search](https://www.dgl.ai/) for related models to innovate new ideas from or use them as baselines for experiments. Moreover, DGL provides many state-of-the-art [GNN layers and modules](https://docs.dgl.ai/api/python/nn.html) for users to build new model architectures. DGL is one of the preferred platforms for many standard graph deep learning benchmarks including [OGB](https://ogb.stanford.edu/) and [GNNBenchmarks](https://github.com/graphdeeplearning/benchmarking-gnns).

### Easy to learn and use

### Analysis Tools

DGL provides a plenty of learning materials for all kinds of users from ML researcher to domain experts. The [Blitz Introduction to DGL](https://docs.dgl.ai/tutorials/blitz/index.html) is a 120-minute tour of the basics of graph machine learning. The [User Guide](https://docs.dgl.ai/guide/index.html) explains in more details the concepts of graphs as well as the training methodology. All of them include code snippets in DGL that are runnable and ready to be plugged into one’s own pipeline.

## Get Started
Users can install DGL from [pip and conda](https://www.dgl.ai/pages/start.html). Advanced users can follow the [instructions](https://docs.dgl.ai/install/index.html#install-from-source) to install from source.

For absolute beginners, start with [the Blitz Introduction to DGL](https://docs.dgl.ai/tutorials/blitz/index.html). It covers the basic concepts of common graph machine learning tasks and a step-by-step on building Graph Neural Networks (GNNs) to solve them.

For acquainted users who wish to learn more,

* Learn DGL by [example implementations](https://www.dgl.ai/) of popular GNN models.
* Read the [User Guide](https://docs.dgl.ai/guide/index.html) ([中文版链接](https://docs.dgl.ai/guide_cn/index.html)), which explains the concepts and usage of DGL in much more details.
* Go through the tutorials for advanced features like [stochastic training of GNNs](https://docs.dgl.ai/tutorials/large/index.html), training on [multi-GPU](https://docs.dgl.ai/tutorials/multi/index.html) or [multi-machine](https://docs.dgl.ai/tutorials/dist/index.html).
* [Study classical papers](https://docs.dgl.ai/tutorials/models/index.html) on graph machine learning alongside DGL.
* Search for the usage of a specific API in the [API reference manual](https://docs.dgl.ai/api/python/index.html), which organizes all DGL APIs by their namespace.

All the learning materials are available at our [documentation site](https://docs.dgl.ai/). If you are new to deep learning in general,
check out the open source book [Dive into Deep Learning](https://d2l.ai/).

