# QuadraLib

QuadraLib is an easy-to-use, Pytorch library for providing implementations of multiple state-of-the-art Quadratic Deep Neuron Networks (QDNNs). It aims to offer an identical playground for QDNNs so that machine learning researchers can readily compare and analyze a new idea in multiple learning tasks, such as image classification, image generation, and object detection. 


<p align="center">
  <img src="https://github.com/zarekxu/QuadraLib/blob/main/Figures/architecture.PNG" alt="library architecture" width="1200">
  <br>
  <b>Figure</b>: QuadraLib Architecture
</p>


## Highlighted Features

### Multiple Types of Quadratic Neuron Supporting

QDNNs is a new but rapidly evovling research topic, which brrow many piror knowledge from the first-order DNNs. The current QDNNs work use different DL framework and distinct benchmarks, which is not easy to comprehensive evaluate. We summarized the state-of-the-art QDNN neuron and structure design, and include them in QuadraLib. 

### Models and benchmarks for QDNN researchers

QuadraLib collects current [QDNN papers](https://github.com/zarekxu/QuadraLib/blob/main/SOTA%20Papers/paper_list.md) and their key designs. It also provides many state-of-the-art [QDNN layers](https://docs.dgl.ai/api/python/nn.html) for users to build new model architectures.

### Analysis Tools

QuadraLib provides several analysis tools to help users to analyze activation, gradients, and weights of their generated models. These tools are written as functions in [Jupyter Notebook files] for easy usage and modification.


## Requirements

- Python (**>=3.6**)
- PyTorch (**>=1.4.0**)
- Tensorboard(**>=1.4.0**) (for ***visualization***)
- Other dependencies (pyyaml, easydict)

## Get Started 

Users can find the two application tasks in [./image_classification]() and [./object_detection](). More recognition tasks will be added in the future. 
All the configurations are defined in yaml filed ``config.yaml``, please check any files in `./image_classification/experimets` for more details. 

simply run the cmd for the training:

```bash
## 1 GPU for vgg11 training from scratch
python train.py --work-path ./experiments/cifar10/vgg19
``` 
```bash
## 1 GPU for vgg11 training from checkpoint
python train.py --work-path ./experiments/cifar10/vgg19
``` 


We use yaml file ``config.yaml`` to save the parameters, check any files in `./experimets` for more details.  
You can see the training curve via tensorboard, ``tensorboard --logdir path-to-event --port your-port``.  
The training log will be dumped via logging, check ``log.txt`` in your work path.


## Get Started






## Cite

If you use QuadraLib in a scientific publication, we would appreciate citations to the following paper:
```
@article{wang2019dgl,
    title={Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks},
    author={Minjie Wang and Da Zheng and Zihao Ye and Quan Gan and Mufei Li and Xiang Song and Jinjing Zhou and Chao Ma and Lingfan Yu and Yu Gai and Tianjun Xiao and Tong He and George Karypis and Jinyang Li and Zheng Zhang},
    year={2022},
    journal={arXiv preprint arXiv:1909.01315}
}
```
