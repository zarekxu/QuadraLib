## Image Classification

For image classifiation, we define all the configuration parameters in yaml filed ``config.yaml``, please check any files in `./image_classification/experimets` for more details. 

The training step is simple, by runing the cmd for the training:

```bash
## 1 GPU for qvgg7 training from scratch
python train.py --work-path ./experiment/cifar_vgg/
``` 

```bash
## 1 GPU for qresnet14 training from checkpoint
python train.py --resume --work-path ./experiment/cifar_resnet/
``` 
The checkpoint folder and tensorboard cache folder are defined in train.py, users can modify with their own path. 

You can see the training curve via tensorboard, ``tensorboard --logdir path-to-event --port your-port``.  

Two pre-trained QDNN models (QVGG-7 and QResNet14) can be found in [./checkpoint](https://github.com/zarekxu/QuadraLib/tree/main/image_classification/checkpoint)



### Auto-builder

The guideline of using auto-builder to identify a QDNN model structure has following steps:   
1) Replace an existing first-order DNN with target quadratic layers in both config.yaml file and add model name in model definition file. 
2) Traing the model and save .pth file in ./checkpoint folder.
3) Configure config_auto_builder.yaml file. Specifically, layernumber represent the total convolution layer number in the model and layer_index is their layer index, masking_ratio indicate how the masked weights ratio in each layer. 
4) Run auto_builder.py file to measuer the accuracy importance of each layer. 
5) Calcualte the find score according to the equation ![equation](https://latex.codecogs.com/png.image?\dpi{80}&space;RI=\frac{P_{M}&space;P_{W}}{\delta_A}&space;).


### Performance

| architecture          | params | batch size | epoch | CIFAR-10 test acc (%) | CIFAR-100 test acc (%) |
| :-------------------- | :----: | :--------: | :---: | :--------------: | :---------------: |
| VGG-13                |  14.7M |    256     |  200  |      93.61       |       73.88       |
| QVGG-7                |  12.0M |    256     |  200  |      94.13       |       74.35       |
| ResNet32              |  0.48M |    256     |  200  |      92.83       |       69.13       |
| QResNet14             |  0.39M |    256     |  200  |      93.23       |       69.58       |


<!-- Driver Version: 470.103.01   CUDA Version: 11.4

| architecture          | params | batch size | epoch | CIFAR-10 test acc (%) | CIFAR-100 test acc (%) |
| :-------------------- | :----: | :--------: | :---: | :--------------: | :---------------: |
| QVGG-13               |  44.2M |    256     |  200  |      94.04       |       74.35       |
| QResNet-14            |  0.52M |    256     |  200  |      93.53       |       69.48       |
| QMobileNet            |  3.32M |    256     |  200  |      92.61       |       69.71       | -->
