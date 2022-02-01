## Image Classification

For image classifiation, we define all the configuration parameters in yaml filed ``config.yaml``, please check any files in `./image_classification/experimets` for more details. 

The training step is simple, by runing the cmd for the training:

```bash
## 1 GPU for qvgg7 training from scratch
python train.py --work-path ./experiments/cifar_vgg/
``` 

```bash
## 1 GPU for qresnet14 training from checkpoint
python train.py --resume --work-path ./experiments/cifar_resnet/
``` 
The checkpoint folder and tensorboard cache folder are defined in train.py, users can modify with their own path. 

You can see the training curve via tensorboard, ``tensorboard --logdir path-to-event --port your-port``.  

Two pre-trained QDNN models (QVGG-7 and QResNet14) can be found in [./checkpoint](https://github.com/zarekxu/QuadraLib/tree/main/image_classification/checkpoint)



### Auto-builder
For auto-builder, QuadraLib provides the auto_builder.py to calculate the accuracy importance of each layer. Users need to first 


run the cmd to get the important score of each layer. 


### Performance

| architecture          | params | batch size | epoch | CIFAR-10 test acc (%) | CIFAR-100 test acc (%) |
| :-------------------- | :----: | :--------: | :---: | :--------------: | :---------------: |
| VGG-13                |  14.7M |    256     |  200  |      93.61       |       xx.xx       |
| QVGG-7                |  12.0M |    256     |  200  |      94.13       |       xx.xx       |
| ResNet32              |  0.48M |    256     |  200  |      92.83       |       xx.xx       |
| QResNet14             |  0.39M |    256     |  200  |      93.23       |       xx.xx       |
