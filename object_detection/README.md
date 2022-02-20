### Object Detection

Currently, QuadraLib includes Single Shot MultiBox Detector (SSD) structure and the backbone model is VGG-16. More detction design (e.g. Mask RCNN and YOLO) will be supported in the future. The SSD repo is highly based on [a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) on Github. 

The pre-processed dataset can be downloaded via [link](https://drive.google.com/file/d/1_RxZuPjWJ0IDVCt2DJ15dsAmRtF6SwWy/view?usp=sharing). Once downloaded, overlay ```./data/pascalVOC```

Simply run the cmd for the SSD training:

```bash
# If this is your time training SSD on pascal VOC, this scipt generates indices for the dataset
python create_data_lists.py

## Training
python train.py --work-path ./experiment/pascal-voc_ssd
```

To use other networks and other dataset, you have to write your own code in ```./dataloader.py``` for a Dataloader, ```./models/__init__.py``` for behaviour of the optimizer, and add the network's implementation in ```./models/```.

By default, the checkpoints of trained models are saved in ```./checkpoint/```, datasets are saved in ```./data/```, and configurations for different training settings are stored in ```./experiement/```. You have to specify the path of the training settings while running the ```./train.py```.