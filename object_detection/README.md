### Object Detection

Currently, QuadraLib includes Single Shot MultiBox Detector (SSD) structure and the backbone model is VGG-16. More detction design (e.g. Mask RCNN and YOLO) will be supported in the future. The SSD repo is highly based on [a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) on Github. 

The pre-processed dataset can be downloaded via [link](https://drive.google.com/file/d/1_RxZuPjWJ0IDVCt2DJ15dsAmRtF6SwWy/view?usp=sharing). 

Simply run the cmd for the SSD training:

```bash
## 1 GPU for training from scratch
python train.py
``` 
