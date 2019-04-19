# CornerNet: Training and Evaluation Code
Update (4/18/2019): please check out [CornerNet-Lite](https://github.com/princeton-vl/CornerNet-Lite), more efficient variants of CornerNet

Code for reproducing the results in the following paper:

[**CornerNet: Detecting Objects as Paired Keypoints**](https://arxiv.org/abs/1808.01244)  
Hei Law, Jia Deng  
*European Conference on Computer Vision (ECCV), 2018*

## Getting Started
Please first install [Anaconda](https://anaconda.org) and create an Anaconda environment using the provided package list.
```
conda create --name CornerNet --file conda_packagelist.txt
```

After you create the environment, activate it.
```
source activate CornerNet
```

Our current implementation only supports GPU so you need a GPU and need to have CUDA installed on your machine.

### Compiling Corner Pooling Layers
You need to compile the C++ implementation of corner pooling layers. 
```
cd <CornerNet dir>/models/py_utils/_cpools/
python setup.py install --user
```

### Compiling NMS
You also need to compile the NMS code (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/cpu_nms.pyx) and [Soft-NMS](https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx)).
```
cd <CornerNet dir>/external
make
```

### Installing MS COCO APIs
You also need to install the MS COCO APIs.
```
cd <CornerNet dir>/data
git clone git@github.com:cocodataset/cocoapi.git coco
cd <CornerNet dir>/data/coco/PythonAPI
make
```

### Downloading MS COCO Data
- Download the training/validation split we use in our paper from [here](https://drive.google.com/file/d/1dop4188xo5lXDkGtOZUzy2SHOD_COXz4/view?usp=sharing) (originally from [Faster R-CNN](https://github.com/rbgirshick/py-faster-rcnn/tree/master/data))
- Unzip the file and place `annotations` under `<CornerNet dir>/data/coco`
- Download the images (2014 Train, 2014 Val, 2017 Test) from [here](http://cocodataset.org/#download)
- Create 3 directories, `trainval2014`, `minival2014` and `testdev2017`, under `<CornerNet dir>/data/coco/images/`
- Copy the training/validation/testing images to the corresponding directories according to the annotation files

## Training and Evaluation
To train and evaluate a network, you will need to create a configuration file, which defines the hyperparameters, and a model file, which defines the network architecture. The configuration file should be in JSON format and placed in `config/`. Each configuration file should have a corresponding model file in `models/`. i.e. If there is a `<model>.json` in `config/`, there should be a `<model>.py` in `models/`. There is only one exception which we will mention later.

To train a model:
```
python train.py <model>
```

We provide the configuration file (`CornerNet.json`) and the model file (`CornerNet.py`) for CornerNet in this repo. 

To train CornerNet:
```
python train.py CornerNet
```
We also provide a trained model for `CornerNet`, which is trained for 500k iterations using 10 Titan X (PASCAL) GPUs. You can download it from [here](https://drive.google.com/open?id=16bbMAyykdZr2_7afiMZrvvn4xkYa-LYk) and put it under `<CornerNet dir>/cache/nnet/CornerNet` (You may need to create this directory by yourself if it does not exist). If you want to train you own CornerNet, please adjust the batch size in `CornerNet.json` to accommodate the number of GPUs that are available to you.

To use the trained model:
```
python test.py CornerNet --testiter 500000 --split <split>
```

If you want to test different hyperparameters in testing and do not want to overwrite the original configuration file, you can do so by creating a configuration file with a suffix (`<model>-<suffix>.json`). You **DO NOT** need to create `<model>-<suffix>.py` in `models/`.

To use the new configuration file:
```
python test.py <model> --testiter <iter> --split <split> --suffix <suffix>
```

We also include a configuration file for multi-scale evaluation, which is `CornerNet-multi_scale.json`, in this repo. 

To use the multi-scale configuration file:
```
python test.py CornerNet --testiter <iter> --split <split> --suffix multi_scale
```
