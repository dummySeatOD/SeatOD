# SeatOD
This repository contains code for the paper SEAT-OD: Semantic-aware Testing for Object Detection Systems.

![system-overview](https://github.com/dummySeatOD/SeatOD/blob/main/picture/system.png)

Document Organization：
```
SEAT-OD/
        README.md
        TestCaseGeneration/
        Metrics/
        utils/
        SEA/
```
where TestCaseGeneration/ is the test sample generation module, Model/ is the model for testing and utils/ is the toolbox with normalized functions, the SEAT framework is in SEA/.

## Test Case Generation
We use styleGAN2 as the basis for test case generation. That is, we first train a GAN with the Kitti dataset for feature visualization. 

![generation](https://github.com/dummySeatOD/SeatOD/blob/main/picture/generation.png)

### 1.0  Install
This implementation is tested with Anaconda Python  3.8(ubuntu18.04), PyTorch  1.8.1 and Cuda  11.1
Need xx.pkl for pre-trained GAN.

### 2.0  Generation
Test case generation using styleGAN2, by run
```
    python case.py
```

The results are saved in file test-case/


## SEA Metrics and Selection
We use Stereo R-CNN as the basis for SEA testing. stereo R-CNN is a typical mainstream vision-based multi-sensor fusion target detection system. Stereo R-CNN focuses on accurate 3D object detection and estimation using image-only data in autonomous driving scenarios. It features simultaneous object detection and association for stereo images, 3D box estimation using 2D information, accurate dense alignment for 3D box refinement. 

### 1.0 Install

This implementation is tested with Anaconda Python 3.6 and Pytorch 1.7.0

**1.1 Create Conda Environment:**
```
    conda create -n env_stereo python=3.6
    conda activate env_stereo
```
**1.2 Install PyTorch 1.7.0**

Go to https://pytorch.org/ and choose your cuda version and run the command generated from the website

**1.3 Install Requirements, and Build:**
```
    pip install -r requirements.txt
    ./build.sh
```

### 2.0 Dataset Preparation
Download the left image, right image, calibration, labels and point clouds (optional for visualization) from [KITTI Object Benchmark](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d). Make sure the structure looks like:
```
data/kitti/
    object/training/
        image_2/
        image_3/
        label_2/
        calib/
        velodyne/
    splits/
        val.txt
        train.txt
        trainval.txt
```


### 3.0 Get Deepfeatures and Labels
We obtained deepfeature for SEA testing, by run:
```
    cd Model
    ./test.sh
```
Then, the deepfeatures we need are saved in df/ and the corresponding labels are saved in sc/. 

### 4.0 SEA Metrics Testing

SEA testing using the obtained deepfeature, by run:
```
    cd SEA
    python SEA.py
```
Then, the SEA_D, SEA_B, SEA_S are shown in terminal.
```
Terminal:
    SEA_D: xx
    SEA_B: xx
    SEA_S: xx
```
### 5.0 SEA Selection and Retrain
![selection-retrain](https://github.com/dummySeatOD/SeatOD/blob/main/picture/selection-result-pic.pngs)
Use SEA system for test sample selection：
```
    cd SEA
    python select.py
```
Then we get the case selection list in select.txt, then, we use the cases selection to retrain model.
