# Gaussian_yolo_HSV_traffic_classification
Using Gaussian Yolo in object detection and classifying Traffic_lights using HSV

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 


## Prerequisites
### Needed libraries
> torch==1.0.0

> numpy==1.15.2

> matplotlib==3.0.2

> opencv_python==3.4.4.19

> tensorboardX==1.4

> PyYAML>=4.2b1

> pycocotools==2.0.0

> seaborn==0.9.0

> scikit-image

```
pip install -r requirements/requirements.txt
```

### Download Gaussian Yolov3 Weights for ms-coco Dataset
```
FILE_ID="1zAFDSga9XLrsUBNHV3S2SvL1YWEsDB_p"
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -sLb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o gaussian_yolov3_coco.pth 

```
### Download MS-COCO Dataset 
```bash requirements/getcoco.sh
```


## Running the scripts 
### Evaluation on COCO 
```
python train.py --cfg config/gaussian_yolov3_eval.cfg --eval_interval 1  --checkpoint gaussian_yolov3_coco.pth
```
### 2D detection 
> Images 

```
python run.py
```
> Videos

```
python video.py
```

### Stereo Vision 
> Images
```
python run_stereo.py
```
>Video
```
python video_stereo.py
```


## Generating Plots 
```
python graphs.py
```
