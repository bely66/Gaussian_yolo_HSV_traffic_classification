

import sys
import argparse
import yaml
from Traffic_Classifier_HSV import classify_color
import cv2
import torch
from torch.autograd import Variable

from models.yolov3 import *
from utils.utils import *
from utils.parse_yolo_weights import parse_yolo_weights
from utils.vis_bbox import vis_bbox

import matplotlib.pyplot as plt

class Infer():
    def __init__(self,detect_thresh = 0.5,gpu=-1):
        # Choose config
        cfg_path = './config/gaussian_yolov3_eval.cfg'
        self.vid = 0
        # Specify checkpoint file which contains the weight of the model you want to use
        ckpt_path = 'gaussian_yolov3_coco.pth'


        self.coco_class_names, self.coco_class_ids, self.coco_class_colors = get_coco_label_names()
        self.names = []
        for i in self.coco_class_names :
          self.names.append(i)
        for i in ["red traffic light","yellow traffic light","green traffic light"] :
          self.names.append(i)

        self.detected = ["red traffic light","yellow traffic light","green traffic light","bus","car","truck","motorcycle","bicycle","traffic light"]
        # Detection threshold
        self.detect_thresh = detect_thresh

        # Use CPU if gpu < 0 else use GPU
        self.gpu = gpu

        with open(cfg_path, 'r') as f:
            cfg = yaml.load(f)

        self.model_config = cfg['MODEL']
        self.imgsize = cfg['TEST']['IMGSIZE']
        self.confthre = cfg['TEST']['CONFTHRE']
        self.nmsthre = cfg['TEST']['NMSTHRE']
        self.gaussian = cfg['MODEL']['GAUSSIAN']

        # if detect_thresh is not specified, the parameter defined in config file is used
        if self.detect_thresh:
            self.confthre = self.detect_thresh

        self.model = YOLOv3(self.model_config)

        # Load weight from the checkpoint
        print("loading checkpoint %s" % (ckpt_path))
        state = torch.load(ckpt_path)

        if 'model_state_dict' in state.keys():
            self.model.load_state_dict(state['model_state_dict'])
        else:
            self.model.load_state_dict(state)

        self.model.eval()

        if self.gpu >= 0:
            # Send model to GPU
            self.model.cuda()




    def infer(self,image_path,frame=None,video=False,out="out"):
        # Load image
        if video :
            img_orig = frame
        else :
            img_orig = image_path
        img_t = cv2.cvtColor(img_orig,cv2.COLOR_BGR2RGB)

        # Preprocess image
        img_raw = img_orig.copy()[:, :, ::-1].transpose((2, 0, 1))
        img, info_img = preprocess(img_orig, self.imgsize, jitter=0)  # info = (h, w, nh, nw, dx, dy)
        img = np.transpose(img / 255., (2, 0, 1))
        img = torch.from_numpy(img).float().unsqueeze(0)

        if self.gpu >= 0:
            # Send model to GPU
            img = Variable(img.type(torch.cuda.FloatTensor))
        else:
            img = Variable(img.type(torch.FloatTensor))


        with torch.no_grad():
            outputs = self.model(img)
            outputs = postprocess(outputs, 80, self.confthre, self.nmsthre)

        if outputs[0] is None:
            print("No Objects Deteted!!")
            #sys.exit(0)
        else :
        # Visualize detected bboxes

            bboxes = list()
            classes = list()
            scores = list()
            colors = list()
            sigmas = list()
            names = list()
            for output in (outputs[0]):

                if self.gpu >=0 :
                    output = output.cpu()

                x1, y1, x2, y2, conf, cls_conf, cls_pred = output[:7]
                cls_id = self.coco_class_ids[int(cls_pred)]
                if self.coco_class_names[cls_id] in self.detected :
                    if self.gaussian:
                        sigma_x, sigma_y, sigma_w, sigma_h = output[7:]
                        sigmas.append([sigma_x, sigma_y, sigma_w, sigma_h])


                    box = yolobox2label([y1, x1, y2, x2], info_img)

                    if(self.coco_class_names[cls_id]== "traffic light") and (cls_conf*conf)>0.7:

                      cls_id = classify_color(img_t,box)


                    bboxes.append(box)
                    classes.append(cls_id)
                    scores.append(cls_conf * conf)
                    colors.append(self.coco_class_colors[int(cls_pred)])
                    names.append(self.names[cls_id])

            # image size scale used for sigma visualization
            h, w, nh, nw, _, _ = info_img
            sigma_scale_img = (w / nw, h / nh)
            if len(bboxes)>0:
                fig, ax = vis_bbox(
                    img_raw, bboxes, label=classes, score=scores, label_names=self.names, sigma=sigmas,
                    sigma_scale_img=sigma_scale_img,
                    sigma_scale_xy=2., sigma_scale_wh=2.,  # 2-sigma
                    show_inner_bound=False,  # do not show inner rectangle for simplicity
                    instance_colors=colors, linewidth=3)
                if video :
                    print("saving the image as : ",out+str(self.vid)+".jpg")
                    fig.savefig(out+str(self.vid)+".jpg")
                    self.vid += 1

                else :
                    print("saving the image as : ",out+str(self.vid)+".jpg")
                    fig.savefig(out)
                return bboxes , names


            else :
                print("No objects Detected")
