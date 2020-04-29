from glob import glob
import Disparity.Opencv as fdm
import argparse
import cv2
import numpy as np
from Inference import Infer
import matplotlib.pyplot as plt
from Traffic_Classifier_HSV import *
from skimage.color import rgb2grey,rgb2hsv
import sys

def get_distance(d_ma,box):
    obj = d_ma[int(box[0]):int(box[2]),int(box[1]-10):int(box[3]-20)]
    dist =(700*.12) /(obj.ravel())

    for i in range(len(dist)) :
        if dist[i]>1000 :
            dist[i]=20
    return (dist)

gpu = int(input("Enter 0 for gpu \n Enter -1 for cpu \n"))

if (gpu) != 0 or gpu != -1 :
    print("you entered a wrong input \n restart and try again")
    sys.exit()


files = glob("Images/stereo/*")
Inference_class = Infer(detect_thresh = 0.5,gpu=gpu)


for i , file in enumerate(files):
    # detection and generating boxes
    f = cv2.imread(file)
    l = f[:,:f.shape[1]//2]
    r = f[:,f.shape[1]//2:]
    name = str(i)
    bboxes,classes,_ = Inference_class.infer(image_path=l,out="out/Hist/"+name+"-0-"+".jpg")

    #Generating Disparity Map
    d_map = fdm.process_frame(l,r)//4
    d_d = np.copy(d_map)
    #iterating on the boxes of every object
    for b in range(len(bboxes)):
        # if the class is traffic generate the results
        if classes[b].find("traffic")!= -1 :
            roi = crop_roi_image(l,bboxes[b])
            cv2.imwrite("out/Hist/"+name+"-roi-"+".jpg",roi)
            roi = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
            hsv = rgb2hsv(roi)
            sat_m = high_saturation_region_mask(hsv)
            val_m = high_value_region_mask(hsv)
            final_mask = np.logical_and(sat_m, val_m)
            fig , ax = plt.subplots(1,2,figsize=(10,15))
            ax[0].hist(sat_m.ravel())
            ax[0].set_title("Saturation Mask")
            ax[1].imshow(final_mask,cmap="gray")
            ax[1].set_title("final_mask")

            fig.savefig("out/Hist/"+name+"-sat&val-.jpg")

        y,x,y2,x2 = bboxes[b]
        dist = get_distance(d_map,bboxes[b])
        l_c = cv2.cvtColor(l,cv2.COLOR_BGR2RGB)
        cv2.rectangle(l_c,(x,y),(x2,y2),255,2)
        fig , ax = plt.subplots(1,2,figsize=(10,15))

        ax[0].hist(dist.ravel(),30)
        #fig.xlim(0,40)
        ax[0].set_title(classes[b]+" distance histogram")
        ax[1].imshow(l_c)
        ax[1].set_title("original image")
        fig.savefig("out/Hist/"+name+"-distance_hist-"+classes[b]+" "+".jpg")

        cv2.rectangle(d_d,(x-10,y-10),(x2-10,y2-10),255,2)

    cv2.imwrite("out/Hist/"+name+"-annot-"+".jpg",d_d)
