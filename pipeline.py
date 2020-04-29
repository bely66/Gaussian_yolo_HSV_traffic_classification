from glob import glob
from Disparity.Opencv import process_frame
import cv2
import numpy as np
from Inference import Infer
from collections import defaultdict
import pandas as pd
from sort.sort import *

## Addresses
STEREO_VIDEO = "Images/s_video/*"
STEREO_IMAGES = "Images/stereo/*"
GPU = -1
##MODULES
def get_distance(d_ma,box):
    obj = d_ma[int(box[0]):int(box[2]),int(box[1]):int(box[3])]
    dist =(650*.12) /(obj.ravel())

    for i in range(len(dist)) :
        if dist[i]>1000 :
            dist[i]=10
    return np.mean(dist)
## MODULES FOR VIDEOS

def read_frame(file_name,video=True):
    if video :
        cap = cv2.VideoCapture(file_name)
        frame_width = int(cap.get(3))

        frame_height = int(cap.get(4))

        return cap , frame_width , frame_height
    else :
        img = cv2.imread(file_name)
        return img

def process_frame(cap,frame_number,mot_tracker=None,video = True):




    i= frame_number


    # Capture frame-by-frame
    d["frame_"+str(i)]
    ret, frame = cap.read()

    l = frame[:,:frame.shape[1]//2]
    r = frame[:,frame.shape[1]//2:]
    frame_f = l

    dets = []

    d["frame_"+str(i)]["name"]=[]
    d["frame_"+str(i)]["X_1"]=[]
    d["frame_"+str(i)]["Y_1"]=[]
    d["frame_"+str(i)]["X_2"]=[]
    d["frame_"+str(i)]["Y_2"]=[]
    d["frame_"+str(i)]["Z"]=[]

    # Our operations on the frame come here
    bboxes,classes,scores=Inference_class.infer(frame=frame_f,video=True,out="out/s_video/out.jpg")

    d_map = process_frame(l,r)
    cv2.imwrite("out/Disparity/"+str(i)+".png",d_map)

    if bboxes != None :

        if video :
            for b in range(len(bboxes)) :


                det = bboxes[b]
                det.append(scores[b])
                dets.append(det)


            dets = np.array(dets)
            trackers = mot_tracker.update(dets) # updating sort with the tracked objects
            for t in range(len(trackers)) :
                y,x,y2,x2 = trackers[t][:4] # getting the coordinates
                id = int(trackers[t][-1]) # getting the object id
                cv2.putText(l, classes[t]+str(id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0,0,255), 2, cv2.LINE_AA)
                cv2.rectangle(l,(int(x),int(y)),(int(x2),int(y2)),(0,255,0),2)
                d["frame_"+str(i)]["name"].append(classes[t]+str(id))
                d["frame_"+str(i)]["X_1"].append(bboxes[b][1].item())
                d["frame_"+str(i)]["Y_1"].append(bboxes[b][0].item())
                d["frame_"+str(i)]["X_2"].append(bboxes[b][3].item())
                d["frame_"+str(i)]["Y_2"].append(bboxes[b][2].item())
                dist = get_distance(d_map,bboxes[t])
                d["frame_"+str(i)]["Z"].append(dist)






Inference_class = Inferw(detect_thresh = 0.5,gpu=GPU)
for iter , n in enumerate(files):
    d = defaultdict(dict)



    out = cv2.VideoWriter("out/s_video/"+str(iter)+'outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width//2,frame_height))
