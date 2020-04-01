from glob import glob
from Disparity.Opencv import process_frame
import argparse
import cv2
import numpy as np
from Inference import Infer
from collections import defaultdict
import pandas as pd
from sort.sort import *

parser = argparse.ArgumentParser()


parser.add_argument('--gpu',dest = "gpu" ,type=int, default = 0,
                    help='enter the number of gpu')


args = parser.parse_args()


files = glob("Images/s_video/*.avi")

def get_distance(d_ma,box):
    obj = d_ma[int(box[0]):int(box[2]),int(box[1]):int(box[3])]
    dist =(650*.12) /(obj.ravel())

    for i in range(len(dist)) :
        if dist[i]>1000 :
            dist[i]=10
    return np.mean(dist)

Inference_class = Infer(detect_thresh = 0.5,gpu=args.gpu)
for iter , n in enumerate(files):
    d = defaultdict(dict)

    cap = cv2.VideoCapture(n)
    frame_width = int(cap.get(3))

    frame_height = int(cap.get(4))
    out = cv2.VideoWriter("out/s_video/"+str(iter)+'outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width//2,frame_height))

    print("Now processing Video : ",n)
    print(".............")
    i= 0
    mot_tracker = Sort()
    while(True):
        # Capture frame-by-frame
        d["frame_"+str(i)]
        ret, frame = cap.read()

        l = frame[:,:frame.shape[1]//2]
        r = frame[:,frame.shape[1]//2:]
        dets = []

        d["frame_"+str(i)]["name"]=[]
        d["frame_"+str(i)]["X_1"]=[]
        d["frame_"+str(i)]["Y_1"]=[]
        d["frame_"+str(i)]["X_2"]=[]
        d["frame_"+str(i)]["Y_2"]=[]
        #d["frame_"+str(i)]["Z"]=[]

        # Our operations on the frame come here
        bboxes,classes,scores=Inference_class.infer(frame=l,video=True,out="out/s_video/out.jpg")

        #d_map = process_frame(l,r)
        #cv2.imwrite("out/Disparity/"+str(i)+".png",d_map)

        if bboxes != None :
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

        cv2.imshow("frame",l)
        i+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        out.write(l)

    o = pd.DataFrame(d)
    o = o.T
    o.to_csv("out/s_video/Frames.csv")







# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
