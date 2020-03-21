from glob import glob
import Disparity.FastDP as fdm
import argparse
import cv2
import numpy as np
from Inference import Infer
from collections import defaultdict
import pandas as pd
parser = argparse.ArgumentParser()

parser.add_argument('--gpu',dest = "gpu" ,type=int, default = -1,
                    help='enter the number of gpu')


args = parser.parse_args()


files = glob("Images/s_video/*.avi")

def get_distance(d_ma,box):
    obj = d_ma[int(box[0]):int(box[2]),int(box[1]-10):int(box[3]-20)]
    dist =(650*.12) /(obj.ravel())

    for i in range(len(dist)) :
        if dist[i]>1000 :
            dist[i]=10
    return np.mean(dist)

Inference_class = Infer(detect_thresh = 0.5,gpu=args.gpu)
for i , n in enumerate(files):
    d = defaultdict(dict)

    cap = cv2.VideoCapture(n)
    frame_width = int(cap.get(3))

    frame_height = int(cap.get(4))
    out = cv2.VideoWriter("out/s_video/"+str(i)+'outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width//2,frame_height))

    print("Now processing Video : ",n)
    print(".............")
    d[i]
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        cv2.imshow("frame",frame)
        l = frame[:,:frame.shape[1]//2]
        r = frame[:,frame.shape[1]//2:]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        d[i]["name"]=[]
        d[i]["X"]=[]
        d[i]["Y"]=[]
        d[i]["Z"]=[]

        # Our operations on the frame come here
        bboxes,classes=Inference_class.infer(frame=l,video=True,out="out/s_video/out.jpg")

        d_map = fdm.generate_disparity_map(l,r,"Disparity/"+str(n))

        if bboxes not None :
            for b in range(len(bboxes)) :
                y,x,y2,x2 = bboxes[b]
                cv2.putText(l, classes[b], (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0,0,255), 2, cv2.LINE_AA)
                cv2.rectangle(l,(x,y),(x2,y2),(0,255,0),2)
            d[i]["name"].append(classes[b])
            d[i]["X"].append((bboxes[b][1].item()+bboxes[b][3].item())//2)
            d[i]["Y"].append((bboxes[b][0].item()+bboxes[b][2].item())//2)
            dist = get_distance(d_map,bboxes[b])
            d[i]["Z"].append(dist)
        out.write(l)

    o = pd.DataFrame(d)
    o.to_csv("out/Disparity/"+str(i)+".csv")







# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
