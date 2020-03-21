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



parser.add_argument('--image_out',dest = "out" , default = "out.jpg",
                    help='enter the path to the output')

args = parser.parse_args()
# get the stereo images
files = glob("Images/stereo/*.png")

def get_distance(d_ma,box):
    obj = d_ma[int(box[0]):int(box[2]),int(box[1]-10):int(box[3]-20)]
    dist =(650*.12) /(obj.ravel())
    cv2.imwrite("a.png",obj)
    for i in range(len(dist)) :
        if dist[i]>1000 :
            dist[i]=10
    return np.mean(dist)


Inference_class = Infer(detect_thresh = 0.5,gpu=args.gpu)
d = defaultdict(dict)
for n,f in enumerate(files) :
    d[n]
    i = cv2.imread(f)
    l = f[:,:f.shape[1]//2]
    r = f[:,f.shape[1]//2:]
    bboxes,classes = Inference_class.infer(image_path=l,out="out/Disparity/0-"+str(n)+".jpg")
    d_map = fdm.generate_disparity_map(l,r,str(n))
    d[n]["name"]=[]
    d[n]["X"]=[]
    d[n]["Y"]=[]
    d[n]["Z"]=[]
    for b in range(len(bboxes)) :
        d[n]["name"].append(classes[b])
        d[n]["X"].append((bboxes[b][1]+bboxes[b][3])//2)
        d[n]["Y"].append((bboxes[b][0]+bboxes[b][2])//2)

        dist = get_distance(d_map,bboxes[b])
        d[n]["Z"].append(dist)
o = pd.DataFrame(d)
o.to_csv("out/Disparity/"+str(n)+".csv")
