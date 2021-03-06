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

    for i in range(len(dist)) :
        if dist[i]>1000 :
            dist[i]=10
    return np.mean(dist)


Inference_class = Infer(detect_thresh = 0.5,gpu=args.gpu)
d = defaultdict(dict)
for n,f in enumerate(files) :
    d["frame_"+str(n)]
    f = cv2.imread(f)
    l = f[:,:f.shape[1]//2]
    r = f[:,f.shape[1]//2:]
    bboxes,classes = Inference_class.infer(image_path=l,out="out/Disparity/0-"+str(n)+".jpg")
    d_map = fdm.generate_disparity_map(l,r,str(n))
    d["frame_"+str(n)]["name"]=[]
    d["frame_"+str(n)]["X"]=[]
    d["frame_"+str(n)]["Y"]=[]
    d["frame_"+str(n)]["Z"]=[]
    for b in range(len(bboxes)) :
        d["frame_"+str(n)]["name"].append(classes[b])
        d["frame_"+str(n)]["X"].append((bboxes[b][1].item()+bboxes[b][3].item())//2)
        d["frame_"+str(n)]["Y"].append((bboxes[b][0].item()+bboxes[b][2].item())//2)

        dist = get_distance(d_map,bboxes[b])
        d["frame_"+str(n)]["Z"].append(dist)
o = pd.DataFrame(d)
o = o.T
o.to_csv("out/Disparity/images.csv")
