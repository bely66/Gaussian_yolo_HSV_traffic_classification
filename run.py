from glob import glob
import cv2
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gpu',dest = "gpu" ,type=int, default = -1,
                    help='enter the number of gpu')



parser.add_argument('--image_out',dest = "out" , default = "out.jpg",
                    help='enter the path to the output')

args = parser.parse_args()
files = glob("Images/normal/*")
from Inference import Infer

Inference_class = Infer(detect_thresh = 0.5,gpu=args.gpu)
for n,i in enumerate(files) :
    i = cv2.imread(i)
    bboxes,classes = Inference_class.infer(image_path=i,out="out/Classification/"+str(n)+".jpg")
