

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',dest = "gpu" ,type=int, default = -1,
                    help='enter the number of gpu')

parser.add_argument('--image_path',dest = "image" , default = "Images/Traffic.jpg",
                    help='enter the path of the image')
parser.add_argument('--image_out',dest = "out" , default = "out.jpg",
                    help='enter the path to the output')

args = parser.parse_args()


from Inference import Infer
Inference_class = Infer(detect_thresh = 0.5,gpu=args.gpu)
Inference_class.infer(args.image,args.out)
