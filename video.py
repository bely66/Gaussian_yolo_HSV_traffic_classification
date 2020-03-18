from Inference import Infer
import cv2
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--gpu',dest = "gpu" ,type=int, default = -1,
                    help='enter the number of gpu')

parser.add_argument('--video_path',dest = "video" , default = "Images/Traffic.jpg",
                    help='enter the path of the video')
parser.add_argument('--image_out',dest = "out" , default = "out.jpg",
                    help='enter the path to the output')

args = parser.parse_args()


Inference_class = Infer(detect_thresh = 0.5,gpu=args.gpu)


cap = cv2.VideoCapture(args.video)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cv2.imshow("frame",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Our operations on the frame come here
    Inference_class.infer(frame=frame,video=True,out=args.out)





# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
