from Inference import Infer
import cv2
import argparse
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument('--gpu',dest = "gpu" ,type=int, default = -1,
                    help='enter the number of gpu')

parser.add_argument('--video_path',dest = "video" , default = "Images/Traffic.jpg",
                    help='enter the path of the video')
parser.add_argument('--image_out',dest = "out" , default = "out.jpg",
                    help='enter the path to the output')

args = parser.parse_args()


Inference_class = Infer(detect_thresh = 0.5,gpu=args.gpu)

files = glob("Images/video/*")
for f in  files :
    cap = cv2.VideoCapture(f)
    frame_width = int(cap.get(3))

    frame_height = int(cap.get(4))
    out = cv2.VideoWriter("out/Video/outpy.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))


    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Our operations on the frame come here
        bboxes,classes=Inference_class.infer(frame=frame,video=True,out="out/Video/out.jpg")
        if bboxes != None :
            for b in range(len(bboxes)) :
                y,x,y2,x2 = bboxes[b]
                cv2.putText(frame, classes[b], (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                           1, (0,0,255), 2, cv2.LINE_AA)
                cv2.rectangle(frame,(x,y),(x2,y2),(0,255,0),2)
        out.write(frame)


        cv2.imshow("frame",frame)







# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()
