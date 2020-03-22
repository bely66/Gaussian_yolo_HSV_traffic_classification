import cv2
from glob import glob

def compute_disparity_pyramid(im_left,im_right):

        stereo = cv2.StereoBM_create(numDisparities=64,
                                       blockSize=13)
        # stereo = cv2.StereoSGBM_create(minDisparity=0,
        #                                numDisparities=64,
        #                                blockSize=11)

        # Compute disparity at full resolution and downsample
        disp = stereo.compute(im_left,im_right).astype(float) / 16.
        return disp

files = glob("Images/test/*.png")


print(files)
for i , f in enumerate(files) :
    img = cv2.imread(f,0)
    l = img[:,:img.shape[1]//2]
    r = img[:,img.shape[1]//2:]

    d_map =    compute_disparity_pyramid(l,r)


    cv2.imwrite("out/Disparity/opencv-disp-"+str(f),d_map)
