import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageFilter
from skimage.measure import compare_ssim
import argparse
import imutils

#input of frames 
img1 = cv2.imread('img0045.jpg')
img2= cv2.imread('img0062.jpg')

#different thresholding methods applied on the image
ret,thresh1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)

#ret,thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
#ret,thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
#ret,thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
#ret,thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)


#Plot all the thresholding images of the frame
'''
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']

for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
'''


plt.imshow(thresh1,'gray')
plt.show()


#converting the thresholded image to gray image
grayA=cv2.cvtColor(thresh1,cv2.COLOR_BGR2GRAY)
grayB=cv2.cvtColor(thresh2,cv2.COLOR_BGR2GRAY)


#applying structural similarity index over the two images
#score gives a number from [-1,1]. 1 showing a perfect match between the images
(score,diff)=compare_ssim(thresh2,thresh1,full=True,multichannel=True)
diff=(diff*255).astype("uint8")

print("SSIM: {}".format(score))
diff=np.array(diff)

diff=Image.fromarray(diff)
plt.imshow(diff,'gray')
plt.show()
