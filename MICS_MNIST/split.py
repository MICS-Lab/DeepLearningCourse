#read image with numpy

import numpy as np
import os
import cv2


for k in range(10):
    
    path = f'MICS_MNIST/{k}_final.png'
    img = np.array(cv2.imread(path,0))
    #get black components in image
    ret,thresh = cv2.threshold(img,200,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours[1:]):
        x,y,w,h = cv2.boundingRect(contour)
        # print(x,y,w,h)
        s = max(w,h)+6
        cx = x + w//2
        cy = y + h//2
        x = cx - s//2-1
        y = cy - s//2-1
        splitted_img = img[y:y+s+2, x:x+s+2]
        #add padding
        padding = 1+s//20
        splitted_img = cv2.copyMakeBorder(splitted_img, padding,padding,padding,padding, cv2.BORDER_CONSTANT, value=[255,255,255])
        #resize to 35x35
        splitted_img = cv2.resize(splitted_img, (35, 35))
        cv2.imwrite(f'MICS_MNIST/splitted/{k}_{i}.png', splitted_img)

    # cv2.imshow('img',img)
    # cv2.waitKey(0)
