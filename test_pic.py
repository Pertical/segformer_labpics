

import os 
import random 

import numpy as np
from PIL import Image
from tqdm import tqdm

import cv2
import sys


VOCdevkit_path      = 'VOCdevkit'



if __name__ == "__main__":

    random.seed(0)
    #segfilepath     = os.path.join(VOCdevkit_path, 'VOC2012/SegmentationClass')

    segfilepath = os.path.join(VOCdevkit_path, 'LabPics/SegmentationClass')

    classes_nums        = np.zeros([256], int)

    for i in range(1):

        
        seg = os.listdir(segfilepath)[i]


        print(seg)
        if seg.endswith(".png"):
            
            seg = os.path.join(segfilepath, seg)
          
            voc_img = Image.open(seg)
            voc_img = np.array(voc_img, np.uint8)

            classes_nums += np.bincount(np.reshape(voc_img, [-1]), minlength=256)
            np.set_printoptions(threshold=sys.maxsize)
            print(voc_img.shape)
            print(np.unique(voc_img))


    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |"%("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
            print('-' * 37)