from image import ImageStack
from active_fire_detection import *
import rasterio
import numpy as np
import os
import cv2
import math 
import sys


IMAGES_DIR = '../images/stack/'
IMAGE_NAME = 'T36RTU_20200829T083611_20m_stack.tif'
IMAGE_PATH = os.path.join(IMAGES_DIR, IMAGE_NAME)


method = 'cicala'
output_path = os.path.join('../images/stack/', method)

TH = 2.8

with rasterio.open(IMAGE_PATH) as src:

    img_stack = ImageStack(src)

    afd = ActiveFireIndex(method)
    # afd = DellaglioAFD()
    # afd = 'DellaglioAFD'()

    mask = afd.fit(img_stack) > TH

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, '{}_th_{}.tif'.format(method, TH))

    meta = src.meta
    meta.update(count = 1, dtype = rasterio.uint8)                
    img = np.asarray(mask, dtype=np.uint8) * 255

    with rasterio.open(output_path, 'w', **meta) as dst:        
        dst.write_band(1, img)  

    
print('Done!')

