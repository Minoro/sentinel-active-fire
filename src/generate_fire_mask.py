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
image_path = os.path.join(IMAGES_DIR, IMAGE_NAME)


method = 'liangrocapart'
output_path = os.path.join('../images/stack/', method)

TH = 1.7

ALPHA = 0.5

with rasterio.open(image_path) as src:

    img_stack = ImageStack(src)

    afd = ActiveFireIndex(method)

    if method == 'cicala':
        mask = afd.fit(img_stack, alpha=ALPHA)

    elif method == 'dellaglio3' or method == 'dellaglio4':
        
        # Open 10m image
        image_path_10m = image_path.replace('20m', '10m')
        
        with rasterio.open(image_path_10m) as src_10m:
            img_stack_10m = ImageStack(src_10m)
            mask = afd.fit(img_stack, img_stack_10m=img_stack_10m)

    else:
        mask = afd.fit(img_stack)


    # O método já gera a imagem limiarizada
    if method != 'liangrocapart':
        mask = mask > TH

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path = os.path.join(output_path, '{}_th_{}.tif'.format(method, TH))

    meta = src.meta
    meta.update(count = 1, dtype = rasterio.uint8)                
    img = np.asarray(mask, dtype=np.uint8) * 255

    with rasterio.open(output_path, 'w', **meta) as dst:        
        dst.write_band(1, img)  

    
print('Done!')

