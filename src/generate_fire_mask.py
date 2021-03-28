from image import ImageStack
from active_fire_detection import *
import rasterio
import numpy as np
import os
import cv2
import math 
import sys
from glob import glob


IMAGES_DIR = '../images/stack/'
IMAGE_NAME = 'T36RTU_20200829T083611_20m_stack.tif'
image_path = os.path.join(IMAGES_DIR, IMAGE_NAME)


# method = 'cicala'
# output_path = os.path.join('../images/stack/', method)

# TH = 2.7

# ALPHA = 0.5

# with rasterio.open(image_path) as src:

#     img_stack = ImageStack(src)

#     afd = ActiveFireIndex(method)

#     if method == 'cicala':
#         mask = afd.fit(img_stack, alpha=ALPHA)

#     elif method == 'dellaglio3' or method == 'dellaglio4':
        
#         # Open 10m image
#         image_path_10m = image_path.replace('20m', '10m')
        
#         with rasterio.open(image_path_10m) as src_10m:
#             img_stack_10m = ImageStack(src_10m)
#             mask = afd.fit(img_stack, img_stack_10m=img_stack_10m)

#     else:
#         mask = afd.fit(img_stack)


#     # O método já gera a imagem limiarizada
#     if method != 'liangrocapart':
#         mask = mask > TH

#     if not os.path.exists(output_path):
#         os.makedirs(output_path)

#     output_path = os.path.join(output_path, '{}_th_{}.tif'.format(method, TH))

#     meta = src.meta
#     meta.update(count = 1, dtype = rasterio.uint8)                
#     img = np.asarray(mask, dtype=np.uint8) * 255

#     with rasterio.open(output_path, 'w', **meta) as dst:        
#         dst.write_band(1, img)  

    
# print('Done!')


images_20m_path = glob(os.path.join(IMAGES_DIR, '*20m*.tif'))


print('Num. 20m images found: ', len(images_20m_path))

tests_config = [
    {
        'method': 'baseline',
        'threshold': 1.6,
        'params': {}
    },
    {
        'method': 'baseline',
        'threshold': 1.7,
        'params': {}
    },
    {
        'method': 'baseline',
        'threshold': 1.8,
        'params': {}
    },
    {
        'method': 'cicala',
        'threshold': 2.6,
        'params': { 'alpha': 0.5 }
    },
    {
        'method': 'cicala',
        'threshold': 2.71,
        'params': { 'alpha': 0.5 }
    },
    {
        'method': 'cicala',
        'threshold': 2.7,
        'params': { 'alpha': 0.3 }
    },
    {
        'method': 'cicala',
        'threshold': 2.8,
        'params': { 'alpha': 0.8 }
    },
    {
        'method': 'liangrocapart',
        'threshold': 0.5,
        'params': {}
    },
    {
        'method': 'dellaglio',
        'threshold': 1.5,
        'params': {}
    },
    {
        'method': 'dellaglio',
        'threshold': 1.7,
        'params': {}
    },
    {
        'method': 'dellaglio',
        'threshold': 1.8,
        'params': {}
    },
    {
        'method': 'dellaglio3',
        'threshold': 1.5,
        'params': { 'image_stack_10m': None }
    },
    {
        'method': 'dellaglio3',
        'threshold': 1.7,
        'params': { 'image_stack_10m': None }
    },
    {
        'method': 'dellaglio3',
        'threshold': 1.8,
        'params': { 'image_stack_10m': None }
    },
    {
        'method': 'dellaglio4',
        'threshold': 1.5,
        'params': { 'image_stack_10m': None }
    },
    {
        'method': 'dellaglio4',
        'threshold': 1.7,
        'params': { 'image_stack_10m': None }
    },
    {
        'method': 'dellaglio4',
        'threshold': 1.8,
        'params': { 'image_stack_10m': None }
    },
]


for image_path in images_20m_path:

    with rasterio.open(image_path) as src:
        meta = src.meta

        for test_config in tests_config:
        
            method = test_config['method']
            params = test_config['params']
            threshold = test_config['threshold']

            afi = ActiveFireIndex(method)

            img_stack = ImageStack(src)

            if 'image_stack_10m' in params:
                # Open a 10m image to yse
                image_path_10m = image_path.replace('20m', '10m')
                
                if not os.path.exists(image_path_10m):
                    print('Image {} not found for {} method'.format(image_path_10m, method))
                    continue
                
                with rasterio.open(image_path_10m) as src_10m:
                    img_stack_10m = ImageStack(src_10m)
                    mask = afi.transform(img_stack, img_stack_10m=img_stack_10m)

            else:
                mask = afi.transform(img_stack, **params)

            # Liangrocapart's method already apply a threshold 
            if method != 'liangrocapart':
                mask = mask > threshold


            image_name = os.path.basename(image_path)
            image_name = os.path.splitext(image_name)[0]

            output_path = os.path.join(IMAGES_DIR, image_name, method)

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            output_path = os.path.join(output_path, '{}_th_{}.tif'.format(method, threshold))

            meta.update(count = 1, dtype = rasterio.uint8)                
            img = np.asarray(mask, dtype=np.uint8) * 255

            with rasterio.open(output_path, 'w', **meta) as dst:        
                dst.write_band(1, img)  

