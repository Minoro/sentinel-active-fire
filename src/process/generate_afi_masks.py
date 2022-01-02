import sys

sys.path.append('../')


from image.sentinel import ImageStack, BufferedImageStack, load_buffered_stack_bands
from image.converter import get_gml_geometry
from active_fire.general import ActiveFireIndex

from tqdm import tqdm
import rasterio
import os
import numpy as np
# import cv2

IMAGES_STACK_DIR = '../../resources/images/stack'
QI_DATA_DIR = '../../resources/images/images/qi_data'

OUTPUT_DIR = '../resources/images/output_txt'

SAVE_AS_TXT = True

ALGORITHMS = [
    {'method': 'Baseline'},
    {'method': 'Liangrocapart'},
    {'method': 'Sahm'},
    {'method': 'PierreMarkuse'},
    {'method': 'Yongxue'},
    # {'method': 'Cicala'},
    # {'method': 'Dellaglio'},
]


def get_stack_names():    
    stacks = os.listdir(IMAGES_STACK_DIR)
    names = [stack.replace('_10m_stack', '').replace('_20m_stack', '').replace('_60m_stack', '') for stack in stacks ]

    return list(set(names))

def get_algorithms():
    algorithms = []
    
    for algorithm in ALGORITHMS:
        alg = algorithm.copy()
        alg['afi'] = ActiveFireIndex(alg['method'])

        algorithms.append(alg)

    return algorithms

if __name__ == '__main__':
    
    stack_names = get_stack_names()
    algorithms = get_algorithms()

    for stack_name in tqdm(stack_names):
        
        stack_name = stack_name.replace('.tif', '')
        img_stack = load_buffered_stack_bands(IMAGES_STACK_DIR, stack_name, (12, 11, '8A'))

        for algorithm in algorithms:
            method = algorithm['method']

            afi = algorithm['afi']
            
            mask = afi.transform(img_stack)
            
            output_dir = os.path.join(OUTPUT_DIR, method, stack_name)    
            os.makedirs(output_dir, exist_ok=True)

            if SAVE_AS_TXT:
                # Save as TXT
                np.savetxt(os.path.join(output_dir, '{}_mask.txt'.format(stack_name)), (mask != 0).astype(int), fmt='%i')
            else:
                # Save as png
                mask = mask * 255
                # cv2.imwrite(os.path.join(output_dir, '{}_mask.png'.format(stack_name)), mask)

                meta = img_stack.metas[12]

                # Active Fire Mask
                meta.update(count=1)
                output_mask = os.path.join(output_dir, '{}_mask.tif'.format(stack_name))
                with rasterio.open(output_mask, 'w', **meta) as dst:
                    dst.write_band(1, (mask).astype(rasterio.uint16))   






