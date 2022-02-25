import sys

from importlib_metadata import metadata

sys.path.append('../')


from image.sentinel import ImageStack, BufferedImageStack, load_buffered_stack_bands
from image.converter import get_gml_geometry
from active_fire.general import ActiveFireIndex
from utils.reflectance_conversion import get_image_metadata

from tqdm import tqdm
import rasterio
import os
import numpy as np
from glob import glob
# import cv2

IMAGES_STACK_DIR = '../../resources/images/stack'
QI_DATA_DIR = '../../resources/images/images/qi_data'

OUTPUT_DIR = '../../resources/images/output_txt'
# METADATA_DIR = '../resources/metadata'
METADATA_DIR = '../../resources/Sentinel2/metadata'

SAVE_AS_TXT = True

ALGORITHMS = [
    {'method': 'Baseline'}, # P: : 0.005999406879034109  R:  0.7987444225381343  IoU:  0.005990351623328869  F-score:  0.01190936198078334
    {'method': 'Liangrocapart'}, # P: : 0.902453531598513  R:  0.6297602988481893  IoU:  0.5896240163217721  F-score:  0.7418408507517418
    {'method': 'Sahm'}, # P: : 0.09443004471312343  R:  0.6256615129189582  IoU:  0.08938021613127974  F-score:  0.16409370173564572
    {'method': 'PierreMarkuse'}, # P: : 0.8566532258064516  R:  0.2204524229532012  IoU:  0.21260945709281961  F-score:  0.3506643558636626
    {'method': 'Yongxue'}, #P: : 0.5469818459509181  R:  0.8175780844661201  IoU:  0.4874864655839134  F-score:  0.6554499511261775
    # {'method': 'Cicala'},
    # {'method': 'Dellaglio'},
]


def get_stack_names():    
    # stacks = os.listdir(IMAGES_STACK_DIR)
    # names = [stack.replace('_10m_stack', '').replace('_20m_stack', '').replace('_60m_stack', '') for stack in stacks ]

    # return list(set(names))

    stacks = os.listdir(IMAGES_STACK_DIR)
    return stacks

def get_metadata_file(stack_name):

    stack_name_parts = stack_name.split('_')
    tile_name = stack_name_parts[0] + '_' + stack_name_parts[1]
    print(tile_name)

    print(glob(os.path.join(METADATA_DIR, 'L1C_*', '*.xml'), recursive=True))
    
    print(os.path.join(METADATA_DIR,'L1C_{}_*_{}'.format(stack_name_parts[0], stack_name_parts[1]), '*.xml'))
    metadata_file = glob(os.path.join(METADATA_DIR,'L1C_{}_*_{}'.format(stack_name_parts[0], stack_name_parts[1]), '*.xml'))

    print(metadata_file)
    sys.exit()

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
        
        img_buffer = BufferedImageStack()

        metadata = get_metadata_file(stack_name)
        print(metadata)
        sys.exit()

        stack_path = os.path.join(IMAGES_STACK_DIR, stack_name)
        with rasterio.open(stack_path) as dataset:
            img_stack = ImageStack(dataset)
            img_buffer.load_band_from_stack(img_stack, 12)
            img_buffer.load_band_from_stack(img_stack, 11)
            img_buffer.load_band_from_stack(img_stack, '8A')

        


        for algorithm in algorithms:
            method = algorithm['method']

            afi = algorithm['afi']
            
            mask = afi.transform(img_buffer)
            
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






