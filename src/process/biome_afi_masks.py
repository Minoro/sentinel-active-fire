import sys

sys.path.append('../')

from image.sentinel import ImageStack, BufferedImageStack, load_buffered_stack_bands
from image.converter import get_gml_geometry
from active_fire.biome import resolve_biome_and_apply_afd, apply_biome_afd


import os
from PIL import Image
import numpy as np
import cv2

IMAGES_PATH = '../../images/stack'
BIOMES_SHAPE_FILE = '../../resources/ecoregions/Ecoregions2017.shp'
OUTPUT_PATH = '../../images/output'




if __name__ == '__main__':

    images = [image.replace('_10m_stack', '').replace('_20m_stack', '').replace('_60m_stack', '') for image in os.listdir(IMAGES_PATH)]
    images = list(set(images))

    print('Num. Images:', len(images))
    for image in images:
        try:
            mask, image_stack = resolve_biome_and_apply_afd(BIOMES_SHAPE_FILE, IMAGES_PATH, image)
        except Exception as e:
            print('Error processing: {} - Skiping image.'.format(image))
            print(e)

            continue

        # print(mask.shape)
        # im = Image.fromarray(mask * 255)
        # im.save(os.path.join(OUTPUT_PATH, '{}_mask.png'.format(image)))
        cv2.imwrite(os.path.join(OUTPUT_PATH, '{}_mask.png'.format(image)), mask*255)


        img = np.zeros((5490, 5490, 3))
        img[:,:,0] = image_stack.read('8A')
        img[:,:,1] = image_stack.read(11)
        img[:,:,2] = image_stack.read(12)

        # img = buffered_stack.read()

        cv2.imwrite(os.path.join(OUTPUT_PATH, '{}.png'.format(image)), img*255)
