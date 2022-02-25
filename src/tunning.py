import sys

sys.path.append('../')

from image.sentinel import BufferedImageStack, load_buffered_stack_bands
from image.converter import get_gml_geometry, get_cloud_mask, reflectance_to_radiance
from active_fire.general import CicalaAFI, LiangrocapartAFI, YongxueAFI
# from utils import reflectance_conversion

from utils import metadata

from PIL import Image

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

import os
import numpy as np
import cv2 

# D
# IMAGE = 'T34SFH_20180723T092031'
# MASK = 'T34SFH_20180723T092031_mask.png'


# IMAGE = 'T34HFH_20181025T081009'
# MASK = 'T34HFH_20181025T081009_mask.png'



# G
IMAGE = 'T09UYV_20180808T193901'
MASK = 'T09UYV_20180808T193901_mask.png'

gt = Image.open(os.path.join('../images/output/G', MASK))
gt = (np.array(gt) > 0)


img_stack = load_buffered_stack_bands('../images/stack/', IMAGE, (12, 11, '8A'))

cloud_mask = get_cloud_mask('../images/qi_data/{}/MSK_CLOUDS_B00.gml'.format(IMAGE), gt.shape, img_stack.metas[12]['transform'])

cv2.imwrite('../images/cicala/g/cloud.png', cloud_mask*255)


metadata = metadata.get_image_metadata('../images/metadata/L1C_T09UYV_A016341_20180808T194437/MTD_TL.xml', '../images/metadata/L1C_T09UYV_A016341_20180808T194437/MTD_MSIL1C.xml')

print(metadata)

# img_stack.set_band(12, reflectance_conversion.get_radiance(img_stack.read(12) * 10000, 12, metadata))
# img_stack.set_band(11, reflectance_conversion.get_radiance(img_stack.read(11) * 10000, 11, metadata))
# img_stack.set_band('8A', reflectance_conversion.get_radiance(img_stack.read('8A') * 10000, '8A', metadata))
# print(img_stack.read(12).mean())

afi = CicalaAFI()
# for i in np.arange(10, 20, 0.5):


gt = gt.flatten()
best = 0
best_th = 0

bottom = 0
top = 110
step = 10

iteration = 0
num_iterations = 4


while iteration <= num_iterations:
    print('Iteration: ', iteration)
    for i in np.arange(bottom, top, step):
        print('Threshold: ', i)

        mask = afi.transform(img_stack, metadata, alpha=0.5, th=i)

        print(gt.sum())
        print(mask.sum())

        score = f1_score(gt, mask.flatten(), average='macro')
        print('F1: ', score)
        if score > best:
            best = score
            best_th = i
    
    print('Best score: ', best, ' Best TH: ', best_th)

    bottom = best_th - step
    top = best_th + step
    step = step / 10

    iteration += 1



    # cv2.imwrite('../images/cicala/d/Cicala_{}.png'.format(i), mask*255)
    # cv2.imwrite('../images/cicala/d/Cicala_{}_cloudless.png'.format(i), (mask & cloud_mask)*255)




# print('F1: ', f1_score(gt.flatten(), mask.flatten(), average='macro'))
# print('P: ', precision_score(gt.flatten(), mask.flatten(), average='macro'))
# print('R: ', recall_score(gt.flatten(), mask.flatten(), average='macro'))
# print('Acc: ', accuracy_score(gt.flatten(), mask.flatten()))



