import rasterio
import rasterio.mask
# from rasterio.enums import Resampling
from image.sentinel import ImageStack, BufferedImageStack, load_buffered_stack_bands
from image.converter import get_gml_geometry
from active_fire.biome import MediterraneanForestAFD, apply_biome_afd
import numpy as np
from osgeo import ogr
import gdal
import json

import os
import cv2

IMAGES_DIR = '../images/stack'
QI_DATA_DIR = '../images/qi_data'
OUTPUT_DIR = '../images/output'

SAMPLES = [
    {'label': 'A', 'stack': 'T50MKB_20180928T022551', 'biome': 'TropicalMoistForest'},
    {'label': 'B', 'stack': 'T20LQH_20190821T142041', 'biome': 'TropicalDryForest'},
    {'label': 'C', 'stack': 'T53LPC_20180829T010731', 'biome': 'Savanna'},
    {'label': 'D', 'stack': 'T34SFH_20180723T092031', 'biome': 'MediterraneanForest'},
    {'label': 'E', 'stack': 'T10TFK_20181111T185621', 'biome': 'MediterraneanForest'},
    {'label': 'F', 'stack': 'T34HFH_20181025T081009', 'biome': 'MediterraneanForest'},
    {'label': 'G', 'stack': 'T09UYV_20180808T193901', 'biome': 'ConiferForest'},
    {'label': 'H', 'stack': 'T50WPT_20180719T034529', 'biome': 'Taiga'},
]

for sample in SAMPLES:
    print('Processando: {} - {} - {}'.format(sample['label'], sample['stack'], sample['biome']))
    mask, buffered_stack = apply_biome_afd(sample['biome'], IMAGES_DIR, sample['stack'])
    # print(mask.shape)

    meta = buffered_stack.metas[12]
    
    shape = (meta['height'], meta['width'])
    gml_cloud_mask_path = os.path.join(QI_DATA_DIR, sample['stack'], 'MSK_CLOUDS_B00.gml')
    print(gml_cloud_mask_path)
    cloud_geometry = get_gml_geometry(gml_cloud_mask_path)
    cloud_mask = np.ones(shape, dtype=np.bool)
    if cloud_geometry is not None:
        cloud_mask = rasterio.features.geometry_mask(cloud_geometry, shape, meta['transform'])

    output_path = os.path.join(OUTPUT_DIR, sample['label'])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Active Fire Mask
    meta.update(count=1)
    output_mask = os.path.join(output_path, '{}_mask.tif'.format(sample['stack']))
    with rasterio.open(output_mask, 'w', **meta) as dst:
        dst.write_band(1, (mask * 255).astype(rasterio.uint16))   

    output_mask = os.path.join(output_path, '{}_mask.png'.format(sample['stack']))
    cv2.imwrite(output_mask, (mask * 255))

    # Active Fire Mask Without Clouds
    cloudless_mask = mask & cloud_mask
    output_mask = os.path.join(output_path, '{}_cloudless_mask.tif'.format(sample['stack']))
    with rasterio.open(output_mask, 'w', **meta) as dst:
        dst.write_band(1, (cloudless_mask * 255).astype(rasterio.uint16))   

    output_mask = os.path.join(output_path, '{}_cloudless_mask.png'.format(sample['stack']))
    cv2.imwrite(output_mask, (cloudless_mask * 255))


    # Cloud Mask
    output_mask = os.path.join(output_path, '{}_cloud_mask.tif'.format(sample['stack']))
    with rasterio.open(output_mask, 'w', **meta) as dst:
        dst.write_band(1, (cloud_mask * 255).astype(rasterio.uint16))   

    output_mask = os.path.join(output_path, '{}_cloud_mask.png'.format(sample['stack']))
    cv2.imwrite(output_mask, (cloud_mask * 255))




    meta = buffered_stack.metas[12]
    meta.update(count=3)
    print(meta)
    print(buffered_stack.read(12).shape)

    output_rgb = os.path.join(output_path, '{}_rgb.tif'.format(sample['stack']))
    with rasterio.open(output_rgb, 'w', **meta) as dst:
        dst.write_band(1, (buffered_stack.read(12) * 255).astype(rasterio.uint16))   
        dst.write_band(2, (buffered_stack.read(11) * 255).astype(rasterio.uint16))   
        dst.write_band(3, (buffered_stack.read(4) * 255).astype(rasterio.uint16))   


    output_rgb = os.path.join(output_path, '{}_rgb.png'.format(sample['stack']))
    cv2.imwrite(output_rgb, (buffered_stack.read() * 255))



# def get_gml_geometry(gml_file):
#     reader = ogr.Open(gml_file)
#     layer = reader.GetLayer()
#     geometry = []
#     for i in range(layer.GetFeatureCount()):
#         feature = layer.GetFeature(i)
#         # yield json.loads(feature.ExportToJson())
#         # print(feature.ExportToJson())
#         json_feature = json.loads(feature.ExportToJson())
#         # print(type(json_feature['geometry']))
#         geometry.append(json_feature['geometry'])
    
#     return geometry

# print(rasterio.__version__)

# with rasterio.open('../images/original/T09UXV_20180808T193901_B12.tif') as dataset:
#     c12 = dataset.read(1)
#     print(dataset.transform)
#     print(dataset.crs)
#     print(dataset.meta)

# print(c12.shape)




# geometry = get_gml_geometry('../images/tiles_50_M_KB_S2A_MSIL1C_20180928T022551_N0206_R046_T50MKB_20180928T060436.SAFE_GRANULE_L1C_T50MKB_A017060_20180928T024358_QI_DATA_MSK_CLOUDS_B00.gml')

# with rasterio.open('../images/original/T50MKB_20180928T022551_B12.tif') as src:
#     out_image, out_transform = rasterio.mask.mask(src, geometry, crop=False)
#     out_meta = src.meta
#     print(out_meta)
#     print(src.nodata)
#     print(src.bounds)
#     print(out_image.shape)
#     print(out_image.sum())
#     print(out_image.min())
#     print(out_image.max())

#     # print(src.read_mask(1))

# out_meta.update({"driver": "GTiff",
#                  "height": out_image.shape[1],
#                  "width": out_image.shape[2],
#                  "transform": out_transform})


# # out_image = out_image /10000.0
# with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
#     dest.write( out_image )
#     # dest.write( (out_image * 255).astype(rasterio.uint16) )


# geometry = get_gml_geometry('../images/tiles_50_M_KB_S2A_MSIL1C_20180928T022551_N0206_R046_T50MKB_20180928T060436.SAFE_GRANULE_L1C_T50MKB_A017060_20180928T024358_QI_DATA_MSK_CLOUDS_B00.gml')

# with rasterio.open('../images/original/T50MKB_20180928T022551_B12.tif') as src:
#     out_image, out_transform, _ = rasterio.mask.raster_geometry_mask(sr, cgeometry)
#     out_meta = src.meta
#     print(out_meta)
#     print(src.nodata)
#     print(src.bounds)
#     print(out_image.shape)
#     print(out_image.sum())
#     print(out_image.min())
#     print(out_image.max())

#     # print(src.read_mask(1))

# out_meta.update({"driver": "GTiff",
#                  "height": out_image.shape[0],
#                  "width": out_image.shape[1],
#                  "transform": out_transform})


# # out_image = out_image /10000.0
# with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
#     # dest.write( out_image )
#     dest.write_band(1, ((~out_image) * 255).astype(rasterio.uint16) )


# with rasterio.open('../images/original/T20LQH_20190821T142041_B12.tif') as src:
#     b12 = src.read(1)
#     print((b12 >= 65535).sum())

# geometry = get_gml_geometry('/home/minoro/Downloads/tiles_20_L_QH_S2A_MSIL1C_20190821T142041_N0208_R010_T20LQH_20190821T155923.SAFE_GRANULE_L1C_T20LQH_A021743_20190821T142039_QI_DATA_MSK_NODATA_B12.gml')
# print(geometry)



# with rasterio.open('../images/original/T20LQH_20190821T142041_B12_nodata.tif') as src:
    
#     b12 = src.read(1, masked=True)
#     meta = src.meta
#     print(meta)

#     msk = b12.mask
#     print(type(msk))
#     print(msk.shape)
#     print(msk.sum())

#     cv2.imwrite('nodata.png', (msk * 255))

    # print(src.read_mask(1))

# out_meta.update({"driver": "GTiff",
#                  "height": out_image.shape[0],
#                  "width": out_image.shape[1],
#                  "transform": out_transform})


# # # out_image = out_image /10000.0
# with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
#     # dest.write( out_image )
#     dest.write_band(1, ((~out_image) * 255).astype(rasterio.uint16) )





###########################
# buffered_stack = load_buffered_stack_bands(IMAGES_DIR, 'T20LQH_20190821T142041', (12,11,4))

# meta = buffered_stack.metas[12]
# print(meta)

# with rasterio.open('saida.tif', 'w', **meta) as dst:
#     dst.write_band(1, (buffered_stack.read(12) * 255).astype(rasterio.uint16))   
#     dst.write_band(2, (buffered_stack.read(11) * 255).astype(rasterio.uint16))   
#     dst.write_band(3, (buffered_stack.read(4) * 255).astype(rasterio.uint16))   

# cv2.imwrite('mask_b12.png', (buffered_stack.read_mask(12) * 255))
# cv2.imwrite('mask_b11.png', (buffered_stack.read_mask(11) * 255))
# cv2.imwrite('mask_b4.png', (buffered_stack.read_mask(4) * 255))

# cv2.imwrite('mask_final.png', (buffered_stack.read_mask() * 255))


# cv2.imwrite('saida_b12.png', (buffered_stack.read(12) * 255))
# cv2.imwrite('saida_b11.png', (buffered_stack.read(11) * 255))
# cv2.imwrite('saida_b4.png', (buffered_stack.read(4) * 255))

# cv2.imwrite('saida.png', (buffered_stack.read() * 255))









# geometry = get_gml_geometry('../images/tiles_50_M_KB_S2A_MSIL1C_20180928T022551_N0206_R046_T50MKB_20180928T060436.SAFE_GRANULE_L1C_T50MKB_A017060_20180928T024358_QI_DATA_MSK_CLOUDS_B00.gml')

# with rasterio.open('../images/original/T50MKB_20180928T022551_B12.tif') as src:
#     out_image, out_transform = rasterio.mask.mask(src, geometry, crop=False)
#     out_meta = src.meta
#     print(out_meta)
#     print(src.nodata)
#     print(src.bounds)
#     print(out_image.shape)
#     print(out_image.sum())
#     print(out_image.min())
#     print(out_image.max())

#     print(src.meta)
#     transform = src.transform
#     shape = src.shape[0], src.shape[1]


# print(transform)
# print(shape)

# mask = rasterio.features.geometry_mask(geometry, shape, transform)
# print(mask.shape)
# cv2.imwrite('clouds.png', mask*255)

#     # print(src.read_mask(1))

# out_meta.update({"driver": "GTiff",
#                  "height": out_image.shape[1],
#                  "width": out_image.shape[2],
#                  "transform": out_transform})


# out_image = out_image /10000.0
# with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
    # dest.write( out_image )
    # dest.write( (out_image * 255).astype(rasterio.uint16) )