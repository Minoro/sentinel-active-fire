import rasterio
import gdal
import os
import glob
from osgeo import ogr
import json
import numpy as np
import rasterio
from rasterio.mask import mask
from PIL import Image, ImageDraw


IMAGES_DIR = "../../images/original/"
STACK_DIR = '../../images/stack/'

def convert_dir_jp2_to_tiff(input_dir, output_dir = None):
    files = glob.glob(os.path.join(input_dir, '*.jp2'))

    num_files = len(files)
    for index, file in enumerate(files, start=1):
        print('{}/{} - {}'.format(index, num_files, file))
        jp2_to_tiff(file, output_dir)


def jp2_to_tiff(file, output_path = None):
    dst_dataset = file.replace('.jp2', '.tif')
    
    if output_path is not None:
        file_name = os.path.basename(dst_dataset)
        dst_dataset = os.path.join(output_path, file_name)

    if not os.path.exists(dst_dataset):
        try:
            dataset = gdal.Open(file)
            ds1 = gdal.Translate(dst_dataset, dataset)
            
        except Exception as e:
            print(e)

def build_stacks(rootPath, output_path):
        
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pattern = '*_B01.jp2'

    files = glob.glob(rootPath + pattern)
    files.sort()


    for ind,file in enumerate(files):


        print(str(ind+1) + '/' + str(len(files)), '-', file[:-8])
        
        file_list10 = []
        file_list20 = []
        file_list60 = []
        files_out = []
        
        file_out = file[:-8] + '_10m_stack.tif'
        file_out = file_out.replace(rootPath, output_path)

        if not os.path.exists(file_out):

            jp2_to_tiff(file[:-8] + '_B02.jp2')
            jp2_to_tiff(file[:-8] + '_B03.jp2')
            jp2_to_tiff(file[:-8] + '_B04.jp2')
            jp2_to_tiff(file[:-8] + '_B08.jp2')

            
            file_list10.append(file[:-8] + '_B02.tif')
            file_list10.append(file[:-8] + '_B03.tif')
            file_list10.append(file[:-8] + '_B04.tif')
            file_list10.append(file[:-8] + '_B08.tif')
            
    #         files_out.append(file_out)
            
            with rasterio.open(file_list10[0]) as src0:
                meta = src0.meta
                meta.update(count = len(file_list10))
                meta.update(nodata = 0)                


            with rasterio.open(file_out, 'w', **meta) as dst:
                for i, layer in enumerate(file_list10, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(i, src1.read(1).astype(rasterio.uint16))   
                                
            
            
            
        file_out = file[:-8] + '_20m_stack.tif'
        file_out = file_out.replace(rootPath, output_path)
        if not os.path.exists(file_out):

            jp2_to_tiff(file[:-8] + '_B05.jp2')
            jp2_to_tiff(file[:-8] + '_B06.jp2')
            jp2_to_tiff(file[:-8] + '_B07.jp2')
            jp2_to_tiff(file[:-8] + '_B8A.jp2')
            jp2_to_tiff(file[:-8] + '_B11.jp2')
            jp2_to_tiff(file[:-8] + '_B12.jp2')
            
            file_list20.append(file[:-8] + '_B05.tif')
            file_list20.append(file[:-8] + '_B06.tif')
            file_list20.append(file[:-8] + '_B07.tif')
            file_list20.append(file[:-8] + '_B8A.tif')
            file_list20.append(file[:-8] + '_B11.tif')
            file_list20.append(file[:-8] + '_B12.tif')
            
    #         files_out.append(file_out)
            
            with rasterio.open(file_list20[0]) as src0:
                meta = src0.meta
                meta.update(count = len(file_list20))          
                meta.update(nodata = 0)                

            with rasterio.open(file_out, 'w', **meta) as dst:
                for i, layer in enumerate(file_list20, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(i, src1.read(1).astype(rasterio.uint16))   
                        
            
        
        
        
        file_out = file[:-8] + '_60m_stack.tif'
        file_out = file_out.replace(rootPath, output_path)
        if not os.path.exists(file_out):

            jp2_to_tiff(file[:-8] + '_B01.jp2')
            jp2_to_tiff(file[:-8] + '_B09.jp2')
            jp2_to_tiff(file[:-8] + '_B10.jp2')

            
            file_list60.append(file[:-8] + '_B01.tif')
            file_list60.append(file[:-8] + '_B09.tif')
            file_list60.append(file[:-8] + '_B10.tif')
            
    #         files_out.append(file_out)
            
            with rasterio.open(file_list60[0]) as src0:
                meta = src0.meta
                meta.update(count = len(file_list60))                
                meta.update(nodata = 0)                

            with rasterio.open(file_out, 'w', **meta) as dst:
                for i, layer in enumerate(file_list60, start=1):
                    with rasterio.open(layer) as src1:
                        dst.write_band(i, src1.read(1).astype(rasterio.uint16))  



def get_gml_geometry(gml_file):
    reader = ogr.Open(gml_file)
    layer = reader.GetLayer()
    if layer is None:
        return None

    geometries = []
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        # yield json.loads(feature.ExportToJson())
        # print(feature.ExportToJson())
        json_feature = json.loads(feature.ExportToJson())
        # print(type(json_feature['geometry']))
        geometries.append(json_feature['geometry'])
    
    return geometries


def get_cloud_mask(gml_cloud_mask_path, mask_shape, transform):
    cloud_geometry = get_gml_geometry(gml_cloud_mask_path)
    cloud_mask = np.ones(mask_shape, dtype=np.bool)
    if cloud_geometry is not None:
        cloud_mask = rasterio.features.geometry_mask(cloud_geometry, mask_shape, transform)

    return cloud_mask

# if __name__ == '__main__':
    # convert_dir_jp2_to_tiff(IMAGES_DIR)
    # build_stack(IMAGES_DIR, STACK_DIR)
    # print(dir(gdal.Translate))
    # gml_to_geojson('../../images/tiles_50_M_KB_S2A_MSIL1C_20180928T022551_N0206_R046_T50MKB_20180928T060436.SAFE_GRANULE_L1C_T50MKB_A017060_20180928T024358_QI_DATA_MSK_CLOUDS_B00.gml', '')