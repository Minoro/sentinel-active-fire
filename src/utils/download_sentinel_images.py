
import sys

sys.path.append('../')

import geopandas as gpd
from datetime import datetime, timedelta  
from image.downloader import SentinelDownloader
from image.converter import convert_dir_jp2_to_tiff, get_cloud_mask
from image.sentinel import BufferedImageStack
from active_fire.general import ActiveFireIndex
import os
from glob import glob
import shutil
#  import rasterio

# import cv2

# Sentinel geodataframe with grid information
SAMPLES_SENTINEL_LAND_GRID_GEODATAFRAME = '../../resources/sentinel_grid/sentinel_land_grid.geojson'


DOWNLOAD_PATH = '../../resources/images/download/'
OUTPUT_PATH = '../../resources/images/original/'
OUTPUT_QI_DATA = '../../resources/images/qi_data/'

TEMP_PATH = '../../resources/images/tmp/'

START_DATE = '2020-08-01' # date included
END_DATE = '2020-09-01' # date NOT included

KEEP_FIRE_ONLY = True

SENTINEL_BANDS = ('B01','B02','B03','B04', 'B05','B06','B07','B08','B8A', 'B09','B10','B11','B12')
CLASSIFICATION_BANDS = ('B8A', 'B11', 'B12')
NON_CLASSIFICATION_BANDS = tuple(set(SENTINEL_BANDS) - set(CLASSIFICATION_BANDS))

# MODELS_PATH = []
ACTIVE_FIRE_ALGORITHMS = [
    {'method': 'Sahm'},
    {'method': 'PierreMarkuse'},
    {'method': 'Yongxue'},
]


def check_fire_in_tile(tile):

    img_buffer = BufferedImageStack()

    band_path = '{}_B12.tif'.format(tile)
    img_buffer.load_file_as_band(band_path, 12)

    band_path = '{}_B11.tif'.format(tile)
    img_buffer.load_file_as_band(band_path, 11)

    band_path = '{}_B8A.tif'.format(tile)
    img_buffer.load_file_as_band(band_path, '8A')

    gml_cloud_mask_path = os.path.join(tile.replace('tiff', 'qi_data'), 'MSK_CLOUDS_B00.gml')
    cloud_mask = get_cloud_mask(gml_cloud_mask_path, mask.shape, img_buffer.metas[12]['transform'])
    
    for algorithm in ACTIVE_FIRE_ALGORITHMS:

        afi = ActiveFireIndex(algorithm['method'])
        mask = afi.transform(img_buffer)
        print('{} - Num. fire pixels: '.format(algorithm['method'], num_fire_pixels))

        num_fire_pixels = (mask & cloud_mask).sum()
        if num_fire_pixels > 0:
            return True

    return False


def download_granules(downloader, granules, bands_to_download):
    for granule in granules:
        downloader.download_granules_to([granule], download_path, bands_to_download)
        downloader.download_granules_clouds_gml([granule], qi_data_path)
            
        # Convert to tiff
        convert_dir_jp2_to_tiff(download_path, tmp_tiff_path)
        
        if KEEP_FIRE_ONLY:
            print('[INFO] Checking for fire')
            has_fire = False
            
            tiles_bands = glob(os.path.join(tmp_tiff_path, '*.tif'))
            # Remove the band sufix from the name (_B12.tif, _B11.tif and _B8A.tif)
            tiles = [tile_band[:-8] for tile_band in tiles_bands]
            # Remove duplicates
            tiles = list(set(tiles))
            
            for tile in tiles:
                has_fire = check_fire_in_tile(tile)
                if has_fire:
                    # Download the others bands                        
                    downloader.download_granules_to([granule], download_path, NON_CLASSIFICATION_BANDS)
                    convert_dir_jp2_to_tiff(download_path, tmp_tiff_path)
                else:
                    # Remove files without fire
                    for file_path in glob(os.path.join(tmp_tiff_path, '*.tif')):
                        os.remove(file_path)
        
    

'''
Download the sentinel tiles and keep only the images with fire  
'''
if __name__ == '__main__':
    
    start_date = datetime.strptime(START_DATE, '%Y-%m-%d')
    end_date = datetime.strptime(END_DATE, '%Y-%m-%d')
    
    gdf = gpd.read_file(SAMPLES_SENTINEL_LAND_GRID_GEODATAFRAME)
    
    print(f'Num. tiles: {len(gdf)}')

    bands_to_download = None
    if KEEP_FIRE_ONLY:
        bands_to_download = CLASSIFICATION_BANDS
    
    downloader = SentinelDownloader()

    for index, row in gdf.iterrows():    
   
        download_path = os.path.join(TEMP_PATH, row['name'], 'jp2')
        tmp_tiff_path = os.path.join(TEMP_PATH, row['name'], 'tiff')
        qi_data_path = os.path.join(TEMP_PATH, row['name'], 'qi_data')

        os.makedirs(download_path, exist_ok=True)
        os.makedirs(tmp_tiff_path, exist_ok=True)
        os.makedirs(qi_data_path, exist_ok=True)
   
        granules = downloader.search_tile_name(row['name']) \
                    .search_dates(start_date, end_date) \
                    .get_granule_info()

        downloader.clear_search()

        num_granules = len(granules)
        print(f'[INFO] Num. Found: {num_granules}')
        download_granules(downloader, granules, bands_to_download)

        
