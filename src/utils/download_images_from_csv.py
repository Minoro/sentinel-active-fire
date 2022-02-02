
import sys

sys.path.append('../')

import os
import pandas as pd
import requests
import numpy as np
from glob import glob
import shutil
from joblib import Parallel, delayed


from active_fire.general import ActiveFireIndex
from image.sentinel import BufferedImageStack
from image.converter import convert_dir_jp2_to_tiff, get_cloud_mask

N_JOBS = 2

INPUT_CSV = '../../resources/s2_continent_202008_list_to_download.csv'

DOWNLOAD_PATH = '../../resources/images/download/'
OUTPUT_PATH = '../../resources/images/original/'
OUTPUT_QI_DATA = '../../resources/images/qi_data/'

LOG_PATH = os.path.join(DOWNLOAD_PATH, 'log')


SENTINEL_BANDS = ('B01','B02','B03','B04', 'B05','B06','B07','B08','B8A', 'B09','B10','B11','B12')
CLASSIFICATION_BANDS = ('B8A', 'B11', 'B12')
NON_CLASSIFICATION_BANDS = tuple(set(SENTINEL_BANDS) - set(CLASSIFICATION_BANDS))


ACTIVE_FIRE_ALGORITHMS = [
    {'method': 'Liangrocapart'},
]


def check_fire_in_tile(tiff_path, cloud_path=None):

    # Load the bands to a buffer
    bands_files = glob(os.path.join(tiff_path, '*.tif'))
    img_buffer = BufferedImageStack()
    for band_file in bands_files:
        band = 12
        if band_file.endswith('_B11.tif'):
            band = 11
        elif band_file.endswith('_B8A.tif'):
            band = '8A'
        
        img_buffer.load_file_as_band(band_file, band)

    image_shape = img_buffer.read(12).shape
    cloud_mask = np.ones(image_shape, dtype=np.bool) 
    

    if cloud_path is not None and os.path.exists(cloud_path):
        cloud_mask = get_cloud_mask(cloud_path, image_shape, img_buffer.metas[12]['transform'])
    
    for algorithm in ACTIVE_FIRE_ALGORITHMS:
        try:
            afi = ActiveFireIndex(algorithm['method'])
            mask = afi.transform(img_buffer)
            num_fire_pixels = (mask & cloud_mask).sum()

            # print('{} - Num. fire pixels: {}'.format(algorithm['method'], num_fire_pixels))

            if num_fire_pixels > 0:
                return True
        except Exception as e:
            # Abort in case of error
            return False

    return False

def download_mask_cloud(file):

    try:
        tile_name = os.path.basename(file)
        
        # Download the cloud mask
        cloud_url = file.replace('IMG_DATA', 'QI_DATA').replace(tile_name, 'MSK_CLOUDS_B00.gml')
        response = requests.get(cloud_url)
        cloud_folder = os.path.join(OUTPUT_QI_DATA, tile_name)
        os.makedirs(cloud_folder, exist_ok=True)
        cloud_file = os.path.join(cloud_folder, 'MSK_CLOUDS_B00.gml') 
        
        with open(cloud_file, 'wb') as f:
            f.write(response.content)

        return cloud_file

    except Exception as e:
        log_file = os.path.join(LOG_PATH, '{}_MSK_CLOUDS_B00.log'.format(tile_name))
        with open(log_file, 'w+') as f:
            f.write(str(e))

    return None


def download_sentinel_bands(file, bands_to_download):
    tile_name = os.path.basename(file)
    
    try:     
        download_path = os.path.join(DOWNLOAD_PATH, tile_name)
        for band in bands_to_download:
            band_url = '{}_{}.jp2'.format(file, band)
            file_name = os.path.basename(band_url)
            log_file = os.path.join(LOG_PATH, file_name.replace('.jp2', '.txt'))

            if os.path.exists(log_file):
                continue
            
            response = requests.get(band_url)
            output_file = os.path.join(download_path, file_name)
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            with open(log_file, 'w+') as f:
                f.write('200\n')

        
        return download_path

    except Exception as e:
        log_file = os.path.join(LOG_PATH, '{}.log'.format(tile_name))
        with open(log_file, 'w+') as f:
            f.write(str(e))

    return None


def download_file(file):
    tile_name = os.path.basename(file)
    log_file = os.path.join(LOG_PATH, '{}.log'.format(tile_name))
    
    # Get the grid name. Ex: T01GEM
    grid_name = tile_name.split('_')[0] 

    # Ignore if the tile has already been processed (log exists)
    if os.path.exists(log_file):
        return
    
    # Ignore if there are another image of the grid with fire
    if grid_name in fire_grid:
        with open(log_file, 'w+') as f:
            f.write('TILE-FIRE')
        return

    tmp_dir = os.path.join(DOWNLOAD_PATH, tile_name, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # Download the cloud mask
    cloud_file = download_mask_cloud(file)
    download_path = download_sentinel_bands(file, CLASSIFICATION_BANDS)
    convert_dir_jp2_to_tiff(download_path, tmp_dir, verbose=False)
        
    has_fire = check_fire_in_tile(tmp_dir, cloud_path=cloud_file)

    if has_fire:
        # Save 
        fire_grid.append(grid_name)

        # Download the missing bands
        download_path = download_sentinel_bands(file, NON_CLASSIFICATION_BANDS)
        
        # Convert the bands to TIFF
        convert_dir_jp2_to_tiff(download_path, tmp_dir, verbose=False)

        # Move the downloaded files to the destination
        downloaded_files = glob(os.path.join(tmp_dir, '*.tif'))
        for downloaded_file in downloaded_files:
            file_name = os.path.basename(downloaded_file)
            shutil.move(downloaded_file, os.path.join(OUTPUT_PATH, file_name))

        with open(log_file, 'w+') as f:
            f.write('FIRE')

    else:
        # remove the cloud mask file
        os.remove(cloud_file)
        # Register that the image was processed
        with open(log_file, 'w+') as f:
            f.write('NO-FIRE')

    # Delete the temporary files
    shutil.rmtree(download_path)

if __name__ == '__main__':
    
    fire_grid = []

    os.makedirs(DOWNLOAD_PATH, exist_ok=True)
    os.makedirs(OUTPUT_QI_DATA, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)

    df = pd.read_csv(INPUT_CSV,  sep=';')
    files = df.url.unique()
    files = [f for f in files if str(f) != 'nan']
    files.sort()


    # for ind, file in enumerate(files[:1]):
    #     download_file(file)
        
    Parallel(n_jobs=N_JOBS, verbose=100)(delayed(download_file)(file) for file in files)
