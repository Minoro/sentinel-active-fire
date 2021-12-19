import rasterio
from rasterio.enums import Resampling
import numpy as np
import os
import sys
import cv2

from utils.reflectance_conversion import get_image_metadata, get_radiance

STACK_10M_BANDS_MAP = {
    2: 1,
    3: 2,
    4: 3,
    8: 4,
}

STACK_20M_BANDS_MAP = {
    5: 1,
    6: 2,
    7: 3,
    8: 4,
    '8A': 4,
    '8a': 4,
    11: 5,
    12: 6,
}

STACK_60M_BANDS_MAP = {
    1: 1,
    9: 2,
    10: 3,
}

QUANTIFICATION_VALUE = 10000.0
# QUANTIFICATION_VALUE = 65535.0

SATURATION_VALUE = 65535
NO_DATA_VALUE = 0

class ImageStack:

    def __init__(self, dataset, mtd_tl_xml = None, mtd_msil_xml = None):

        self.dataset = dataset
        self.meta = dataset.meta
        self.transform = dataset.transform
        self.mtd_tl_xml = mtd_tl_xml
        self.mtd_msil_xml = mtd_msil_xml

        # Map the channels
        self.map = STACK_20M_BANDS_MAP
        if self.meta['count'] == 3:
            self.map = STACK_60M_BANDS_MAP
        elif self.meta['count'] == 4:
            self.map = STACK_10M_BANDS_MAP

        # Load the image metadata if needed
        self.xml_metadata = {}
        if mtd_tl_xml is not None and mtd_msil_xml is not None:
            self.xml_metadata = get_image_metadata(self.mtd_tl_xml, self.mtd_msil_xml)

        self.masks = {}
        for stack_band, image_band in self.map.items():
            self.masks[stack_band] = self.dataset.read_masks(image_band)

    def read_raw(self, band):
        return self.dataset.read(self.map[band])

    def read(self, band):
        return self.dataset.read(self.map[band]) / QUANTIFICATION_VALUE

    def read_scaled(self, band, scale=1.0, resampling = None):
        
        if resampling is None:
            resampling = Resampling.nearest

        out_shape = (int(self.dataset.height * scale), int(self.dataset.width * scale))
        data = self.dataset.read(
            self.map[band],
            out_shape=(
                1,
                out_shape[0],
                out_shape[1]
            ),
            resampling=resampling
        )

        self.masks[band] = self.dataset.read_masks(self.map[band], out_shape=out_shape, resampling=resampling)

        if 'quantification_value' in self.xml_metadata:
            return data / self.xml_metadata['quantification_value']

        return data / QUANTIFICATION_VALUE

    def read_radiance(self, band, scale=1.0):
        if self.mtd_tl_xml is None or self.mtd_msil_xml is None:
            raise ValueError('The XML-metada must be informed')

        if scale != 1.0:
            band_value = self.read_scaled(band, scale)
        else:
            band_value = self.read(band) 
 
        return get_radiance(band_value, str(band), self.xml_metadata)
    

    def get_saturated(self):
        bands = np.zeros((self.dataset.height, self.dataset.width))

        for band_name in self.map:
            bands = np.logical_or(bands, self.read_raw(band_name) == SATURATION_VALUE)

        return bands
    


class BufferedImageStack:

    def __init__(self) -> None:
        self.buffer = {}
        self.metas = {}
        self.transform = None
        self.masks = {}

    def load_band_from_stack(self, img_stack : ImageStack, band, scale = 1.0):
        """Load a specific band to memory. 

        Args:
            img_stack (ImageStack): [description]
            band ([type]): [description]
            scale (float, optional): [description]. Defaults to 1.0.
        """
        self.transform = img_stack.transform
        if scale == 1.0:
            data = img_stack.read(band)
        else:
            data = img_stack.read_scaled(band, scale)
    
        self.metas[band] = img_stack.meta
        self.buffer[band] = data
        
        # store the valid pixel max as boolean
        self.masks[band] = (img_stack.masks[band] > 0)

    def read(self, band = None):
        """Read a band loaded in memory.
        If a the band is not informed (band=None), it will load all bands with cannels-last

        Args:
            band (mixed, optional): band identifier. Defaults to None.

        Returns:
            np.array: band value
        """
        if band is None:
            return np.moveaxis(np.array([*self.buffer.values()]), 0, -1)
        
        if band not in self.buffer:
            return None
        
        return self.buffer[band]

    def read_mask(self, band = None):
        if band is not None:
            mask = self.masks[band]
        else:
            first_band = next(iter(self.buffer))
            mask = np.ones(self.buffer[first_band].shape, dtype=bool)
            for b in self.buffer:
                band_mask = self.masks[b]
                # cv2.imwrite('../mask_b{}.png'.format(b), (band_mask*255)) 
                mask = mask & band_mask

        return mask

    def set_band(self, band_number, band_value):
        self.buffer[band_number] = band_value

    def apply_valid_data_mask_to_stack(self):
        msk = self.read_mask()    
        for band in self.buffer:
            band_value = self.read(band)
            band_value = band_value*msk
            self.set_band(band, band_value)

    def get_center_coord_band(self, band=None):
        if band is None:
            key = next(iter(self.metas))
        else:
            key = band
        meta = self.metas[key]
        return rasterio.transform.xy(meta['transform'], meta['height'] // 2, meta['width'] // 2)

            

def load_buffered_stack_bands(image_dir, stack_partial_name, bands, spatial_resolution=20):
    """Load the bands from the image stacks. Each stack has all bands of a spacial resolution.
    The loaded bands will be resampled to a specified spacial resolution.

    Args:
        image_dir (str): Base path to the image stacks
        stack_partial_name (str): Partial name of the stack, without the spacial resolution sufix
        bands (tuple): bands to load.
        spatial_resolution (int, optional): spacial resolution to resample the bands. Defaults to 20.

    Returns:
        BufferedImageStack: bands loaded
    """

    assert spatial_resolution == 10 or spatial_resolution == 20 or spatial_resolution == 60
    
    buffered_stack = BufferedImageStack()

    stack_partial_name = stack_partial_name.replace('.tif', '')
    stack_10m = os.path.join(image_dir, '{}_10m_stack.tif'.format(stack_partial_name))            
    stack_20m = os.path.join(image_dir, '{}_20m_stack.tif'.format(stack_partial_name))            
    stack_60m = os.path.join(image_dir, '{}_60m_stack.tif'.format(stack_partial_name))            

    for band in bands:
        scale = 1.0
        image_path = stack_20m
        if band in STACK_10M_BANDS_MAP:
            scale = 10/spatial_resolution
            image_path = stack_10m

        elif band in STACK_60M_BANDS_MAP:
            scale = 60/spatial_resolution
            image_path = stack_60m
        
        with rasterio.open(image_path) as src:
            img_stack = ImageStack(src)
            buffered_stack.load_band_from_stack(img_stack, band, scale=scale)

    # mask the bands to use only valid values
    # msk = buffered_stack.read_mask()    
    # for band in bands:
    #     band_value = buffered_stack.read(band)
    #     band_value = band_value*msk
    #     buffered_stack.set_band(band, band_value)
        
        # print('Band {}: {}'.format(band, band_value.sum()))

    return buffered_stack