from rasterio.enums import Resampling

from utils.reflectance_conversion import get_conversion_metadata, get_radiance

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
    11: 5,
    12: 6,
}

STACK_60M_BANDS_MAP = {
    1: 1,
    9: 2,
    10: 3,
}

QUANTIFICATION_VALUE = 10000.0

class ImageStack:

    def __init__(self, dataset, mtd_tl_xml = None, mtd_msil_xml = None):

        self.dataset = dataset
        self.meta = dataset.meta
        self.mtd_tl_xml = mtd_tl_xml
        self.mtd_msil_xml = mtd_msil_xml

        self.map = STACK_20M_BANDS_MAP

        if self.meta['count'] == 3:
            self.map = STACK_60M_BANDS_MAP
        elif self.meta['count'] == 4:
            self.map = STACK_10M_BANDS_MAP

    def read(self, band):
        return self.dataset.read(self.map[band])

    def read_scaled(self, band, scale=1.0):
    
        data = self.dataset.read(
            self.map[band],
            out_shape=(
                1,
                int(self.dataset.height * scale),
                int(self.dataset.width * scale)
            ),
            resampling=Resampling.bilinear
        )

        return data

    def read_reflectance(self, band, scale=1.0):
        if self.mtd_tl_xml is None or self.mtd_msil_xml is None:
            raise ValueError('The XML-metada must be informed')

        if scale != 1.0:
            band_value = self.read_scaled(band, scale)
        else:
            band_value = self.read(band) 
            
        conversion_metadata = get_conversion_metadata(self.mtd_tl_xml, self.mtd_msil_xml)
        return get_radiance(band_value, str(band), conversion_metadata)
        