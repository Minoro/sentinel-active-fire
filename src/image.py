from rasterio.enums import Resampling

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

    def __init__(self, dataset):

        self.dataset = dataset
        self.meta = dataset.meta
        self.map = STACK_20M_BANDS_MAP

        if self.meta['count'] == 3:
            self.map = STACK_60M_BANDS_MAP
        elif self.meta['count'] == 4:
            self.map = STACK_10M_BANDS_MAP

    def read(self, band):
        return self.dataset.read(self.map[band]) / QUANTIFICATION_VALUE

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

        return data / QUANTIFICATION_VALUE