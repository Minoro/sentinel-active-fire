
STACK_20M_BANDS_MAP = {
    5: 1,
    6: 2,
    7: 3,
    8: 4,
    11: 5,
    12: 6
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


    def read(self, band):
        return self.dataset.read(self.map[band]) / QUANTIFICATION_VALUE

