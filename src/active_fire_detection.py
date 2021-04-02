import numpy as np
from image import ImageStack
import importlib

class ActiveFireIndex:

    def __init__(self, method='dellaglio'):
        self.method = method
        self.algorithm = self.resolve_algorithm()
        print(self.algorithm)

    def transform(self, img_stack : ImageStack, **kwargs):
        return self.algorithm.transform(img_stack, **kwargs)

    def resolve_algorithm(self):
        """Instanciate the aldorithm by name
        """

        module = importlib.import_module('active_fire_detection', '.')
        algorithm = getattr(module, '{}AFI'.format(self.method.capitalize()))

        return algorithm()



class BaselineAFI:

    def transform(self, img_stack, **kwargs):
        b12 = img_stack.read_reflectance(12)
        b8 = img_stack.read_reflectance('8A')

        return generalized_normalized_difference_index(b12, b8)
        

class CicalaAFI:
    """Cicala et al, 2018
    DOI: 10.1109/EE1.2018.8385269
    """

    def transform(self, img_stack, **kwargs):
        return self.cicala_afi3(img_stack, kwargs['alpha'])

    def cicala_baseline_afi(self, img_stack : ImageStack):
       
        b12 = img_stack.read_reflectance(12)
        b8 = img_stack.read_reflectance('8A')

        return generalized_normalized_difference_index(b12, b8)


    def cicala_afi3(self, img_stack : ImageStack, alpha=0.5):
        b12 = img_stack.read_reflectance(12)
        b11 = img_stack.read_reflectance(11)
        b8 = img_stack.read_reflectance('8A')

        AFI_index = generalized_normalized_difference_index(b12, b8) + generalized_normalized_difference_index(b12, b11) + ( alpha * generalized_normalized_difference_index(b8, b11) )

        return AFI_index


class LiangrocapartAFI:
    """Liangrocapart et al, 2020
    DOI: 10.1109/ECTI-CON49241.2020.9158262
    """

    def transform(self, img_stack, **kwargs):
        return self.calculate_afi(img_stack)

    def calculate_afi(self, img_stack):
        
        b12 = img_stack.read_reflectance(12)
        b11 = img_stack.read_reflectance(11)
        b8 = img_stack.read_reflectance('8A')

        ndi1 = normalized_difference_index(b11, b8)
        ndi2 = normalized_difference_index(b12, b11)

        hcf = self.high_temperature_crown_fire(b12, ndi1, ndi2)
        tcf = self.typical_crown_fire(b12, ndi1, ndi2)
        sma = self.smolder_area(b12, ndi1, ndi2)

        return hcf  + tcf + sma

    def high_temperature_crown_fire(self, b12, ndi1, ndi2, th=1.2):

        hcf = b12 > th
        positive_index = np.logical_and((ndi1 > 0), (ndi2 > 0))
        ndi2_lt_ndi1 = (ndi2 < ndi1)

        return np.logical_and(np.logical_and(hcf, positive_index), ndi2_lt_ndi1)   

    def typical_crown_fire(self, b12, ndi1, ndi2, th=1.0):

        tcf = b12 > th
        ndi2_gt_ndi1 = (ndi2 > ndi1)

        return np.logical_and(tcf, ndi2_gt_ndi1)


    def smolder_area(self, b12, ndi1, ndi2, lower_th=0.5, higher_th=1.0):

        sma = np.logical_and(b12 >= lower_th, b12 <= higher_th)
        return np.logical_and(sma, (ndi2 > 0.2))


    def remains_area(self, b12, ndi1, ndi2, th=0.5):

        rma = b12 < th
        return np.logical_and(rma, (ndi1 > -0.27))


class DellaglioAFI:
    """Dell’aglio, D. A. G., Grambardella, C., Gargiulo, M., et al. (2020).
    Dell’aglio, D. et al. test 5 different AFI. The AFI 1 and 2 are covered in the CicallaAFI.
    The bests results are produced by Dell’aglio, D. et al. AFI5
    """

    def transform(self, img_stack, **kwargs):
        return self.calculate_afi5(img_stack)

    def calculate_afi5(self, img_stack : ImageStack):
        
        b12 = img_stack.read_reflectance(12)
        b11 = img_stack.read_reflectance(11)

        return generalized_normalized_difference_index(b12, b11)

    def calculate_afi3(self, img_stack, img_stack_10m):
        
        b12 = img_stack.read_reflectance(12)
        b8 = img_stack.read_reflectance(8, scale=0.5)

        return generalized_normalized_difference_index(b12, b8)

    def calculate_afi4(self, img_stack, img_stack_10m):

        b11 = img_stack.read_reflectance(11)
        b8 = img_stack.read_reflectance(8, scale=0.5)

        return generalized_normalized_difference_index(b11, b8)

class Dellaglio3AFI:

    def transform(self, img_stack, **kwargs):
        return self.calculate_afi3(img_stack, kwargs['img_stack_10m'])

    def calculate_afi3(self, img_stack, img_stack_10m):
        
        b12 = img_stack.read_reflectance(12)
        b8 = img_stack_10m.read_reflectance(8, scale=0.5)

        return generalized_normalized_difference_index(b12, b8)


class Dellaglio4AFI:


    def transform(self, img_stack, **kwargs):
        return self.calculate_afi4(img_stack, kwargs['img_stack_10m'])

    def calculate_afi4(self, img_stack, img_stack_10m):

        b11 = img_stack.read_reflectance(11)
        b8 = img_stack_10m.read_reflectance(8, scale=0.5)

        return generalized_normalized_difference_index(b11, b8)


class SahmAFI:
    """https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/active_fire_detection/script.js"""

    def transform(self, img_stack, **kwargs):
        b12 = img_stack.read_reflectance(12)
        b11 = img_stack.read_reflectance(11)

        afi = normalized_difference_index(b12, b11)

        return np.logical_or(afi > 0.4, b12 > 1.0)


class PierreMarkuseAFI:

    def __init__(self):
        self.sensitivity = 1.0

    def transform(self, img_stack, **kwargs):
        b12 = img_stack.read_reflectance(12)
        b11 = img_stack.read_reflectance(11)

        afi_zone2 = (b12 + b11) > (2.0 / self.sensitivity)

        return afi_zone2

def generalized_normalized_difference_index(b1, b2):
    """Compute de Generalized Normalized Difference Index, with is B1/B2
    """
    
    # avoid zero division
    p2 = np.where(b2 == 0, np.finfo(float).eps, b2)
    return b1/p2

    # return np.divide(b1, b2, out=np.zeros_like(b1, dtype=float), where=b2!=0)

def normalized_difference_index(b1, b2):
    """Compute the Normalised Difference Index. (b1 - b2) / (b1 + b2).
    """
    ndi = b1 - b2
    div = b1 + b2 
    div = np.where(div == 0, np.finfo(float).eps, div)

    ndi = ndi / div

    return ndi