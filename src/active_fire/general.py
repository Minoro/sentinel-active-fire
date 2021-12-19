import numpy as np
from image.sentinel import BufferedImageStack
import importlib
import cv2
from scipy import ndimage

class ActiveFireIndex:

    def __init__(self, method='baseline'):
        self.method = method
        self.algorithm = self.resolve_algorithm()
        # print(self.algorithm)

    def transform(self, img_stack : BufferedImageStack, **kwargs):
        return self.algorithm.transform(img_stack, **kwargs)

    def resolve_algorithm(self):
        """Instanciate the algorithm by name
        """

        module = importlib.import_module('active_fire.general', '.')
        algorithm = getattr(module, '{}AFI'.format(self.method))

        return algorithm()



class BaselineAFI:

    def transform(self, buffered_stack : BufferedImageStack, **kwargs):
        # b12 = img_stack.read_radiance(12)
        # b8 = img_stack.read_radiance('8A')
        b12 = buffered_stack.read(12) 
        b8 = buffered_stack.read('8A') 

        afi = normalized_difference_index(b12, b8)
        afi = afi > (afi.mean() + 3*afi.std())  
        valid_data_mask =  buffered_stack.read_mask()
        
        return  afi & valid_data_mask
        

class CicalaAFI:
    """Cicala et al, 2018
    DOI: 10.1109/EE1.2018.8385269
    """
    
    # def __init__(self, threshold=0.5):
    #     self.threshold = threshold


    def transform(self, img_stack : BufferedImageStack, alpha=0.5, **kwargs):
        return self.cicala_afi3(img_stack, alpha)

    def cicala_baseline_afi(self, img_stack : BufferedImageStack):
       
        b12 = img_stack.read(12)
        b8 = img_stack.read('8A')
            
        return generalized_normalized_difference_index(b12, b8) > 0.5


    def cicala_afi3(self, img_stack : BufferedImageStack, alpha=0.001):
        b12 = img_stack.read(12)
        b11 = img_stack.read(11)
        b8 = img_stack.read('8A')
        
        t1 = generalized_normalized_difference_index(b12, b8)
        
        t2 = generalized_normalized_difference_index(b12, b11)
        
        t3 = generalized_normalized_difference_index(b8, b11)
        
        afi_index = t1 + t2 + ( alpha * t3 )
        
        return afi_index

class LiangrocapartAFI:
    """Liangrocapart et al, 2020
    DOI: 10.1109/ECTI-CON49241.2020.9158262
    """

    def transform(self, img_stack, **kwargs):
        return self.calculate_afi(img_stack)

    def calculate_afi(self, img_stack):
        
        # b12 = img_stack.read_radiance(12)
        # b11 = img_stack.read_radiance(11)
        # b8 = img_stack.read_radiance('8A')

        b12 = img_stack.read(12)
        b11 = img_stack.read(11)
        b8 = img_stack.read('8A')

        ndi1 = normalized_difference_index(b11, b8)
        ndi2 = normalized_difference_index(b12, b11)

        hcf = self.high_temperature_crown_fire(b12, ndi1, ndi2)
        tcf = self.typical_crown_fire(b12, ndi1, ndi2)
        sma = self.smolder_area(b12, ndi1, ndi2)

        return hcf | tcf | sma

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

    def __init__(self, threshold=0.5, version=5):
        self.threshold = threshold
        self.version = version

    def transform(self, img_stack, **kwargs):

        if self.version == 5:
            return self.calculate_afi5(img_stack) > self.threshold
        elif self.version == 4:
            return self.calculate_afi4(img_stack) > self.threshold
        elif self.version == 3:
            return self.calculate_afi3(img_stack, kwargs['img_stack_10m']) > self.threshold
        elif self.version == 2:
            return self.calculate_afi2(img_stack) > self.threshold
        elif self.version == 1:
            return self.calculate_afi1(img_stack) > self.threshold
        else:
            raise ValueError('Version unknown')

    def calculate_afi1(self, img_stack, alpha=0.001):
        b12 = img_stack.read(12)
        b11 = img_stack.read(11)
        b8 = img_stack.read('8A')

        t1 = generalized_normalized_difference_index(b12, b8)
        t2 = generalized_normalized_difference_index(b12, b11)
        t3 = generalized_normalized_difference_index(b8, b11)

        return t1 + t2 + (alpha * t3)

    def calculate_afi2(self, img_stack):
        b12 = img_stack.read(12)
        b8 = img_stack.read('8A')

        return generalized_normalized_difference_index(b12, b8)


    def calculate_afi3(self, img_stack, img_stack_10m):
        
        # b12 = img_stack.read_radiance(12)
        # b8 = img_stack.read_radiance(8, scale=0.5)

        b12 = img_stack.read(12)
        b8 = img_stack.read(8)

        return generalized_normalized_difference_index(b12, b8)

    def calculate_afi4(self, img_stack, img_stack_10m):

        b11 = img_stack.read_radiance(11)
        b8 = img_stack.read_radiance(8, scale=0.5)

        return generalized_normalized_difference_index(b11, b8)

    def calculate_afi5(self, img_stack : BufferedImageStack):
        
        # b12 = img_stack.read_radiance(12)
        # b11 = img_stack.read_radiance(11)

        b12 = img_stack.read(12)
        b11 = img_stack.read(11)

        return generalized_normalized_difference_index(b12, b11) 

    

class Dellaglio3AFI:

    def transform(self, img_stack, **kwargs):
        return self.calculate_afi3(img_stack, kwargs['img_stack_10m'])

    def calculate_afi3(self, img_stack, img_stack_10m):
        
        b12 = img_stack.read_radiance(12)
        b8 = img_stack_10m.read_radiance(8, scale=0.5)

        return generalized_normalized_difference_index(b12, b8)


class Dellaglio4AFI:


    def transform(self, img_stack, **kwargs):
        return self.calculate_afi4(img_stack, kwargs['img_stack_10m'])

    def calculate_afi4(self, img_stack, img_stack_10m):

        b11 = img_stack.read_radiance(11)
        b8 = img_stack_10m.read_radiance(8, scale=0.5)

        return generalized_normalized_difference_index(b11, b8)



class Dellaglio5AFI:
    """Dell’aglio, D. A. G., Grambardella, C., Gargiulo, M., et al. (2020).
    Dell’aglio, D. et al. test 5 different AFI. The AFI 1 and 2 are covered in the CicallaAFI.
    The bests results are produced by Dell’aglio, D. et al. AFI5
    """

    def transform(self, img_stack, **kwargs):
        return self.calculate_afi5(img_stack)

    def calculate_afi5(self, img_stack : BufferedImageStack):
        
        # b12 = img_stack.read_radiance(12)
        # b11 = img_stack.read_radiance(11)

        b12 = img_stack.read(12)
        b11 = img_stack.read(11)

        return generalized_normalized_difference_index(b12, b11)

class SahmAFI:
    """https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/active_fire_detection/script.js"""

    def transform(self, buffered_stack : BufferedImageStack, **kwargs):
        b12 = buffered_stack.read(12) 
        b11 = buffered_stack.read(11) 

        afi = normalized_difference_index(b12, b11)

        valid_data_mask =  buffered_stack.read_mask()

        return np.logical_or(afi > 0.4, b12 > 1.0) & valid_data_mask


class PierreMarkuseAFI:
    """https://pierre-markuse.net/2018/04/30/visualizing-wildfires-burn-scars-sentinel-hub-eo-browser/"""

    def __init__(self):
        self.sensitivity = 1.0

    def transform(self, img_stack, **kwargs):
        # b12 = img_stack.read_radiance(12)
        # b11 = img_stack.read_radiance(11)

        b12 = img_stack.read(12)
        b11 = img_stack.read(11)

        afi_zone2 = (b12 + b11) > (2.0 / self.sensitivity)


        valid_data_mask =  img_stack.read_mask()
        return afi_zone2 & valid_data_mask


class YongxueAFI:

    def transform(self, buffered_stack, **kwargs):
        
        b12 = buffered_stack.read(12)
        b11 = buffered_stack.read(11)
        b8 = buffered_stack.read('8A')

        # thermal anomaly index
        # tai = (b12 - b11) / b8
        tai = generalized_normalized_difference_index((b12 - b11), b8)

        # Step 1 - consider only non-zero values
        tai_p = tai.copy()
        tai_p[ tai<0 ] = 0

        # Step 2 - compute the mean of tai_p in a 15x15 window 
        kernel = np.ones((15,15)) 
        kernel = kernel / kernel.size
        tai_mean = ndimage.convolve(tai_p, weights=kernel) 

        # generate the initial segmentation mask
        segmentation = (tai_p - tai_mean) > 0.45

        # Step 3 - Create a buffer of 15-pixel around every pixel detected in the previous step
        structure  = np.ones((15,15))
        buffer = ndimage.morphology.binary_dilation(segmentation, structure=structure)
        
        # Step 4 - identify pixels with TAI >= 0.45 in the buffer (15-pixel)
        hta_pixels = ((tai >= 0.45) & buffer)

        # refine the detections
        hta_pixels = hta_pixels & ((b12 - b11) > (b11 - b8)) & (b12 > 0.15)

        # Take in consideration only the satured pixels in a 8-pixel neighborhood of the previous identified pixels
        structure  = np.ones((3,3))
        buffer = ndimage.morphology.binary_dilation(hta_pixels, structure=structure)
        satured = (b12 >= 1) & (b11 >= 1)
        satured = buffer & satured

        # Combine the privous detecion with the satured pixels
        hta_pixels = hta_pixels | satured

        false_alarm_control = ~( (b11 <= 0.05) | (b8 <= 0.01) )
        valid_data_mask =  buffered_stack.read_mask()

        return hta_pixels & false_alarm_control & valid_data_mask




def generalized_normalized_difference_index(b1, b2):
    """Compute de Generalized Normalized Difference Index, with is B1/B2
    """
    
    # avoid zero division
    # p2 = np.where(b2 == 0, np.finfo(float).eps, b2)
    # return b1/p2
    
    return np.divide(b1, b2, out=np.zeros_like(b1, dtype=np.float), where=b2!=0)

def normalized_difference_index(b1, b2):
    """Compute the Normalised Difference Index. (b1 - b2) / (b1 + b2).
    """
    ndi = b1 - b2
    div = b1 + b2 
    div = np.where(div == 0, np.finfo(float).eps, div)

    ndi = ndi / div

    return ndi