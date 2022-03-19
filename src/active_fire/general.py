import numpy as np
from image.sentinel import BufferedImageStack
import importlib
from scipy import ndimage
import cv2
import joblib
from image.converter import reflectance_to_radiance, band_reflectance_to_radiance

class ActiveFireIndex:

    def __init__(self, method='baseline'):
        self.method = method
        self.algorithm = self.resolve_algorithm()
        # print(self.algorithm)

    def transform(self, buffered_stack : BufferedImageStack, *args, **kwargs):
        return self.algorithm.transform(buffered_stack, *args, **kwargs)

    def resolve_algorithm(self):
        """Instanciate the algorithm by name
        """

        module = importlib.import_module('active_fire.general', '.')
        algorithm = getattr(module, '{}AFI'.format(self.method))

        return algorithm()
        

class CicalaAFI:
    """Cicala et al, 2018
    DOI: 10.1109/EE1.2018.8385269
    """
    
    # def __init__(self, threshold=0.5):
    #     self.threshold = threshold


    def transform(self, buffered_stack : BufferedImageStack, metadata, alpha=0.5, th=5.0, **kwargs):
        # Transform the reflectance to radiance
        buffered_stack = reflectance_to_radiance(buffered_stack, metadata)
        
        afi = self.cicala_afi3(buffered_stack, alpha)
        afi = afi > th

        valid_data_mask =  buffered_stack.read_mask()

        return afi & valid_data_mask

    def cicala_baseline_afi(self, buffered_stack : BufferedImageStack):
       
        b12 = buffered_stack.read(12)
        b8 = buffered_stack.read('8A')
            
        return generalized_normalized_difference_index(b12, b8) > 0.5


    def cicala_afi3(self, buffered_stack : BufferedImageStack, alpha=0.001):
        b12 = buffered_stack.read(12)
        b11 = buffered_stack.read(11)
        b8 = buffered_stack.read('8A')
        
        t1 = generalized_normalized_difference_index(b12, b8)
        
        t2 = generalized_normalized_difference_index(b12, b11)
        
        t3 = generalized_normalized_difference_index(b8, b11)
        
        afi_index = t1 + t2 + ( alpha * t3 )
        
        return afi_index

class LiangrocapartAFI:
    """Liangrocapart et al, 2020
    DOI: 10.1109/ECTI-CON49241.2020.9158262
    """

    def transform(self, buffered_stack, **kwargs):
        return self.calculate_afi(buffered_stack)

    def calculate_afi(self, buffered_stack):
        
        b12 = buffered_stack.read(12)
        b11 = buffered_stack.read(11)
        b8 = buffered_stack.read('8A')

        ndi1 = normalized_difference_index(b11, b8)
        ndi2 = normalized_difference_index(b12, b11)

        hcf = self.high_temperature_crown_fire(b12, ndi1, ndi2)
        tcf = self.typical_crown_fire(b12, ndi1, ndi2)
        sma = self.smolder_area(b12, ndi1, ndi2)

        rma = self.remains_area(b12, ndi1, ndi2)

        valid_data_mask =  buffered_stack.read_mask()

        return (hcf | tcf | sma ) & valid_data_mask

    def high_temperature_crown_fire(self, b12, ndi1, ndi2, th=1.2):

        hcf = b12 > th
        positive_index = (ndi1 > ndi2) > 0

        return np.logical_and(hcf, positive_index)   

    def typical_crown_fire(self, b12, ndi1, ndi2, th=1.0):

        tcf = b12 > th
        return tcf


    def smolder_area(self, b12, ndi1, ndi2, lower_th=0.8, higher_th=1.0):

        sma = np.logical_and(b12 >= lower_th, b12 <= higher_th)
        return np.logical_and(sma, (ndi2 > 0.2))


    def remains_area(self, b12, ndi1, ndi2, th=0.8):
        """NOTE: Not used by transform method"""
        rma = b12 < th
        return np.logical_and(rma, (ndi1 > -0.27))


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

    def transform(self, buffered_stack, **kwargs):
        # b12 = buffered_stack.read_radiance(12)
        # b11 = buffered_stack.read_radiance(11)

        b12 = buffered_stack.read(12)
        b11 = buffered_stack.read(11)

        afi_zone2 = (b12 + b11) > (2.0 / self.sensitivity)


        valid_data_mask =  buffered_stack.read_mask()
        return afi_zone2 & valid_data_mask


class YongxueAFI:

    def transform(self, buffered_stack, **kwargs):
        
        b12 = buffered_stack.read(12)
        b11 = buffered_stack.read(11)
        b8 = buffered_stack.read('8A')

        # thermal anomaly index
        # tai = (b12 - b11) / b8
        tai = generalized_normalized_difference_index((b12 - b11), b8)

        # Step 1 - clip negative values
        tai_p = tai.copy()
        tai_p[ tai<0 ] = 0

        # Step 2 - compute the mean of tai_p in a 15x15 window 
        tai_mean = cv2.blur(tai_p, ksize=(15,15))

        # generate the initial segmentation mask
        segmentation = (tai_p - tai_mean) > 0.45

        # Step 3 - Create a buffer of 15-pixel around every pixel detected in the previous step
        structure  = np.ones((15,15))
        buffer = ndimage.morphology.binary_dilation(segmentation, structure=structure)
        
        # Step 4 - identify pixels with TAI >= 0.45 in the buffer (15-pixel)
        hta_pixels = ((tai >= 0.45) & buffer)

        # refine the detections
        hta_pixels = hta_pixels & ((b12 - b11) > (b11 - b8)) & (b12 > 0.15)

        # Take in consideration only the saturated pixels in a 8-pixel neighborhood of the previous identified pixels
        structure  = np.ones((3,3))
        buffer = ndimage.morphology.binary_dilation(hta_pixels, structure=structure)
        satured = (b12 >= 1) & (b11 >= 1)
        satured = buffer & satured
        
        # Combine the previous detecion with the satured pixels
        hta_pixels = hta_pixels | satured

        false_alarm_control = ~( (b11 <= 0.05) | (b8 <= 0.01) )
        valid_data_mask =  buffered_stack.read_mask()

        return hta_pixels & false_alarm_control & valid_data_mask

class KatoNakamuraAFI:

    def transform(self, buffered_stack, metadata, **kwargs):

        # if 'metadata' not in kwargs:
        #     raise ValueError('The metadata argument must be informed')

        b12 = buffered_stack.read(12)
        b11 = buffered_stack.read(11)
        b8 = buffered_stack.read('8A')

        mask = generalized_normalized_difference_index(b12, b8) > 5
        mask = (mask) & (b8 < 0.6) 

        l12 = band_reflectance_to_radiance(b12, 12, metadata)
        mask = mask & (l12 > 0.3)

        false_alarm_control = generalized_normalized_difference_index((b12 - b8),  (b11 - b8))
        false_alarm_control = (1.65 < false_alarm_control) & (false_alarm_control < 33)

        valid_data_mask = buffered_stack.read_mask()

        return mask & false_alarm_control & valid_data_mask


class MurphyAFI:
    

    def transform(self, buffered_stack, **kwargs):
        
        p7 = buffered_stack.read(12)
        p6 = buffered_stack.read(11)
        p5 = buffered_stack.read('8A')
        
        unamb_fires = ( generalized_normalized_difference_index(p7, p6) >= 1.4) & (generalized_normalized_difference_index(p7,p5) >= 1.4) & (p7 >= 0.15)
        
        if np.any (unamb_fires):
            neighborhood = cv2.dilate(unamb_fires.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))).astype(unamb_fires.dtype)

            saturated = (buffered_stack.get_saturated_mask(12)) | (buffered_stack.get_saturated_mask(11))
            potential_fires = (((generalized_normalized_difference_index(p6, p5) >= 2) & (p6 >= 0.5)) | saturated)
            potential_fires = potential_fires & neighborhood
            final_mask = (unamb_fires | potential_fires)
        else:
            final_mask = unamb_fires

        return (final_mask.astype(np.bool))


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