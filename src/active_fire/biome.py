import importlib
from image.sentinel import BufferedImageStack, load_buffered_stack_bands
import rasterio

class BiomeAFD:

    def criteria_1(self, b4, b12, coefficient_a, coefficient_b):
        c1 = (b4 <= (coefficient_a * b12 + coefficient_b))
        return c1

    def criteria_2(self, img, value):
        c2 = img >= value
        return c2

    def criteria_3(self, b11, b12, coefficient_c=0.0,  coefficient_d=1.0):
        c3 = (b11 >= coefficient_c) | (b12 >= coefficient_d)
        return c3

class TropicalMoistForestAFD(BiomeAFD):

    def transform(self, buffered_img : BufferedImageStack):
        """Generate an active fire detection mask for Tropical & Subtropical Moist Broadleaf Forests
        C1 = B4 <= (1.045 * B12) - 0.071
        C2 = B12/B11 >= 1
        mask = C1 AND C2

        Args:
            buffered_img (BufferedImageStack): Image buffer with the bands 4, 11 and 12 loaded

        Returns:
            np.array: active fire mask
        """
        b4 = buffered_img.read(4)
        b11 = buffered_img.read(11)
        b12 = buffered_img.read(12)

        c1 = super().criteria_1(b4, b12, 1.045, -0.071)
        c2 = super().criteria_2(b12/b11, 1)

        return (c1 & c2)


class TropicalDryForestAFD(BiomeAFD):
    
    def transform(self, buffered_img : BufferedImageStack):
        """Generate an activa fire detection mask for Tropical & Subtropical Dry Broadleaf Forests
        mask = B4 <= (0.681 * B12) - 0.071

        Args:
            buffered_img (BufferedImageStack): Image buffer with the bands 4 and 12 loaded

        Returns:
            np.array: active fire mask
        """
        b4 = buffered_img.read(4)
        b12 = buffered_img.read(12)

        c1 = super().criteria_1(b4, b12, 0.681, -0.052)
        return c1

class SavannaAFD(BiomeAFD):

    def transform(self, buffered_img : BufferedImageStack):
        """Generate an activa fire detection mask for Tropical & Subtropical Grassland, Savannas & Shrublands
        mask = B4 <= (0.677 * B12) - 0.052

        Args:
            buffered_img (BufferedImageStack): Image buffer with the bands 4 and 12 loaded

        Returns:
            np.array: active fire mask
        """
        b4 = buffered_img.read(4)
        b12 = buffered_img.read(12)
        
        c1 = super().criteria_1(b4, b12, 0.677, -0.052)
        return c1

class MediterraneanForestAFD(BiomeAFD):
    
    def transform(self, buffered_img : BufferedImageStack):
        """Generate an activa fire detection mask for Mediterranean Forests, Woodlands & Scrub
        C1 = B4 <= (0.743 * B12) - 0.068
        C2 = B12 >= 0.355
        C3 = (B11 >= 0.475) OR (B12 >= 1.0)
        mask = C1 AND C2 AND C3

        Args:
            buffered_img (BufferedImageStack): Image buffer with the bands 4, 11 and 12 loaded

        Returns:
            np.array: active fire mask
        """
        b4 = buffered_img.read(4)
        b11 = buffered_img.read(11)
        b12 = buffered_img.read(12)

        c1 = super().criteria_1(b4, b12, 0.743, -0.068)
        c2 = super().criteria_2(b12, 0.355)
        c3 = super().criteria_3(b11, b12, 0.475, 1.0)
        meta = buffered_img.metas[12]
        meta.update(count=1)

        return (c1 & c2 & c3)


class ConiferForestAFD(BiomeAFD):

    def transform(self, buffered_img : BufferedImageStack):
        """Generate an activa fire detection mask for Temperate Conifer Forests
        mask = B4 <= (0.504 * B12) - 0.198

        Args:
            buffered_img (BufferedImageStack): Image buffer with the bands 4 and 12 loaded

        Returns:
            np.array: active fire mask
        """
        b4 = buffered_img.read(4)
        b12 = buffered_img.read(12)
        
        c1 = super().criteria_1(b4, b12, 0.504, -0.198)
        return c1

    
class TaigaAFD(BiomeAFD):

    def transform(self, buffered_img : BufferedImageStack):
        """Generate an activa fire detection mask for Boreal Forests/Taiga
        mask = B4 <= (0.727 * B12) - 0.11

        Args:
            buffered_img (BufferedImageStack): Image buffer with the bands 4 and 12 loaded

        Returns:
            np.array: active fire mask
        """
        b4 = buffered_img.read(4)
        b12 = buffered_img.read(12)
        
        c1 = super().criteria_1(b4, b12, 0.727, -0.11)
        return c1

def apply_biome_afd(biome_name, image_dir, stack_partial_name):
    """Apply an biome method to segmentate Active Fire  in a Sentinel image.
    It will load the image channels from the stack and apply the specified method.
    An stack has all channels to a specific spatial resolution.
    The nodata mask will be combined with an AND operation and will be applyed to each channel.

    Args:
        biome_name (str): Biome method's name
        image_dir (str): Path to the image stack
        stack_partial_name (str): Stack partial name (without the spacial resolution sufix)

    Returns:
        tuple: The mask and the sentinel image buffer
    """
    module = importlib.import_module('active_fire.biome', '.')
    algorithm = getattr(module, '{}AFD'.format(biome_name))
    algorithm = algorithm()

    # Load the bands needed from the image stacks
    # bands = (4, 12)
    # if biome_name == 'TropicalMoistForest' or biome_name == 'MediterraneanForest':
    #     bands = (4, 11, 12) 
    bands = (4, 11, 12) 
    buffered_stack = load_buffered_stack_bands(image_dir, stack_partial_name, bands)
    buffered_stack.apply_valid_data_mask_to_stack()
    
    return algorithm.transform(buffered_stack), buffered_stack
