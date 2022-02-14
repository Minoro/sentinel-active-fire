# import sys
# sys.path.append('../')
import xml.etree.ElementTree as ET
import os
import rasterio
import math
import numpy as np

# from image import ImageStack, ImageInMemory

XML_FILE_PATH = '../../xml/'

def get_image_metadata(mtd_tl_xml, mtd_msil_xml):
    tree = ET.parse(mtd_tl_xml)
    incidence_angle = get_metadata_tl(tree)

    tree = ET.parse(mtd_msil_xml)
    reflectance_conversion_data = get_metadata_msil(tree)

    return {**incidence_angle, **reflectance_conversion_data}

def get_metadata_tl(xml_tree):
    root_xml = xml_tree.getroot()
    incidence_angle = {}

    for child in root_xml.iter('Mean_Sun_Angle'):
        incidence_angle['mean_zenith_angle'] = child.find('ZENITH_ANGLE').text
        incidence_angle['mean_zenith_angle_unit'] = child.find('ZENITH_ANGLE').attrib['unit']

        incidence_angle['mean_zenith_angle'] = child.find('AZIMUTH_ANGLE').text
        incidence_angle['mean_zenith_angle_unit'] = child.find('AZIMUTH_ANGLE').attrib['unit']

    for child in root_xml.iter('Mean_Viewing_Incidence_Angle_List'):
        mean_viewing_incidence_angle = child.findall('Mean_Viewing_Incidence_Angle')
        for mean_angle in mean_viewing_incidence_angle:
            band_id = mean_angle.attrib['bandId']

            incidence_angle['zenith_' + band_id] = mean_angle.find('ZENITH_ANGLE').text
            incidence_angle['zenith_unit_' + band_id] = mean_angle.find('ZENITH_ANGLE').attrib['unit']

            
            incidence_angle['azimuth_' + band_id] = mean_angle.find('AZIMUTH_ANGLE').text
            incidence_angle['azimuth_unit_' + band_id] = mean_angle.find('AZIMUTH_ANGLE').attrib['unit']

    return incidence_angle

def get_metadata_msil(xml_tree):
    root_xml = xml_tree.getroot()

    reflectance_conversion = {}

    for u in root_xml.iter('U'):
        reflectance_conversion['U'] = u.text

    for quantification in root_xml.iter('QUANTIFICATION_VALUE'):
        reflectance_conversion['quantification_value'] = quantification.text

    for child in root_xml.iter('Solar_Irradiance_List'):
        for solar_irradiance in child.findall('SOLAR_IRRADIANCE'):
            key = 'solar_irradiance_' + solar_irradiance.attrib['bandId']
            reflectance_conversion[key] = solar_irradiance.text

    for child in root_xml.iter('Spectral_Information'):
        reflectance_conversion[child.attrib['physicalBand']] = child.attrib['bandId']

    return reflectance_conversion


def get_radiance(band_value, band, metadata):

    # band_value = img_stack.read(band)

    band_id = str(metadata['B' + str(band)])

    solar_angle_correction = math.radians(float(metadata['zenith_' + band_id]))
    solar_angle_correction = math.cos(solar_angle_correction)

    solar_irradiance = float(metadata['solar_irradiance_' + band_id])
    
    d2 = 1.0 / float(metadata['U'])

    quantification_value = 10000 # default value
    if 'quantification_value' in metadata:
        quantification_value = float(metadata['quantification_value'])

    rtoa = band_value / quantification_value
    radiance = (rtoa * solar_irradiance * solar_angle_correction ) / (math.pi * d2)

    return radiance

