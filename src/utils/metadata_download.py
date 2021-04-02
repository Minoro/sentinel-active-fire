import pandas as pd
import urllib.parse
from tqdm import tqdm
import requests
import os 

INPUT_FILE = '../../S2_2020_Ago_para_baixar.xlsx'
OUTPUT_PATH = '../../xml/'
MTD_TL_XML_FILE_NAME = 'MTD_TL.xml'


def download_xml_file(url, output_file):
    try:
        r = requests.get(url)
        with open(output_file, 'w') as f:
            f.write(r.text)
    except Exception as e:
        raise Exception('Error downloading {} to {} - Message: {}'.format(url, output_file, str(e))) 


if __name__ == '__main__':

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    df = pd.read_excel(INPUT_FILE)

    with open(os.path.join(OUTPUT_PATH, 'download_xml.log'), 'w+') as log:
        for row in tqdm(df.to_dict(orient="records")):
            try:
                url = row['url']
                mtd_xml_url = url.split('IMG_DATA')[0]
                mtd_xml_url = urllib.parse.urljoin(mtd_xml_url, MTD_TL_XML_FILE_NAME)

                img_name = url.split('/')[-1]
                mtd_xml_output = os.path.join(OUTPUT_PATH, img_name + '_' + MTD_TL_XML_FILE_NAME)

                download_xml_file(mtd_xml_url, mtd_xml_output)

                url = row['url']
                mtd_msil_xml_url = url.split('GRANULE')[0]
                mtd_msil_file_name = 'MTD_MSIL2A.xml'
                if 'MSIL1C' in mtd_msil_xml_url:
                    mtd_msil_file_name = 'MTD_MSIL1C.xml'
                elif 'MSIL2B' in mtd_msil_xml_url:
                    mtd_msil_file_name = 'MTD_MSIL2B.xml'
                    
                mtd_msil_xml_url = urllib.parse.urljoin(mtd_msil_xml_url, mtd_msil_file_name)
                
                mtd_xml_output = img_name + '_' + 'MTD_MSIL.xml' 
                mtd_xml_output = os.path.join(OUTPUT_PATH, mtd_xml_output)
                download_xml_file(mtd_msil_xml_url, mtd_xml_output)
            except Exception as e:
                log.write(str(e))

    print('Done')
