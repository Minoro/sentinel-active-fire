''' Download images from Google Earth Engine.
This script needs earthengine package.
'''
from requests.api import request
import ee
import os
import requests
from datetime import datetime, timedelta  

# Samples used by https://doi.org/10.1016/j.jag.2021.102347
SAMPLES_PAPER_HU_BAN_NASCETTI = [
    {'sample': 'A', 'date': '2018-09-28', 'roi': (114.777, -3.251)},
    {'sample': 'B', 'date': '2019-08-21', 'roi': (-60.616, -15.624)},
    {'sample': 'C', 'date': '2018-08-29', 'roi': (136.089, -15.676)},
    {'sample': 'D', 'date': '2018-07-23', 'roi': (23.166, 38.010)},
    {'sample': 'E', 'date': '2018-11-11', 'roi': (-121.390, 39.813)},
    {'sample': 'F', 'date': '2018-10-25', 'roi': (22.431, -33.870)},
    {'sample': 'G', 'date': '2018-08-08', 'roi': (-125.947, 53.846)},
    {'sample': 'H', 'date': '2018-07-19', 'roi': (119.99, 65.47)},
]


class EarthEngineCollectionSearch:

    def __init__(self, collection_name) -> None:
        ee.Initialize()
        self.collection_name = collection_name
        self.collection = ee.ImageCollection(self.collection_name)


    def search_point(self, point):
        """Filter a specific point in the Google Earth Engine Collection.

        Args:
            point (tuple): Lat-Long point

        Returns:
            self: Collection filtered
        """
        bounds = ee.Geometry.Point(point[0], point[1])
        self.collection = self.collection.filterBounds(bounds)

        return self

    def search_dates(self, start_date, end_date):
        """Filter a specific date range in the Google Earth Engine Collection.
        The date must be in the format 'Y-m-d'

        Args:
            start_date (str): initial date
            end_date (str): final date

        Returns:
            self: Collection filtered by date
        """
        self.collection = self.collection.filterDate(start_date, end_date)

        return self

    def get_collection(self):
        """Get the Google Earth Engine Collection

        Returns:
            ee.ImageCollection: Google Earth Engine Collection
        """
        return self.collection

    def first(self):
        """Get the first result from collection

        Returns:
            ee.Image: Google Earth Engine Image
        """
        return self.collection.first()

    def clear_search(self):
        """Remove all filters from collection
        """
        self.collection = ee.ImageCollection(self.collection_name)


class SentinelDownloader(EarthEngineCollectionSearch):

    def __init__(self) -> None:
        super().__init__('COPERNICUS/S2')
        self.collection_name = 'COPERNICUS/S2'
        # self.ee_searcher = EarthEngineCollectionSearch(self.collection_name)
        self.granule_info = []
        self.base_url = 'https://storage.googleapis.com/gcp-public-data-sentinel-2/tiles'


    def get_granule_info(self):
        """Parse the granule information from Google Earth Engine search results.

        Returns:
            dict: Sentinel's granule information
        """
        self.granule_info = []
        collection_info = self.collection.getInfo()

        for feature in collection_info['features']:
            self.granule_info.append({
                'id': feature['id'],
                'granule_id': feature['properties']['GRANULE_ID'], 
                'product_id': feature['properties']['PRODUCT_ID'], 
            })

        return self.granule_info

    def download_granules_to(self, granules, output_path):
        """Downloads a specific granule from Sentinel's image.
        Each band will be downloaded as a jp2 file.
        The granule information MUST have the product_id and the granule_id information.

        Args:
            granules (dict): granule information.
            output_path (str): directory to save the images.
        """
        for granule in granules:
            
            granule_id = granule['granule_id']
            safe_path = self.get_granule_safe_path(granule)
            file_url = os.path.join(safe_path, 'GRANULE', granule_id, 'IMG_DATA')

            product_id = granule['product_id']
            product_parts = product_id.split('_')
            grid_info = product_parts[-2]
            image_prefix = '{}_{}'.format(grid_info, product_parts[2])

            bands = ['B01','B02','B03','B04',
                'B05','B06','B07','B08','B8A',
                'B09','B10','B11','B12']

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            for b in bands:
                image_name = '{}_{}.jp2'.format(image_prefix, b)
                image_url = os.path.join(file_url, image_name)

                print(image_url)        
                output_file = os.path.join(output_path, image_name)
                response = requests.get(image_url)

                if response.status_code != 200:
                    print('[ERROR] Erro ao baixar banda: {}'.format(image_url))
                    continue
                
                with open(output_file, 'wb') as f:
                    f.write(response.content)

    def download_granules_clouds_gml(self, granules, output_path):

        for granule in granules:
            print(granule)
            product_id = granule['product_id']
            product_parts = product_id.split('_')
            grid_info = product_parts[-2]
            process_time = product_parts[2]
            grid_info = '{}_{}'.format(grid_info, process_time)

            output_dir = os.path.join(output_path, grid_info)
            self.download_granule_qidata(granule, 'MSK_CLOUDS_B00.gml', output_dir)

    
    def download_granule_qidata(self, granule, qi_file, output_path):
        safe_path = self.get_granule_safe_path(granule)
        file_url = os.path.join(safe_path, 'GRANULE', granule['granule_id'], 'QI_DATA', qi_file)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_file = os.path.join(output_path, qi_file)
        response = requests.get(file_url)
        if response.status_code != 200:
            raise Exception('[ERROR] Erro ao baixar banda: {}'.format(file_url))
            
        with open(output_file, 'wb') as f:
            f.write(response.content)



    def get_granule_safe_path(self, granule):
        product_id = granule['product_id']

        product_parts = product_id.split('_')
        grid_info = product_parts[-2]
        
        grid_square = grid_info[-2:]
        latitude_band = grid_info[-3:-2]
        utm_zone = grid_info[1:3]
        
        file_url = os.path.join(self.base_url, utm_zone, latitude_band,grid_square, '{}.SAFE'.format(product_id))
        
        return file_url

if __name__ == '__main__':

    downloader = SentinelDownloader()
    for sample in SAMPLES_PAPER_HU_BAN_NASCETTI:

        print('[INFO] Searching: {}'.format(sample['sample']))
        start_date = datetime.strptime(sample['date'], '%Y-%m-%d')
        end_date = start_date + timedelta(days=1)
        
        start_date = start_date.strftime("%Y-%m-%d")
        end_date = end_date.strftime("%Y-%m-%d")

        granules = downloader.search_dates(start_date, end_date) \
            .search_point(sample['roi']) \
            .get_granule_info()

        print('[INFO] Imagens encontradas: {}'.format(len(granules)))
        downloader.download_granules_to(granules, '../../images/original')
        downloader.download_granules_clouds_gml(granules, '../../images/qi_data/')
        print('[INFO] Download concluido')
        downloader.clear_search()

    print('Done!')