import sys

sys.path.append('../')

from datetime import datetime, timedelta  

from image.downloader import SentinelDownloader


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