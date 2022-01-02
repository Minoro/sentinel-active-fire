import geopandas as gpd
import os
import matplotlib.pyplot as plt 
from glob import glob
import fiona
import sys
from tqdm import tqdm


CONTINENTS_FOLDER = '../../resources/continentes'
GRID_FOLDER = '../../resources/sentinel_grid/'

OUTPUT_FILE_NAME = 'sentinel_land_grid.geojson'


'''
Load the shapefile of the continents and the sentinel grid.
Intersects the sentinel grid with the continents shapefile to get the grid with land. 
'''
if __name__ == '__main__':

    print('Loading sentinel grid...')
    # Load the sentinel grid
    gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
    gdf_grid = gpd.read_file(os.path.join(GRID_FOLDER, 'sentinel_grid.kml'), driver='KML')

    print('Processing the shapefiles...')
    shapefiles = glob(os.path.join(CONTINENTS_FOLDER, '*.shp'))

    grid_square_land = []

    for shapefile in shapefiles:
        gdf = gpd.read_file(shapefile)
        print(f"Shapefile of {gdf.loc[0]['CONTINENT']}")

        for index, grid_row in tqdm(gdf_grid.iterrows()):

            # Check the intersection between the continent shape and the sentinel grid
            for grid_geometry in grid_row.geometry:
                gdf_intersection = gdf.intersects(grid_geometry)
                if gdf_intersection.any():
                    grid_square_land.append({
                        'name': grid_row.Name,
                        'description': grid_row.Description,
                        'continent': gdf.loc[0]['CONTINENT'],
                        'geometry': grid_geometry
                    })


    sentinel_grid = gpd.GeoDataFrame( grid_square_land )
    num_grids = len(sentinel_grid)
    print(f'Num. tiles: {num_grids}')

    with open(os.path.join(GRID_FOLDER, OUTPUT_FILE_NAME), 'w') as f:
        f.write(sentinel_grid.to_json())

    print('Done!')