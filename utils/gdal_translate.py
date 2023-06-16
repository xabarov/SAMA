from osgeo import gdal
import os
import cv2

import geopandas as gpd
from shapely import Point
import rasterio


# def get_extent(geo_transform, raster_x_size, raster_y_size):
#     min_x = geo_transform[0]
#     max_y = geo_transform[3]
#     max_x = min_x + geo_transform[1] * raster_x_size
#     min_y = max_y + geo_transform[5] * raster_y_size
#     points = [Point(min_x, max_y), Point(max_x, min_y)]
#     gdf = gpd.GeoDataFrame(geometry=points, crs=3857)
#     gdf = gdf.to_crs(4326)
#
#     points = [(point.x, point.y) for point in gdf['geometry']]
#
#     return points

def get_extent(img_filename):
    dataset = rasterio.open(img_filename)
    bounds = dataset.bounds
    points = [Point(bounds.left, bounds.top), Point(bounds.right, bounds.bottom)]
    gdf = gpd.GeoDataFrame(geometry=points, crs={'init': 'epsg:3395'})
    gdf = gdf.to_crs({'init': 'epsg:4326'})

    points = [(point.x, point.y) for point in gdf['geometry']]

    return points


def get_data(geotiff_path, print_info=True):
    gdalData = gdal.Open(geotiff_path)
    if gdalData:
        if print_info:
            print(f'Driver short name {gdalData.GetDriver().ShortName}')
            print(f'Driver long name {gdalData.GetDriver().LongName}')
            print(f'Raster size {gdalData.RasterXSize}, {gdalData.RasterYSize}')
            print(f"Number of bands {gdalData.RasterCount}")
            print(f"Projection {gdalData.GetProjection()}")
            geo_transform = gdalData.GetGeoTransform()
            print(f"Geo transform {geo_transform}")
            print(f"Extent {get_extent(geotiff_path)}")

    return gdalData


def get_band(gdalData, band=1):
    gdalBand = gdalData.GetRasterBand(band)
    return gdalBand.ReadAsArray()


def convert_geotiff(geotiff_path, save_path=None, bands=None):
    options_list = [
        '-ot Byte',
        '-of JPEG']

    if not bands:
        gdalData = get_data(geotiff_path)
        bands_num = gdalData.RasterCount

        for i in range(bands_num):
            options_list.append(f'-b {i + 1}')

    else:
        for b in bands:
            options_list.append(f'-b {b}')

    options_list.extend(['-r lanczos',
                         '-co',
                         '-scale'])

    # options_list = [
    #     '-ot Byte',
    #     '-of JPEG',
    #     '-b 1',
    #     '-b 2',
    #     '-b 3',
    #     # '-outsize 1200 0',
    #     # '-expand rgb',
    #     '-r lanczos',
    #     '-co',
    #     '-scale'
    # ]

    options_string = " ".join(options_list)

    if not save_path:
        save_path = os.path.join(os.path.dirname(geotiff_path),
                                 os.path.basename(geotiff_path).split('.')[0] + '.jpg')

    gdal.Translate(
        save_path,
        geotiff_path,
        options=options_string
    )

    return cv2.imread(save_path)


if __name__ == '__main__':
    geotiff_path = 'F:\python\\ai_annotator\projects\geotiff\kewaune.tif'
    save_path = 'F:\python\\ai_annotator\\assets\\emsland.jpg'

    # im = convert_geotiff(geotiff_path, save_path)
    # # print(im)

    # data = get_data(geotiff_path, print_info=True)
    get_extent(geotiff_path)
    # band_1 = get_band(data, 3)
    # cv2.imwrite('band1.jpg', band_1)
