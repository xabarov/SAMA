import os
from PIL import Image
import geopandas as gpd
from shapely import Point
import rasterio


class GeoTIFF:

    def __init__(self, geotiff_path=None):
        self.im_pil = None
        self.size = None
        self.width = None
        self.height = None
        self.bands_num = None
        self.raster_x_size = None
        self.raster_y_size = None
        self.geotiff_path = geotiff_path
        if geotiff_path:
            self.read(geotiff_path)

    def read(self, geotiff_path):
        self.im_pil = Image.open(geotiff_path)
        self.size = self.im_pil.size
        self.width = self.im_pil.tag[256]
        self.height = self.im_pil.tag[257]
        self.bands_num = self.im_pil.tag[277][0]
        raster_pixel_size = self.im_pil.tag[33550]
        self.raster_x_size = raster_pixel_size[0]
        self.raster_y_size = raster_pixel_size[1]

    def translate(self, geotiff_path=None, save_path=None):
        if not geotiff_path:
            geotiff_path = self.geotiff_path
        if geotiff_path:
            image = Image.open(geotiff_path)
            jpeg_image = image.convert('RGB')
            if not save_path:
                save_path = os.path.join(os.path.dirname(geotiff_path),
                                         os.path.basename(geotiff_path).split('.')[0] + '.jpg')

            jpeg_image.save(save_path, 'JPEG')

    def __str__(self):
        info = f"Size: {self.im_pil.size}\n"
        info += f"Bands num: {self.bands_num}"
        return info


def get_extent(img_filename, from_crs='epsg:3395', to_crs='epsg:4326'):
    dataset = rasterio.open(img_filename)
    bounds = dataset.bounds
    points = [Point(bounds.left, bounds.top), Point(bounds.right, bounds.bottom)]
    gdf = gpd.GeoDataFrame(geometry=points, crs={'init': from_crs})
    gdf = gdf.to_crs({'init': to_crs})

    points = [(point.x, point.y) for point in gdf['geometry']]

    return points


def print_pil_data(geotiff_path):
    im = Image.open(geotiff_path)
    print("Dimensions : {}".format(im.size))
    for id in im.tag:
        print("{} : {}".format(id, im.tag[id]))


if __name__ == '__main__':
    geotiff_path = 'test_data/test.tif'
    print_pil_data(geotiff_path)
    # tiff = GeoTIFF(geotiff_path)
    # tiff.translate()
