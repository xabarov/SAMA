from PySide2 import QtCore
from ui.signals_and_slots import LoadPercentConnection
from utils.gdal_translate import convert_geotiff, get_data
import os
import utils.help_functions as hf


class GdalWorker(QtCore.QThread):

    def __init__(self, geotiff_path, save_folder=None, bands=None):
        super(GdalWorker, self).__init__()
        self.translate_conn = LoadPercentConnection()
        self.geotiff_path = geotiff_path
        self.bands = bands
        self.save_folder = save_folder
        self.bands_full_names = []
        self.bands_names = []
        self.lrm = None

    def run(self):
        self.bands_images = self.save_bands_from_geotiff_as_jpg()

    def save_bands_from_geotiff_as_jpg(self):

        self.translate_conn.percent.emit(0)

        if not self.save_folder:
            self.save_folder = os.path.basename(self.geotiff_path).split('.')[0]
            self.save_folder = os.path.join(os.path.dirname(self.geotiff_path), self.save_folder)
            if not os.path.exists(self.save_folder):
                os.makedirs(self.save_folder)

        if not self.bands:
            gdalData = get_data(self.geotiff_path)
            bands_num = gdalData.RasterCount
            self.bands = [i + 1 for i in range(bands_num)]

        print("Start bands convertion...")
        bands_full_names = []
        for i, b in enumerate(self.bands):
            print(f"\tBand {b}...")
            save_band_name = os.path.join(self.save_folder, f'band {b}.jpg')
            bands_full_names.append(save_band_name)
            convert_geotiff(self.geotiff_path, save_band_name, bands=[b])
            self.translate_conn.percent.emit(int(100.0 * (i + 1) / len(self.bands)))

        print("All bands has been converted!")

        self.translate_conn.percent.emit(100)
        self.bands_full_names = bands_full_names
        self.bands_names = [os.path.basename(name) for name in self.bands_full_names]

    def get_bands_full_names(self):
        return self.bands_full_names

    def get_bands_names(self):
        return self.bands_names

    def get_lrm(self, from_crs='epsg:3395', to_crs='epsg:4326'):
        lrm = hf.try_read_lrm(self.geotiff_path, from_crs=from_crs, to_crs=to_crs)
        if lrm:
            self.lrm = lrm

        return lrm
