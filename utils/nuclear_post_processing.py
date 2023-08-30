import math
import os
import pickle

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PySide2 import QtCore
from shapely import Polygon

from ui.signals_and_slots import LoadPercentConnection, InfoConnection
from utils.calc_methods import DNTheam, GrigStructs
from utils.primitives import DNWLine, DNWPoint, DNPoly, DNWPoly
from utils.sam_fragment import create_masks as create_sam_pkl
from utils.settings_handler import AppSettings


class PostProcessingWorker(QtCore.QThread):

    def __init__(self, yolo_txt_name, tek_image_path, edges_stats, lrm, save_folder, sam_path):
        """
        yolo_txt_name - результаты классификации CNN_Worker
        tek_image_path - путь к изображению
        edges_stats - путь к файлу со статистикой
        """
        super(PostProcessingWorker, self).__init__()
        self.yolo_txt_name = yolo_txt_name
        self.tek_image_path = tek_image_path
        self.edges_stats = edges_stats
        self.lrm = lrm
        self.save_folder = save_folder
        self.sam_path = sam_path
        self.settings = AppSettings()

        if 'sam_hq' in sam_path:
            self.use_hq = True
        else:
            self.use_hq = False

        self.polygons = []

        self.psnt_connection = LoadPercentConnection()
        self.info_connection = InfoConnection()

    def calc_points_per_side(self, min_obj_width_meters):
        image = Image.open(self.tek_image_path)
        image_width = image.width

        min_obj_width_px = min_obj_width_meters / self.lrm
        step_px = min_obj_width_px / 2.0
        return math.floor(image_width / step_px)

    def bns_detection(self):
        self.psnt_connection.percent.emit(0)

        ToQGisObj = DNToQGis(self.tek_image_path,
                             self.yolo_txt_name,
                             self.edges_stats)

        self.psnt_connection.percent.emit(5)
        info_message = f"Начинаю поиск зоны расположения БНС..." if self.settings.read_lang() == 'RU' else f"Start finding BNS local zone..."
        self.info_connection.info_message.emit(info_message)

        bns_zones = ToQGisObj.LocalZoneBNS(self.lrm, 180, 500)

        self.psnt_connection.percent.emit(30)

        crop_names = []
        if not bns_zones:
            self.psnt_connection.percent.emit(100)
            return

        im = Image.open(self.tek_image_path)

        for i, coords in enumerate(bns_zones['Coords']):
            left_upper = coords[0]
            right_bottom = coords[1]
            im_crop = im.crop((*left_upper, *right_bottom))

            crop_name = os.path.join(self.save_folder, f'crop{i}.jpg')
            crop_names.append(crop_name)

            im_crop.save(crop_name)

        self.psnt_connection.percent.emit(40)

        info_message = f"Зона расположения БНС найдена. Начинаю кластеризацию методом SAM" if self.settings.read_lang() == 'RU' else f"Found BNS local zone. Start clustering with SAM.."
        self.info_connection.info_message.emit(info_message)

        step = 0
        steps = len(crop_names) * 2

        for i, crop_name in enumerate(crop_names):
            pkl_name = os.path.join(self.save_folder, f'crop{i}.pkl')

            points_per_side = self.calc_points_per_side(min_obj_width_meters=80)
            print(f"Points per side = {points_per_side}")

            create_sam_pkl(crop_name, checkpoint=self.sam_path,
                           device=self.settings.read_platform(), output_path=None,
                           one_image_name=os.path.join(self.save_folder, f'crop{i}_sam.jpg'),
                           pickle_name=pkl_name, use_sam_hq=self.use_hq, points_per_side=points_per_side)

            step += 1
            self.psnt_connection.percent.emit(40 + 60 * float(step) / steps)

            with open(pkl_name, 'rb') as f:
                Mass = pickle.load(f)
                ContBNS = ToQGisObj.FinedBNS(Mass, bns_zones['Coords'][i], self.lrm)

                info_message = f"Кластеризация методом SAM завершена. Создаю контуры БНС..." if self.settings.read_lang() == 'RU' else f"SAM finished. Start building contours..."
                self.info_connection.info_message.emit(info_message)

                for points in ContBNS:
                    cls_num = 5  # bns
                    # self.view.add_polygon_to_scene(cls_num, points, color=color, id=shape_id)
                    self.polygons.append({'cls_num': cls_num, 'points': points})

            step += 1
            self.psnt_connection.percent.emit(40 + 60 * float(step) / steps)

    def run(self):
        self.bns_detection()


# Сигналы, управляющие прогрессбаром вставлять сюда
class DNProgressBar:
    def __init__(self):
        self.proc = 0
        self.state = ""
        self.IsProcFin = False

    # Изменение состояния прогрессбара
    def ChangeProcState(self, SummVar: int, CurVar: int, NameState: str):
        self.proc = float(CurVar / SummVar) * 100
        self.state = NameState

    # Функция, вызываемая при завершении процесса
    def FinishProc(self):
        self.IsProcFin = True


# Класс для встраивания в QGis
class DNToQGis:
    def __init__(self, PathToImg: str, PathToCNNRes: str, PathToModelFile: str, MinArea=150, MinL=10):

        # Начальные параметры для детектирования зданий (в пикселях)
        self.MinArea = MinArea  # Минимальная площадь сегмента
        self.MinL = MinL  # Минимальный линейный размер сегмента
        self.PathToImg = PathToImg  # Путь к файлу - изображению
        self.PathToCNNRes = PathToCNNRes  # Путь к файлу - результату работы СНС
        self.PathToModelFile = PathToModelFile  # Путь к файлу - модели
        self.ClassNums = {'RO_P': 0,
                          'RO_S': 1,
                          'MZ_V': 2,
                          'MZ_Ot': 3,
                          'RU_Ot': 4,
                          'Bns_Ot': 5,
                          'Gr_b': 6,
                          'Gr_V_S': 7,
                          'Gr_V_P': 8,
                          'Gr_B_Act': 9,
                          'Disch': 10,
                          'Disel': 11}

        self.image = Image.open(PathToImg)
        self.HImg = self.image.height
        self.WImg = self.image.width

        # Читаем файл результатов СНС
        file = open(PathToCNNRes, "r")
        self.Elements = []
        while True:
            line = file.readline()
            line.strip()
            if not line: break
            Data = line.strip().split(' ')
            xT = Data[1::2]
            yT = Data[2::2]
            x = np.array(xT, np.float32)
            y = np.array(yT, np.float32)
            # xy = list(zip(x, y))
            Element = {
                'NumCls': int(Data[0]),
                'x': x,
                'y': y
            }
            self.Elements.append(Element)
        file.close()

    def PrintConturs(self, Conturs: []):
        RGBMAss = np.array(self.image).astype("uint8")

        for Contur in Conturs:
            for i in range(len(Contur)):
                j = i + 1
                if j == len(Contur): j = 0
                p1 = Contur[i]
                p2 = Contur[j]
                cv.line(RGBMAss, p1, p2, (255, 255, 0), 2)
        plt.imshow(RGBMAss)
        plt.show()

    # Преобразование результатов НС в полигоны Шейп и пиксельные
    def ElemsToPoly(self, Elems: []):
        PolysPT = []
        PolysSH = []
        NumCl = []
        for i in range(len(Elems)):
            Elems[i]['x'] = np.array(Elems[i]['x'] * self.WImg, int)
            Elems[i]['y'] = np.array(Elems[i]['y'] * self.HImg, int)
            pts = np.zeros([len(Elems[i]['x']), 2], dtype=np.int32)
            pts[:, 0] = Elems[i]['x']
            pts[:, 1] = Elems[i]['y']
            PolysPT.append(pts)
            PolysSH.append(Polygon(pts))
            NumCl.append(Elems[i]['NumCls'])

        return {'PolysPT': PolysPT, 'PolysSH': PolysSH, 'NumCl': NumCl}

    # Функция получения конкретных классов из результатов НС
    def FinedClassElement(self, NumsClass: []):
        index = [i for i in range(0, len(self.Elements)) if self.Elements[i]['NumCls'] in NumsClass]
        Res = []
        for i in index:
            Res.append(self.Elements[i])
        return Res

    # Получение списка существующих полигонов
    @classmethod
    def GetPolysNames(cls, CatName: str):
        IsDirExist = os.path.isdir(CatName)
        # Если дирректория с именем файла изображения есть, записываем полигон туда
        if not IsDirExist:
            return []

        else:
            # Получение списка существующих полигонов
            PolysClass = os.walk(CatName, topdown=True, onerror=None, followlinks=False)
            ListFiles = []
            ListFolders = []
            ListPath = []
            for Path, Dirs, FilesInFolds in PolysClass:
                ListFiles.append(FilesInFolds)
                ListFolders.append(Dirs)
                ListPath.append(Path)

            Result = []
            for Files in ListFiles:
                for File in Files:
                    Result.append(File.split('.')[:-1][0])  # Убираем расширение у файлов

            return Result

    # Функция возвращает полигон с указанным именем (создает его, если его не существует, или возвращает ранее созданный)
    @classmethod
    def CreatePoly(cls, ImgPath: str, Polyname: str, Polygon: []):
        # Проверяем наличие уже созданных полигонов
        CatName = ImgPath.split('.')[:-1][0]
        IsDirExist = os.path.isdir(CatName)
        # Если дирректория с именем файла изображения есть, записываем полигон туда
        if not IsDirExist:
            os.mkdir(CatName)

        # Получение списка существующих полигонов
        PolysClass = os.walk(ImgPath.split('.')[:-1][0], topdown=True, onerror=None, followlinks=False)

        ListFiles = []
        ListFolders = []
        ListPath = []
        for Path, Dirs, FilesInFolds in PolysClass:
            ListFiles.append(FilesInFolds)
            ListFolders.append(Dirs)
            ListPath.append(Path)

        # Проверяем соответствует ли имя полигона имени уже записанного файла
        IsFileHere = False
        for Files in ListFiles:
            for File in Files:
                NFile = File.split('.')[:-1][0]  # Убираем расширение у файлов
                if NFile == Polyname:
                    IsFileHere = True
                    break

            if IsFileHere:
                break
        FileNameCurElem = CatName + '/' + Polyname + ".pol"
        # Если файл с полигоном существует, то просто читаем его
        if IsFileHere:
            Poly = DNPoly(FileNameCurElem)

        # Если файла такого не существут, создаем файл
        else:
            # Создаем объект DNWPoly
            WPts = []
            i = 0
            for Pt in Polygon:
                WPts.append(DNWPoint(Pt[0], Pt[1], Pt[0] + Pt[1] + i))
                i += 1

            WLines = []
            for i in range(len(WPts)):
                j = i + 1
                if j == len(WPts): j = 0
                WLines.append(DNWLine(WPts[i], WPts[j], WPts[i].x + WPts[j].x + WPts[i].y + WPts[j].y + i + j))

            WPoly = DNWPoly(WPts, WLines, Polyname)

            # Читаем картинку
            Img = Image.open(ImgPath)

            # Создаем файл полигона
            DNPoly.WriteFile(WPoly, FileNameCurElem, Img)
            Poly = DNPoly(FileNameCurElem)

        return Poly

    # Функция возвращает имя полигона с указанными координатами или False, если такого полигона нет
    @classmethod
    def FinedPoly(cls, ImgPath: str, Polygon: []):

        # Получение списка имен всех полигонов
        NamesPoly = DNToQGis.GetPolysNames(ImgPath)

        # Если полигоны отсутствуют, возвращаем False
        if len(NamesPoly) == 0:
            return False

        CatName = ImgPath.split('.')[:-1][0]
        # Ищем среди существующих полигонов тот, который подходит под координаты
        for NamePoly in NamesPoly:
            FileNameCurElem = CatName + '/' + NamePoly + ".pol"
            Poly = DNPoly(FileNameCurElem)
            IsPolyThis = True
            # Если количество точек в полигоне не совпадает, то это не наш полигон
            if len(Poly.WPoly.Points) != len(Polygon):
                IsPolyThis = False
            else:
                # Проверяем присутствие каждой точки в полигоне
                for WPt in Poly.WPoly.Points:
                    Pt = [WPt.x, WPt.y]
                    if not Pt in Polygon:
                        IsPolyThis = False
                        break

            if IsPolyThis:
                return NamePoly
        return False

    # Функция генерит уникальное имя полигона
    @classmethod
    def GenUniName(cls, ResultFilePath: str, BaseName: str):
        # Получение списка имен всех полигонов
        NamesPoly = DNToQGis.GetPolysNames(ResultFilePath)

        NameCurPoly = BaseName + "_"
        i = 0
        while 1:
            NameCurPoly = BaseName + "_" + str(i)
            i += 1
            if not NameCurPoly in NamesPoly:
                break
        return NameCurPoly

    # Функция получения относительных координат
    @classmethod
    def CalcOtnCoord(cls, Conturs, W: int, H: int):
        ContOtn = []
        for Cont in Conturs:
            x = Cont[:, 0]
            y = Cont[:, 1]
            xOtn = x / W
            yOtn = y / H
            BuildOtn = np.zeros([len(xOtn), 2], dtype=np.float32)
            BuildOtn[:, 0] = xOtn
            BuildOtn[:, 1] = yOtn
            ContOtn.append(BuildOtn)

        return ContOtn

    # Функция записи координат в файл
    @classmethod
    def WriteContursFile(cls, FilePath, Conturs):
        # Запись результатов в файл
        # Проверка, есть ли каталог, куда будет записан файл результатов
        ResultFilePath = FilePath.replace('\\', '/')
        IsDirExist = os.path.isdir(ResultFilePath)
        # Если дирректории нет, создавем ее
        if not IsDirExist:
            os.mkdir(ResultFilePath)

        # Создаем текстовый файл с именем картинки
        FileTxtName = DNToQGis.GenUniName(ResultFilePath, "Result") + ".txt"
        FileTxtName = ResultFilePath + '/' + FileTxtName
        f = open(FileTxtName, 'w')

        for Build in Conturs:
            StrWrite = "0"
            for Pt in Build:
                StrWrite += " " + str(Pt[0]) + " " + str(Pt[1])
            StrWrite += "\n"
            f.write(StrWrite)
        f.close()

    ######## Функции для Ромы
    # Функции локализации области интереса
    # Локализация области для поиска РО (MaxDist - размер буферной зоны в пределах которой ищем РО)
    def LocalZoneRO(self, LRMImg: float, MaxDist=140):
        GObj = GrigStructs(LRMImg, self.PathToImg, self.PathToCNNRes, self.PathToModelFile)
        Res = GObj.LocalZoneRO(MaxDist)
        return Res

    # Локализация области для поиска МЗ (MaxDist - размер буферной зоны в пределах которой ищем МЗ,
    # она же - пороговое расстояние для принятия решения, что данный МЗ относится к данному РО
    # MinDPorDist - минимальная разница расстояний между разными РО и одним МЗ, для того,
    # чтобы принять решение, относятся, ли эти РО к данному МЗ)
    def LocalZoneMZ(self, LRMImg: float, MaxDist=30, MinDPorDist=12):
        GObj = GrigStructs(LRMImg, self.PathToImg, self.PathToCNNRes, self.PathToModelFile)
        Res = GObj.LocalZoneMZ(MaxDist, MinDPorDist)
        return Res

    def LocalZoneBNS(self, LRMImg: float, MaxDistGroupGR=190, MaxDistGroupObj=70):
        GObj = GrigStructs(LRMImg, self.PathToImg, self.PathToCNNRes, self.PathToModelFile)
        Res = GObj.LocalZoneBNSGR(MaxDistGroupGR, MaxDistGroupObj)
        return Res

    # Функции обнаружения объектов
    def FinedMZ(self, Mass: [], RectImg: [], LRM, MinW=25, MaxW=90, MaxDist=200):
        # Проверка достаточности ЛРМ для обнаружения МЗ
        MinWP = float(MinW / LRM)
        if MinWP < self.MinL:
            print("Детальности изображения недостаточно для выполнения задачи")
            return

        ProgressBar = DNProgressBar()  # Объект класса для визуализации ProgressBar

        # Подготовка массива для классификации
        NCl, H, W = np.shape(Mass)
        ClsMass = np.zeros([W, H], dtype=np.uint8)
        for n in range(len(Mass)):
            P = np.column_stack(np.where(np.array(Mass[n]) == 1))
            ClsMass[P[:, 1], P[:, 0]] = n + 1

        # Определение зданий в заданном массиве
        BuildsС = DNTheam.DetectBuild2(ClsMass, self.MinArea, self.MinL, ProgressBar)
        StartP = RectImg[0]
        for i in range(len(BuildsС)):
            BuildsС[i][:, 0] = BuildsС[i][:, 0] + StartP[0]
            BuildsС[i][:, 1] = BuildsС[i][:, 1] + StartP[1]

        # Получение контуров РО
        Elems = self.FinedClassElement([self.ClassNums['RO_P'],
                                        self.ClassNums['RO_S']])

        Polys = self.ElemsToPoly(Elems)
        ContsRO = Polys['PolysPT'].copy()

        # Определение машинного зала
        MZConts = DNTheam.FinedMZ_RO(BuildsС, ContsRO, LRM, MinW, MaxW, MaxDist)

        ProgressBar.FinishProc()
        return MZConts

    def FinedROpr(self, Mass: [], RectImg: [], LRM, MinW=25, MaxW=90, MaxDist=200):
        # Проверка достаточности ЛРМ для обнаружения МЗ
        MinWP = float(MinW / LRM)
        if MinWP < self.MinL:
            print("Детальности изображения недостаточно для выполнения задачи")
            return

        ProgressBar = DNProgressBar()  # Объект класса для визуализации ProgressBar

        # Подготовка массива для классификации
        NCl, H, W = np.shape(Mass)
        ClsMass = np.zeros([W, H], dtype=np.uint8)
        for n in range(len(Mass)):
            P = np.column_stack(np.where(np.array(Mass[n]) == 1))
            ClsMass[P[:, 1], P[:, 0]] = n + 1

        # Определение зданий в заданном массиве
        BuildsС = DNTheam.DetectBuild2(ClsMass, self.MinArea, self.MinL, ProgressBar)
        StartP = RectImg[0]
        for i in range(len(BuildsС)):
            BuildsС[i][:, 0] = BuildsС[i][:, 0] + StartP[0]
            BuildsС[i][:, 1] = BuildsС[i][:, 1] + StartP[1]

        # Получение контуров МЗ
        Elems = self.FinedClassElement([self.ClassNums['MZ_V'],
                                        self.ClassNums['MZ_Ot']])

        Polys = self.ElemsToPoly(Elems)
        ContsMZ = Polys['PolysPT'].copy()

        # Определение реакторного отделения
        ROConts = DNTheam.FinedMZ_RO(BuildsС, ContsMZ, LRM, MinW, MaxW, MaxDist)

        ProgressBar.FinishProc()
        return ROConts

    def FinedROCir(self, Mass: [], RectImg: [], LRM, MinR=70, MaxR=240, MaxDist=200):
        # Проверка достаточности ЛРМ для обнаружения МЗ
        MinWP = float(MinR / LRM)
        if MinWP < self.MinL:
            print("Детальности изображения недостаточно для выполнения задачи")
            return

        ProgressBar = DNProgressBar()  # Объект класса для визуализации ProgressBar

        # Подготовка массива для классификации
        NCl, H, W = np.shape(Mass)
        ClsMass = np.zeros([W, H], dtype=np.uint8)
        for n in range(len(Mass)):
            P = np.column_stack(np.where(np.array(Mass[n]) == 1))
            ClsMass[P[:, 1], P[:, 0]] = n + 1

        # Определение зданий круглой формы в заданном массиве
        BuildsС = DNTheam.DetectCircleBuild(ClsMass, self.MinArea, self.MinL, 0.2)
        StartP = RectImg[0]
        for i in range(len(BuildsС)):
            BuildsС[i][:, 0] = BuildsС[i][:, 0] + StartP[0]
            BuildsС[i][:, 1] = BuildsС[i][:, 1] + StartP[1]

        # Получение контуров МЗ
        Elems = self.FinedClassElement([self.ClassNums['MZ_V'],
                                        self.ClassNums['MZ_Ot']])

        Polys = self.ElemsToPoly(Elems)
        ContsMZ = Polys['PolysPT'].copy()

        # Определение РО круглой формы
        ROConts = DNTheam.FinedMZ_RO(BuildsС, ContsMZ, LRM, MinR, MaxR, MaxDist)

        ProgressBar.FinishProc()
        return ROConts

    def FinedBNS(self, Mass: [], RectImg: [], LRM, MinW=7, MaxW=15):
        # Проверка достаточности ЛРМ для обнаружения МЗ
        MinWP = float(MinW / LRM)
        if MinWP < self.MinL:
            print("Детальности изображения недостаточно для выполнения задачи")
            return

        ProgressBar = DNProgressBar()  # Объект класса для визуализации ProgressBar

        # Подготовка массива для классификации
        NCl, H, W = np.shape(Mass)
        ClsMass = np.zeros([W, H], dtype=np.uint8)
        for n in range(len(Mass)):
            P = np.column_stack(np.where(np.array(Mass[n]) == 1))
            ClsMass[P[:, 1], P[:, 0]] = n + 1

        # Определение зданий в заданном массиве
        BuildsС = DNTheam.DetectBuild2(ClsMass, self.MinArea, self.MinL, ProgressBar)
        StartP = RectImg[0]
        for i in range(len(BuildsС)):
            BuildsС[i][:, 0] = BuildsС[i][:, 0] + StartP[0]
            BuildsС[i][:, 1] = BuildsС[i][:, 1] + StartP[1]

        # Получение контуров градирен
        Elems = self.FinedClassElement([self.ClassNums['Gr_b'],
                                        self.ClassNums['Gr_V_S'],
                                        self.ClassNums['Gr_V_P'],
                                        self.ClassNums['Gr_B_Act']])

        Polys = self.ElemsToPoly(Elems)
        ContsGR = Polys['PolysPT'].copy()

        # Определение машинного зала
        BNSConts = DNTheam.FinedBNS(BuildsС, ContsGR, LRM, MinW, MaxW)

        ProgressBar.FinishProc()
        return BNSConts


if __name__ == '__main__':
    def calc_points_per_side(min_obj_width_meters, tek_image_path, lrm):
        image = Image.open(tek_image_path)
        image_width = image.width

        min_obj_width_px = min_obj_width_meters / lrm
        step_px = min_obj_width_px / 2.0
        return math.floor(image_width / step_px)


    tek_image_path = "../nuclear_power/crop0.jpg"
    lrm = 0.9
    print(calc_points_per_side(100, tek_image_path, lrm))
