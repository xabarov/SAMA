import pickle
import numpy as np
from matplotlib import path
import statistics as st

# Классы для отрисовки полигонов
class DNWPoint:
    def __init__(self, x: int, y: int, IdPoint: int):
        self.x = x
        self.y = y
        self.IdPoint = IdPoint

    def WriteToFile(self, f):
        bt = {'x': self.x, 'y': self.y, 'IdPoint': self.IdPoint}
        pickle.dump(bt, f)

    @classmethod
    def ReadFromFile(cls, f):
        P = pickle.load(f)
        return DNWPoint(P['x'], P['y'], P['IdPoint'])


class DNWLine:
    def __init__(self, P1: DNWPoint, P2: DNWPoint, IdLine: int):
        self.P1 = P1
        self.P2 = P2
        self.IdLine = IdLine

    def WriteToFile(self, f):
        self.P1.WriteToFile(f)
        self.P2.WriteToFile(f)
        pickle.dump(self.IdLine, f)

    @classmethod
    def ReadFromFile(cls, f):
        P1 = DNWPoint.ReadFromFile(f)
        P2 = DNWPoint.ReadFromFile(f)
        Id = pickle.load(f)
        return DNWLine(P1, P2, Id)


class DNWPoly:
    def __init__(self, Points: [] = [], Lines: [] = [], NamePoly: str = ""):
        self.Points = []
        self.Lines = []
        self.NamePoly = NamePoly

        for Point in Points:
            self.Points.append(Point)

        for Line in Lines:
            self.Lines.append(Line)

    def AddPoint(self, P: DNWPoint):
        self.Points.append(P)

    def AddLine(self, L: DNWLine):
        self.Lines.append(L)

    def SetNamePoly(self, NamePoly: str):
        self.NamePoly = NamePoly

    def WriteToFile(self, f):
        pickle.dump(self.NamePoly, f)
        pickle.dump(len(self.Points), f)
        for P in self.Points:
            P.WriteToFile(f)

        pickle.dump(len(self.Lines), f)
        for L in self.Lines:
            L.WriteToFile(f)

    @classmethod
    def ReadFromFile(cls, f):
        NamePoly = pickle.load(f)
        PCount = pickle.load(f)
        P = []
        for i in range(PCount):
            P.append(DNWPoint.ReadFromFile(f))

        LCount = pickle.load(f)
        L = []
        for i in range(LCount):
            L.append(DNWLine.ReadFromFile(f))

        return DNWPoly(P, L, NamePoly)


class DNWPoly_s:
    def __init__(self, Poly_s: [] = [], NumCurPoly: int = None):
        self.NumCurPoly = NumCurPoly
        self.Poly_s = []
        for Poly in Poly_s:
            self.Poly_s.append(Poly)

    def ChangeCurPoly(self, NameCurPoly: str):
        # Узнаем номер текущего полигона
        self.NumCurPoly = -1
        for Poly in self.Poly_s:
            self.NumCurPoly += 1
            if Poly.NamePoly == NameCurPoly:
                break

    def AddPoly(self, Poly: DNWPoly):
        self.Poly_s.append(Poly)
        self.NumCurPoly = len(self.Poly_s) - 1

    def DelCurPoly(self):
        self.Poly_s.pop(self.NumCurPoly)
        self.NumCurPoly = None

    def SetNameCurPoly(self, Name: str):
        if self.NumCurPoly == None:
            return

        self.Poly_s[self.NumCurPoly].SetNamePoly(Name)


# Класс полигона для обработки изображения
class DNPoly:
    ChProp = {'NamesCls': '',
              'ParamsCls': []}

    def __init__(self, FileName: str):
        self.FileName = FileName
        Data = DNPoly.ReadFile(FileName)
        self.WPoly = Data[0]
        self.Mask = Data[1]
        self.XMin = Data[2]
        self.YMin = Data[3]
        self.W = Data[4]
        self.H = Data[5]
        self.NamesCh = Data[6]
        self.ChsP = Data[7]
        self.DataMass = Data[8]

    # Проверка, существует ли определенный канал полигона и определение индекса канала по его параметрам
    def IsChanInPoly(self, NameCh='', ChP={'NamesCls': '', 'ParamsCls': []}):
        # Проверка, существует ли канал с таким именем
        if not NameCh in self.NamesCh:
            return False

        Res = False
        for i in range(len(self.NamesCh)):
            # Проверяем на соответствие параметров
            if NameCh == self.NamesCh[i] and ChP['NamesCls'] == self.ChsP[i]['NamesCls']:
                if len(ChP['ParamsCls']) == len(self.ChsP[i]['ParamsCls']):
                    AllParams = True
                    for P in ChP['ParamsCls']:
                        if not P in self.ChsP[i]['ParamsCls']:
                            AllParams = False

                    if AllParams: return i
        return Res

    # Проверка, на существование множества каналов
    def IsChansInPoly(self, NamesCh=[], ChsP=[]):
        if not len(NamesCh) == len(ChsP):
            return False

        for i in range(len(NamesCh)):
            if not self.IsChanInPoly(NamesCh[i], ChsP[i]):
                return False
        return True

    @classmethod
    # Маска возвращается в виде bool массива размерностью [W,H] (первая координата: x)
    def GetMaskPol(cls, Pol: DNWPoly):
        # Ищим минимальные и максимальные значения координат
        x = []
        y = []
        PMass = []
        for P in Pol.Points:
            x.append(P.x)
            y.append(P.y)
            PMass.append((P.x, P.y))

        XMin = min(x)
        XMax = max(x)
        YMin = min(y)
        YMax = max(y)

        # Заполнение массива координат точек
        XList = list(range(XMin, XMax + 1))
        YList = list(range(YMin, YMax + 1))
        P = []

        WPol = XMax - XMin + 1
        HPol = YMax - YMin + 1

        for YVal in YList:
            for XVal in XList:
                P.append((XVal, YVal))

        # Проверка на попадание каждой точки из массива во внутрь полигона
        pPath = path.Path(PMass)
        PMass = pPath.contains_points(P)
        PMass = np.array(PMass).reshape(-1, WPol).transpose()

        return {'Mass': PMass, 'XMin': XMin, 'YMin': YMin, 'WPol': WPol, 'HPol': HPol}

    # Записываем файл полигона (Когда полигон впервые создается, не для редактирования)
    @classmethod
    def WriteFile(cls, P: DNWPoly, FileName: str, PilImg):
        f = open(FileName, 'wb')

        # Записываем полигон как графический объект
        P.WriteToFile(f)

        # Записываем маску пикселей
        Mask = cls.GetMaskPol(P)
        pickle.dump(Mask['Mass'], f)
        pickle.dump(Mask['XMin'], f)
        pickle.dump(Mask['YMin'], f)
        pickle.dump(Mask['WPol'], f)
        pickle.dump(Mask['HPol'], f)

        # Записываем массив наименований каналов
        NamesCh = ['R', 'G', 'B']
        pickle.dump(NamesCh, f)

        # Записываем параметры каналов (для RGB - параметры пустые, они используются в алгоритмах классификации)
        ChsProp = [cls.ChProp, cls.ChProp, cls.ChProp]
        pickle.dump(ChsProp, f)

        # Записываем яркости пикселей
        CroppImg = PilImg.crop((Mask['XMin'], Mask['YMin'], Mask['XMin'] + Mask['WPol'], Mask['YMin'] + Mask['HPol']))
        RGBMAss = np.array(CroppImg).astype("int32")

        pickle.dump(np.array(RGBMAss[:, :, 0]).transpose(), f)
        pickle.dump(np.array(RGBMAss[:, :, 1]).transpose(), f)
        pickle.dump(np.array(RGBMAss[:, :, 2]).transpose(), f)

        f.close()

    # Перезаписываем файл полигона после редактирования
    def ReWriteFile(self, P: DNWPoly, NamesCh=[], ChsP=[], DataMass=[]):
        if not len(NamesCh) == len(DataMass) or not len(ChsP) == len(DataMass):
            print('Количество имена каналов не совпадает с количеством данных')
            return

        f = open(self.FileName, 'wb')

        # Записываем полигон как графический объект
        P.WriteToFile(f)

        # Записываем маску пикселей
        Mask = self.GetMaskPol(P)
        pickle.dump(Mask['Mass'], f)
        pickle.dump(Mask['XMin'], f)
        pickle.dump(Mask['YMin'], f)
        pickle.dump(Mask['WPol'], f)
        pickle.dump(Mask['HPol'], f)

        # Записываем массив наименований каналов
        pickle.dump(NamesCh, f)
        self.NamesCh = NamesCh.copy()  # Это было изменено недавно Если пойдут какие-нибудь глюки, то, возможно, из-за этой фразы

        # Записываем свойства каналов
        pickle.dump(ChsP, f)
        self.ChsP = ChsP.copy()

        # Записываем яркости пикселей
        self.DataMass = self.DataMass.tolist()
        self.DataMass.clear()
        for i in range(len(NamesCh)):
            pickle.dump(np.array(DataMass[i]).astype("int32"), f)
            self.DataMass.append(DataMass[i])

        self.DataMass = np.array(self.DataMass).astype("int32")

        f.close()

        self.WPoly = P
        self.Mask = Mask['Mass']
        self.XMin = Mask['XMin']
        self.YMin = Mask['YMin']
        self.W = Mask['WPol']
        self.H = Mask['HPol']
        self.NamesCh = NamesCh

    # Добавляем новые каналы в полигон
    def AddCh(self, NamesNewCh=[], NewChP=[], DataNewCh=[]):
        if not len(NamesNewCh) == len(DataNewCh) or not len(NewChP) == len(DataNewCh):
            print('Количество имена каналов не совпадает с количеством данных:',len(NamesNewCh),len(DataNewCh),len(NewChP))
            return
        NamesCh = np.hstack([self.NamesCh, NamesNewCh.copy()])
        NamesCh = NamesCh.tolist()

        # Копируем параметры канала в новую переменную, чтобы небыло связи с сылкой на DNPoly.ChProp
        NewChPCopy = []
        for ChP in NewChP:
            NewChPCopy.append(ChP.copy())

        ChsP = np.hstack([self.ChsP, NewChPCopy])
        ChsP = ChsP.tolist()

        DataMass = self.DataMass.copy()
        DataMass.tolist()
        DataMass = DataMass.tolist()
        for i in range(len(NamesNewCh)):
            DataMass.append(DataNewCh[i])

        DataMass = np.array(DataMass).astype("int32")

        self.ReWriteFile(self.WPoly, NamesCh, ChsP, DataMass)

    @classmethod
    def ReadFile(cls, FileName: str):
        f = open(FileName, 'rb')

        # Читаем полигон как графический объект
        WPoly = DNWPoly.ReadFromFile(f)

        # Читаем маску пикселей
        Mask = pickle.load(f)
        XMin = pickle.load(f)
        YMin = pickle.load(f)
        W = pickle.load(f)
        H = pickle.load(f)

        # Читаем наименования каналов
        NamesCh = pickle.load(f)

        # Читаем свойства каналов
        ChsProp = pickle.load(f)

        # Читаем яркости пикселей
        DataMass = []
        for s in NamesCh:
            DataMass.append(pickle.load(f))

        DataMass = np.array(DataMass)

        f.close()
        return (WPoly, Mask, XMin, YMin, W, H, NamesCh, ChsProp, DataMass)

    # Читаем канал из полигона
    def GetCh(self, NumCh: int):
        DataMass = DNPoly.ReadFile(self.FileName)[8]
        return DataMass[NumCh]

