import numpy as np
import statistics as st
import cv2 as cv
import math as mt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from utils.primitives import DNPoly
from matplotlib import path
from PIL import Image, ImageTk
from shapely import LineString, Point, Polygon, MultiPolygon, BufferCapStyle, BufferJoinStyle
from shapely.ops import unary_union
import pandas as pd


class DNMathAdd:
    # Функции определения взаимного расположения отрезков и прямых
    # Определение точки пересечения отрезков (отрезки задаются точками)
    @classmethod
    def SegmentsPointIntersection(cls, P11: [], P12: [], P21: [], P22: []):
        # Определение пересекаются ли отрезки принципиально
        D = (P12[0] - P11[0]) * (P21[1] - P22[1]) - (P12[1] - P11[1]) * (P21[0] - P22[0])
        D1 = (P12[0] - P11[0]) * (P21[1] - P11[1]) - (P12[1] - P11[1]) * (P21[0] - P11[0])
        D2 = (P21[0] - P11[0]) * (P21[1] - P22[1]) - (P21[1] - P11[1]) * (P21[0] - P22[0])

        Res = {'IsSegInter': False,
               'x': 0,
               'y': 0}
        if not D == 0:
            t = float(D1) / float(D)
            s = float(D2) / float(D)
            if t >= 0 and t <= 1 and s >= 0 and s <= 1:
                Res['IsSegInter'] = True  # Отрезки пересекаются
                # Ищем точку пересечения
                # Частные случаи (когда один из отрезков перпендикулярен оси x)
                if (P12[0] - P11[0]) == 0:
                    Res['x'] = P12[0]
                    K2 = float(P22[1] - P21[1]) / float(P22[0] - P21[0])
                    d2 = (P22[0] * P21[1] - P21[0] * P22[1]) / float(P22[0] - P21[0])
                    Res['y'] = K2 * Res['x'] + d2

                elif (P22[0] - P21[0]) == 0:
                    Res['x'] = P22[0]
                    K1 = float(P12[1] - P11[1]) / float(P12[0] - P11[0])
                    d1 = (P12[0] * P11[1] - P11[0] * P12[1]) / float(P12[0] - P11[0])
                    Res['y'] = K1 * Res['x'] + d1

                else:
                    K1 = float(P12[1] - P11[1]) / float(P12[0] - P11[0])
                    d1 = (P12[0] * P11[1] - P11[0] * P12[1]) / float(P12[0] - P11[0])
                    K2 = float(P22[1] - P21[1]) / float(P22[0] - P21[0])
                    d2 = (P22[0] * P21[1] - P21[0] * P22[1]) / float(P22[0] - P21[0])
                    Res['x'] = (d2 - d1) / (K1 - K2)
                    Res['y'] = K1 * (d2 - d1) / (K1 - K2) + d1

        return Res

    # Определение расстояния между отрезками и координат перпендикуляров из концов отрезков к другим отрезкам
    @classmethod
    def SegmentsDistans(cls, P11: [], P12: [], P21: [], P22: []):

        Res = {'IsSegInter': False,  # Пересекаются ли отрезки
               'Per': [],  # Перпендикуляры (координаты)
               'Dist': []  # Длина перпендикуляра
               }
        # Проверка пересекаются ли отрезик
        Seg = DNMathAdd.SegmentsPointIntersection(P11, P12, P21, P22)
        Res['IsSegInter'] = Seg['IsSegInter']
        if Seg['IsSegInter']: return Res

        # Если отрезки не пересекаются, ищем перпендикуляры
        # Если хотя бы один из отрезков вертикальный, меняем координаты всех отрезков местами (делаем этот отрезок горизонтальным)
        IsCoordMiror = False
        if P11[0] == P12[0] or P21[0] == P22[0]:
            P11[0], P11[1] = P11[1], P11[0]
            P12[0], P12[1] = P12[1], P12[0]
            P21[0], P21[1] = P21[1], P21[0]
            P22[0], P22[1] = P22[1], P22[0]
            IsCoordMiror = True

        # Если и после перемены координат местами один из отрезков вертикальный, то отрезки перпендикулярны
        if P11[0] == P12[0] or P21[0] == P22[0]:
            if IsCoordMiror:
                P11[0], P11[1] = P11[1], P11[0]
                P12[0], P12[1] = P12[1], P12[0]
                P21[0], P21[1] = P21[1], P21[0]
                P22[0], P22[1] = P22[1], P22[0]
            Res['Dist'] = [None, None, None, None]
            Res['Per'] = [None, None, None, None]
            # Res['Dist'] = np.array(Res['Dist'])
            # Res['Per'] = np.array(Res['Per'])
            return Res

        # Ищем перпендикуляры для каждой из точек двух отрезков
        Per = [[P11, P12], [P21, P22]]

        i = 1
        for Ps in Per:
            for P in Ps:
                # Координаты точки откуда опускаем перпендикуляр
                x3 = P[0]
                y3 = P[1]

                # Координаты отрезка, от которого берем перпендикуляр
                x1 = Ps[0][0]
                y1 = Ps[0][1]
                x2 = Ps[1][0]
                y2 = Ps[1][1]

                # Вычисляем точку пересечения перпендикуляра к отрезку по уравнению прямой и скалярному произведению векторов
                xl1 = Per[i][0][0]
                yl1 = Per[i][0][1]
                xl2 = Per[i][1][0]
                yl2 = Per[i][1][1]

                k = (yl1 - yl2) / (xl1 - xl2)
                d = yl1 - k * xl1

                if (k * y2 - k * y1 + x2 - x1) == 0:
                    Res['Per'].append(None)

                else:
                    xp = (x3 * x2 - x3 * x1 + y2 * y3 - y1 * y3 + y1 * d - y2 * d) / (k * y2 - k * y1 + x2 - x1)
                    yp = k * xp + d

                    # Проверяем, принадлежит ли точка отрезку, если да, то записываем координаты перпендикуляра
                    if xp >= min(xl1, xl2) and xp <= max(xl1, xl2) and yp >= min(yl1, yl2) and yp <= max(yl1, yl2):
                        # Округляем до целых координаты начала и конца перпендикуляра
                        # x3=np.around(x3,decimals=1)
                        # y3=np.around(y3,decimals=1)
                        # xp=np.around(xp,decimals=1)
                        # yp=np.around(yp,decimals=1)
                        if IsCoordMiror:
                            Res['Per'].append([[y3, x3], [yp, xp]])
                        else:
                            Res['Per'].append([[x3, y3], [xp, yp]])
                    # Если точка отрезку не пренадлежит, записываем пустоту (нужно для того, чтобы четко понимать из какой точки опускается перпендикуляр)
                    else:
                        Res['Per'].append(None)
            i = 0

        # Вычисляем длину перпендикуляров
        for Line in Res['Per']:
            # Если линия есть, рассчитываем расстояние
            if not Line == None:
                x = []
                y = []
                for P in Line:
                    x.append(P[0])
                    y.append(P[1])
                Dist = np.sqrt((x[0] - x[1]) * (x[0] - x[1]) + (y[0] - y[1]) * (y[0] - y[1]))
                Res['Dist'].append(Dist)
            # Если линии нет, записываем пустоту
            else:
                Res['Dist'].append(None)

        # Если вначале координаты были зеркально отображены, возвращаем все обратно
        if IsCoordMiror:
            P11[0], P11[1] = P11[1], P11[0]
            P12[0], P12[1] = P12[1], P12[0]
            P21[0], P21[1] = P21[1], P21[0]
            P22[0], P22[1] = P22[1], P22[0]

        # Переводим все списки в массивы numpy
        # Res['Dist']=np.array(Res['Dist'])
        # Res['Per']=np.array(Res['Per'])

        return Res

    # Определение угла и расстояния между прямыми
    @classmethod
    def LinesDistCorner(cls, P11: [], P12: [], P21: [], P22: []):

        # Рассчитываем угол между прямыми
        wl1 = P11[0] - P12[0]
        hl1 = P11[1] - P12[1]

        wl2 = P21[0] - P22[0]
        hl2 = P21[1] - P22[1]

        if wl1 == 0:
            alfa1 = 90
        else:
            alfa1 = np.arctan(hl1 / wl1) * 180 / np.pi

        if wl2 == 0:
            alfa2 = 90
        else:
            alfa2 = np.arctan(hl2 / wl2) * 180 / np.pi

        corner = abs(alfa1 - alfa2)
        d_corner = 3
        d = 0

        # Рассчитываем расстояния между прямыми, как еслибы они были параллельны
        x = [P11[0], P12[0]]
        y = [P11[1], P12[1]]

        # Формируем основное уравнение прямой
        A = 0
        B = 0
        C = 0
        if not (x[0] == x[1] or y[0] == y[1]):
            A = 1 / (x[1] - x[0])
            B = -1 / (y[1] - y[0])
            C = -x[0] * A - y[0] * B

        elif y[0] == y[1]:
            B = 1.
            C = -1 * y[0] / B

        elif x[0] == x[1]:
            A = 1.
            C = -1 * x[0] / A

        # Расстояние между почти параллельными прямыми рассчитываем как минимальное значение расстояния от точки одного отрезка до прямой
        d = min(abs((A * P21[0] + B * P21[1] + C) / np.sqrt(A * A + B * B)),
                abs((A * P22[0] + B * P22[1] + C) / np.sqrt(A * A + B * B)))
        return [corner, d]

    # Определение расстояния между точкой и прямой, образованной отрезком,
    # Определение координат перпендикуляра от точки до прямой
    # Определение, принадлежит ли перпендикуляр отрезку
    # Определение расстояния между перпендикуляром и концом отрезка, если перпендикуляр не принадлежит отрезку
    @classmethod
    def PointLineDist(cls, P1: [], P2: [], P: []):

        # Формируем основное уравнение прямой
        x = [P1[0], P2[0]]
        y = [P1[1], P2[1]]
        A = 0
        B = 0
        C = 0
        if not (x[0] == x[1] or y[0] == y[1]):
            A = 1 / (x[1] - x[0])
            B = -1 / (y[1] - y[0])
            C = -x[0] * A - y[0] * B

        elif y[0] == y[1]:
            B = 1.
            C = -1 * y[0] / B

        elif x[0] == x[1]:
            A = 1.
            C = -1 * x[0] / A

        # Расстояние между точкой и прямой
        d = abs((A * P[0] + B * P[1] + C) / np.sqrt(A * A + B * B))

        # Координаты точки - перпендикуляра от заданной точки до прямой
        x1 = P[0]
        y1 = P[1]
        if not (A == 0 or B == 0):
            R = A * y1 - B * x1
            x = ((-B * R / A) - C) / (A + (B * B / A))
            y = B * x / A + R / A
        elif not B == 0:
            x = x1
            y = -C / B
        elif not A == 0:
            y = y1
            x = -C / A

        # Проверка, лежит ли точка на отрезке
        xMax = max(P1[0], P2[0])
        xMin = min(P1[0], P2[0])
        yMax = max(P1[1], P2[1])
        yMin = min(P1[1], P2[1])

        IsPInSed = x >= xMin and x <= xMax and y >= yMin and y <= yMax

        # Если точка не лежит на отрезке, определяем минимальное расстояние от точки до концов отрезка
        dPMin = -1
        Indx = -1
        if not IsPInSed:
            A = [DNMathAdd.CalcEvkl([x, y], P1), DNMathAdd.CalcEvkl([x, y], P2)]
            dPMin = min(A)
            Indx = A.index(dPMin)

        return [d, [x, y], IsPInSed, dPMin, Indx]

    # Определение угла между векторами (P*1 - начальная точка вектора, P*2 - конечная точка вектора
    # учитывается порядок векторов: угол строится от первого вектора до второго: по часовой стрелке +, против часовой стрелки -)
    @classmethod
    def SegmentsAngle(cls, P11: [], P12: [], P21: [], P22: []):
        # Преобразовываем координаты (параллельным переносом п0ереносим начало второго вектора к первому
        Dx1 = P11[0]
        Dy1 = P11[1]

        Dx2 = P21[0]
        Dy2 = P21[1]

        Pp11 = [P11[0] - Dx1, P11[1] - Dy1]
        Pp12 = [P12[0] - Dx1, P12[1] - Dy1]
        Pp21 = [P21[0] - Dx2, P21[1] - Dy2]
        Pp22 = [P22[0] - Dx2, P22[1] - Dy2]

        # Определяем угол между векторами (от 0 до 180)
        alfa = (Pp12[0] * Pp22[0] + Pp12[1] * Pp22[1]) / (
                np.sqrt(Pp12[0] * Pp12[0] + Pp12[1] * Pp12[1]) * np.sqrt(Pp22[0] * Pp22[0] + Pp22[1] * Pp22[1]))
        alfa = np.arccos(alfa) * 180 / np.pi

        # Определяем знак угла между векторами
        if Pp12[0] > 0 or Pp22[0] > 0:
            if Pp12[1] < Pp22[1]: alfa = -alfa

        if Pp12[0] < 0 or Pp22[0] < 0:
            if Pp12[1] > Pp22[1]: alfa = -alfa

        return alfa

    @classmethod
    def CalcEvkl(cls, P1: [], P2: []):
        x1 = P1[0]
        y1 = P1[1]
        x2 = P2[0]
        y2 = P2[1]

        return np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


class DNTheam:
    # Методы сегментации, классификации, кластеризации
    # Функция подготовки данных для методов кластеризации
    @classmethod
    def DataPrepForClast(cls, Pol: DNPoly, NumsCh: [], MaskPix=None):
        DataMass = np.array(Pol.DataMass)

        # Узнаем координаты точек, принадлежащие полигону из маски
        if MaskPix is None:
            P = np.column_stack(np.where(Pol.Mask))
        else:
            P = np.column_stack(np.where(MaskPix))
        Result = []
        # Добавляем поканально яркости точек в результирующий массив
        for NCh in NumsCh:
            Data = DataMass[NCh]
            Result.append(Data[P[:, 0], P[:, 1]])

        Result = np.array(Result).transpose()

        return Result

    # Функции статистической обработки результатов классификации
    @classmethod
    def GetClassStaticParams(cls, ClsMass: [], Pol: DNPoly, NumsCh: []):

        # Вычисляем общее рассеивание точек по полигону
        P = np.column_stack(np.where(Pol.Mask))
        S_Mass = np.zeros([len(P[:, 0])], dtype=np.float64)
        S_Pol = len(P[:, 0])  # Общая площадь всего полигона
        PolMiddle = []  # Средняя яркость по полигону

        for NCh in NumsCh:
            Mass = Pol.DataMass[NCh, P[:, 0], P[:, 1]]
            PolMiddle.append(st.mean(np.array(Mass).astype("float")))
            S_Mass += ((Mass - PolMiddle[-1]) * (Mass - PolMiddle[-1]))

        PolMiddle = np.array(PolMiddle)

        S = sum(S_Mass)
        S_Middle = 1. * S / len(P[:, 0])

        # Вычисляем среднее, СКО и площадь и внутриклассовый разброс для каждого класса
        MiddleCls = []
        SKOCls = []
        N = []
        W = []
        NumsCls = []
        SP = []  # Процент площади от всего полигона

        for NCls in np.unique(ClsMass):
            if not NCls < 0:
                NumsCls.append(NCls)
                P = np.column_stack(np.where(ClsMass == NCls))
                N.append(len(P[:, 0]))
                SP.append(1. * N[-1] / S_Pol)
                # Среднее СКО для каждого канала
                ChMiddle = []
                ChSKO = []
                D_Mass = np.zeros([len(P[:, 0])], dtype=np.float64)
                for NCh in NumsCh:
                    Mass = np.array(Pol.DataMass[NCh, P[:, 0], P[:, 1]])

                    # Вычисляем среднее и СКО для каждого канала
                    ChMiddle.append(st.mean(Mass.astype("float")))
                    ChSKO.append(st.pstdev(Mass.astype("float")))

                    # Расстояние от каждой точки кластера до центра кластера
                    D_Mass += (Mass - ChMiddle[-1]) * (Mass - ChMiddle[-1])

                MiddleCls.append(ChMiddle)
                SKOCls.append(ChSKO)
                W.append(sum(D_Mass))

        # Перевод все в массивы numpy
        MiddleCls = np.array(MiddleCls)
        SKOCls = np.array(SKOCls)
        W = np.array(W)
        W_s = sum(W)

        DispCls = 1. * W / N  # Средний разброс яркостей в кластере (дисперстия) (Можно было вычислить через СКО)

        # Среднее расстояние центра кластера от других кластеров
        MCls = []
        iM = 0
        for Middle in MiddleCls:
            # Получаем новый массив средних значений яркостей без текущего
            NewMiddle = np.delete(MiddleCls, iM, axis=0)

            # Вычисляем расстояние от центра кластера до других центров кластеров
            i = 0
            D = np.zeros([len(NewMiddle)])

            for MiddleCh in Middle:
                D += (NewMiddle[:, i] - MiddleCh) * (NewMiddle[:, i] - MiddleCh)
                i += 1

            # Берем среднее значение до других центров
            MCls.append(st.mean(D))
            iM += 1

        MCls = np.array(MCls)
        # Вычисляем два коэффициента:
        KofDispCls = DispCls / MCls  # Отношение разброса яркостей внутри кластера к разбросу яркостей между кластерами
        KofDispCls2 = DispCls / S_Middle  # Отношение разброса яркостей внутри кластера к разбросу яркостей по полигону в целом

        T = 1. - W_s / S

        # Возвращаем: номера кластеров, среднее значение яркостей в каналах, СКО яркостей в каналах, площадь кластеров,
        # относительную площадь кластеров,
        # показатель доли разброса яркостей внутри кластеров относительно общего разброса яркостей
        return {"NumsCls": NumsCls, "MiddleCls": MiddleCls,
                "SKOCls": SKOCls, "NCls": N, "S%Cls": SP,
                "KofDispCls": KofDispCls, "KofDispCls2": KofDispCls2, "T": T}

    # Функция разделения результатов классификации на сегменты
    @classmethod
    def ClsToSegments(cls, ClsMass: []):
        # Формируем массив результата (необработанные пиксели: -5)
        W, H = np.shape(ClsMass)
        Res = np.full([W, H], -5)

        # Сразу заполняем те области, которые обрабатывать не надо
        P = np.column_stack(np.where(np.array(ClsMass) == -1))
        Res[P[:, 0], P[:, 1]] = -1

        # Определяем максимальное значение номера класса, чтобы номера сегментов не пересекались с номерами классов
        MaxNCls = max(np.unique(ClsMass)) + 1

        # Копируем ClsMass в новый массив, чтобы не повредить исходный результат классификации
        Img = np.zeros([W, H], dtype=np.int32)
        np.copyto(Img, ClsMass)

        # Последовательно применяем волшебную палочку к каждой необработанной точке
        NSeg = 0
        while True:
            Pos = np.argwhere(Res == -5)
            if Pos.size == 0: break

            x = Pos[0, 0]
            y = Pos[0, 1]
            Val = NSeg + int(MaxNCls)
            cv.floodFill(Img, None, (y, x), Val)

            P = np.column_stack(np.where(np.array(Img) == Val))
            Res[P[:, 0], P[:, 1]] = NSeg
            NSeg += 1

        return Res

    # Фильтрация сегментов по площади
    @classmethod
    def SegmentsAreaFilter(cls, SegMass: [], MinArea: int, MaxArea: int):
        W, H = np.shape(SegMass)
        Res = np.full([W, H], -1)
        np.copyto(Res, SegMass)

        for NSeg in np.unique(SegMass):
            P = np.argwhere(SegMass == NSeg)
            if not (P.size >= MinArea and P.size <= MaxArea):
                Res[P[:, 0], P[:, 1]] = -1

        NumsSeg = np.unique(Res)
        MaxNSeg = max(NumsSeg) + 1
        NResSeg = 0
        for NSeg in NumsSeg:
            P = np.argwhere(Res == NSeg)
            Res[P[:, 0], P[:, 1]] = int(MaxNSeg) + NResSeg
            NResSeg += 1

        Res = Res - int(MaxNSeg)
        return Res

    # Кластеризация К-Средних
    @classmethod
    def ClastKMeans(cls, Pol: DNPoly, NumsCh: [], NClasters, MaskPix=None):

        Data = cls.DataPrepForClast(Pol, NumsCh, MaskPix)
        ClastModel = KMeans(n_clusters=NClasters)
        ClastModel.fit_predict(Data)

        Img = np.full([Pol.W, Pol.H], -1)
        if MaskPix is None:
            P = np.column_stack(np.where(Pol.Mask))
        else:
            P = np.column_stack(np.where(MaskPix))
        Img[P[:, 0], P[:, 1]] = ClastModel.labels_
        return Img

    # Кластеризация DBSCAN
    @classmethod
    def ClastDBScan(cls, Pol: DNPoly, NumsCh: [], eps: float, min_samples: int, MaskPix=None):

        # Определяем дохера кластеров
        NumCl = 150

        # Проверяем, есть ли предыдущие результаты кластеризации K-средних
        if not "K-Means_150" in Pol.NamesCh:
            print("Этап кластеризации К-средних")
            # Кластеризуем данные методом K-средних
            ClsKMeans = cls.ClastKMeans(Pol, NumsCh, NumCl, MaskPix)
            Pol.AddCh(["K-Means_150"], [ClsKMeans])
            print("K-средних закончил работу")

        else:
            # Узнаем индекс массива классификации
            i = Pol.NamesCh.index("K-Means_150")
            ClsKMeans = Pol.DataMass[i]

        # Определяем среднее значение яркостей в каждом классе
        DataForClaster = []
        for NCls in range(NumCl):
            P = np.column_stack(np.where(ClsKMeans == NCls))
            P = np.array(P).transpose()
            DataForClasterChan = []

            NCh = 0
            for DataCh in Pol.DataMass:
                if NCh in NumsCh:
                    DataForClasterChan.append(st.mean(np.array(DataCh[P[0], P[1]]).astype('float64')))
                NCh += 1

            DataForClaster.append(DataForClasterChan)

        # Подаем средние значения яркостей на кластеризацию
        ClastModel = DBSCAN(eps=eps, min_samples=min_samples)
        ClastModel.fit_predict(DataForClaster)

        # Рассчитываем новые центры кластеров
        NewCenters = []
        for NCl in np.unique(ClastModel.labels_):
            Index = np.column_stack(np.where(np.array(ClastModel.labels_) == NCl))
            if not NCl == -1:
                NewCenters.append(np.mean(np.array(DataForClaster)[Index[:, 0]], axis=0))

            else:
                for i in Index[:, 0]:
                    NewCenters.append(DataForClaster[i])

        # Вычисляем евклидово расстояние от каждой точки изображения до нового кластера
        Data = cls.DataPrepForClast(Pol, NumsCh, MaskPix)
        EvklMass = []
        for Cent in NewCenters:
            EvklClass = np.zeros(len(Data))
            NCh = 0
            for BCent in Cent:
                EvklClass += (Data[:, NCh] - BCent) * (Data[:, NCh] - BCent)
                NCh += 1

            EvklClass = np.sqrt(EvklClass)
            EvklMass.append(EvklClass)

        # Получаем индексы минимальных значений евклидовых расстояний (это и есть новые номера классов)
        EvklMassIndex = np.argmin(EvklMass, axis=0)

        # Получаем список уникальных значений массива
        Img = np.full([Pol.W, Pol.H], -1)

        if MaskPix is None:
            P = np.column_stack(np.where(Pol.Mask))
        else:
            P = np.column_stack(np.where(MaskPix))

        Img[P[:, 0], P[:, 1]] = EvklMassIndex
        return Img

        # fig=plt.figure()
        # rows=1
        # cols=2
        #
        # fig.add_subplot(rows,cols,1)
        # plt.imshow(Im1.transpose())
        #
        # fig.add_subplot(rows,cols,2)
        # plt.imshow(Im2.transpose())
        #
        # plt.show()

    # Кластеризация OPTICS
    @classmethod
    def ClastOPTICS(cls, Pol: DNPoly, NumsCh: [], min_samples: int, MaskPix=None):

        # Определяем дохера кластеров
        NumCl = 150

        # Проверяем, есть ли предыдущие результаты кластеризации K-средних
        if not "K-Means_150" in Pol.NamesCh:
            print("Этап кластеризации К-средних")
            # Кластеризуем данные методом K-средних
            ClsKMeans = cls.ClastKMeans(Pol, NumsCh, NumCl, MaskPix)
            DNPoly.ChProp['NamesCls'] = ''
            DNPoly.ChProp['ParamsCls'] = []
            Pol.AddCh(["K-Means_150"], [DNPoly.ChProp], [ClsKMeans])

            print("K-средних закончил работу")

        else:
            # Узнаем индекс массива классификации
            i = Pol.NamesCh.index("K-Means_150")
            ClsKMeans = Pol.DataMass[i]

        # Определяем среднее значение яркостей в каждом классе
        DataForClaster = []
        for NCls in range(NumCl):
            P = np.column_stack(np.where(ClsKMeans == NCls))
            P = np.array(P).transpose()
            DataForClasterChan = []

            NCh = 0
            for DataCh in Pol.DataMass:
                if NCh in NumsCh:
                    DataForClasterChan.append(st.mean(np.array(DataCh[P[0], P[1]]).astype('float64')))
                NCh += 1

            DataForClaster.append(DataForClasterChan)

        # Подаем средние значения яркостей на кластеризацию
        for i in range(20):
            eps = i * 3 + 1
            ClastModel = DBSCAN(min_samples=min_samples, eps=eps)
            ClastModel.fit_predict(DataForClaster)
            print(eps, '\t', len(np.unique(ClastModel.labels_)))

        ClastModel = DBSCAN(min_samples=min_samples, eps=30)
        ClastModel.fit_predict(DataForClaster)

        # Рассчитываем новые центры кластеров
        NewCenters = []
        for NCl in np.unique(ClastModel.labels_):
            Index = np.column_stack(np.where(np.array(ClastModel.labels_) == NCl))
            if not NCl == -1:
                NewCenters.append(np.mean(np.array(DataForClaster)[Index[:, 0]], axis=0))

            else:
                for i in Index[:, 0]:
                    NewCenters.append(DataForClaster[i])

        # Вычисляем евклидово расстояние от каждой точки изображения до нового кластера
        Data = cls.DataPrepForClast(Pol, NumsCh, MaskPix)
        EvklMass = []
        for Cent in NewCenters:
            EvklClass = np.zeros(len(Data))
            NCh = 0
            for BCent in Cent:
                EvklClass += (Data[:, NCh] - BCent) * (Data[:, NCh] - BCent)
                NCh += 1

            EvklClass = np.sqrt(EvklClass)
            EvklMass.append(EvklClass)

        # Получаем индексы минимальных значений евклидовых расстояний (это и есть новые номера классов)
        EvklMassIndex = np.argmin(EvklMass, axis=0)

        # Получаем список уникальных значений массива
        Img = np.full([Pol.W, Pol.H], -1)

        if MaskPix is None:
            P = np.column_stack(np.where(Pol.Mask))
        else:
            P = np.column_stack(np.where(MaskPix))

        Img[P[:, 0], P[:, 1]] = EvklMassIndex

        plt.imshow(Img.transpose())
        plt.show()

        return Img

        # fig=plt.figure()
        # rows=1
        # cols=2
        #
        # fig.add_subplot(rows,cols,1)
        # plt.imshow(Im1.transpose())
        #
        # fig.add_subplot(rows,cols,2)
        # plt.imshow(Im2.transpose())
        #
        # plt.show()

    # Геометрические признаки
    # Определение контуров сегментов класса
    @classmethod
    def GetContursClass(cls, ClsMass, NumClass: int):
        # Выделение определенного класса из массива классификации
        Mass = cv.inRange(ClsMass, NumClass, NumClass)

        # Нахождение контуров полигонов заданного класса
        Сonturs, hierarchy = cv.findContours(Mass, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        # В координатах точек контуров x и y поменены местами, поэтому меняем их обратно
        PConturs = []
        for Contur in Сonturs:
            PConturs.append(np.array(Contur)[:, 0, [1, 0]])
        return PConturs

    # Определение площади сегмента обозначенного контуром
    @classmethod
    def GetAreaContur(cls, Contur: []):
        Area = cv.contourArea(Contur)

        # # Получаем маску контура
        # Points = []
        # XMin = min(Contur[:, 0])
        # XMax = max(Contur[:, 0])
        # YMin = min(Contur[:, 1])
        # YMax = max(Contur[:, 1])
        #
        # XList = list(range(XMin, XMax+1))
        # YList = list(range(YMin, YMax+1))
        #
        # for y in YList:
        #     for x in XList:
        #         Points.append((x, y))
        #
        # pPath = path.Path(Contur)
        # PolMass = pPath.contains_points(Points)
        # PolMass = np.array(PolMass).reshape(-1, XMax - XMin+1)
        #
        # # Считаем количество пикселей в контуре
        # P=np.column_stack(np.where(PolMass))
        # Area=len(P)
        #
        #
        # # Если контур вырождается в линию или точку, то его площадь считаем по точкам контура
        # if Area == 0:
        #     Area=len(Contur)

        return Area

    # Упорядочивание аппроксимированных линий
    @classmethod
    def FLinesOrder(cls, lines: []):
        LineNoOrder = []
        MaxV = max(lines[0])
        for line in lines:
            LineNoOrder.append(line)
            if max(line) > MaxV:
                MaxV = max(line)

        # Выбираем отрезок, с которого начнем упорядочивание
        # Оба конца отрезка по наименьшему расстоянию должны быть смежны с одним и тем же отрезком

        SIndx = 0
        for i in range(len(LineNoOrder)):
            ps1 = [LineNoOrder[i][:2], LineNoOrder[i][2:]]
            IndSm = [0, 0]
            IndP = 0
            for p1 in ps1:
                DMin = MaxV * MaxV
                for j in range(len(LineNoOrder)):
                    if not i == j:
                        ps2 = [LineNoOrder[j][:2], LineNoOrder[j][2:]]
                        for p2 in ps2:
                            D = np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
                            if D < DMin:
                                DMin = D
                                IndSm[IndP] = j
                IndP += 1
            if IndSm[0] == IndSm[1]:
                SIndx = i
                break

        LineOrder = []
        LineOrder.append(LineNoOrder[SIndx])
        LineNoOrder.pop(SIndx)

        while len(LineNoOrder) > 0:
            dMin = MaxV * MaxV  # WContur * HContur
            i = 0
            indexL = i
            # Подбираем к текущей линии самую близкую по расстоянию
            PInLineOrder = True
            P = []
            for Line in LineNoOrder:
                p1 = LineOrder[-1][2:]
                p2 = Line[:2]
                p3 = Line[2:]

                # Проверяем начало и конец линии
                d1 = np.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]))
                d2 = np.sqrt((p1[0] - p3[0]) * (p1[0] - p3[0]) + (p1[1] - p3[1]) * (p1[1] - p3[1]))

                if d1 < dMin:
                    dMin = d1
                    indexL = i

                # Если концы линии перепутаны, запоминаем это
                if d2 < dMin:
                    dMin = d2
                    indexL = i
                    PInLineOrder = False

                i += 1
            if PInLineOrder:
                P = LineNoOrder[indexL]

            # Если концы линии перепутаны в упорядоченный массив записываем как надо
            else:
                P = [LineNoOrder[indexL][2], LineNoOrder[indexL][3], LineNoOrder[indexL][0], LineNoOrder[indexL][1]]

            LineOrder.append(P)
            LineNoOrder.pop(indexL)
        return LineOrder

    # Аппроксимация контура прямыми
    @classmethod
    def ApproxLinesContur(cls, Contur: [], minLLength: int, maxLGap: int):

        # Определяем габаритные размеры контура
        XMin = min(Contur[:, 0])
        XMax = max(Contur[:, 0])
        YMin = min(Contur[:, 1])
        YMax = max(Contur[:, 1])

        WContur = XMax - XMin
        HContur = YMax - YMin

        # Генерирование бинарной картинки контура
        ConturImg = np.zeros([WContur + 1, HContur + 1], dtype=np.uint8)
        for p in Contur:
            ConturImg[p[0] - XMin][p[1] - YMin] = 255

        # Аппроксимация контура линиями
        # maxLGap=4
        # minLLength=10

        lines = cv.HoughLinesP(ConturImg, 1, (np.pi / 180) * 0.5, minLLength, minLineLength=minLLength,
                               maxLineGap=maxLGap)  # DNTODO: Параметры 2 и 4 надо настраивать

        if lines is None:
            return []

        linesTrue = []
        for Line in lines:
            linesTrue.append(Line[0])
        linesTrue = np.array(linesTrue)

        # print(lines)
        # print(linesTrue)
        # print(type(lines))
        # print(type(linesTrue))

        LineOrder = DNTheam.FLinesOrder(linesTrue)

        # Редактирование результатов аппроксимации
        if len(LineOrder) > 1:
            # Расчет центра масс контура
            xc = np.mean(Contur[:, 0]) - XMin
            yc = np.mean(Contur[:, 1]) - YMin
            i = 1
            for L1 in LineOrder:
                L2 = LineOrder[i]
                # 1. Укорачиваем линии до их перекрестия (укарачиваем наименьшую часть)
                Res = DNMathAdd.SegmentsPointIntersection(L1[:2], L1[2:], L2[:2], L2[2:])
                if Res['IsSegInter']:
                    xP = Res['x']
                    yP = Res['y']

                    D1 = [DNMathAdd.CalcEvkl([xP, yP], L1[:2]), DNMathAdd.CalcEvkl([xP, yP], L1[2:])]
                    D2 = [DNMathAdd.CalcEvkl([xP, yP], L2[:2]), DNMathAdd.CalcEvkl([xP, yP], L2[2:])]

                    if D1[0] < D1[1]:
                        L1[0], L1[1] = int(xP), int(yP)
                    else:
                        L1[2], L1[3] = int(xP), int(yP)

                    if D2[0] < D2[1]:
                        L2[0], L2[1] = int(xP), int(yP)
                    else:
                        L2[2], L2[3] = int(xP), int(yP)

                    LineOrder[i - 1] = L1
                    LineOrder[i] = L2

                i += 1
                if i == len(LineOrder): i = 0

            # 2. Фильтруем параллельные или почти параллельные линии на маленьком расстоянии друг от друга
            LFilt = LineOrder.copy()
            LFilt = np.array(LFilt)

            # if len(LFilt)==13:
            # print(len(LFilt))

            # Фильтруем отрезки до тех пор, пока максимальное расстояние между любыми парами отрезков не превышает maxLGap
            while 1:
                if len(LFilt) <= 1: break
                # Получаем матрицу расстояний всех отрезков
                MatrRes = []
                MatrDist = np.zeros([len(LFilt), len(LFilt)], dtype=np.float32)
                x = 0
                y = 0
                for L1 in LFilt:
                    StrRes = []
                    for L2 in LFilt:
                        # Формируем массив точек рассматриваемых отрезков
                        PLines = [L1.tolist()[:2], L1.tolist()[2:], L2.tolist()[:2], L2.tolist()[2:]]
                        PLines = np.around(PLines, decimals=0)

                        Res = DNMathAdd.SegmentsDistans(L1[:2], L1[2:], L2[:2], L2[2:])
                        StrRes.append(Res)
                        # Получаем индексы существующих перпендикуляров
                        I = np.column_stack(np.where(np.array(Res['Dist']) != None))
                        if len(I) == 0:  # Нет перпендикуляров к отрезкам
                            MatrDist[x, y] = -1
                        # Если перпендикуляры есть
                        else:
                            # #Рассматриваем только те перпендикуляры, которые кончаются не вконце отрезков
                            # MaxD=-1
                            # for i in I:
                            #     PerInt=[int(Res['Per'][i[0]][1][0]),int(Res['Per'][i[0]][1][1])]
                            #     if not PerInt in PLines.tolist():
                            #         if Res['Dist'][i[0]]>MaxD:
                            #             MaxD=Res['Dist'][i[0]]

                            MatrDist[x, y] = max(np.array(Res['Dist'])[I])
                        x += 1
                    MatrRes.append(StrRes)
                    y += 1
                    x = 0

                # Расстояние между отрезками округляем до десятых, чтобы получить чистый 0
                MatrDist = np.around(MatrDist, decimals=1)

                # Если в матрице расстояний все отрезки находятся друг от друга на приличном расстоянии то и нечего фильтровать
                DissIn = np.column_stack(np.where(MatrDist > 0))
                if len(DissIn) == 0: break
                if min(MatrDist[DissIn[:, 0], DissIn[:, 1]]) >= maxLGap:
                    break

                # В противном случае фильтруем отрезки
                else:
                    # Получаем индексы отрезков расположенных близко друг к другу
                    DissIn = np.column_stack(np.where((MatrDist < maxLGap) & (MatrDist > 0)))
                    DissIn = DissIn.tolist()
                    # DissIn=DissIn[:int(len(DissIn)/2)]
                    # Убираем зеркальные  индексы
                    for Indx in DissIn:
                        x = Indx[0]
                        y = Indx[1]
                        if [y, x] in DissIn and not x == y:
                            I = DissIn.index([y, x])
                            DissIn.pop(I)

                    InDelSeg = []  # Индексы удаленных или отредактированных отрезков (чтобы дальше их уже не считать)
                    for Indx in DissIn:
                        x = Indx[0]
                        y = Indx[1]

                        if not x in InDelSeg and not y in InDelSeg:
                            # Вытаскиваем координаты перпендикуляров у близких друг к другу отрезков
                            Res = MatrRes[x][y]

                            # Координаты отрезков между которыми близкое расстояние
                            L1 = [LFilt[x].tolist()[:2], LFilt[x].tolist()[2:]]
                            L2 = [LFilt[y].tolist()[:2], LFilt[y].tolist()[2:]]

                            # print(Res)
                            # print(DNMathAdd.SegmentsDistans(L1[0],L1[1],L2[0],L2[1]))

                            # Если один из отрезков полностью проецируется на другой, просто удаляем его и пересчитываем матрицы
                            # Отрезок полностью проецируется на другой, если существуют оба перпендикуляра из обеих концов этого отрезка
                            if not Res['Per'][0] == None and not Res['Per'][1] == None:
                                LFilt = np.delete(LFilt, x, axis=0)
                                InDelSeg.append(x)
                                break

                            if not Res['Per'][2] == None and not Res['Per'][3] == None:
                                LFilt = np.delete(LFilt, y, axis=0)
                                InDelSeg.append(y)
                                break

                            # Если отрезки пересекаются только частично
                            else:
                                # print(Res)
                                # print(L1,L2)
                                # print(DNMathAdd.SegmentsDistans(L1[0],L1[1],L2[0],L2[1]))
                                # print("\n")
                                # Выбираем перпендикуляр с которым будем рабоать
                                # Сначала по минимальному расстоянию
                                DistMass = np.array(Res['Dist'])
                                I = np.column_stack(np.where(DistMass != None))[:, 0]
                                MinDist = min(DistMass[I])
                                IMinD = Res['Dist'].index(MinDist)

                                # Если перпендикуляр идет в точку какого-нибудь из отрезков то надо менять рабочий перпендикуляр
                                # PerInt=[int(Res['Per'][IMinD][1][0]), int(Res['Per'][IMinD][1][1])]
                                # if PerInt in L1 or PerInt in L2:
                                #     k=np.column_stack(np.where(I!=IMinD))[:,0]
                                #     #I=np.delete(I,k)
                                #     MinDist=min(DistMass[I[k]])
                                #     IMinD=Res['Dist'].index(MinDist)

                                xp = int(Res['Per'][IMinD][1][0])
                                yp = int(Res['Per'][IMinD][1][1])

                                # Находим отрезок и точку от которой укорачиваем
                                # Перпендикуляр от первого отрезка (значит укорачиваем второй отрезок)
                                if IMinD == 0 or IMinD == 1:
                                    # Выбираем точку, которую будем корректировать
                                    if 2 in I:  # Есть перпендикуляр от первой точки второго отрезка
                                        # Расчитываем прирощение к xp,yp чтобы в дальнейшем, при пересчете отрезки уже не проецировались друг на друга
                                        # Перпендикуляр не будет принадлежать отрезку
                                        DX = mt.ceil((L2[1][0] - L2[0][0]) / abs((L2[1][0] - L2[0][0]) + 0.1))
                                        DY = mt.ceil((L2[1][1] - L2[0][1]) / abs((L2[1][1] - L2[0][1]) + 0.1))
                                        xp, yp = (xp + DX), (yp + DY)
                                        # Корректируем первую точку второго отрезка
                                        L2[0][0], L2[0][1] = xp, yp

                                    elif 3 in I:  # Есть перпендикуляр от второй точки второго отрезка
                                        DX = mt.ceil((L2[0][0] - L2[1][0]) / abs((L2[0][0] - L2[1][0]) + 0.1))
                                        DY = mt.ceil((L2[0][1] - L2[1][1]) / abs((L2[0][1] - L2[1][1]) + 0.1))
                                        xp, yp = (xp + DX), (yp + DY)
                                        # Корректируем вторую точку второго отрезка
                                        L2[1][0], L2[1][1] = xp, yp

                                    else:  # Если перпендикуляр только один, пододвигаем ближайшую к перпендикуляру точку
                                        D1 = DNMathAdd.CalcEvkl([xp, yp], L2[0])
                                        D2 = DNMathAdd.CalcEvkl([xp, yp], L2[1])
                                        if min(D1, D2) == D1:
                                            DX = mt.ceil((L2[1][0] - L2[0][0]) / abs((L2[1][0] - L2[0][0]) + 0.1))
                                            DY = mt.ceil((L2[1][1] - L2[0][1]) / abs((L2[1][1] - L2[0][1]) + 0.1))
                                            xp, yp = (xp + DX), (yp + DY)
                                            L2[0][0], L2[0][1] = xp, yp

                                        else:
                                            DX = mt.ceil((L2[0][0] - L2[1][0]) / abs((L2[0][0] - L2[1][0]) + 0.1))
                                            DY = mt.ceil((L2[0][1] - L2[1][1]) / abs((L2[0][1] - L2[1][1]) + 0.1))
                                            xp, yp = (xp + DX), (yp + DY)
                                            L2[1][0], L2[1][1] = xp, yp

                                    InDelSeg.append(y)

                                # Перпендикуляр от второго отрезка (значит укорачиваем первый отрезок)
                                if IMinD == 2 or IMinD == 3:
                                    if 0 in I:
                                        DX = mt.ceil((L1[1][0] - L1[0][0]) / abs((L1[1][0] - L1[0][0]) + 0.1))
                                        DY = mt.ceil((L1[1][1] - L1[0][1]) / abs((L1[1][1] - L1[0][1]) + 0.1))
                                        xp, yp = (xp + DX), (yp + DY)
                                        # Корректируем первую точку первого отрезка
                                        L1[0][0], L1[0][1] = xp, yp

                                    elif 1 in I:
                                        DX = mt.ceil((L1[0][0] - L1[1][0]) / abs((L1[0][0] - L1[1][0]) + 0.1))
                                        DY = mt.ceil((L1[0][1] - L1[1][1]) / abs((L1[0][1] - L1[1][1]) + 0.1))
                                        xp, yp = (xp + DX), (yp + DY)
                                        # Корректируем вторую точку первого отрезка
                                        L1[1][0], L1[1][1] = xp, yp

                                    else:  # Если перпендикуляр только один, пододвигаем ближайшую к перпендикуляру точку
                                        D1 = DNMathAdd.CalcEvkl([xp, yp], L1[0])
                                        D2 = DNMathAdd.CalcEvkl([xp, yp], L1[1])
                                        if min(D1, D2) == D1:
                                            DX = mt.ceil((L1[1][0] - L1[0][0]) / abs((L1[1][0] - L1[0][0]) + 0.1))
                                            DY = mt.ceil((L1[1][1] - L1[0][1]) / abs((L1[1][1] - L1[0][1]) + 0.1))
                                            xp, yp = (xp + DX), (yp + DY)
                                            L1[0][0], L1[0][1] = xp, yp
                                        else:
                                            DX = mt.ceil((L1[0][0] - L1[1][0]) / abs((L1[0][0] - L1[1][0]) + 0.1))
                                            DY = mt.ceil((L1[0][1] - L1[1][1]) / abs((L1[0][1] - L1[1][1]) + 0.1))
                                            xp, yp = (xp + DX), (yp + DY)
                                            L1[1][0], L1[1][1] = xp, yp

                                    InDelSeg.append(x)

                                LFilt[x] = [L1[0][0], L1[0][1], L1[1][0], L1[1][1]]
                                LFilt[y] = [L2[0][0], L2[0][1], L2[1][0], L2[1][1]]

                                # print(L1,L2)
                                # print(x,y)
                                # print('\n')

            # Переводим координаты точек линий в координаты изображения
            LineOrder = np.array(LFilt)
            # for L in LFilt:
            #     L1 = [L.tolist()[:2], L.tolist()[2:]]
            #     L2 = [L.tolist()[:2], L.tolist()[2:]]
            #     print(L1)
            #     print(L2)
            #     print("\n")
            # print("\n")
        else:
            LineOrder = np.array(LineOrder)

        # Переупорядочиваем линии с учетом удаленных и отфильтрованных
        LineOrder2 = []
        for L in LineOrder:
            LineOrder2.append(L)
        LineOrder2 = np.array(LineOrder2)

        LineOrder2 = DNTheam.FLinesOrder(LineOrder2)
        LineOrder2 = np.array(LineOrder2)

        x1 = LineOrder2[:, 1] + XMin
        y1 = LineOrder2[:, 0] + YMin
        x2 = LineOrder2[:, 3] + XMin
        y2 = LineOrder2[:, 2] + YMin

        LineOrderImg = np.vstack([x1, y1, x2, y2]).transpose()

        # Удаление линий, которые в результате преобразований выродились в 0
        LineOrderImgNoNull = []
        for Line in LineOrderImg:
            p1 = Line[:2]
            p2 = Line[2:]
            d = (p1[0] - p2[0]) + (p1[1] - p2[1])
            if not d == 0:
                LineOrderImgNoNull.append(Line)
        LineOrderImgNoNull = np.array(LineOrderImgNoNull)

        return LineOrderImgNoNull

    # Аппроксимация контура окружностями
    @classmethod
    def ApproxCircleContur(cls, Contur: [], DPor: float, HPor: float):
        # Определяем габаритные размеры контура
        XMin = min(Contur[:, 0])
        XMax = max(Contur[:, 0])
        YMin = min(Contur[:, 1])
        YMax = max(Contur[:, 1])

        WContur = XMax - XMin + 1
        HContur = YMax - YMin + 1

        # Генерирование бинарной картинки из контура
        X_Mass = list(range(XMin, XMax + 1))
        Y_Mass = list(range(YMin, YMax + 1))

        P_Mass = []
        for x in X_Mass:
            for y in Y_Mass:
                P_Mass.append((x, y))

        pPath = path.Path(Contur)
        Mask = pPath.contains_points(P_Mass)
        Mask = np.array(Mask).reshape(-1, HContur)

        SegImg = np.zeros([WContur, HContur], dtype=np.uint8)
        S_Seg = 0
        for P in P_Mass:
            if Mask[P[0] - XMin, P[1] - YMin]:
                SegImg[P[0] - XMin][P[1] - YMin] = 255
                S_Seg += 1

        # Аппроксимация сегмента окружностью
        Circles = cv.HoughCircles(SegImg, cv.HOUGH_GRADIENT, 1, min(WContur, HContur) / 2,
                                  param1=255, param2=1,
                                  minRadius=int(min(WContur, HContur) / 2) - 1,
                                  maxRadius=int(max(WContur, HContur) / 2))

        if Circles is None:
            return []

        Circles = np.uint16(np.around(Circles))
        Razn = 1000
        NumRes = -1
        i = 0

        for Circle in Circles[0, :]:
            C = (Circle[0], Circle[1])
            R = Circle[2]
            S_C = np.pi * R * R
            # cv.circle(SegImg,C,R,128,2)

            if S_Seg / S_C < HPor and S_Seg / S_C > DPor \
                    and abs(1 - S_Seg / S_C) < Razn:
                Razn = abs(1 - S_Seg / S_C)
                NumRes = i
            i += 1

            # print(S_Seg/S_C)
            # plt.imshow(SegImg)
            # plt.show()

        if NumRes < 0:
            return []

        xC = Circles[0, NumRes][0] + XMin
        yC = Circles[0, NumRes][1] + YMin
        R = Circles[0, NumRes][2]
        return [xC, yC, R]

    # Поиск взаимно перпендикулярных и параллельных прямых
    @classmethod
    def FineParallelPerpendLines(cls, Lines: [], d_alfa):
        # Копируем линии в новыймассив, чтобы не испортить исходный
        LinesC = Lines.copy()

        # Определяем углы наклона всех прямых
        alfaL = []
        for Line in LinesC:
            p1 = Line[:2]
            p2 = Line[2:]
            wl = p1[0] - p2[0]
            hl = p1[1] - p2[1]

            if wl == 0:
                alfa = 90
            else:
                alfa = np.arctan(hl / wl) * 180 / np.pi

            alfaL.append(alfa)

        alfaL = np.array(alfaL)

        LinesP = []  # Результат будет записан как массив массивов взаимно перпендикулярных и параллельных прямых

        while not len(LinesC) == 0:
            LinesP.append([LinesC[0]])
            alfaP = alfaL[0]

            LinesC = np.delete(LinesC, 0, 0)
            alfaL = np.delete(alfaL, 0, 0)

            # Узнаем индексы всех параллельных и перпендикулярных линий
            IndxP = []
            for i in range(len(LinesC)):
                corner = alfaP - alfaL[i]
                if abs(corner) < d_alfa or \
                        (abs(corner) > 90 - d_alfa and abs(corner) < 90 + d_alfa) or \
                        (abs(corner) > 180 - d_alfa and abs(corner) < 180 + d_alfa):
                    IndxP.append(i)

            for i in IndxP:
                LinesP[-1].append(LinesC[i])

            LinesC = np.delete(LinesC, IndxP, 0)
            alfaL = np.delete(alfaL, IndxP, 0)

        # Оставляем только массивы, состоящие из нескольких линий (больше чем одной)

        IndxP = []
        for i in range(len(LinesP)):
            if len(LinesP[i]) >= 2:
                IndxP.append(i)

        Res = []
        for i in IndxP:
            Res.append(LinesP[i])

        #        Res = np.array(Res)
        return Res

    # Объединение параллельных линий с маленьким расстоянием (на вход подается выход из функции FineParallelPerpendLines)
    @classmethod
    def LinesPObed(cls, LinesP: [], alfa_por, DistPor):
        if len(LinesP) == 0:
            return []
        LinesP = DNTheam.FLinesOrder(LinesP)

        # Разбиваем массив линий на две группы параллельных линий
        P11 = LinesP[0][:2]
        P12 = LinesP[0][2:]
        IndxPar1 = []
        IndxPar2 = []
        IndxPar1.append(0)
        for i in range(len(LinesP)):
            if not i == 0:
                P21 = LinesP[i][:2]
                P22 = LinesP[i][2:]
                D = DNMathAdd.LinesDistCorner(P11, P12, P21, P22)
                if D[0] < alfa_por or (D[0] <= 180 + alfa_por and D[0] >= 180 - alfa_por):
                    IndxPar1.append(i)
                else:
                    IndxPar2.append(i)

        IndxPar = [IndxPar1, IndxPar2]

        # Составляем два массива линий (одна группа линий перпендикулярна другой группе)
        LMass = []
        for Indx in IndxPar:
            Lines = []
            for i in Indx:
                Lines.append(LinesP[i])
            LMass.append(Lines)

        # Ищем линии, которые можно объединить
        ResultLines = []
        for Lines in LMass:
            while 1:  # Цикл, пока нечего будет объединять
                IsObed = False
                # Ищем пару линий для объединения
                IndLineObed1 = 0
                LineObed = []
                for Line1 in Lines:
                    P11 = Line1[:2]
                    P12 = Line1[2:]
                    L1 = DNMathAdd.CalcEvkl(P11, P12)
                    IndLineObed2 = 0
                    DLMin = 1000  # Если минимальное расстояние между точками отрезков больше 1000, то их не объединяем
                    for Line2 in Lines:
                        if not np.all(Line1 == Line2):
                            P21 = Line2[:2]
                            P22 = Line2[2:]
                            D = DNMathAdd.LinesDistCorner(P11, P12, P21, P22)
                            L2 = DNMathAdd.CalcEvkl(P21, P22)
                            L = max(L1, L2)
                            DLP = [DNMathAdd.CalcEvkl(P11, P21), DNMathAdd.CalcEvkl(P11, P22),
                                   DNMathAdd.CalcEvkl(P12, P21), DNMathAdd.CalcEvkl(P12, P22)]
                            DL = min(DLP)
                            # Условие объединения линий
                            if D[1] <= DistPor and DL < DistPor:  # L/2:
                                IsObed = True
                                IndP = DLP.index(DL)
                                # Формируем объединяющую линию
                                if IndP == 0:
                                    LineObed = [P12[0], P12[1], P22[0], P22[1]]
                                elif IndP == 1:
                                    LineObed = [P12[0], P12[1], P21[0], P21[1]]
                                elif IndP == 2:
                                    LineObed = [P11[0], P11[1], P22[0], P22[1]]
                                elif IndP == 3:
                                    LineObed = [P11[0], P11[1], P21[0], P21[1]]
                                LineObed = np.array(LineObed)
                                break
                        IndLineObed2 += 1
                    # Если найдены пары линий, которые можно объединить
                    if IsObed:
                        Lines[IndLineObed1] = LineObed
                        Lines.pop(IndLineObed2)
                        break

                    IndLineObed1 += 1

                # Если объединять больше нечего, выходим из цикла
                if not IsObed: break

            # Записываем результат с уже объединенными линиями в новый массив
            for Line in Lines:
                ResultLines.append(Line)

        ResultLines = DNTheam.FLinesOrder(ResultLines)
        return ResultLines

    # Объединение в группы параллельных и перпендикулярных линий с маленьким расстоянием
    # (на вход подается выход из функции FineParallelPerpendLines)
    @classmethod
    def LinesPPGrOb(cls, LinesP: [], DistPorDel, DistPorOb):

        if len(LinesP) == 0:
            return []

        LinesC = LinesP.copy()

        ResultGrLines = []

        i = 0
        while 1:
            if i == len(LinesC):
                break
            Ps1 = [LinesC[i][:2], LinesC[i][2:]]
            IsLineDel = False
            for j in range(len(LinesC)):
                D = []
                if not i == j:
                    Ps2 = [LinesC[j][:2], LinesC[j][2:]]
                    for P in Ps1:
                        D.append(DNMathAdd.PointLineDist(Ps2[0], Ps2[1], P))

                    # Удаляем параллельные близкие друг к другу отрезки
                    if D[0][2] and D[1][2] and D[0][0] < DistPorDel and D[1][0] < DistPorDel:
                        LinesC.pop(i)
                        IsLineDel = True
                        break

                    # Объединяем близкие и параллельные друг к другу отрезки
                    # Если первая точка i-того отрезка выходит за пределы j-того отрезка
                    elif not D[0][2] and D[1][2] and D[0][0] < DistPorOb and D[1][0] < DistPorOb:
                        if D[0][4] == 0:
                            LinesC[j] = [Ps1[0][0], Ps1[0][1], Ps2[1][0], Ps2[1][1]]
                        if D[0][4] == 1:
                            LinesC[j] = [Ps2[0][0], Ps2[0][1], Ps1[0][0], Ps1[0][1]]

                        LinesC.pop(i)
                        IsLineDel = True
                        # Начинаем проверять все отрезки заново, начиная с объединенного
                        if j < i: i = j
                        break

                    elif D[0][2] and not D[1][2] and D[0][0] < DistPorOb and D[1][0] < DistPorOb:
                        if D[1][4] == 0:
                            LinesC[j] = [Ps1[1][0], Ps1[1][1], Ps2[1][0], Ps2[1][1]]
                        if D[1][4] == 1:
                            LinesC[j] = [Ps2[0][0], Ps2[0][1], Ps1[1][0], Ps1[1][1]]

                        LinesC.pop(i)
                        IsLineDel = True
                        # Начинаем проверять все отрезки заново, начиная с объединенного
                        if j < i: i = j
                        break

            if not IsLineDel:
                i += 1
            else:
                print(i, len(LinesC))

        # Объединяем близкие и параллельные друг к другу отрезки

        return LinesC

        # # Ищем линии, которые можно объединить
        # ResultGrLines=[]
        # while 1:
        #     if len(LinesC)==0:
        #         break
        #     # Массив линий, входящих в одну группу
        #     LocalGrLines=[LinesC[0]]
        #     LinesC.pop(0)
        #
        #     # Проверяем каждую линию в группе
        #     for Line in LocalGrLines:
        #         P1 = Line[:2]
        #         P2 = Line[2:]
        #
        #         # с каждой линией вне группы
        #         j=0
        #         while j<len(LinesC):
        #             Ps=[LinesC[j][:2],LinesC[j][2:]]
        #             D=[]
        #             for P in Ps:
        #                 D.append(DNMathAdd.PointLineDist(P1,P2,P))
        #
        #             # Если оба конца отрезка лежат в пределах линии из группы и расстояние их мало
        #             # удаляем этот отрезок
        #             if D[0][2] and D[1][2] and D[0][0]<DistPorDel and D[1][0]<DistPorDel:
        #                 LinesC.pop(j)
        #
        #             # Если любой из концов отрезка лежит на расстоянии объединения в группу, объединяем
        #             elif D[0][0]<=DistPorOb or D[1][0]<=DistPorOb:
        #                 LocalGrLines.append(LinesC[j])
        #                 LinesC.pop(j)
        #
        #             # В противном случае переходим к следующему отрезку
        #             else: j+=1
        #
        #     ResultGrLines.append(LocalGrLines)

        return ResultGrLines

    # Убираем отрезки, которые параллельны друг другу, но не находятся друг против друга (смещены) или находятся на значительном расстоянии друг от друга
    @classmethod
    def ParalelSedimentFilterShift(cls, LinesP: [], d_alf, DistPor):
        if len(LinesP) == 0:
            return []
        LinesP = DNTheam.FLinesOrder(LinesP)

        # Разбиваем массив линий на две группы параллельных линий
        P11 = LinesP[0][:2]
        P12 = LinesP[0][2:]
        IndxPar1 = []
        IndxPar2 = []
        IndxPar1.append(0)
        for i in range(len(LinesP)):
            if not i == 0:
                P21 = LinesP[i][:2]
                P22 = LinesP[i][2:]
                D = DNMathAdd.LinesDistCorner(P11, P12, P21, P22)
                if D[0] < d_alf or (D[0] <= 180 + d_alf and D[0] >= 180 - d_alf):
                    IndxPar1.append(i)
                else:
                    IndxPar2.append(i)

        IndxPar = [IndxPar1, IndxPar2]

        # Составляем два массива линий (одна группа линий перпендикулярна другой группе)
        LMass = []
        for Indx in IndxPar:
            Lines = []
            for i in Indx:
                Lines.append(LinesP[i])
            LMass.append(Lines)

        # Убираем параллельные линии не из одной точки которых нельзя опустить перпендикуляр на другие параллельне линии
        ResultLines = []
        b = True
        for Lines in LMass:
            i = int(b)
            for Line1 in Lines:
                P11 = Line1[:2]
                P12 = Line1[2:]
                L1 = DNMathAdd.CalcEvkl(P11, P12)
                IsPerpFind = False
                for Line2 in Lines:
                    if not np.all(Line1 == Line2):
                        P21 = Line2[:2]
                        P22 = Line2[2:]
                        L2 = DNMathAdd.CalcEvkl(P21, P22)
                        Dist = DNMathAdd.LinesDistCorner(P11, P12, P21, P22)
                        LMax = max(L1, L2)
                        Corn = []
                        D = DNMathAdd.LinesDistCorner(P11, P12, P11, P21)
                        Corn.append(D[0])
                        D = DNMathAdd.LinesDistCorner(P11, P12, P11, P22)
                        Corn.append(D[0])
                        D = DNMathAdd.LinesDistCorner(P11, P12, P12, P21)
                        Corn.append(D[0])
                        D = DNMathAdd.LinesDistCorner(P11, P12, P12, P22)
                        Corn.append(D[0])
                        Corn = np.array(Corn)
                        R1 = Corn >= 87
                        R2 = Corn <= 93
                        alf_i = 0
                        for alf in Corn:
                            # Условие:
                            # 1. Угол между одним из концов отрезка и другим отрезком равен 90 грд
                            # 2. Разница между длинами линий менее чем в 2 раза
                            # 3. Расстояние между линиями меньше длины наибольшей линии
                            if alf >= 90 - d_alf and alf <= 90 + d_alf and L1 >= LMax / 2 and Dist[1] < LMax:
                                # 4. Второй конец отрезка находится напротив первого отрезка
                                D = []
                                if alf_i == 0:
                                    D.append(DNMathAdd.PointLineDist(P11, P12, P22))
                                    D.append(DNMathAdd.PointLineDist(P21, P22, P12))
                                elif alf_i == 1:
                                    D.append(DNMathAdd.PointLineDist(P11, P12, P21))
                                    D.append(DNMathAdd.PointLineDist(P21, P22, P12))
                                elif alf_i == 2:
                                    D.append(DNMathAdd.PointLineDist(P11, P12, P22))
                                    D.append(DNMathAdd.PointLineDist(P21, P22, P11))
                                elif alf_i == 3:
                                    D.append(DNMathAdd.PointLineDist(P11, P12, P21))
                                    D.append(DNMathAdd.PointLineDist(P21, P22, P11))

                                # Если вторая точка внутри отрезка
                                if D[0][2] or D[1][2]:
                                    IsPerpFind = True
                                    break
                                # # Если вторая точка снаружи отрезка но на минимальном расстоянии
                                # elif min(DNMathAdd.CalcEvkl(D[0][1],P11),DNMathAdd.CalcEvkl(D[1],P12))<DistPor:
                                #     IsPerpFind = True
                                #     break

                            alf_i += 1
                        if IsPerpFind:
                            break
                if IsPerpFind:
                    ResultLines.append(Line1)

                # Если параллельные линии смещены, но
                else:
                    for Line2 in LMass[i]:
                        P21 = Line2[:2]
                        P22 = Line2[2:]
                        # в окрестностях предположительно удаляемой линии находится отрезок перпендикулярный ей
                        D = min(DNMathAdd.CalcEvkl(P11, P21), DNMathAdd.CalcEvkl(P11, P22),
                                DNMathAdd.CalcEvkl(P12, P21), DNMathAdd.CalcEvkl(P12, P22))
                        if D < DistPor:
                            ResultLines.append(Line1)
                            break

            b = not b

        if not len(ResultLines) == 0:
            ResultLines = DNTheam.FLinesOrder(ResultLines)

        return ResultLines

    # Убираем отрезки, которые более двух выходят из одной точки (буквы "т" из трех отрезков, кресты и т.д.)
    @classmethod
    def FilterCrossLine(cls, LinesP: [], d_alf, DistPor):
        if len(LinesP) == 0:
            return []
        LinesP = DNTheam.FLinesOrder(LinesP)

        # Разбиваем массив линий на две группы параллельных линий
        P11 = LinesP[0][:2]
        P12 = LinesP[0][2:]
        IndxPar1 = []
        IndxPar2 = []
        IndxPar1.append(0)
        for i in range(len(LinesP)):
            if not i == 0:
                P21 = LinesP[i][:2]
                P22 = LinesP[i][2:]
                D = DNMathAdd.LinesDistCorner(P11, P12, P21, P22)
                if D[0] < d_alf or (D[0] <= 180 + d_alf and D[0] >= 180 - d_alf):
                    IndxPar1.append(i)
                else:
                    IndxPar2.append(i)

        IndxPar = [IndxPar1, IndxPar2]

        # Составляем два массива линий (одна группа линий перпендикулярна другой группе)
        LMass = []
        for Indx in IndxPar:
            Lines = []
            for i in Indx:
                Lines.append(LinesP[i])
            LMass.append(Lines)

        # Проверяеь на наличие крестов
        b = True
        LinesDel = []
        for Lines in LMass:
            for Line1 in Lines:
                P11 = np.array(Line1[:2]).tolist()
                P12 = np.array(Line1[2:]).tolist()
                # Ищем близкие к отрезку перпендикуляры
                i = int(b)
                PProv = []
                LinesGroup = []  # Группировки линий по близости и перпендикулярности
                for Line2 in LMass[i]:
                    P21 = np.array(Line2[:2]).tolist()
                    P22 = np.array(Line2[2:]).tolist()
                    D = [DNMathAdd.CalcEvkl(P11, P21), DNMathAdd.CalcEvkl(P12, P21), DNMathAdd.CalcEvkl(P11, P22),
                         DNMathAdd.CalcEvkl(P12, P22)]
                    DMin = min(D)
                    IndxDMin = D.index(DMin)
                    if DMin <= DistPor:
                        if IndxDMin == 0 or IndxDMin == 2:
                            if not P11 in PProv:
                                PProv.append(P11)
                                LinesGroup.append([Line2])
                            else:
                                Indx = PProv.index(P11)
                                LinesGroup[Indx].append(Line2)
                        if IndxDMin == 1 or IndxDMin == 3:
                            if not P12 in PProv:
                                PProv.append(P12)
                                LinesGroup.append([Line2])
                            else:
                                Indx = PProv.index(P12)
                                LinesGroup[Indx].append(Line2)

                for LinesG in LinesGroup:
                    if len(LinesG) >= 2:
                        for Line in LinesG:
                            LinesDel.append(np.array(Line).tolist())

            b = not b

        Result = []
        for Line in LinesP:
            if not np.array(Line).tolist() in LinesDel:
                Result.append(np.array(Line))

        return Result

    # Убираем перпендикулярные отрезки, которые идут к середине других отрезков (буквы "Т")
    @classmethod
    def FilterTauLine(cls, LinesP: [], d_alf, DistPor):
        if len(LinesP) == 0:
            return []
        LinesP = DNTheam.FLinesOrder(LinesP)

        # Разбиваем массив линий на две группы параллельных линий
        P11 = LinesP[0][:2]
        P12 = LinesP[0][2:]
        IndxPar1 = []
        IndxPar2 = []
        IndxPar1.append(0)
        for i in range(len(LinesP)):
            if not i == 0:
                P21 = LinesP[i][:2]
                P22 = LinesP[i][2:]
                D = DNMathAdd.LinesDistCorner(P11, P12, P21, P22)
                if D[0] < d_alf or (D[0] <= 180 + d_alf and D[0] >= 180 - d_alf):
                    IndxPar1.append(i)
                else:
                    IndxPar2.append(i)

        IndxPar = [IndxPar1, IndxPar2]

        # Составляем два массива линий (одна группа линий перпендикулярна другой группе)
        LMass = []
        for Indx in IndxPar:
            Lines = []
            for i in Indx:
                Lines.append(LinesP[i])
            LMass.append(Lines)

        # Определяем линии, которые перпендикулярны и идут к середине отрезков
        b = True
        LinesDel = []
        LinesDelIndxMass = []
        for Lines in LMass:
            for Line1 in Lines:
                P11 = np.array(Line1[:2]).tolist()
                P12 = np.array(Line1[2:]).tolist()
                # Ищем перпендикуляры в который упирается отрезок
                i = int(b)
                for Line2 in LMass[i]:
                    P21 = np.array(Line2[:2]).tolist()
                    P22 = np.array(Line2[2:]).tolist()

                    # Проверка, пересекаются ли отрезки
                    D = DNMathAdd.SegmentsPointIntersection(P11, P12, P21, P22)
                    # Если отрезки пересекаются
                    if D['IsSegInter']:
                        # Считаем, что отрезок нужно удалять, если расстояние от точки пересечения
                        # до обоих концов отрезков больше порогового значения
                        # заносим такие линии в кандидаты на удаление
                        P = [D['x'], D['y']]
                        Dist = min(DNMathAdd.CalcEvkl(P11, P), DNMathAdd.CalcEvkl(P12, P))
                        if Dist > DistPor and not np.array(Line2).tolist() in LinesDel:
                            LinesDel.append(np.array(Line2).tolist())
                            LinesDelIndxMass.append(i)
                    # Если отрезки не пересекаются
                    else:
                        # Считаем минимальное расстояние от конца одного отрезка до другого
                        D = []
                        D.append(DNMathAdd.PointLineDist(P11, P12, P21))
                        D.append(DNMathAdd.PointLineDist(P11, P12, P22))
                        Indx = [D[0][0], D[1][0]].index(min(D[0][0], D[1][0]))
                        P = D[Indx][1]
                        # Если точка перпендикуляра лежит на отрезке, а расстояние от исходной точки до отрезка меньше порогового
                        if D[Indx][2] and D[Indx][0] <= DistPor:
                            Dist = min(DNMathAdd.CalcEvkl(P11, P), DNMathAdd.CalcEvkl(P12, P))
                            if Dist > DistPor and not np.array(Line2).tolist() in LinesDel:
                                LinesDel.append(np.array(Line2).tolist())
                                LinesDelIndxMass.append(i)

            b = not b

        Result = []
        for Line in LinesP:
            if not np.array(Line).tolist() in LinesDel:
                Result.append(np.array(Line))

        return Result

    # Пересекаются ли линии в многоугольнике (проверка на правильность многоугольника)
    @classmethod
    def PolyIs(cls, PtsOrder: [], DistPor):
        IsPPoly = True
        # Проверяем каждый отрезок в полигоне
        for i in range(len(PtsOrder)):
            j = i + 1
            if j == len(PtsOrder):
                j = 0
            P11 = PtsOrder[i]
            P12 = PtsOrder[j]

            # Сопостовляем с каждым не смежным отрезком в полигоне
            for k in range(len(PtsOrder)):
                if not (k == i or k == j):
                    z = k + 1
                    if z == len(PtsOrder):
                        z = 0
                    if not (z == i or z == j):
                        P21 = PtsOrder[k]
                        P22 = PtsOrder[z]
                        # Если отрезок не приходит в ту же точку из которой исходит другой отрезок
                        if not (P21 == P11 or P21 == P12 or P22 == P11 or P22 == P12):
                            D = DNMathAdd.SegmentsPointIntersection(P11, P12, P21, P22)
                            # Если отрезки пересекаются
                            if D['IsSegInter']:
                                # Проверяем, на каком расстоянии от концов отрезка лежит точка пересечения
                                P = [D['x'], D['y']]
                                Dist = [DNMathAdd.CalcEvkl(P, P11), DNMathAdd.CalcEvkl(P, P12)]
                                IndxD = Dist.index(min(Dist))
                                # Если отрезки пересекаются,но точка пересечения близка к концу отрезка
                                # то редактируем полигон
                                if Dist[IndxD] <= DistPor:
                                    if IndxD == 0: PtsOrder[i] = P
                                    if IndxD == 1: PtsOrder[j] = P
                                else:
                                    IsPPoly = False
                                    return [IsPPoly, PtsOrder]
        return [IsPPoly, PtsOrder]

    # Группируем близкие друг к другу отрезки (отрезки должны либо пересекаться), либо концами находится близко друг к другу
    @classmethod
    def SedimentsGrop(cls, Lines: [], DistPorL, DistPorP):
        IndexG = []  # Индексы сгруппированных линий
        IndexNoObr = []  # Индексы не обработанных линий
        for i in range(len(Lines)):
            IndexNoObr.append(i)
        # Проверяем каждую линию на предмет возможности ее объединения с другими линиями
        for i in range(len(Lines)):
            if i in IndexNoObr:
                Gr = []  # Индексы линий, которые будут объеденены
                IndexNoObr.pop(IndexNoObr.index(i))
                Gr.append(i)
                # Крутим цикл до тех пор, пока объединять будет нечего
                while 1:
                    IsObed = False  # Произошло ли объединение
                    for j in Gr:  # На объединение проверяем каждую линию в группе с каждой необработанной линией
                        P11 = Lines[j][:2]
                        P12 = Lines[j][2:]
                        for n in IndexNoObr:
                            P21 = Lines[n][:2]
                            P22 = Lines[n][2:]
                            D = []
                            D.append(DNMathAdd.PointLineDist(P11, P12, P21))
                            D.append(DNMathAdd.PointLineDist(P11, P12, P22))
                            D.append(DNMathAdd.PointLineDist(P21, P22, P11))
                            D.append(DNMathAdd.PointLineDist(P21, P22, P12))

                            DistMin = min(D[0][0], D[1][0], D[2][0], D[3][0])
                            IndxDMin = [D[0][0], D[1][0], D[2][0], D[3][0]].index(DistMin)
                            # Если расстояние от какого либо из края отрезка до прямой меньше порогового и...
                            # точка перпендикуляра лежит внутри отрезка
                            # или точка точка перпендикуляра вне отрезка, но
                            if (DistMin <= DistPorL and D[IndxDMin][2]) or \
                                    (DistMin <= DistPorL and not D[IndxDMin][2] and D[IndxDMin][3] <= DistPorP):
                                IndexNoObr.pop(IndexNoObr.index(n))
                                Gr.append(n)
                                IsObed = True
                                break
                        # Если произошло объединение с какой-либо из линий возвращаемся в цикл while
                        if IsObed:
                            break
                    # Если объединять больше нечего в данной группе, выходим из while
                    if not IsObed:
                        IndexG.append(Gr)
                        break

        # Объединяем линии в группы
        Res = []
        for Gr in IndexG:
            LinesG = []
            for Indx in Gr:
                LinesG.append(Lines[Indx])
            Res.append(LinesG)

        return Res

    # Определение габаритных размеров фигуры обозначенной контуром (по минимально описанному прямоугольнику)
    @classmethod
    def GetWidHeighContur(cls, Contur: []):
        Rect = cv.minAreaRect(Contur)
        WContur = Rect[1][0]
        HContur = Rect[1][1]
        SMin = WContur * HContur
        alfaMin = Rect[2]

        # # Определяем габаритные размеры контура
        # XMin = min(Contur[:,0])
        # XMax = max(Contur[:,0])
        # YMin = min(Contur[:,1])
        # YMax = max(Contur[:,1])
        #
        # # Переводим координаты контура в локальные координаты
        # XMass=Contur[:,0]-XMin
        # YMass=Contur[:,1]-YMin
        #
        # D=2 # Ширина полей вокруг картинки
        # XMass=XMass+D
        # YMass=YMass+D
        # W=XMax-XMin+D*2
        # H=YMax-YMin+D*2
        #
        # # Генерация картинки (получение из контура сегмента, т.к. дальнейшая работа будет проводится с сегментом,
        # # для того, чтобы конечный повернутый контур был замкнутым)
        # Img=np.zeros([W,H],dtype=np.uint8)
        # Img[XMass,YMass]=255
        #
        # P=[]
        # for x in range(W):
        #     for y in range(H):
        #         P.append((x,y))
        #
        # Cont=np.array((XMass,YMass)).transpose()
        # pPath=path.Path(Cont)
        # MassSeg=pPath.contains_points(P)
        # MassSeg=np.array(MassSeg).reshape(W,-1)
        #
        # Img=Img+(MassSeg*255)
        # Img=Img.astype("uint8")
        #
        # # Помещаем картинку в центр более большой картинки, чтобы при повороте все пиксели влезли в большую картинку
        # NewW=int(np.sqrt(W*W+H*H))+1
        # NewH=int(np.sqrt(W*W+H*H))+1
        #
        # ShiftMatr=np.float32([[1,0,NewH/2-H/2], [0,1,NewW/2-W/2]])
        # ShiftImg=cv.warpAffine(Img,ShiftMatr,(NewH,NewW))
        #
        # # Центр новоой картинки
        # CImg=(NewH/2,NewW/2)
        # SMin=NewH*NewW
        # alfaMin=0
        # WContur=NewW
        # HContur=NewH
        #
        # for alfa in range(90):
        #     RotMatr=cv.getRotationMatrix2D(CImg,alfa,1)
        #     RotImg=cv.warpAffine(ShiftImg,RotMatr,(NewW,NewH))
        #
        #     # Новое повернутое изображение записывается не в бинарном виде, а в градациях серого
        #     # Поэтому бинаризуем его по порогу 128
        #     BinRotImg = cv.inRange(RotImg,128,255)
        #
        #     # Ищим минимальные значения x и y нового изображения
        #     P=np.column_stack(np.where(BinRotImg==255))
        #     XMin=min(P[:,1])
        #     YMin=min(P[:,0])
        #
        #     XMax=max(P[:,1])
        #     YMax=max(P[:,0])
        #
        #     WImg=XMax-XMin+1
        #     HImg=YMax-YMin+1
        #     S=WImg*HImg
        #
        #     if S<SMin:
        #         SMin=S
        #         alfaMin=alfa
        #         WContur=WImg
        #         HContur=HImg

        if WContur > HContur:
            HContur, WContur = WContur, HContur

        return {"W": WContur, "H": HContur, "S": SMin, "alfa": alfaMin}

    # Расчет наименьшего расстояния между контурами
    @classmethod
    def CalcMinDistCont(cls, Contur1: [], Contur2: []):
        if len(Contur1) == 0 or len(Contur2) == 0:
            return -1
        DMin = DNMathAdd.CalcEvkl(Contur1[0], Contur2[0])
        P1 = []
        P2 = []
        NDx1 = 0
        NDx2 = 0
        i = 0
        j = 0
        for Pt1 in Contur1:
            for Pt2 in Contur2:
                D = DNMathAdd.CalcEvkl(Pt1, Pt2)
                if D < DMin:
                    DMin = D
                    P1 = Pt1
                    P2 = Pt2
                    NDx1 = i
                    NDx2 = j
                j += 1
            i += 1

        return [DMin, P1, P2, NDx1, NDx2]

    # Рабочая функция по отрисовке контура и линий
    @classmethod
    def PrintContLines(cls, Contur: [], Lines: []):
        W = max(Contur[:, 0]) - min(Contur[:, 0])
        H = max(Contur[:, 1]) - min(Contur[:, 1])
        imgCont = np.zeros([W + 2, H + 2], dtype=np.uint8)

        imgLinesMass = []
        for i in range(len(Lines)):
            imgLinesMass.append(np.zeros([W + 2, H + 2], dtype=np.uint8))

        x = Contur[:, 0] - min(Contur[:, 0]) + 1
        y = Contur[:, 1] - min(Contur[:, 1]) + 1
        imgCont[x, y] = 255

        NumImg = 0
        for LinesImg in Lines:
            NumObj = 0
            for LinesObj in LinesImg:
                for Line in LinesObj:
                    x = [Line[0], Line[2]]
                    y = [Line[1], Line[3]]
                    xp = x - min(Contur[:, 0]) + 1
                    yp = y - min(Contur[:, 1]) + 1
                    cv.line(imgLinesMass[NumImg], [yp[0], xp[0]], [yp[1], xp[1]], NumObj + 1, 1)
                NumObj += 1
            NumImg += 1

        fig = plt.figure()
        rows = 1
        cols = len(Lines) + 1

        fig.add_subplot(rows, cols, 1)
        plt.imshow(imgCont.transpose())

        for i in range(len(Lines)):
            fig.add_subplot(rows, cols, i + 2)
            plt.imshow(imgLinesMass[i].transpose())

        plt.show()

    # Тематические задачи
    # Выделение зданий
    @classmethod
    def DetectBuild(cls, ClsMass: [], ProgressBar=None):
        W, H = np.shape(ClsMass)
        # Получение контуров всех сегментов во всех классах
        ContursImg = []
        for NCls in np.unique(ClsMass):
            # Подготовка нового массива для обработки (капец, сколько памяти тратится)
            Img = np.zeros([W, H], dtype=np.uint8)
            P = np.column_stack(np.where(np.array(ClsMass) == NCls))
            Img[P[:, 0], P[:, 1]] = 255

            # Получение контуров сегментов
            Conturs = cls.GetContursClass(Img, 255)
            ContursImg.append(Conturs)

        ImgCont = np.zeros([W, H], dtype=np.uint8)
        ContursAll = []

        for ContursCls in ContursImg:
            for Contur in ContursCls:
                ImgCont[Contur[:, 0], Contur[:, 1]] = 255
                ContursAll.append(Contur)

        # Фильтрация контуров, которые относятся к границам полигона
        ContursNoBord = []
        for Contur in ContursAll:
            VarMass = ClsMass[Contur[:, 0], Contur[:, 1]]
            if not np.all(VarMass < 0):
                ContursNoBord.append(Contur)

        NumsSeg = []  # Номера сегментов, которые последовательно будут отфильтрованы по признакам
        ContursParam = []
        for i in range(len(ContursNoBord)):
            NumsSeg.append(i)
            ContursParam.append({'Area': -1,
                                 'W': -1,
                                 'H': -1,
                                 'SMin': -1,
                                 'Lines': [],
                                 'PLines': -1})

        BuildingPolys = []  # Массив, куда будут загоняться полигоны, относящиеся к зданиям

        # 1. Определяем параметры контуров (площадь, длину, ширину, площадь описанного прямоугольника)
        # Пороговые значения: S<150, W<10, S/SRecMin<0.5
        i = 0
        for Contur in ContursNoBord:
            Area = cls.GetAreaContur(Contur)
            WidHeigh = cls.GetWidHeighContur(Contur)
            ContursParam[i]['Area'] = Area
            ContursParam[i]['W'] = WidHeigh['W']
            ContursParam[i]['H'] = WidHeigh['H']
            ContursParam[i]['SMin'] = WidHeigh['S']
            i += 1

        # 2. Аппроксимация контура прямыми (какая часть контура аппроксимируется прямыми)
        for i in range(len(ContursNoBord)):
            ProgressBar.ChangeProcState(len(ContursNoBord), i, "Анализ геометрических характеристик сегментов")

            LinesCont = []

            # Аппроксимируем только контура подходящей площади и линейных размеров
            Lim = 800  # Лимит длины контура
            N = 0
            if ContursParam[i]['Area'] >= 150 and ContursParam[i]['W'] > 10:
                if len(ContursNoBord[i]) < Lim:
                    LinesCont = cls.ApproxLinesContur(ContursNoBord[i], 10, 10)
                # Если контур больше 1000 пикселей, для ускорения аппроксимации разбиваем его на составные части
                else:
                    LinesCont = []
                    N = int(len(ContursNoBord[i]) / Lim)
                    NOst = len(ContursNoBord[i]) % Lim
                    for l in range(N):
                        PartContur = ContursNoBord[i][l * Lim:l * Lim + Lim]
                        LinesPart = cls.ApproxLinesContur(PartContur, 10, 10)
                        for Line in LinesPart:
                            LinesCont.append(Line)
                    if NOst > 0:
                        PartContur = ContursNoBord[i][N * Lim:N * Lim + NOst]
                        LinesPart = cls.ApproxLinesContur(PartContur, 10, 10)
                        for Line in LinesPart:
                            LinesCont.append(Line)

                if not len(LinesCont) == 0:  # Если контур аппроксимировался хотя бы одной линией
                    ContursParam[i]['Lines'].append(LinesCont)

                    # Фильтруем мусор (удаляем линии меньше 7 пикселей)
                    LinesCont7 = []
                    for Line in LinesCont:
                        P1 = Line[:2]
                        P2 = Line[2:]
                        if DNMathAdd.CalcEvkl(P1, P2) > 7:
                            LinesCont7.append(Line)

                    # Установка пороговых значений для алгоритмов фильтрации линий
                    d_alfa = 7  # Угол отклонения от 90 или 180 грд
                    DistPor = 5  # Расстояние между линиями для объединения
                    DistPorGr = 15  # Параметр для объединения линии в группы

                    # if i==783:
                    #     print(i)

                    # Ищем взаимно перпендикулярные и параллельные линии
                    LinesP = DNTheam.FineParallelPerpendLines(LinesCont7, d_alfa)

                    LinesPOb = []
                    for Lines in LinesP:
                        # Объединяем часть параллельных линий, расположенных близко друг к другу
                        LinesOb = DNTheam.LinesPObed(Lines, 7, 3)
                        # Убираем отрезки, которые параллельны друг другу, но смещены
                        LinesOb = DNTheam.ParalelSedimentFilterShift(LinesOb, d_alfa, DistPor)
                        if len(LinesOb) > 0:
                            LinesPOb.append(LinesOb)

                    # Считаем количество углов у объектов
                    LinesFilter = []
                    Indx = 0
                    for Lines in LinesPOb:
                        Lines = DNTheam.FLinesOrder(Lines)
                        N = 0
                        DMax = 0
                        for j in range(len(Lines)):
                            P11 = [Lines[j][0], Lines[j][1]]
                            P12 = [Lines[j][2], Lines[j][3]]
                            k = j + 1
                            if k == len(Lines): k = 0
                            P21 = [Lines[k][0], Lines[k][1]]
                            P22 = [Lines[k][2], Lines[k][3]]
                            D = DNMathAdd.LinesDistCorner(P11, P12, P21, P22)
                            if D[0] >= 90 - d_alfa and D[0] <= 90 + d_alfa:
                                N += 1
                            if D[1] > DMax: DMax = D[1]
                        # Оставляем объекты, с количеством прямых углов не менее 2 (не обнаружилась одна стенка у здания)
                        if N >= 2:
                            LinesFilter.append(Lines)

                        # Оставляем объекты, состоящие из двух параллельных отрезков
                        elif N == 0 and DMax >= DistPor:
                            LinesFilter.append(Lines)

                    # Убираем прямые, менее 10 пикселей (мелоч)
                    LinesFilter10 = []
                    for Lines in LinesFilter:
                        Lines10 = []
                        for Line in Lines:
                            P1 = Line[:2]
                            P2 = Line[2:]
                            D = np.sqrt((P1[0] - P2[0]) * (P1[0] - P2[0]) + (P1[1] - P2[1]) * (P1[1] - P2[1]))
                            if D >= 10:
                                Lines10.append(Line)
                        if len(Lines10) >= 2:
                            LinesFilter10.append(Lines10)

                    # Группировка линий по пространственному признаку
                    LinesGroup = []
                    for Lines in LinesFilter10:
                        LinesG = DNTheam.SedimentsGrop(Lines, DistPorGr, DistPorGr)
                        for Gr in LinesG:
                            LinesGroup.append(Gr)

                    # Удаление крестов
                    LinesGroupNoCross = []
                    for Lines in LinesGroup:
                        Lines = DNTheam.FilterCrossLine(Lines, d_alfa, DistPorGr)
                        LinesGroupNoCross.append(Lines)

                    # Удаление букв "Т"
                    LinesGroupNoTau = []
                    for Lines in LinesGroupNoCross:
                        Lines = DNTheam.FilterTauLine(Lines, d_alfa, DistPorGr)
                        LinesGroupNoTau.append(Lines)

                    # Удаление кочерег
                    LinesGroup2 = []
                    for Lines in LinesGroupNoTau:
                        if len(Lines) == 2:
                            P11 = Lines[0][:2]
                            P12 = Lines[0][2:]
                            P21 = Lines[1][:2]
                            P22 = Lines[1][2:]
                            D = DNMathAdd.LinesDistCorner(P11, P12, P21, P22)
                            if (D[0] >= 180 - d_alfa and D[0] <= 180 + d_alfa) or (D[0] >= -d_alfa and D[0] <= d_alfa):
                                LinesGroup2.append(Lines)
                        elif len(Lines) > 2:
                            LinesGroup2.append(Lines)
                    # if i==1873:
                    # print(i)
                    # #DNTheam.PrintContLines(ContursNoBord[i],[[LinesCont],[LinesCont7],LinesP,LinesPOb,LinesGroup,LinesGroup2])

                    # Упорядочиваем новый массив линий
                    # print(i)
                    BuildingPolysinCont = []
                    for Lines in LinesGroup2:
                        BuildingPoly = []
                        Lines = DNTheam.FLinesOrder(Lines)
                        for Line in Lines:
                            BuildingPoly.append(Line[:2])
                            BuildingPoly.append(Line[2:])
                        # Проверка является ли полученный полигон односвязной областью
                        Pts = []
                        for Pt in BuildingPoly:
                            Pts.append([Pt[1], Pt[0]])

                        IsBuild = DNTheam.PolyIs(Pts, DistPor)

                        # print(IsBuild[0])
                        # print(Pts)
                        if IsBuild[0]:
                            BuildingPolysinCont.append(IsBuild[1])

                    for BuildingPoly in BuildingPolysinCont:
                        BuildingPolys.append(BuildingPoly)
                    # print(BuildingPolys)
                    # DNTheam.PrintContLines(ContursNoBord[i], [[LinesCont], LinesP, LinesGroup2])
                    # print(i)

        ProgressBar.ChangeProcState(len(ContursNoBord), len(ContursNoBord),
                                    "Анализ геометрических характеристик сегментов")
        # Визуализация результата
        PolysPts = []
        if not len(BuildingPolys) == 0:
            for Poly in BuildingPolys:
                PolyPts = []
                for Pt in Poly:
                    PolyPts.append([int(Pt[0]), int(Pt[1])])
                PolysPts.append(PolyPts)

        # Перевод контуров в массив numpy
        PolysPtsNP = []
        for Poly in PolysPts:
            Poly = np.array(Poly)
            PolysPtsNP.append(Poly)

        # Аппроксимация полигонов прямоугольниками
        Rects = []
        for Poly in PolysPtsNP:
            Rect = cv.minAreaRect(Poly)
            box = cv.boxPoints(Rect)
            box = np.int0(box)
            Rects.append(box)

        # Сравнение формы полигонов с формой описанных пряпоугольников
        NumPolRect = []

        for i in range(len(PolysPtsNP)):
            if cv.matchShapes(PolysPtsNP[i], Rects[i], 1, 0) < 0.72:
                NumPolRect.append(i)

        ProgressBar.ChangeProcState(len(NumPolRect), 0, "Фильтрация пересекающихся полигонов")
        # Определение пар пересекающихся прямоугольников полигонов
        NumPolRectC = NumPolRect.copy()
        NumsRectPar = []
        i = 0
        while 1:
            XMin = min(Rects[NumPolRectC[i]][:, 0])
            XMax = max(Rects[NumPolRectC[i]][:, 0])
            YMin = min(Rects[NumPolRectC[i]][:, 1])
            YMax = max(Rects[NumPolRectC[i]][:, 1])
            RectPar = []
            RectPar.append(NumPolRectC[i])
            NumPolRectC.pop(i)

            ProgressBar.ChangeProcState(len(NumPolRect), len(NumPolRect) - len(NumPolRectC),
                                        "Фильтрация пересекающихся полигонов")

            if i >= len(NumPolRectC):
                break

            for x in range(XMin, XMax):
                for y in range(YMin, YMax):
                    j = 0
                    while 1:
                        if j >= len(NumPolRectC):
                            break
                        if cv.pointPolygonTest(Rects[RectPar[0]], [x, y], False) == 1 and \
                                cv.pointPolygonTest(Rects[NumPolRectC[j]], [x, y], False) == 1 and \
                                not NumPolRectC[j] in RectPar:
                            RectPar.append(NumPolRectC[j])

                        j += 1

            if len(RectPar) > 1:
                NumsRectPar.append(RectPar)

        # Фильтрация пересекающихся полигонов
        NumDelRect = []
        for Par in NumsRectPar:
            Kof = []
            Kof.append(cv.matchShapes(PolysPtsNP[Par[0]], Rects[Par[0]], 1, 0))
            Kof.append(cv.matchShapes(PolysPtsNP[Par[1]], Rects[Par[1]], 1, 0))
            KofMax = max(Kof)
            Ind = Kof.index(KofMax)
            NumDelRect.append(Par[Ind])

        # Формируем окончательный список полигонов после фильтрации
        RectsFilter = []
        PolysFilter = []

        for i in NumPolRect:
            if not i in NumDelRect:
                PolysFilter.append(PolysPtsNP[i])
                RectsFilter.append(Rects[i])

        return RectsFilter

    @classmethod
    # MinArea и MinWidth задаются в пикселях
    def DetectBuild2(cls, ClsMass: [], MinArea: int, MinWidth: int, ProgressBar=None):
        W, H = np.shape(ClsMass)
        # 1. Получение контуров всех сегментов во всех классах
        ContursAll = []

        for NCls in np.unique(ClsMass):
            if not NCls == 0:
                Img = np.zeros([W, H], dtype=np.uint8)
                P = np.column_stack(np.where(np.array(ClsMass) == NCls))
                Img[P[:, 0], P[:, 1]] = 255

                # Получение контуров сегментов
                Conturs = cls.GetContursClass(Img, 255)
                for Contur in Conturs:
                    ContursAll.append(Contur)

            # if NCls==85:
            #     print(len(ContursAll),len(Conturs))
            #     plt.imshow(Img)
            #     plt.show()

        # DNTheam.PrintContLines(ContursAll[344],[])
        NumsSeg = []  # Номера сегментов, которые последовательно будут отфильтрованы по признакам
        ContursParam = []
        for i in range(len(ContursAll)):
            NumsSeg.append(i)
            ContursParam.append({'Area': -1,
                                 'W': -1,
                                 'H': -1,
                                 'SMin': -1,
                                 'Lines': [],
                                 'LinesP': [],
                                 'DLinesP': -1,
                                 'DLP_P': -1})

        # 2. Определяем параметры контуров (площадь, длину, ширину, площадь описанного прямоугольника)
        # Пороговые значения: S<150, W<10, S/SRecMin<0.5

        IndxGoodSeg = []  # Индексы сегментов, которые проходят по признакам
        for i in range(len(ContursAll)):
            Area = cls.GetAreaContur(ContursAll[i])
            WidHeigh = cls.GetWidHeighContur(ContursAll[i])
            ContursParam[i]['Area'] = Area
            ContursParam[i]['W'] = WidHeigh['W']
            ContursParam[i]['H'] = WidHeigh['H']
            ContursParam[i]['SMin'] = WidHeigh['S']

            # Фильтрация сегментов по геометрическим характеристикам
            if ContursParam[i]['Area'] >= MinArea \
                    and ContursParam[i]['W'] > MinWidth \
                    and ContursParam[i]['W'] / ContursParam[i]['H'] > 0.2 \
                    and ContursParam[i]['Area'] / ContursParam[i]['SMin'] > 0.65:
                IndxGoodSeg.append(i)

        # print(ContursParam[644])
        # print(ContursParam[644]['W']/ContursParam[644]['H'])
        # print(ContursParam[644]['Area']/ContursParam[644]['SMin'])
        # print(644 in IndxGoodSeg)
        # DNTheam.PrintContLines(ContursAll[644],[])

        # 3. Аппроксимация контура прямыми
        LinesGoodSegs = []  # Линии всех хороших сегментов
        for i in IndxGoodSeg:
            ProgressBar.ChangeProcState(len(ContursAll), i, "Анализ геометрических характеристик сегментов")
            LinesCont = []
            LMin = 10
            GarpMax = 10
            # Аппроксимируем только контура подходящей площади и линейных размеров
            Lim = 800  # Лимит длины контура
            N = 0
            if len(ContursAll[i]) < Lim:
                LinesCont = cls.ApproxLinesContur(ContursAll[i], LMin, GarpMax)
            # Если контур больше 1000 пикселей, для ускорения аппроксимации разбиваем его на составные части
            else:
                LinesCont = []
                N = int(len(ContursAll[i]) / Lim)
                NOst = len(ContursAll[i]) % Lim
                for l in range(N):
                    PartContur = ContursAll[i][l * Lim:l * Lim + Lim]
                    LinesPart = cls.ApproxLinesContur(PartContur, LMin, GarpMax)
                    for Line in LinesPart:
                        LinesCont.append(Line)
                if NOst > 0:
                    PartContur = ContursAll[i][N * Lim:N * Lim + NOst]
                    LinesPart = cls.ApproxLinesContur(PartContur, LMin, GarpMax)
                    for Line in LinesPart:
                        LinesCont.append(Line)

            # Фильтруем мусор (удаляем линии меньше 7 пикселей)
            LinesCont7 = []
            if not len(LinesCont) == 0:  # Если контур аппроксимировался хотя бы одной линией
                for Line in LinesCont:
                    P1 = Line[:2]
                    P2 = Line[2:]
                    if DNMathAdd.CalcEvkl(P1, P2) > 7:
                        LinesCont7.append(Line)
                        ContursParam[i]['Lines'].append(Line)

        # 4. Поиск в каждом сегменте взаимно перпендикулярных и параллельных линий
        d_alfa = 7  # Допуск угла, какие линии считать параллельными и перпендикулярными
        IndxFilter = []
        for i in IndxGoodSeg:
            # if i==644:
            #     print(i)
            if not len(ContursParam[i]['Lines']) == 0:
                LinesP = DNTheam.FineParallelPerpendLines(ContursParam[i]['Lines'], d_alfa)

                # Ищем наиболее длинную группу параллельных-перпендикулярных прямых
                IndxMaxL = -1
                DMax = 0.
                IndxP = 0
                for Lines in LinesP:
                    # Lines=DNTheam.LinesPObed(Lines,d_alfa,3)
                    D = 0.
                    for Line in Lines:
                        P1 = Line[:2]
                        P2 = Line[2:]
                        D += DNMathAdd.CalcEvkl(P1, P2)

                    if D > DMax:
                        DMax = D
                        IndxMaxL = IndxP

                    IndxP += 1

                if IndxMaxL >= 0:
                    for Line in LinesP[IndxMaxL]:
                        ContursParam[i]['LinesP'].append(Line)
                    ContursParam[i]['DLinesP'] = DMax
                    ContursParam[i]['DLP_P'] = float(DMax / len(ContursAll[i]))

                # Отфильтровываем сегменты по отношению длины параллельных-перпендикулярных линий к периметру контура
                if ContursParam[i]['DLP_P'] >= 0.6:  # 0.73:
                    IndxFilter.append(i)

                # print(ContursParam[i]['LinesP'],"\n")
                # if i==644:
                #     print(ContursParam[i]['DLP_P'])
                #     DNTheam.PrintContLines(ContursAll[i], [[ContursParam[i]['Lines']],[ContursParam[i]['LinesP']]])

        IndxGoodSeg = IndxFilter.copy()

        ResultCont = []
        for i in IndxGoodSeg:
            ResultCont.append(ContursAll[i])

        return ResultCont

    @classmethod
    def DetectCircleBuild(cls, ClsMass: [], MinArea: int, MinR: int, DExcPor: float):
        W, H = np.shape(ClsMass)
        # 1. Получение контуров всех сегментов во всех классах
        ContursAll = []

        for NCls in np.unique(ClsMass):
            if not NCls == 0:
                Img = np.zeros([W, H], dtype=np.uint8)
                P = np.column_stack(np.where(np.array(ClsMass) == NCls))
                Img[P[:, 0], P[:, 1]] = 255

                # Получение контуров сегментов
                Conturs = cls.GetContursClass(Img, 255)
                for Contur in Conturs:
                    ContursAll.append(Contur)

        NumsSeg = []  # Номера сегментов, которые последовательно будут отфильтрованы по признакам
        ContursParam = []
        for i in range(len(ContursAll)):
            NumsSeg.append(i)
            ContursParam.append({'Area': -1,
                                 'W': -1,
                                 'H': -1,
                                 'SMin': -1,
                                 'Lines': [],
                                 'LinesP': [],
                                 'DLinesP': -1,
                                 'DLP_P': -1})

        # 2. Определяем параметры контуров (площадь, длину, ширину, площадь описанного прямоугольника)
        # Пороговые значения: S<150, W<10, S/SRecMin<0.5

        IndxGoodSeg = []  # Индексы сегментов, которые проходят по признакам
        for i in range(len(ContursAll)):
            if i == 109:
                print(i)
            Area = cls.GetAreaContur(ContursAll[i])
            WidHeigh = cls.GetWidHeighContur(ContursAll[i])
            ContursParam[i]['Area'] = Area
            ContursParam[i]['W'] = WidHeigh['W']
            ContursParam[i]['H'] = WidHeigh['H']
            ContursParam[i]['SMin'] = WidHeigh['S']

            # Фильтрация сегментов по геометрическим характеристикам
            if ContursParam[i]['Area'] >= MinArea \
                    and ContursParam[i]['W'] > MinR \
                    and abs(1 - ContursParam[i]['W'] / ContursParam[i]['H']) < DExcPor:
                IndxGoodSeg.append(i)
        # 3. Аппроксимация контура окружностями
        IndxFilter = []
        for i in IndxGoodSeg:
            CirclesCont = cls.ApproxCircleContur(ContursAll[i], 0.9, 1.2)

            if not len(CirclesCont) == 0:
                IndxFilter.append(i)

        IndxGoodSeg = IndxFilter.copy()

        ResultCont = []
        for i in IndxGoodSeg:
            ResultCont.append(ContursAll[i])

        return ResultCont

    @classmethod
    def DetectBuild3(cls, ClsMass: []):
        W, H = np.shape(ClsMass)
        # 1. Получение контуров всех сегментов во всех классах
        ContursAll = []

        for NCls in np.unique(ClsMass):
            if not NCls == 0:
                Img = np.zeros([W, H], dtype=np.uint8)
                P = np.column_stack(np.where(np.array(ClsMass) == NCls))
                Img[P[:, 0], P[:, 1]] = 255

                # Получение контуров сегментов
                Conturs = cls.GetContursClass(Img, 255)
                for Contur in Conturs:
                    ContursAll.append(Contur)

        for Contur in ContursAll:
            DNTheam.ApproxCircleContur(Contur)
            DNTheam.PrintContLines(Contur, [])

        IndxGoodSeg = []  # Номера сегментов, которые последовательно будут отфильтрованы по признакам

        ResultCont = []
        for i in IndxGoodSeg:
            ResultCont.append(ContursAll[i])

        return ResultCont

    # Функция определения машинного зала
    @classmethod
    # Линейные размеры заполняются в метрах
    def FinedMZ_RO(cls, ContursAll: [], ContursRO: [], LRM: float, MinW: float, MaxW: float, MaxDist: float):

        # 1. Фильтрация контуров по линейным размерам
        NumCMZ = []
        Indx = 0
        for Contur in ContursAll:
            # Определяем геометрические параметры контуров
            CProp = cls.GetWidHeighContur(Contur)
            Wb = CProp['W'] * LRM
            Hb = CProp['H'] * LRM

            # Фильтрация контуров по линейным размерам
            if Wb > MinW and Wb < MaxW:
                NumCMZ.append(Indx)
            Indx += 1

        # 2. Фильтрация по минимальному расстоянию от РО до предполагаемого
        CopNumCMZ = NumCMZ.copy()
        RezIndx = []
        for ContRO in ContursRO:
            DMin = 100000
            P1 = []
            P2 = []
            IndxMin = -1
            for i in CopNumCMZ:
                D = DNTheam.CalcMinDistCont(ContursAll[i], ContRO)
                if D[0] < DMin:
                    DMin = D[0]
                    IndxMin = i
                    P1 = D[1]
                    P2 = D[2]
            if IndxMin >= 0 and float(DMin * LRM) < MaxDist:
                RezIndx.append(IndxMin)

        NumCMZ = np.unique(RezIndx)

        ContursMZ = []
        for i in NumCMZ:
            if i >= 0:
                ContursMZ.append(ContursAll[i])

        return ContursMZ

    @classmethod
    def FinedBNS(cls, ContursAll: [], ContursGR: [], LRM: float, MinW: float, MaxW: float):

        # 1. Фильтрация контуров по линейным размерам
        NumCBNS = []
        Indx = 0
        for Contur in ContursAll:
            # Определяем геометрические параметры контуров
            CProp = cls.GetWidHeighContur(Contur)
            Wb = CProp['W'] * LRM
            Hb = CProp['H'] * LRM

            # Фильтрация контуров по линейным размерам
            if Wb > MinW and Wb < MaxW:
                NumCBNS.append(Indx)
            Indx += 1

        # #2. Фильтрация по минимальному расстоянию от градирни
        # # Преобразуем контуры зданий в шейп полигоны
        PolBNS = [Polygon(ContursAll[i]) for i in NumCBNS]
        IndxBNS = []
        for ContGR in ContursGR:
            # Для каждой градирни ищем соответствующую ей БНС по минимальному расстоянию
            PolGr = Polygon(ContGR)
            DistMass = PolGr.distance(PolBNS)
            DistMin = min(DistMass)
            IndxMin = np.column_stack(np.where(DistMass == DistMin))
            IndxBNS += [i[0] for i in IndxMin]

        NumCBNS = np.unique(np.array(IndxBNS))
        ContursBNS = [np.array(PolBNS[i].exterior.coords, int) for i in NumCBNS]
        return ContursBNS

    # Предварительная классификация найденных зданий
    @classmethod
    def BuildClassif(cls, Conturs: [], LRM: float):
        ClsId = []
        for Contur in Conturs:
            ContProp = cls.GetWidHeighContur(Contur)

    @classmethod
    # Сгенерировать контур графического примитива
    def GenElementContur(cls, W: int, H: int, TypeElem: int):
        Result = []
        # Тип элемента: прямоугольник
        if TypeElem == 0:
            for x in range(W - 1):
                Result.append([x - (W - 1) / 2, -(H - 1) / 2])
            for y in range(H - 1):
                Result.append([(W - 1) / 2, y - (H - 1) / 2])
            for x in range(W - 1):
                Result.append([(W - 1) / 2 - x, (H - 1) / 2])
            for y in range(H - 1):
                Result.append([-(W - 1) / 2, (H - 1) / 2 - y])

        # Тип элемента: эллипс
        if TypeElem == 1:
            # Для эллипса реализуем алгоритм Брезендхейма
            delta = [[0, -1], [1, -1], [1, 0]]
            Result.append([-W, 0])
            n = 0;
            # Делаем четверть эллипса
            while True:
                eps = 1000
                i = 0
                for d in delta:
                    x = Result[-1][0] + d[0]
                    y = Result[-1][1] + d[1]
                    if eps > abs(((x * x) / (W * W) + (y * y) / (H * H)) - 1):
                        eps = abs(((x * x) / (W * W) + (y * y) / (H * H)) - 1)
                        n = i
                    i += 1

                x = Result[-1][0] + delta[n][0]
                y = Result[-1][1] + delta[n][1]
                Result.append([x, y])
                if Result[-1][0] >= 0: break

            # Симметрично распространяем четверть эллипса на остальные четверти
            for R in Result[-2::-1]:
                Result.append([-R[0], R[1]])

            for R in Result[-2:0:-1]:
                Result.append([R[0], -R[1]])

        return Result


class GrigStructs:
    def __init__(self, LRM: float, PathToImgFile: str, PathToCNNFile, PathToModelFile: str):
        # Задаем ЛРМ
        self.lrm = LRM

        # Читаем картинку
        self.image = Image.open(PathToImgFile)
        self.HImg = self.image.height
        self.WImg = self.image.width

        # Читаем файл результатов СНС
        file = open(PathToCNNFile, "r")
        self.Elements = []
        while True:
            line = file.readline()
            line = line.strip()
            if not line: break
            Data = line.split(' ')
            xT = Data[1::2]
            yT = Data[2::2]
            x = np.array(xT, np.float32)
            y = np.array(yT, np.float32)
            xy = list(zip(x, y))
            Element = {
                'NumCls': int(Data[0]),
                'x': x,
                'y': y
            }
            self.Elements.append(Element)
        file.close()

        # Читаем таблицу ребер, созданную по эталонам
        self.rebra_data = pd.read_csv(PathToModelFile, delimiter=';', encoding='utf_8', encoding_errors='replace')

        # Определяем номера классоов в результатах НС
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

    # Преобразование результатов НС в полигоны Шейп и пиксельные
    def ElemsToPoly(self, Elems):
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

    # Преобразование шейп полигонов в пиксельные
    def SHPToPol(self, PolsSHP: []):
        # Получаем координаты точек полигонов
        ResPT = []
        for Pol in PolsSHP:
            ResPT.append(np.array(Pol.exterior.coords, int))

        # Проверка, не выходят ли координаты точек буферной зоны за пределы изображения
        for i in range(len(ResPT)):
            for j in range(len(ResPT[i])):
                if ResPT[i][j, 0] < 0:
                    ResPT[i][j, 0] = 0
                if ResPT[i][j, 1] < 0:
                    ResPT[i][j, 1] = 0
                if ResPT[i][j, 0] >= self.WImg:
                    ResPT[i][j, 0] = self.WImg - 1
                if ResPT[i][j, 1] >= self.HImg:
                    ResPT[i][j, 1] = self.HImg - 1

        return ResPT

    # Объединение полигонов по критерию близости друг к другу
    # MaxDist - без учета ЛРМ (в пикселях)
    @classmethod
    def PolyUnion(cls, PolsSHP: [], MaxDist: float):
        GrP_C = PolsSHP.copy()
        while True:
            IsUnion = False  # Произошло ли объединение:
            for i in range(len(GrP_C)):
                # Рассчитываем дистанцию от i-того полигона до всех остальных
                DistMass = GrP_C[i].distance(GrP_C)

                # Узнаем индексы полигонов, до которых дистанция меньше максимальной, чтобы включать их в группу
                Indx = np.column_stack(np.where(np.array(DistMass) < MaxDist))
                Indx_Ob = [j[0] for j in Indx]

                # Если есть что объединять
                if len(Indx_Ob) > 1:
                    # Объединяем полигоны
                    MPol = [GrP_C[j] for j in Indx_Ob]
                    MPol = MultiPolygon(MPol)
                    P = MPol.convex_hull

                    # Исключаем из изначального списка объединенные полигоны
                    GrP_C = [GrP_C[j] for j in range(len(GrP_C)) if not j in Indx_Ob]
                    # Добавляем объединенный полигон
                    GrP_C.append(P)
                    IsUnion = True
                    break

            # Если объединять больше нечего, выходим из цикла
            if not IsUnion:
                break

        return GrP_C

    # Функция получения конкретных классов из результатов НС
    def FinedClassElement(self, NumsClass: []):
        index = [i for i in range(0, len(self.Elements)) if self.Elements[i]['NumCls'] in NumsClass]
        Res = []
        for i in index:
            Res.append(self.Elements[i])
        return Res

    # Функция создания буферных зон вокуруг полигонов
    def CreateBufZone(self, Pols: [], Dist: int):
        Bufs = []

        # Строим буфер полигонов
        for Pol in Pols:
            Bufs.append(Pol.buffer(Dist))

        # Объединяем пересекающиеся полигоны
        Res = Bufs.copy()
        while 1:
            IsPolUnion = False
            for i in range(len(Res)):
                Ov = Res[i].overlaps(Res)
                IndxOv = np.column_stack(np.where(Ov))
                if not len(IndxOv) == 0:
                    IsPolUnion = True
                    for j in IndxOv[0]:
                        Res[i] = Res[i].union(Res[j])
                    for j in IndxOv[0]:
                        Res.pop(j)
                    break

            # Если нет полигонов, которые надо объединять, выходим из цикла
            if not IsPolUnion:
                break

        # Получаем координаты точек полигонов
        ResPT = []
        for Pol in Res:
            ResPT.append(np.array(Pol.exterior.coords, int))

        # Проверка, не выходят ли координаты точек буферной зоны за пределы изображения
        for i in range(len(ResPT)):
            for j in range(len(ResPT[i])):
                if ResPT[i][j, 0] < 0:
                    ResPT[i][j, 0] = 0
                if ResPT[i][j, 1] < 0:
                    ResPT[i][j, 1] = 0
                if ResPT[i][j, 0] >= self.WImg:
                    ResPT[i][j, 0] = self.WImg - 1
                if ResPT[i][j, 1] >= self.HImg:
                    ResPT[i][j, 1] = self.HImg - 1

        return ResPT

    # Функция создания мини-картинок из указанных полигонов
    def CreateZoneImgs(self, Conturs: []):

        # CroppImg = PilImg.crop((Mask['XMin'],Mask['YMin'],Mask['XMin']+Mask['WPol'],Mask['YMin']+Mask['HPol']))
        # Преобразование картинки в массив данных
        RGBMAss = np.array(self.image).astype("uint8")

        # Формирование маску всех зон
        mask = np.zeros([self.HImg, self.WImg], dtype=bool)
        RecMassImg = []
        for Contur in Conturs:
            # DNTheam.PrintContLines(Contur,[])
            x = Contur[:, 0]
            y = Contur[:, 1]

            XMin = min(x)
            XMax = max(x)
            YMin = min(y)
            YMax = max(y)
            RecMassImg.append([[XMin, YMin], [XMax, YMax]])

            XList = list(range(XMin, XMax + 1))
            YList = list(range(YMin, YMax + 1))
            P = []

            WPol = XMax - XMin + 1
            HPol = YMax - YMin + 1

            for YVal in YList:
                for XVal in XList:
                    P.append((XVal, YVal))

            pPath = path.Path(Contur)
            PMass = pPath.contains_points(P)
            PMass = np.array(PMass).reshape(-1, WPol).transpose()

            P = np.column_stack(np.where(PMass.transpose()))
            P[:, 0] = P[:, 0] + YMin
            P[:, 1] = P[:, 1] + XMin

            mask[P[:, 0], P[:, 1]] = True

        # Накладываем маску на изображение
        mask = mask.astype('uint8')
        RGBMAss = cv.bitwise_and(RGBMAss, RGBMAss, mask=mask)

        # Формируем мини-картинки согласно маскам
        MiniImgMass = []
        Img = Image.fromarray(RGBMAss, 'RGB')
        for Rec in RecMassImg:
            ImgCrop = Img.crop((Rec[0][0], Rec[0][1], Rec[1][0], Rec[1][1]))
            MiniImgMass.append(ImgCrop)

        return {'Coords': RecMassImg, 'Imgs': MiniImgMass}

    # Функция фильтрации пересекающихся полигонов (когда разными методами обнаружилось одно и то же)
    def FilterPol(self, Conturs: [], NumCls: [], IntersPart=0.85):
        PolysCNN = self.ElemsToPoly(self.Elements)
        PolysOth = [Polygon(Conturs[i]) for i in range(len(Conturs))]

        # Проверяем пересечение между полигонами полученными сторонними методами
        IndxDel = []
        for i in range(len(PolysOth)):
            if not i in IndxDel:
                Ov = PolysOth[i].intersects(PolysOth)
                Ov[i] = False
                # Получаем индексы пересекающихся полигонов
                Indx = np.column_stack(np.where(Ov))
                Indx = [j[0] for j in Indx if not j[0] in IndxDel]

                if not len(Indx) == 0:
                    # Получаем пересекающиеся полигоны
                    P_OV = [PolysOth[j] for j in Indx]

                    # Сравниваем площади пересекающихся полигонов с площадью пересечения
                    AreaP = [P_OV[j].area for j in range(len(P_OV))]
                    AreaMean = np.mean(AreaP)

                    for j in Indx:
                        P = PolysOth[i].intersection(PolysOth[j])
                        AreaOV = P.area
                        if (AreaMean / AreaOV) >= IntersPart:
                            IndxDel.append(j)

        ContursRes = [Conturs[i] for i in range(len(Conturs)) if not i in IndxDel]

        return ContursRes

    # Функции построения зон поиска
    def LocalZoneRO(self, MaxR: float):
        MaxR = int(MaxR / self.lrm)

        # Работа с результатами сети
        # Найденные РО и МЗ
        ElemsRO = self.FinedClassElement([self.ClassNums['RO_P'], self.ClassNums['RO_S']])
        ElemsMZ = self.FinedClassElement([self.ClassNums['MZ_V']])

        # Если не найдено ни одного машинного зала, возвращаем пустоту
        if len(ElemsMZ) == 0:
            return None

        # Преобразуем найденные объекты в набор полигонов
        RO_Polys = []
        RO_PolysSH = []
        for i in range(len(ElemsRO)):
            ElemsRO[i]['x'] = np.array(ElemsRO[i]['x'] * self.WImg, int)
            ElemsRO[i]['y'] = np.array(ElemsRO[i]['y'] * self.HImg, int)
            pts = np.zeros([len(ElemsRO[i]['x']), 2], dtype=np.int32)
            pts[:, 0] = ElemsRO[i]['x']
            pts[:, 1] = ElemsRO[i]['y']
            RO_Polys.append(pts)
            RO_PolysSH.append(Polygon(pts))

        MZ_Polys = []
        MZ_PolysSH = []
        for i in range(len(ElemsMZ)):
            ElemsMZ[i]['x'] = np.array(ElemsMZ[i]['x'] * self.WImg, int)
            ElemsMZ[i]['y'] = np.array(ElemsMZ[i]['y'] * self.HImg, int)
            pts = np.zeros([len(ElemsMZ[i]['x']), 2], dtype=np.int32)
            pts[:, 0] = ElemsMZ[i]['x']
            pts[:, 1] = ElemsMZ[i]['y']
            MZ_Polys.append(pts)
            MZ_PolysSH.append(Polygon(pts))

        Zoes = self.CreateBufZone(MZ_PolysSH, MaxR)
        return self.CreateZoneImgs(Zoes)

    def LocalZoneMZ(self, MaxR: float, MinDPorDist: float):
        MaxR = int(MaxR / self.lrm)
        MinDPorDist = int(MinDPorDist / self.lrm)

        # Работа с результатами сети
        # Найденные РО и МЗ
        ElemsRO = self.FinedClassElement([self.ClassNums['RO_P'], self.ClassNums['RO_S']])
        ElemsMZ = self.FinedClassElement([self.ClassNums['MZ_V']])

        # Если не найдено ни одного РО, возвращаем пустоту
        if len(ElemsRO) == 0:
            return None

        # Преобразуем найденные объекты в набор полигонов
        RO_Polys = []
        RO_PolysSH = []

        # Преобразуем полигоны РО
        for i in range(len(ElemsRO)):
            ElemsRO[i]['x'] = np.array(ElemsRO[i]['x'] * self.WImg, int)
            ElemsRO[i]['y'] = np.array(ElemsRO[i]['y'] * self.HImg, int)
            pts = np.zeros([len(ElemsRO[i]['x']), 2], dtype=np.int32)
            pts[:, 0] = ElemsRO[i]['x']
            pts[:, 1] = ElemsRO[i]['y']
            RO_Polys.append(pts)
            RO_PolysSH.append(Polygon(pts))

        MZ_Polys = []
        MZ_PolysSH = []

        # Преобразуем полигоны МЗ
        for i in range(len(ElemsMZ)):
            ElemsMZ[i]['x'] = np.array(ElemsMZ[i]['x'] * self.WImg, int)
            ElemsMZ[i]['y'] = np.array(ElemsMZ[i]['y'] * self.HImg, int)
            pts = np.zeros([len(ElemsMZ[i]['x']), 2], dtype=np.int32)
            pts[:, 0] = ElemsMZ[i]['x']
            pts[:, 1] = ElemsMZ[i]['y']
            MZ_Polys.append(pts)
            MZ_PolysSH.append(Polygon(pts))

        # Для каждого найденного по сети МЗ  определяем его группу реакторов
        IndxGRO = list(range(len(ElemsRO)))
        for i in range(len(ElemsMZ)):
            # Заполнение массива оставшихся реакторов, для которых не нашли своих МЗ
            RO_PolysSHG = []
            for j in IndxGRO:
                RO_PolysSHG.append(RO_PolysSH[j])

            # Определяем расстояния от МЗ до оставшихся неприкаянными реакторов
            DistMass = MZ_PolysSH[i].distance(RO_PolysSHG)
            # Определяем минимальное расстояние между реакторами и конкретным МЗ
            MinDist = min(DistMass)

            # Определяем разницу расстояний между МЗ и другими реакторами
            DDistMass = []
            for j in range(len(IndxGRO)):
                DDistMass.append(DistMass[j] - MinDist)

            # Если разница расстояний меньше пороговой, реакторы принадлежат одной группе
            Indx = np.column_stack(np.where(np.array(DDistMass) < MinDPorDist))
            IndxGRO_list = []
            for j in Indx:
                IndxGRO_list.append(IndxGRO[j[0]])

            if MinDist < MaxR:
                IndxGRO_C = []
                for j in IndxGRO:
                    if not j in Indx:
                        IndxGRO_C.append(j)

                IndxGRO = IndxGRO_C.copy()

        # Если все реакторы при деле возвращаем None
        if len(IndxGRO) == 0:
            return None

        # Рисуем буферные зоны от неприкаянных реакторов
        RO_PolysSHG = []
        for i in IndxGRO:
            RO_PolysSHG.append(RO_PolysSH[i])

        Zoes = self.CreateBufZone(RO_PolysSHG, MaxR)
        return self.CreateZoneImgs(Zoes)

    def LocalZoneBNSGR(self, MaxDistGroupGR: float, MaxDistGroupObj: float):

        # Пороговое расстояние для объединения градирен и объектов в один комплекс
        MaxDistGroupGR = MaxDistGroupGR / self.lrm
        MaxDistGroupObj = MaxDistGroupObj / self.lrm

        # Чтение результатов работы НС
        ElemsRO = self.FinedClassElement([self.ClassNums['RO_P'], self.ClassNums['RO_S']])
        ElemsMZ = self.FinedClassElement([self.ClassNums['MZ_V']])
        ElemsGR = self.FinedClassElement([self.ClassNums['Gr_b'],
                                          self.ClassNums['Gr_V_S'],
                                          self.ClassNums['Gr_V_P'],
                                          self.ClassNums['Gr_B_Act']])

        # Если не найдена пара объектов Градирня - Машинный зал (РО), возвращаем None
        if len(ElemsGR) == 0:
            return None

        if len(ElemsMZ) == 0 and len(ElemsRO) == 0:
            return None

        # В качестве объекта-пары выбираем и РО и МЗ (если они рядом, в последствии они объединятся)
        ElemsObj = [Elem for Elem in ElemsMZ]
        ElemsObj += ([Elem for Elem in ElemsRO])

        # Преобразуем результаты НС в полигоны
        P_Gr = self.ElemsToPoly(ElemsGR)
        P_Obj = self.ElemsToPoly(ElemsObj)

        # Объединяем в общие полигоны градирни
        P_Gr_Ob = GrigStructs.PolyUnion(P_Gr['PolysSH'], MaxDistGroupGR)

        # Объединяем в общие полигоны объекты
        P_Obj_Ob = GrigStructs.PolyUnion(P_Obj['PolysSH'], MaxDistGroupObj)

        # Строим зоны поиска БНС
        # Для каждой группы градирен ищем ближайший объект (МЗ или РО)
        P_ZonesSH = []
        for P_Gr in P_Gr_Ob:
            # Ищем объект, находящийся на минимальном расстоянии от градирен
            Dist = P_Gr.distance(P_Obj_Ob)
            DistMin = min(Dist)
            Indx = np.column_stack(np.where(Dist == DistMin))

            # Создаем зону поиска (объединяем объект и градирню)
            P_ZoneSH = [P_Obj_Ob[i[0]] for i in Indx]
            P_ZoneSH.append(P_Gr)
            MPol = MultiPolygon(P_ZoneSH)
            P_SH = MPol.convex_hull

            # Формируем список зон объединений
            P_ZonesSH.append(P_SH)

        # Переводим зоны из формата шейп в полигоны точек
        P_Zones = self.SHPToPol(P_ZonesSH)

        return self.CreateZoneImgs(P_Zones)
