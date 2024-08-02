#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# geotrans2extent.py: geotransからmatplotlib引数に使用可能なextentに変換する
from osgeo import gdal

def geotrans2extent(geotrans=None, h=None, w=None, path=None):
    if path is not None:
        src = gdal.Open(path)
        geotrans = src.GetGeoTransform()
        h,w = src.ReadAsArray().shape
        del src
    x_min, x_d, _, y_max, _, y_d = geotrans
    x_max, y_min = x_min + w*x_d, y_max + h*y_d
    extent = (x_min, x_max, y_min, y_max)
    return extent