
#!/usr/bin/env python3
# -*- cording: utf-8 -*-
# ndarray -> Geotif 変換して保存する関数

# %%
from osgeo import gdal, ogr, osr
import numpy as np
# %%
def arr2tif(
    arr:np.ndarray,
    out_file_path,
    geotrans, dtype=gdal.GDT_Float64, epsg=4326
    ):

    # geotransは(左上の経度, Δ経度, 0, 左上の緯度, 0, -Δ緯度)
    cols, rows = arr.shape[1], arr.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(out_file_path, cols, rows, 1, dtype)
    outRaster.SetGeoTransform(geotrans)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(arr)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    del outRaster
    print(f'Output {out_file_path}')