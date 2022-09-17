
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
    geotrans, dtype=gdal.GDT_Float64, epsg=4326,
    sinusoidal = False
    ):

    # geotransは(左上の経度, Δ経度, 0, 左上の緯度, 0, -Δ緯度)
    cols, rows = arr.shape[1], arr.shape[0]
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(out_file_path, cols, rows, 1, dtype)
    outRaster.SetGeoTransform(geotrans)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(arr)
    outRasterSRS = osr.SpatialReference()
    if sinusoidal:
        projection = 'PROJCS["unnamed",GEOGCS["Unknown datum based upon the custom spheroid",DATUM["Not specified (based on custom spheroid)",SPHEROID["Custom spheroid",6371007.181,0]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]]],PROJECTION["Sinusoidal"],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1],AXIS["Easting",EAST],AXIS["Northing",NORTH]]'
        outRasterSRS.ImportFromWkt(projection)
    else:
        outRasterSRS.ImportFromEPSG(epsg)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    del outRaster

def arr2tif_readmeta(
    arr: np.ndarray,
    original_dataset,
    out_file_path, dtype=gdal.GDT_Float64
    ):
    """メタデータから情報を抽出し、geotiffに変換する

    Args:
        arr (np.ndarray): 出力したいラスターの中身
        original_dataset (gdal.SubDatasets()): 真似をしたいデータ
        out_file_path (str): 出力先ファイル名
        dtype (type): 出力するデータ型. Defaults to gdal.GDT_Float64.
    """
    # geotransは(左上の経度, Δ経度, 0, 左上の緯度, 0, -Δ緯度)
    cols, rows = arr.shape[1], arr.shape[0]
    geotrans = original_dataset.GetGeoTransform()
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(out_file_path, cols, rows, 1, dtype)
    outRaster.SetGeoTransform(geotrans)
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(arr)
    outRasterSRS = osr.SpatialReference()

    projection = original_dataset.GetProjection()
    outRasterSRS.ImportFromWkt(projection)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    del outRaster