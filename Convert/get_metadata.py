#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# geotiffからメタデータを取得し、dictionaryで返す(jsonを出力する)
# %%
from osgeo import gdal, ogr
import rasterio
import json

# %%
def get_metadata(in_geotiff_path, out_json_path=None):
    """geotiffのメタデータをdict型に出力,jsonに保存

    Args:
        in_geotiff_path (str): メタデータを取得したいgeotiffファイルのパス
        out_json_path (str): 出力先jsonファイルパス. Defaults to None.
    
    Return:
        out_dict(dictionary): メタデータをまとめた辞書
    """
    
    out_dict = {}
    with rasterio.open(in_geotiff_path) as raster:
        transform_rasterio = raster.transform
        src_arr = raster.read(1)

    src = gdal.Open(in_geotiff_path)
    transform_gdal = src.GetGeoTransform()
    proj = src.GetProjection()
    del src

    # dict化
    out_dict['transform_rasterio'] = list(transform_rasterio)[:6]
    out_dict['transform_gdal'] = transform_gdal
    out_dict['projection'] = proj
    out_dict['h'], out_dict['w'] = src_arr.shape

    if out_json_path is not None:
        with open(out_json_path, 'w') as f:
            json.dump(out_dict, f, indent=4)
    return(out_dict)

if __name__=='__main__':
    meta_dict = get_metadata(
        f'D:/ResearchData2/Level5/LULC_change/AUS_h29v12/yearly500_geotiff/LULCcange12000_yearly.A2001001.tif',
        './meta2.json'
    )