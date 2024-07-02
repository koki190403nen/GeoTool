#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from osgeo import gdal
import cv2

# %%
class MethodOfNDVI_GSI:
    def __init__(self):
        self.veg_area = None
        self.bare_area = None
        self.AMarea   = None

    def fit(self, pre_set, post_set, minimumsize=5):
        self.extract_vegarea(pre_set)
        self.extract_barearea(post_set)

        AMarea = (self.veg_area&self.bare_area).astype(np.uint8)


        self.AMarea = self.postprocessed(AMarea, minimumsize)
        return self.AMarea

    def extract_vegarea(self, band_set):

        # 前画像から植生域の抽出
        self.set_(*band_set)
        self.calc_indices()
        ndvi_uint8 = self.convert_uint8(self.ndvi)
        gsi_uint8  = self.convert_uint8(self.gsi)

        ndvi_threshold = self.Otsu(ndvi_uint8)
        gsi_threshold  = self.Otsu(gsi_uint8)

        self.veg_area = (ndvi_uint8>ndvi_threshold) & (gsi_uint8<=gsi_threshold)
    
    def extract_barearea(self, band_set):
        # 後画像から裸地域の抽出
        self.set_(*band_set)
        self.calc_indices()
        ndvi_uint8 = self.convert_uint8(self.ndvi)
        gsi_uint8  = self.convert_uint8(self.gsi)

        ndvi_threshold = self.Otsu(ndvi_uint8)
        gsi_threshold  = self.Otsu(gsi_uint8)

        self.veg_area = (ndvi_uint8<=ndvi_threshold) & (gsi_uint8>gsi_threshold)

    def set_(self, r, g, b, nir, mask=None):
        """NDVI_GSI法による人工改変箇所抽出

        Args:
            r    (Array like): バンドRed   (反射率)
            g    (Array like): バンドGreen (反射率)
            b    (Array like): バンドBlue  (反射率)
            nir  (Array like): バンドNIR   (反射率)
            mask (Array like): マスク画像(使用部分のみ1に) defaults.None
        """
        self.r      = r     .astype(np.float32)
        self.g      = g     .astype(np.float32)
        self.b      = b     .astype(np.float32)
        self.nir    = nir   .astype(np.float32)
        self.mask   = mask if mask is not None else np.ones_like(self.r)



    def calc_indices(self):
        self.ndvi = (self.nir - self.r) / (self.nir + self.r)
        self.gsi  = (self.r - self.b) / (self.r + self.b + self.g)
    
    def convert_uint8(self, idx):
        """-1~1をとる指標値をuint8形式(0~255)に変換
        1%ile-99%ile scalerを使用

        Args:
            idx (Array like (float)): NDVI等の指標値
        """

        p1  = np.percentile(idx[~np.isnan(idx)], 1)
        p99 = np.percentile(idx[~np.isnan(idx)], 99)


        idx_ = (idx - p1) / (p99 - p1)  # 1%ile~99%ile内に正規化 (nullそのまま、範囲は0~1)
        Relu_pls = lambda x: np.where(x<0, 0, np.where(x>1, 1, x))  # 下限を0, 上限を1にする関数
        idx_uint8 = (Relu_pls(np.where(np.isnan(idx_), 0, idx_)) * 255).astype(np.uint8)
        
        return idx_uint8

    def Otsu(self, idx):
        """入力した指標値の閾値を求める

        Args:
            idx (Array like, uint8): dtype uint8の指標値
        """

        
        ret, dst = cv2.threshold(idx[self.mask==1], 0, 255, cv2.THRESH_OTSU)
        return ret
    
    def postprocessed(self, AMarea, minimumsize=5):
        """一定面積未満の抽出箇所を除外

        Args:
            
            minimumsize (int, optional): 閾値とするピクセルの数. Defaults to 5.
        """
        id_size, labeled_img = cv2.connectedComponents(AMarea)  # 塊ごとにラベルを振る
        id_arr , area_arr    = np.unique(labeled_img, return_counts=True)  # 各塊のピクセルサイズを計算する
        area_img             = area_arr[labeled_img]             # 各塊のピクセルに面積を代入する

        preprocessed_img = np.where(area_img==np.nanmax(area_img), 0, np.where(area_img<minimumsize, 0, 1))
        return preprocessed_img