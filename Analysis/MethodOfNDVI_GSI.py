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
        self.r, self.g, self.b, self.nir = None, None, None, None
        self.AA_area = None

    def fit(self, pre_bands, post_bands, mask=None):
        self.set_mask(mask)
        pre_veg_area  = self.extract_veg_area(*pre_bands)
        post_bare_area = self.extract_bare_area(*post_bands)

        self.AA_area = np.where(np.isnan(pre_veg_area)|np.isnan(post_bare_area), np.nan, self.preprocess(((pre_veg_area==1)&(post_bare_area==1)).astype(np.uint8))).astype(np.float32)
        return self.AA_area


    def extract_veg_area(self, r, g, b, nir):
        self.set_bands(r,g,b,nir)
        ndvi, gsi = self.calc_indices()

        ndvi_bin  = self.make_binary(ndvi)
        gsi_bin   = self.make_binary(gsi)
        
        veg_area = np.where(np.isnan(ndvi_bin)|np.isnan(gsi_bin), np.nan, (ndvi_bin==1)&(gsi_bin==0)).astype(np.float32)
        return veg_area
    
    def extract_bare_area(self, r,g,b,nir):
        self.set_bands(r,g,b,nir)
        ndvi, gsi = self.calc_indices()

        ndvi_bin  = self.make_binary(ndvi)
        gsi_bin   = self.make_binary(gsi)
        
        bare_area = np.where(np.isnan(ndvi_bin)|np.isnan(gsi_bin), np.nan, (ndvi_bin==0)&(gsi_bin==1)).astype(np.float32)
        return bare_area


    def calc_indices(self):
        ndvi = (self.nir - self.r) / (self.nir + self.r)
        gsi  = (self.r - self.b) / (self.r + self.g + self.b)
        return ndvi, gsi



    def set_bands(self, r, g, b, nir):
        self.r      = r.astype(np.float32)
        self.g      = g.astype(np.float32)
        self.b      = b.astype(np.float32)
        self.nir    = nir.astype(np.float32)
    
    def set_mask(self, mask=None):
        self.mask   = mask.astype(np.float32) if mask is not None else np.ones_like(self.r)




    
    def calc_gsi(self):
        gsi = (self.r - self.b) / (self.r + self.g + self.b)
        return gsi
    
    def make_binary(self, idx):
        """-1 ~ 1(dtype: float) を 大津の二値化にかける。

        Args:
            idx (Array like, dtype: float): ある指標値 (範囲: -1 ~ 1)
        Returns:
            bin_img (Array like, dtype:float32): 大津の二値化によりバイナリ化された画像 (0: 閾値以下, 1: 閾値より大, nan: マスク部)
        """

        # スケーリング部分
        min, max = np.percentile(idx[~np.isnan(idx)], 1), np.percentile(idx[~np.isnan(idx)], 99)
        idx_uint8 = ((idx - min) / (max - min) * 255).astype(np.uint8)

        threshold, _ = cv2.threshold(idx_uint8[self.mask==1], 0, 255, cv2.THRESH_OTSU)

        bin_img = np.where(self.mask!=1, np.nan, np.where(idx_uint8>threshold, 1, 0)).astype(np.float32)
        return bin_img

    def preprocess(self, bin_img, min_pix=5):
        id_size, labeled_img = cv2.connectedComponents(bin_img)
        id_arr , area_arr    = np.unique(labeled_img, return_counts=True)
        area_img             = area_arr[labeled_img]
        preprocessed_img = np.where(area_img==np.nanmax(area_img), 0, np.where(area_img<min_pix, 0, 1))
        return preprocessed_img