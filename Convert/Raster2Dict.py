#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# raster2dict_mR95pT_NDVI.py: RAWデータから指定した座標値のmR95pTとNDVIを抜き出して書き出す

# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime
import json

# %%
class Raster2Dict:
    def __init__(self, lat, lon):
        self.lat=lat
        self.lon=lon
        self.dataset_dict={
            'meta':{'lat':lat, 'lon':lon}
        }
    
    def capture(self):
        self.get_mR95pT()
        self.get_R95pT()
        self.get_NDVI()
        self.get_PPT()
        self.get_LULC()
        self.get_CDD()

        return self.dataset_dict
    
    def get_mR95pT(self):
        print('Import mR95pT values ...')
        h,w = 1600, 1500
        row, col = int((self.lat-(40)) / (-0.05)), int((self.lon-(-20)) / (0.05))
        
        mR95pT_date_arr = pd.to_datetime(
            [f'{year}/{month}' for year in range(1981, 2021+1) for month in range(1, 12+1)]
            )
        mR95pT_value_ls = []

        for date in mR95pT_date_arr:
            get_img = np.fromfile(
                f'D:/ResearchData3/Level4/mCCIs/Monthly/mR95pT/mR95pT.B{date.strftime("%Y%m")}.float64_h1600w1500.raw',
                count=h*w, dtype=np.float64
            ).reshape(h,w)
            mR95pT_value_ls.append(get_img[row, col])

        mR95pT_dict = {
            'date': list(mR95pT_date_arr.strftime('%Y/%m/%d').values),
            'values': mR95pT_value_ls
        }
        self.dataset_dict['mR95pT'] = mR95pT_dict
        return mR95pT_dict
    
    def get_R95pT(self):
        print('Import R95pT values ...')
        h,w = 1600, 1500
        row, col = int((self.lat-(40)) / (-0.05)), int((self.lon-(-20)) / (0.05))
        
        R95pT_value_ls = []
        R95pT_date_arr = pd.to_datetime(
            [f'{year}/{month}' for year in range(1981, 2021+1) for month in range(1, 12+1)]
            )
        for date in R95pT_date_arr:
            get_img = np.fromfile(
                f'D:/ResearchData3/Level4/CCIs/Monthly/R95pT/R95pT.B{date.strftime("%Y%m")}.float64_h1600w1500.raw',
                count=h*w, dtype=np.float64
            ).reshape(h,w)

            R95pT_value_ls.append(get_img[row, col])
        
        R95pT_dict = {
            'date':list(R95pT_date_arr.strftime('%Y/%m/%d').values),
            'values':R95pT_value_ls
        }
        self.dataset_dict['R95pT'] = R95pT_dict
        return R95pT_dict
    
    def get_NDVI(self):

        print('Import NDVI values ...')
        h,w = 1600, 1500
        row, col = int((self.lat-(40)) / (-0.05)), int((self.lon-(-20)) / (0.05))
        ndvi_date_arr = pd.to_datetime(
            [
                datetime.datetime.strptime(f'{year}/{str(doy).zfill(3)}', '%Y/%j')
                    for year in range(2001, 2020+1) for doy in range(1, 366+1, 16)])
        ndvi_value_ls = []
        for date in ndvi_date_arr:
            get_img = np.fromfile(
                f'D:/ResearchData3/Level3/MOD13C1_RAW/MOD13C1.A{date.strftime("%Y%j")}.int16_h1600w1500.raw',
                count=h*w, dtype=np.int16
            ).reshape(h,w)
            ndvi_value_ls.append(get_img[row, col]/10000)

        ndvi_dict = {
            'date': list(ndvi_date_arr.strftime('%Y/%m/%d').values),
            'values':ndvi_value_ls
        }
        self.dataset_dict['NDVI'] = ndvi_dict
        return ndvi_dict

    def get_PPT(self):
        print('Import PPT values...')
        h,w = 1600, 1500
        row, col = int((self.lat-(40)) / (-0.05)), int((self.lon-(-20)) / (0.05))
        ppt_date_arr = pd.to_datetime(np.arange(
            datetime.datetime(1981, 1, 1),
            datetime.datetime(2021, 12, 31, 1),
            datetime.timedelta(days=1)
        ))
        ppt_value_ls=[]

        for date in ppt_date_arr:
            get_img = np.fromfile(
                f'D:/ResearchData3/Level3/chirps005_RAW_f32/chirps_005.A{date.strftime("%Y%j")}.float32_h1600w1500.raw',
                count=h*w, dtype=np.float32
            ).reshape(h,w).astype(np.float64)
            ppt_value_ls.append(get_img[row, col])
        
        ppt_dict = {
            'date': list(ppt_date_arr.strftime('%Y/%m/%d').values),
            'values': ppt_value_ls
        }
        self.dataset_dict['PPT'] = ppt_dict
        return ppt_dict
    
    def get_LULC(self):
        print('Import LULC ...')
        h,w = 1600, 1500
        row, col = int((self.lat-(40)) / (-0.05)), int((self.lon-(-20)) / (0.05))
        lulc_date_ls = [year for year in range(2002, 2019)]
        lulc_ls = []

        for year in lulc_date_ls:
            get_img = np.fromfile(
                f'D:/ResearchData3/Level3/MCD12C1_RAW/MCD12C1.A{year}001.uint8_h1600w1500.raw',
                count=h*w,dtype=np.uint8
            ).reshape(h,w).astype(np.float64)
            lulc_ls.append(get_img[row, col])
        
        lulc_dict = {
            'date': lulc_date_ls,
            'values': lulc_ls
        }
        self.dataset_dict['LULC'] = lulc_dict
        return lulc_dict
    
    def get_CDD(self):
        print('Import CDD ...')
        h,w = 1600, 1500
        row, col = int((self.lat-(40)) / (-0.05)), int((self.lon-(-20)) / (0.05))
        cdd_date_ls = pd.date_range('1981/1/1', '2021/12/31', freq='MS')+datetime.timedelta(days=14)
        cdd_ls = []

        for date in cdd_date_ls:
            get_img = np.fromfile(
                f'D:/ResearchData3/Level4/CCIs/Monthly/CDD/CDD.B{date.strftime("%Y%m")}.float64_h1600w1500.raw',
                count=h*w, dtype=np.float64
            ).reshape(h,w)
            cdd_ls.append(get_img[row, col])

        cdd_dict = {
            'date':list(cdd_date_ls.strftime('%Y/%m/%d').values),
            'values': cdd_ls
        }

        self.dataset_dict['CDD'] = cdd_dict

        return cdd_dict
    
    def get_PRCPTOT(self):
        print('Import PRCPTOT ...')
        h,w = 1600, 1500
        row, col = int((self.lat-(40)) / (-0.05)), int((self.lon-(-20)) / (0.05))
        prcptot_date_ls = pd.date_range('1981/1/1', '2021/12/31', freq='MS')+datetime.timedelta(days=14)
        prcptot_ls = []
        for date in prcptot_date_ls:
            get_img = np.fromfile(
                f'D:/ResearchData3/Level4/CCIs/Monthly/PRCPTOT/PRCPTOT.B{date.strftime("%Y%m")}.float64_h1600w1500.raw',
                count=h*w, dtype=np.float64
            ).reshape(h,w)
            prcptot_ls.append(get_img[row, col])
        
        prcptot_dict = {
            'date':list(prcptot_date_ls.strftime('%Y/%m/%d').values),
            'values':prcptot_ls
        }

        return prcptot_dict
# %%
if __name__=='__main__':
    r2d = Raster2Dict(lat=-17.215, lon=27.424)
    cdd_dict = r2d.get_PRCPTOT()
