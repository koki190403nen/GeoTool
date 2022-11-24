#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mR95pを計算するクラス

# %%
from operator import index
import numpy as np
import datetime
import pandas as pd
from matplotlib import pyplot as plt

# %%
# 各DOYごとにR95inを算出する

class mR95pBase:
    def __init__(self, normal_start_year=1991, normal_end_year=2020):
        """mR95pを計算する

        Args:
            normal_start_year (int): 平年値算出に使うデータの開始年
            normal_end_year (int): 平年値算出に使うデータの終了年
        """
        self.normal_start_year = normal_start_year  # 平年値の計算開始年
        self.normal_end_year = normal_end_year  # 平年値の計算終了年
        
        # 平年値のdateリスト
        self.normal_date_arr = pd.to_datetime(np.arange(
            datetime.datetime(normal_start_year, 1, 1),
            datetime.datetime(normal_end_year, 12, 31, 1),
            datetime.timedelta(days=1)
            ))
        self.mR95pin = None  # mR95pin
        self.PPT_mean = None  # 平年値の計算
        self.mR95p = None  # 月別のmR95p(mm)
        self.mR95pT = None  # 月別のmR95pを30年平均降水量で正規化

    def calc_mR95pin(self, rain30_arr, window_half=7, min_sample_size=30, Rnnmm=10):
        """各平年値(mR95pin, PPT)を計算する

        Args:
            rain30_arr (_type_): 30年平年値を算出するために使用するデータ
            window_half (int, optional): ウィンドウサイズ. Defaults to 7.
            min_sample_size (int, optional): 計算に最低限必要な降雨日数(30年間). Defaults to 30.
            Rnnmm (float): ユーザー定義の最低豪雨しきい値. Defaults to 10.

        """

        doy_arr = self.normal_date_arr.dayofyear.values  # 入力データのDOYリスト
        
        no366_ppt_arr = rain30_arr[doy_arr!=366].reshape(-1, 365)  # DOY366は計算しない
        clean_ppt_arr = np.concatenate([
            no366_ppt_arr.reshape(-1, 365)[:, -1*window_half:],
            no366_ppt_arr.reshape(-1, 365),
            no366_ppt_arr.reshape(-1, 365)[:, :window_half]
        ], axis=1)
        
        r95pin_ls = []
        for i in range(365):
            target_period = clean_ppt_arr[:, i:i+window_half*2+1]
            if len(target_period[target_period!=0])<min_sample_size:
                r95pin = Rnnmm
            else:
                r95pin = np.percentile(target_period[target_period!=0], 95)
            r95pin_ls.append(r95pin)
        r95pin_ls.append(r95pin)  # DOY366用のデータを入れる
        self.mR95pin = np.array(r95pin_ls)
        self.mR95pin[self.mR95pin<Rnnmm] = Rnnmm

        return self
    
    def calc_PPT_mean(rain30_arr):
        pass

    def calc_normalyear(self, rain30_arr, window_half=7, min_sample_size=30, Rnnmm=10):
        """各平年値(mR95pinと期間別平均総降水量)を計算する

        Args:
            rain30_arr (Array like): 平年値計算に使用する降水量データ(30年分)
            window_half (int): 各DOY計算に使用するウィンドウのサイズ. Defaults to 7.
            min_sample_size (int): 最低限必要な降雨日の日数. Defaults to 30.
            Rnnmm (float): ユーザー定義の最低豪雨しきい値. Defaults to 10.

        """
        self.calc_mR95pin(rain30_arr, window_half, min_sample_size, Rnnmm)
        self.calc_PPT_mean(rain30_arr)
        return self

    def set_normalyear(self, r95pin, ppt_mean):
        """平年値がすでに計算済みの時, クラスにセットする

        Args:
            r95pin (Array like (1D, 366)): DOY別95%ileしきい値
            ppt_mean (Array like): 期間別平均総降水量
        """
        self.mR95pin = r95pin
        self.PPT_mean = ppt_mean
        return self
    
    def calc_mR95pT_single(self, rain_arr, start_doy, end_doy):
        """mR95pTを一つだけ推定する

        Args:
            rain_arr (Array like): 入力データ
            start_doy (int): 入力データの開始DOY
            end_doy (int): 入力データの終了DOY

        Returns:
            Array like: 入力降水量データから求められたmR95pT
        """
        threshold_arr = self.mR95pin[start_doy-1:end_doy-1]
        mR95p = np.nanmean(rain_arr[rain_arr>=threshold_arr])
        PPT_mean = np.nanmean(self.PPT_mean[start_doy-1:end_doy-1])
        return mR95p / PPT_mean

class mR95pMonthly(mR95pBase):
    def __init__(self, normal_start_year, normal_end_year):
        """mR95pを計算する(時間解像度は1ヶ月)

        Args:
            normal_start_year (int): 平年値算出に使うデータの開始年
            normal_end_year (int): 平年値算出に使うデータの終了年
        """
        super().__init__(normal_start_year, normal_end_year)

    def calc_PPT_mean(self, rain30_arr):
        """各スパンごとの平均総降水量を計算(オーバーライド)

        Args:
            rain30_arr (Array like, 1D): 30年平年値を算出するために使用するデータ
        """
        year_arr = self.normal_date_arr.year.values
        month_arr = self.normal_date_arr.month.values
        PRCP_ls = []
        for year in range(self.normal_start_year, self.normal_end_year+1):
            for month in range(1, 12+1):
                all_rain_arr = np.where(
                    (year_arr==year)&(month_arr==month),
                    rain30_arr, np.nan
                )
                PRCP = np.nansum(all_rain_arr)
                PRCP_ls.append(PRCP)
        self.PPT_mean = np.nanmean(np.array(PRCP_ls).reshape(-1, 12), axis=0)
        return self

    def calc_mR95pT(self, rain_arr, start_year, end_year):
        """入力データ・入力期間の期間毎mR95p, mR95pTを計算する

        Args:
            rain_arr (Array like): 計算したい期間の降水量データ
            start_year (int): 計算したい期間の開始年
            end_year (int): 計算したい期間の終了年

        """

        date_arr = pd.to_datetime(np.arange(
            datetime.datetime(start_year, 1, 1),
            datetime.datetime(end_year, 12, 31, 1),
            datetime.timedelta(days=1)
            ))
        doy_arr = date_arr.dayofyear.values
        year_arr = date_arr.year.values
        month_arr = date_arr.month.values
        
        year_unique = np.unique(year_arr)[~np.isnan(np.unique(year_arr))]
        start_year = int(year_unique[0])
        end_year = int(year_unique[-1])
        R95p_ls = []
        threshold_arr = self.mR95pin[list((doy_arr-1).astype(int))]
        for year in range(start_year, end_year+1):
            for month in range(1, 12+1):
                over_rain_arr = np.where(
                    (year_arr==year)&(month_arr==month)&(rain_arr>=threshold_arr),
                    rain_arr, np.nan
                )
                R95p = np.nansum(over_rain_arr)
                R95p_ls.append(R95p)
        self.mR95p = np.array(R95p_ls)
        self.mR95pT = (self.mR95p.reshape(-1, 12)/self.PPT_mean).flatten()  # mR95p/PPT_meanを計算
        return self

# %%
if __name__=='__main__':
    df = pd.read_csv('../../sample/prcp_sample.csv', index_col=0)
    area='Zanbia1'
    print(df.head())
    normal_ppt = df.query('year>=1991 & year<=2020')[area].values
    target_ppt = df.query('year>=1991 & year<=2020')[area].values
    mr95mon = mR95pMonthly(1991, 2020)
    res = mr95mon.calc_normalyear(normal_ppt, Rnnmm=10).calc_mR95pT(target_ppt, 1991, 2020)
    
    plt.bar(range(366), res.mR95pin)
    plt.show()