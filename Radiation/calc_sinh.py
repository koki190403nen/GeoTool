#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# calc_sinh.py: 指定緯度経度の指定日時のsinhを計算する
# %%
import numpy as np
import datetime
# %%
def calc_sinh(lat, lon, date, Ls=135):
    """sinhを計算する
        sinh = (sin lat)(sin delta) + (cos lat)(cos delta)(cos omega)

    Args:
        lat (float or np.ndarray): 緯度lattide(°)
        L (float or np.ndarray): 経度longitude(°)
        date(datetime.datetime): 日時
        Ls (float): 標準子午線の経度(明石市の経度(°)) Defaults to 135.
    
    Return:
        sinh(float or np.ndarray): sinh
    """

    #############################
    year, month, day, hour, minute, second = date.year, date.month, date.day, date.hour, date.minute, date.second
    dn = (datetime.datetime(year, month, day) - datetime.datetime(year, 1, 1)).days+1  # DOY
    JST = hour + minute/60 + second/3600  # 地方標準時(時)

    # omegaの計算 ########################
    Ganma = 2*np.pi*(dn - 1)/365  # ラジアン
    Et = (0.000075 + 0.001868*np.cos(Ganma) - 0.032077*np.sin(Ganma) - 0.014615*np.cos(2*Ganma) - 0.04089*np.sin(2*Ganma)) * 229.18  # 均時差(分)
    Hs =  JST + 4*(lon-Ls)/60 + Et/60  # 真太陽時(時)
    if type(Hs)==np.float64:
        if Hs<12:
            omega = 15*(Hs+12)# 時角(°)
        else:
            omega = 15*(Hs-12)
    elif type(Hs)==np.ndarray:
        omega = np.where(
            Hs<12, 15*(Hs+12), 15*(Hs-12)
        )
    # omegaここまで######################



    # delta(太陽赤緯)の計算##########
    delta = \
        (0.006918 - 0.399912*np.cos(Ganma) + 0.070257*np.sin(Ganma) \
        - 0.006758*np.cos(2*Ganma) + 0.000907*np.sin(2*Ganma) \
        - 0.002697*np.cos(3*Ganma) + 0.00148*np.sin(3*Ganma)) \
        * (180/np.pi)  # 太陽赤緯(°)



    sinh = np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(delta)) + np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(delta))*np.cos(np.deg2rad(omega))
    return sinh
# %%
calc_sinh(
    lat=39, lon=139, 
)