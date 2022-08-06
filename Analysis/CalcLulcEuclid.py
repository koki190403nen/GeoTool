#!/usr/bin/env python3
# -*- coding : utf-8
# CalcLulcEuclid.py: ユークリッド距離を使ってLULCの変化点を計算する

# %%
import numpy as np
from scipy import signal
def calc_euclid_distance(arr:np.array, period=23, height=5000, distance=10):
    """calculate euclid distance from before year period

    Args:
        arr (numpy.array): NDVI時系列データ(平滑化済み)
        period (int): 1年当たりのサンプリング数 Defaults to 23.
        height (int): 異常と判断する際のしきい値 Defaults to 5000.
        distance (int): 近くの変曲点は無視する Defaults to 10.

    Returns:
        distance_arr (numpy.array): 1年前の周期とのユークリッド距離
        peak_iter (numpy.array): 検出したピークの要素番号
    """
    distance_arr = np.ones(len(arr))
    for i in range(len(arr) - period*2):
        before_arr = arr[i : i+period]  # 昨年度の
        after_arr = arr[i+period : i+period*2]
        distance_value = np.linalg.norm(before_arr-after_arr)
        distance_arr[i+period] = distance_value
        peak_iter = signal.find_peaks(distance_arr, height=height, distance=distance)[0]
    return distance_arr, peak_iter