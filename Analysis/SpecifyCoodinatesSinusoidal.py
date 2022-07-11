#!/usr/bin/env python3
# -*- cooding: utf-8 -*-
# SpecifyCoodinatesSinusoidal.py : サンソン図法の緯度経度をもとに写真座標を求める
# %% 指定した緯度経度のピクセルの図形座標を求める
import numpy as np
def SpecifyCoodinatesSinusoidal(lon, lat, lon_0=0, tile_h=29, tile_w=5, pixel=2400, int_return=True):

    world_x = 0.5 + (lon - lon_0)/360 * np.cos(np.radians(lat))
    world_y = lat / 180 - 0.5

    local_x = (np.abs(world_x /(1/36)) - tile_h) * pixel
    local_y = (np.abs(world_y /(1/18)) - tile_w) * pixel

    if int_return:
        local_x = int(local_x)
        local_y = int(local_y)

    return local_x, local_y