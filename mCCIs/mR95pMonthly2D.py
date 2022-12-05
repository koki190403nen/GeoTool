#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mR95pMonthly2D: 月次mR95pの2Dを計算する.
# %%
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime

if __name__=='__main__':
    from mR95p import mR95pMonthlyBase
else:
    from .mR95p import mR95pMonthlyBase

import warnings
warnings.simplefilter('ignore')

# %%
class mR95pMonthly2D(mR95pMonthlyBase):
    def __init__(self, normal_start_year=1991, normal_end_year=2020):
        """月次mR95pを計算する(2D)

        Args:
            normal_start_year (int): 平年値算出に使うデータの開始年
            normal_end_year (int): 平年値算出に使うデータの終了年
        """
        super().__init__(normal_start_year, normal_end_year)
    