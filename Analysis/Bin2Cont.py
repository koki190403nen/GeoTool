#/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class Bin2Cont:

    def __init__(self, in_arr=None):
        """バイナリ時系列の連続具合を数値化する

        Args:
            in_arr (_type_): _description_
        """
        self.in_arr =in_arr

    def fit(self, in_arr=None):
        if (self.in_arr is not None) & (in_arr is None):
            pass
        else:
            self.in_arr = in_arr
        self.calc_serial().calc_length()
        return self

    def calc_serial(self):
        """元の時系列を0,1,2,0,0,1,0のように変換する
        """
        out_ls = []
        for i, val in enumerate(self.in_arr):
            if i==0:
                out_ls.append(val)
                continue

            out_ls.append((out_ls[i-1]+1)*val)
        self.serial = np.array(out_ls)
        return self
    
    def calc_length(self):
        out_ls = []
        for i, (in_val, ser_val) in enumerate(zip(self.in_arr[::-1], self.serial[::-1])):
            if i==0:
                out_ls.append(ser_val)
                continue
            if out_ls[i-1]>ser_val:
                out_ls.append(out_ls[i-1]*in_val)
            elif ser_val>0:
                out_ls.append(ser_val)
            else:
                out_ls.append(0)
                
        self.len_arr = np.array(out_ls[::-1])
        return self