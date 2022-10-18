#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# judge_crow.py: カラスがいるかどうか判定する

# %%
from tkinter.tix import Tree
import numpy as np
from matplotlib import pyplot as plt
import glob
import cv2
import threading
import datetime
import pandas as pd
from . import calc_sinh

# %% 
def get_circle_img(path='E:/ResearchData4/Level1/circle_img/Mask75circle.tif'):
    """マスク用サークル画像をメモリに移す
    """
    circle_3d = cv2.imread(path)
    circle_2d = np.where(circle_3d[:,:,0]==0, np.nan, 1).astype(np.float32)
    return circle_2d

# %%
class ProcessWsiImage(threading.Thread):
    """BI値を計算する

    """

    def __init__(self, wsi_path=None, circle_2d=None, circle_area=None, masking=True):
        """RGB画像からBI画像を作成する
        しきい値を設定することでカラス抜き画像や白飛び抜き画像を作成する

        Args:
            wsi_path (str, optional): 処理を行う全天画像. Defaults to None.
            circle_2d (Array like(2d), optional): マスキング用サークル画像. Defaults to None.
            circle_area (int): マスキング用サークルのピクセル数. Defaults to None
            masking (bool): マスキングを実行するかどうか

        Attributes:
            .wsi_path (str)                         : 全天画像のパス
            .circle_2d (Array like (2d, float32))   : マスク用サークル画像(2d)
            .wsi_3d (Array like (3d, uint8))        : 使用する全天画像(3d)
            .bi_img (Array like (2d, uint8))        : BI画像(2d)
            .crow_binary (Array like (2d, uint8))   : カラス検出の2値画像
            .sun_binary (Array like (2d, uint8))    : 太陽検出の2値画像
            .masked_bi_img (Array like (2d))        : カラス,太陽等のマスキングを行ったあとのbi画像
            .crow_area (float)                      : カラスが占める領域面積を保持
            .crow_is(bool)                          : カラスの有無があるかどうかを判定  Defaults to False
            .self.circle_area (int)                 : マスキングに使用するサークルのピクセル数
            .used_bi (bool)                         : ピクセル数が計算に使用できるかどうかを判定する.  Defaults to True.
            .used_area (int)                        : 計算に使用できるピクセル数
            .used_area_img (Array like (2d))        : 使用できるピクセルの二値画像
            .used_area_rate (float)                 : 使用できるピクセルの割合
            .masking_flag (bool)                    : マスキングを実行するかどうか Default to True.
        """
        super().__init__()  # スレッドクラスをオーバーライド
        self.wsi_path       = wsi_path      # 全天画像のパスを保持
        self.circle_2d      = circle_2d     # サークル画像の読み込み
        self.wsi_3d         = None          # 使用する全天画像(RGB画像)
        self.bi_img         = None          # BI画像(2d)
        self.crow_binary    = None          # カラス検知の二値画像
        self.sun_binary     = None          # 太陽検知の二値画像
        self.masked_bi_img  = None          # マスキングを行ったあとのBI画像
        self.crow_area      = None          # カラスが占める領域の面積
        self.crow_is        = False         # カラスの有無があるかどうかを判定
        self.circle_area    = circle_area   # サークル画像のピクセル数
        self.used_bi        = True          # BIの計算に使用できるかどうかを判定する
        self.used_area      = None          # 使用できるピクセル数
        self.used_area_img  = None          # 使用できるピクセルの二値画像
        self.used_area_rate = None          # 使用できる面積の割合
        self.masking_flag   = masking       # マスキングを実行するかどうか

    def run(self):
        """実行用メソッド
        """

        # 上限と下限を指定してbiを計算
        self.get_wsi_img(self.wsi_path)  # WSI画像を取得
        self.calc_bi_img()  # bi画像を作成

        if self.masking_flag:  # マスキングする場合
            self.bi_masking_integration(min=15, max=250, area=0.95)  # BIをマスキング

        else:  #マスキングしない場合
            self.bi_no_masking()

    def get_wsi_img(self, path):
        """指定パスの全天画像をメモリに取り込む
        Args:
            path (str): 全天画像のパス(全天画像はjpg or png or tif)
        """
        self.wsi_3d = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    
    def bi_masking_integration(self, min=15, max=250, area=0.95):
        """指定強度のピクセルを用いて画像を再構成

        Args:
            min (int, optional): _description_. Defaults to 15.
            max (int, optional): _description_. Defaults to 250.
            area (float, optional): _description_. Defaults to 0.95.
        """
        self.masked_bi_img = np.where(
            (self.bi_img>min)&(self.bi_img<max)&(self.circle_2d==1),
            self.bi_img,
            np.nan
        )
        self.used_area_img = np.where(
            (self.bi_img>min)&(self.bi_img<max)&(self.circle_2d==1),
            1,
            np.nan
        )
        self.used_area_rate = np.nansum(self.used_area_img) / self.circle_area
        self.used_bi = self.used_area_rate>area
    
    def bi_no_masking(self):
        self.masked_bi_img = np.where(self.circle_2d==1, self.bi_img, np.nan)
        self.used_bi = True

    def calc_bi_img(self):
        """取り込んでいるWSIイメージからBI画像を算出

        Returns:
            Array like(2d, uint8); BIイメージ画像
        """
        self.bi_img = np.nansum(self.wsi_3d/3, axis=2).astype(np.uint8)
        return self.bi_img
    
    def detect_crow(self, threshold=35):
        """カラスを検出した二値画像を作成する
        
        Args:
            threshold(int): 画素値が指定以下の時カラスと判定する. Defaults to 35.

        Returns:
            Array like(2d, uint8): カラスを1, それ以外を0とした二値画像
        """
        self.crow_binary = np.where(self.bi_img<threshold, 1, 0).astype(np.uint8)
        return self.crow_binary
    
    def detect_sun(self, threshold):
        """太陽を検出した二値画像を作成する
        Args:
            threshold(int): 画素値が指定以上の時太陽と判定する. Defaults to 250.

        Returns:
            Array like(2d, uint8): 太陽を1, それ以外を0とした二値画像
        """
        self.sun_binary = np.where(self.bi_img>threshold, 1, 0).astype(np.uint8)
        return self.sun_binary
    
    def judge_crow(self, area_min=0.09, area_max=0.6):
        """カラスがいるかいないかを判定する

        Args:
            area_min (float): カラスが領域に占める割合の下限. Defaults to 0.09.
            area_max (float): カラスが領域に占める割合の上限. Defaults to 0.6.

        Returns:
            crow_area (float): カラスが領域にしめる割合
            crow_is (bool): カラスが存在するかどうか
        """
        circle_masked_crow = np.where(self.circle_2d==1 ,self.crow_binary, np.nan)  # サークルでマスキングしたカラス二値画像
        self.crow_area = np.nansum(circle_masked_crow) / np.nansum(self.circle_2d)  # カラスが領域に占める割合
        self.crow_is = True if (self.crow_area>area_min)&(self.crow_area<area_max) else False
        return self.crow_area, self.crow_is

    def bi_masking(self, used_filter=None):
        """画像内のBI
        Args:
            used_filter(list, (str)): カラス・太陽に対してフィルターをかける(select 'crow', 'sun)

        Returns:
            Array like(2d): マスキングを行ったあとのBI画像
        """
        self.masked_bi_img = self.bi_img.astype(np.float32)
        if type(used_filter)==list:
            for filter in used_filter:
                if filter=='crow':
                    self.masked_bi_img = np.where(self.crow_binary==1, np.nan, self.masked_bi_img)
                if filter=='sun':
                    self.masked_bi_img = np.where(self.sun_binary==1, np.nan, self.masked_bi_img)
        
        self.masked_bi_img = np.where(self.circle_2d==1, self.masked_bi_img, np.nan)  # サークルでマスキング
        return self.masked_bi_img

# %%
#マルチスレッド処理(10分間隔5枚ずつ)
class MultiWsiImage:
    def __init__(
        self,
        input_dir_path = 'E:/ResearchData4/Level1/wsi_202209/',
        circle_img_path= 'E:/ResearchData4/Level1/circle_img/Mask75circle.tif',
        masking=True):
        """WSI画像の10分平均処理
        BI10分平均値を計算する。マルチスレッド対応済み

        Args:
            input_dir_path (str, optional)  : 元画像の保存先ディレクトリ. 
            circle_img_path (str, optional) : マスク用画像(tif)のパス.
            masking (bool)                  : マスキングを実行するかどうか Default to True.

        Attributes:
            .input_dir_path (str, path)     : 元画像の保存先ディレクトリ
            .circle_img (Array like, 2d)    : マスク用画像の配列
            .circle_area (int)              : マスク用画像のピクセル数
            .bi_mean (float)                : 画像全体のBI平均
            .bi_std (float)                 : 画像全体のBI標準偏差
            .h (int)                        : 画像の高さ
            .w (int)                        : 画像の幅
            .used_img_num (int)             : 10分平均作成に使用した画像の枚数
            .out_df (pandas.DataFrame)      : 出力用DataFrame
            .masking_flag (bool)            : マスキングを実行するかどうか Default to True.
        """
        
        self.input_dir_path = input_dir_path
        self.circle_img     = get_circle_img(circle_img_path)
        self.circle_area    = np.nansum(self.circle_img)
        self.bi_mean        = None
        self.bi_std         = None
        self.h, self.w      = self.circle_img.shape
        self.used_img_num   = None
        self.out_df         = pd.DataFrame()
        self.masking_flag   = masking
    
    def run(self, start, end, min_sun_height=5, min_used=1, lon=139.48, lat = 35.68):
        """バッチ処理を行う

        Args:
            start (datetime.datetime)   : 処理開始時間
            end (datetime.datetime)     : 処理終了時間
            min_sun_height (float)      : 最低太陽高度(degree)
            min_used (int)              : 処理に使う画像の最低枚数 Defaults to 1
        """

        min_sinh = np.sin(np.deg2rad(min_sun_height))
        loop_date_ls = np.arange(start, end, datetime.timedelta(minutes=10))
        for date_64 in loop_date_ls:
            base_date = pd.to_datetime(date_64)
            if calc_sinh(lon=lon, lat=lat, date=base_date) < min_sinh:  # sinhがしきい値を超えないときは計算しない
                continue
            self.calc_bi_stats_1img(base_date, min_used)
            self.out_df.loc[date_64, 'BI_mean'] = self.bi_mean  # BIの平均値
            self.out_df.loc[date_64, 'BI_std'] = self.bi_std  # BIの標準偏差
            self.out_df.loc[date_64, 'used_img'] = self.used_img_num  # 10分平均を算出するのに使った画像数


            # デバック用
            if (base_date.hour==12)&(base_date.minute==0):
                print(f'Processed:{base_date} | (now:{datetime.datetime.now()}')



    def calc_bi_stats_1img(self, base_date, min_used=1):  # 基準時間前5枚を使ってbiを算出
        """基準時間前5枚を使ってbiを算出

        Args:
            base_date (datetime.datetime): 基準時間
            min_used (int): 処理に使う画像の最低枚数 Defaults to 1
        """

        threads = []  # スレッドをまとめる
        for i in range(5):
            get_img_date = datetime.datetime.strftime(
                base_date - datetime.timedelta(minutes=i*2),
                '%Y%m%d_%H%M'
                )
            get_img_path_ls = glob.glob(f'{self.input_dir_path}/{get_img_date}*.jpg')

            if len(get_img_path_ls)==0:  # 画像が見つからない時はpass
                continue

            pwi = ProcessWsiImage(
                wsi_path    = get_img_path_ls[0],
                circle_2d   = self.circle_img,
                circle_area = self.circle_area,
                masking     = self.masking_flag
            )
            pwi.start()
            threads.append(pwi)
        
        # スレッドが0のときの処理
        if len(threads)==0:
            self.bi_mean = np.nan
            self.bi_std = np.nan
            self.used_img_num = 0
            return

        bi_all_img = []
        self.used_img_num = 0

        # スレッドが存在する時は計算を行う
        for i, thread in enumerate(threads):
            thread.join()
            if thread.used_bi:  # BI画像が処理可能なら以下の操作を行う
                bi_all_img.append(thread.masked_bi_img)
                self.used_img_num += 1
        
        # 所定以上の枚数使用可能な画像がある時は計算を行う
        if self.used_img_num>=min_used:  # 画像が所定枚数以上ある時,平均・標準偏差を計算する
            bi_all_arr = np.array(bi_all_img)
            self.bi_mean = np.nanmean(bi_all_arr) / 255
            self.bi_std = np.nanstd(bi_all_arr) / 255
            return

        # 所定以上の枚数がないときは無視する
        else:
            self.bi_mean = np.nan
            self.bi_std = np.nan
            return


# %% 処理部分
if __name__ =='__main__':
    start = datetime.datetime.now()  # 時間計測用(DEBUG)
    mwi = MultiWsiImage()  # インスタンス化
    mwi.run(
        start=datetime.datetime(2022, 9, 2, 0, 0),
        end = datetime.datetime(2022, 9, 3, 0, 0)
    )
    # 時間計測用(DEBUG)
    end = datetime.datetime.now()
    print(f'time: {(end - start).total_seconds()}s')
