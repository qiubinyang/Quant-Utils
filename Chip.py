# coding: utf-8
# Author:Kevin Qiu
import matplotlib.pyplot as plt
import numpy as np
from grouping import group_by
from jqdatasdk import *


def cut(x, q, axis=0):
    ''' 多维数字化分箱函数
    通过线性变换到列直接对多维数据进行分箱
    x     : 1d/2d二维数据(按轴)
    q     : 分箱数/如未提供则默认为1%100间隔数
    axis  : 支持指定轴计算
    '''
    with np.warnings.catch_warnings():
        '''大数据计算中无法避免传入全nan的数据因此直接忽略'''
        np.warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')    
        x = np.asarray(x)
        if axis==0:
            f,l = np.nanmin(x, axis=axis),np.nanmax(x, axis=axis)
        else:
            f,l = np.nanmin(x, axis=axis)[:,None],np.nanmax(x, axis=axis)[:,None]
        if q==None:
            q = np.nanmean((l-f)/f)*100
    return np.rint((q-1)*(x-f)/(l-f))    

def cost_distribution(x,c,q=120):
    '''对商品价格进行数字化分组'''
    cuts = cut(x,q) # x是价格序列
    # 获取数组尺寸,取得有效值(分组号及筹码表)
    shps = x.shape; 
    mask = np.isfinite(cuts) &  np.isfinite(c)    
    # 输出申请个空数组
    rets = np.full(15,np.nan)
    # 汇总统计,q分布价的筹码数量
    unique, ctbe = group_by(cuts[mask].ravel(),hold=q).sum(c[mask].ravel())
    return ctbe

def calc():
    np.set_printoptions(precision=9)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    np.set_printoptions(suppress=True)
    np.set_printoptions(linewidth=220)

    df = get_price('000001.XSHE', end_date='2018-10-08',count=150, frequency='1d', fields=['close'])
    tr = get_fundamentals_continuously(query(valuation.turnover_ratio).filter(valuation.code.in_(['000001.XSHE'])),end_date='2018-10-08',count=150)['turnover_ratio']

    ratio = (np.nan_to_num(tr.values)/100).ravel()
    close = df.values.ravel()
    ratio[0:-1] *= np.cumprod(1-ratio[::-1],0)[::-1][1:]
    cost = cost_distribution(close,ratio)
    return cost

