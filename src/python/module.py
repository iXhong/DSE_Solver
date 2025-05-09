"""
@name module.py
@author George Liu
@since 2025.04
"""
from scipy.special import roots_legendre
import numpy as np


def gausslegendreGrid(xp1,xp2,Nz,Np):
    """
    x, w in Gauss-Legendre points
    Paramters:
        xp1: left endpoint of momentum p^2
        xp2: right endpoint of momentum p^2

    Return:
        xz,wz: Guass-legendre point in [-1,1]
        xp,wp: Guass-legendre point combined with log transform in [xp1,xp2]

    Note:
        x stands for grid points array; w stands for weight array
    """
    #gausslegendre points for angular integration 
    xz, wz = roots_legendre(Nz)
    #gausslegendre points for momentum integration
    xp, wp = roots_legendre(Np)
    # transform the points from [-1,1] to [-4,4]
    xp1_log, xp2_log = np.log10(xp1), np.log10(xp2)
    xp = (xp2_log-xp1_log)*xp/2 + (xp1_log+xp2_log)/2
    wp = (xp2_log-xp1_log)/2 * wp
    #log transform to [1e-4,1e4]
    xp = 10**xp
    wp = wp*xp*np.log(10)

    return xz,wz,xp,wp


def loggaussPiecewise(xmin, xmax, N_low, N_mid, N_high,Nz, bounds=(1.0, 100.0)):
    """
    生成分段加密、对数尺度、高斯-勒让德积分节点和权重。

    返回：
        xp_full : 所有动量节点 (p²)
        wp_full : 所有权重 (用于积分)
    """
    xz, wz = roots_legendre(Nz)

    mid_lower, mid_upper = bounds

    xp_list = []
    wp_list = []

    # 第一段 [xmin, mid_lower]
    xl, wl = roots_legendre(N_low)
    log_min = np.log10(xmin)
    log_mid_lower = np.log10(mid_lower)
    xp = 10**(0.5*(log_min+log_mid_lower) + 0.5*(log_mid_lower-log_min)*xl)
    wp = (0.5*(log_mid_lower-log_min)) * wl * xp * np.log(10)
    xp_list.append(xp)
    wp_list.append(wp)

    # 第二段 [mid_lower, mid_upper]
    xl, wl = roots_legendre(N_mid)
    log_mid_lower = np.log10(mid_lower)
    log_mid_upper = np.log10(mid_upper)
    xp = 10**(0.5*(log_mid_lower+log_mid_upper) + 0.5*(log_mid_upper-log_mid_lower)*xl)
    wp = (0.5*(log_mid_upper-log_mid_lower)) * wl * xp * np.log(10)
    xp_list.append(xp)
    wp_list.append(wp)

    # 第三段 [mid_upper, xmax]
    xl, wl = roots_legendre(N_high)
    log_mid_upper = np.log10(mid_upper)
    log_max = np.log10(xmax)
    xp = 10**(0.5*(log_mid_upper+log_max) + 0.5*(log_max-log_mid_upper)*xl)
    wp = (0.5*(log_max-log_mid_upper)) * wl * xp * np.log(10)
    xp_list.append(xp)
    wp_list.append(wp)

    # 拼接起来
    xp = np.concatenate(xp_list)
    wp = np.concatenate(wp_list)

    return xz,wz,xp,wp
