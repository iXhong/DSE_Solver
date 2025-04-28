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
    wp = wp*xp*np.log(10)/(2*np.sqrt(xp))

    return xz,wz,xp,wp