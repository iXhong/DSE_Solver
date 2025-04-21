import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


def f(x):
    # return np.exp(-x)
    return 1/x

#矩形积分
def I_rect(f):
    x = np.linspace(0,1,100)
    dx = np.diff(x)
    Int = 0
    for i,x_i in enumerate(x):
        Int += f(x_i)*dx[i-1]

    return Int


#梯形积分
def I_trapez(f):
    x = np.linspace(0,1,100)
    dx = np.diff(x)
    Int = 0
    for i in range(len(x)-1):
        Int += (f(x[i])+f(x[i+1]))/2*dx[i-1]

    return Int

def I_np(f):
    x = np.linspace(0,1,1000)
    Int = np.trapezoid(f(x),x)

    return Int

#注意np.logspace的定义！！
def loggrid(min,max,N):
    grid = np.logspace(np.log10(min),np.log10(max),N)

    return grid

def lingrid(min,max,N):
    min = 1e-4
    max = 1e3
    N = 100

    grid = np.linspace(np.log10(min),np.log10(max),N)

    return grid


def plot():
    grid = loggrid()
    log_grid = np.log10(grid)
    # grid = lingrid()

    fig, ax = plt.subplots()
    # y = np.ones_like(grid)
    y1 = 1/2 * np.ones_like(log_grid)
    # ax.scatter(grid,y)
    ax.scatter(log_grid,y1)
    fig.savefig("loggrid_in_logspace.jpg")
    plt.show()


def I_log(f,min,max,N):
    x_grid = loggrid(min,max,N)
    r = np.trapezoid(f(x_grid),x_grid)

    return r


def test():
#np.trapzoide & np.logspace
    grid = np.logspace(-3,3,100)
    y = 1/grid
    r = np.trapezoid(y, grid)

    return r


def main():

    # r = I_log(f,1e-3,1e3,1000)
    r = test()
    print(r)

    return 0

main()
    
