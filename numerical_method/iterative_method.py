import numpy as np

def f(x):
    return np.exp(-x)
    
def f1(x):
    return np.cos(x)

def f2(x,beta):
    return np.tanh(beta*x)

def iterative_solver(f,init,iter,tol,args):
    x0 = init
    for i in range(iter):
        x = f(x0,args)
        if np.abs(x-x0) < tol:
            break
        x0 = x

    return x


def solve():
    for beta in [0.5,1,1.5]:
        solution = iterative_solver(f2,init=0.1,iter=1000,tol=1e-6,args=beta)
        print(f"beta={beta},x={solution}")
    
solve()