"""
@solve DSE at non-zero tempteature non-zero chmeical potential
@author George Liu
@since 2025.04
"""

import numpy as np
from module import gausslegendreGrid
from numba import jit

sigma = 0.4 #GeV
T = 0.12    #GeV
m0 = 0.5    #GeV
p2_min, p2_max = 1e-4,1e4
max_iter = 1000
eps = 1e-3
Np = 20    #momentum points num
Nz = 25     #angular points num
Nf = 20     #frequency points num
# omega = np.linspace(-Nf,Nf,20)
omega = (2 * np.arange(-10, 10) + 1) * np.pi * T
# z2 = 9.898648e-01
# z4 = 7.796126e-01


def c(T):
    return 16*np.pi**2*T/(3*sigma**6)


def k2_nl(p2,q2,z,n,l):
    """
    return k^2_nl
    """
    return  p2+q2-2*np.sqrt(p2*q2)*z + (omega[n]-omega[l])**2


# def f_nl(p2,q2,z,omega_n,omega_l):
#     k2 = k2_nl(p2,q2,z,omega_n,omega_l)
#     k2 = max(k2,1e-10)
#     return k2*np.exp(-k2/sigma**2)


def solver(xmin,xmax):

    Ai = np.ones((Np,Nf))
    Bi = np.full((Np,Nf),m0)
    Ci = np.ones((Np,Nf))
    A = np.zeros((Np,Nf))
    B = np.zeros((Np,Nf))
    C = np.zeros((Np,Nf))

    xz,wz,xp,wp = gausslegendreGrid(xmin,xmax,Nz,Np)

    @jit(nopython=True)
    def IntreABC(p2,n):
        fA,fB,fC = 0,0,0
        sumA,sumB,sumC = 0,0,0
        for l,_ in enumerate(omega):
            for i in range(Np):
                for j in range(Nz):
                    jaccobian = xp[i]*wp[i]*wz[j]/(2*np.sqrt(xp[i]))
                    denom = (xp[i]*Ai[i,l]**2+Bi[i,l]**2)
                    k2 = max(k2_nl(p2,xp[i],xz[j],n,l),1e-10)
                    pq = np.sqrt(p2*xp[i])*xz[j]
                    kp = p2-pq
                    kq = -xp[i]+pq
                    omega_nl = omega[n]-omega[l]
                    fnl = k2*np.exp(-k2/sigma**2)
                    #integration
                    fA += fnl*Ai[i,l]*jaccobian*(k2*pq+2*kp*kq+2*Ci[i,l]*omega[l]*omega_nl*kp)/(denom*k2*p2)
                    fB += fnl*3*Bi[i,l]*jaccobian/denom
                    fC += fnl*(omega[l]*k2*Ci[i,l]+2*omega_nl*(Ai[i,l]*kq+omega[l]*omega_nl*Ci[i,l]))*jaccobian/(k2*omega[n]*denom)
            
            sumA += fA 
            sumB += fB
            sumC += fC

        Axy = 1 + c(T)*sumA/(2*np.pi)**2
        Bxy = m0 + c(T)*sumB/(2*np.pi)**2
        Cxy = 1 + c(T)*sumC/(2*np.pi)**2

        return Axy,Bxy,Cxy
    
    for iter in range(max_iter):
        error = 0

        for i in range(Np):
            for j in range(Nf):
                A[i,j],B[i,j],C[i,j] = IntreABC(xp[i],j)

        error = np.sum(np.abs(B - Bi))
        Ai = A.copy()
        Bi = B.copy()
        Ci = C.copy()

        print(f"Iteration {iter+1},error={error:.6e}")
        if error < eps:
            print(f"Converged after {iter+1} iterations")
            break

    return A,B,C,xp,omega



if __name__ == "__main__":
    
    print("Ready! Run!")
    A,B,C,xp,omega = solver(p2_min,p2_max)
    file = "./abc.npz"
    np.savez(file=file,A=A,B=B,C=C,p2=xp,omega=omega)

