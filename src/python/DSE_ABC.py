"""
@solve DSE at non-zero tempteature non-zero chmeical potential
@author George Liu
@since 2025.04
"""

import numpy as np
from module import loggaussPiecewise
from numba import njit,prange

sigma = 0.4 #GeV
T = 0.139    #GeV
m0 = 0.005    #GeV
p2_min, p2_max = 1e-4,1e4
max_iter = 1000
Np = 50    #momentum points num
Nz = 25     #angular points num
Nf = 20     #frequency points num
D0 = 0.93   #GeV^2


@njit(parallel=True)
def IntreABC(p2, n, xp, wp, xz, wz, omega, Ai, Bi, Ci):
    fA = 0.0 + 0.0j
    fB = 0.0 + 0.0j
    fC = 0.0 + 0.0j
    for l in prange(Nf):
        omega_nl = omega[n] - omega[l]
        for i in range(Np):
            denom = xp[i] * Ai[i, l]**2 + Bi[i, l]**2 + omega[l]**2*Ci[i,l]**2
            denom = max(np.abs(denom), 1e-10)
            jac = xp[i] * wp[i] * wz/(2*np.sqrt(xp[i]))
            for j in range(Nz):
                k2 = p2 + xp[i] - 2 * np.sqrt(p2 * xp[i]) * xz[j] + (omega[n] - omega[l])**2
                k2_mag = np.abs(k2)
                if k2_mag < 1e-10:
                    k2 = 1e-10 + 0.0j
                pq = np.sqrt(p2 * xp[i]) * xz[j]
                kp = p2 - pq
                kq = -xp[i] + pq
                fnl = k2*np.exp(-k2/sigma**2)
                fA += fnl*Ai[i, l] * jac[j] * (k2 * pq + 2 * kp * kq + 2 * Ci[i, l] * omega[l] * omega_nl * kp) / (denom * k2 * p2)
                fB += fnl*3 * Bi[i, l] * jac[j] / denom
                fC += fnl*(omega[l] * k2 * Ci[i, l] + 2 * omega_nl * (Ai[i, l] * kq + omega[l] * omega_nl * Ci[i, l])) * jac[j] / (k2 * omega[n] * denom)

    cT = D0*16 * np.pi**2 * T / (3 * sigma**6)  # c(T)
    Axy = 1 + cT * fA / (2 * np.pi)**2
    Bxy = m0 + cT * fB / (2 * np.pi)**2
    Cxy = 1 + cT * fC / (2 * np.pi)**2
    return Axy, Bxy, Cxy


def solver(xmin,xmax,mu,eps):

    A = np.zeros((Np, Nf), dtype=np.complex128)
    B = np.zeros((Np, Nf), dtype=np.complex128)
    C = np.zeros((Np, Nf), dtype=np.complex128)
    Ai = np.ones((Np, Nf), dtype=np.complex128)
    Bi = np.full((Np, Nf), m0, dtype=np.complex128)
    Ci = np.ones((Np, Nf), dtype=np.complex128)

    omega = (2 * np.arange(-Nf//2, Nf//2) + 1) * np.pi * T + 1j * mu
    # xz,wz,xp,wp = gausslegendreGrid(xmin,xmax,Nz,Np)
    xz,wz,xp,wp = loggaussPiecewise(xmin,xmax,10,30,10,25)
    
    for iter in range(max_iter):
        error = 0
        for i in range(Np):
            for j in range(Nf):
                A[i,j],B[i,j],C[i,j] = IntreABC(xp[i],j,xp,wp,xz,wz,omega,Ai,Bi,Ci)

        # error = np.sum(np.abs(A - Ai)) + np.sum(np.abs(B - Bi)) + np.sum(np.abs(C - Ci))
        error = (np.sum(np.abs(A - Ai)) + np.sum(np.abs(B - Bi)) + np.sum(np.abs(C - Ci))) / (np.sum(np.abs(A)) + np.sum(np.abs(B)) + np.sum(np.abs(C)))
        Ai[:, :] = A[:, :]
        Bi[:, :] = B[:, :]
        Ci[:, :] = C[:, :]

        print(f"Iteration {iter+1},error={error:.6e}")
        if error < eps:
            print(f"Converged after {iter+1} iterations")
            break

    return A,B,C,xp,omega


def sovle_mus(eps):
    mus = np.linspace(0.0,0.2,10)
    for i,mu in enumerate(mus):
        A,B,C,xp,omega = solver(p2_min,p2_max,mu,eps)
        np.savez(file=f"./data/abc_{i}.npz",A=A,B=B,C=C,p2=xp,omega=omega)



if __name__ == "__main__":

    eps = 1e-7
    print("Ready, Run!")
    # sovle_mus(eps)

    A,B,C,xp,omega = solver(p2_min,p2_max,0.1,eps)
    file = f"./abc_test07.npz"
    np.savez(file=file,A=A,B=B,C=C,p2=xp,omega=omega)