"""
solve DSE at zero temperature & zero chemical potential
@author George Liu
@since 2025.04
"""
import numpy as np
from module import gausslegendreGrid

#define parameter
D = (0.8**3)/0.5
Nf = 4
m0 = 0.0034
xi = 361 #GeV renomalization point
gamma_m = 12/25 #12/(33-2*Nf)
tau = np.e**2 - 1
lambda_qcd = 0.234 #GeV
mt = 0.5 #GeV
omega = 0.5 #GeV


def g(k2):
    """
    Gluon propagator
    interaction model,use Qin-chang model
    """
    k2 = max(k2,1e-10)
    part1 = 8*np.pi**2*D/omega**4 * np.exp(-k2/omega**2)
    part2 = 8*np.pi**2*gamma_m/np.log(tau+(1+k2/lambda_qcd**2)**2)
    part3 = (1-np.exp(-k2/(4*mt**2)))/k2

    return part1 + part2 * part3


def solver(Nz:int,Np:int,xmin:float,xmax:float,max_iter:int,eps:float):
    """
    iteratively  solve A,B
    Paramters: 
        Nz: integration points for angular part
        Np: integration points for momentum part
        xmin:left endpoint of momentum 
        xmax:right endpoint of momentum
        max_iter: the max iteration times
        eps: tolerance for convergence
    Return:
        z2
        z4
        reA: function A as an array
        reB: function B as an array
        xp:  momentum points
    """
    reAi = np.ones(Np)
    reBi = np.full(Np,0.3)
    reA = np.zeros(Np)
    reB = np.zeros(Np)

    xz, wz, xp, wp = gausslegendreGrid(xmin,xmax,Nz,Np)

    def intreAB(p2,fArn,fBrn):
        """
        solve A(p2),B(p2) at point p2
        Paramters:
            p2: the outside momentum point
            fArn: value of fA at renormalization point
            fBrn: value of fB at renormalization point
        Return:
            Ax: A at point p2
            Bx: B at point p2
        """
        fA, fB = 0, 0
        for i in range(Np):
            for j in range(Nz):
                #integrand & weights
                dxw = xp[i]*wp[i]*np.sqrt(1-xz[j]**2)*wz[j]/(xp[i]*reAi[i]**2+reBi[i]**2)
                pqz = np.sqrt(p2*xp[i])*xz[j]
                k2 = p2 +xp[i]-2*pqz
                G_k2 = g(k2)
                #integration
                fA += dxw*reAi[i]*G_k2*(pqz+2*(p2-pqz)*(pqz-xp[i])/k2)
                fB += dxw*reBi[i]*G_k2

        fA = 4*fA/(p2*3*8*np.pi**3)
        fB = 4*fB/(8*np.pi**3)
        Ax = 1 + fA - fArn
        Bx = m0 + fB - fBrn
        
        return Ax,Bx
    
    #iteration part
    for iter in range(max_iter):
        fArn = 1.0
        fBrn = m0
        error = 0

        # calculate fA, fB at renormalization point \xi= 361GeV^2
        fArn, fBrn = intreAB(361,fArn,fBrn)

        #solve A,B at every point
        for i in range(Np):
            reA[i],reB[i] = intreAB(xp[i],fArn,fBrn)
        
        error = np.sum(np.abs(reB - reBi))
        reAi = reA.copy()
        reBi = reB.copy()
        
        print(f"Iteration {iter+1}, error = {error:.6e}")
        if error < eps:
            z2 = 1 - fArn
            z4 = 1 - fBrn/m0
            print(f"Converged after {iter+1} iterations")
            print(f"z2 = {z2:.6e}, z4 = {z4:.6e}, fArn = {fArn:.6e}, fBrn = {fBrn:.6e}")
            break
        
    return z2,z4,reA,reB,xp


if __name__ == "__main__":

    p2_min,p2_max = 1e-4,1e4
    max_iter = 1000
    eps = 1e-3

    z2,z4,reA,reB,xp = solver(Np=100,Nz=25,xmin=p2_min,xmax=p2_max,max_iter=max_iter,eps=eps)

    file = "./abdat.npz"
    M = reB / reA
    np.savez(file=file,A=reA,B=reB,M=M,p2=xp,z2=z2,z4=z4)

