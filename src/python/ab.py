import numpy as np
from scipy.special import roots_legendre

#define parameter
D = (0.8**3)/0.5
Nf = 4
m0 = 0.0034
xi = 361 #GeV
gamma_m = 12/25 #12/(33-2*Nf)
tau = np.e**2 - 1
lambda_qcd = 0.234 #GeV
mt = 0.5 #GeV
omega = 0.5 #GeV


def g(k2):
    """
    interaction model,use Qin-chang model
    """
    k2 = max(k2,1e-10)
    part1 = 8*np.pi**2*D/omega**4 * np.exp(-k2/omega**2)
    part2 = 8*np.pi**2*gamma_m/np.log(tau+(1+k2/lambda_qcd**2)**2)
    part3 = (1-np.exp(-k2/(4*mt**2)))/k2

    return part1 + part2 * part3


def gausslegendreGrid(xp1,xp2,Nz,Np):
    """
    x, w in Gauss-Legendre points
    Args:
        xp1: left endpoint of p^2
        xp2: right endpoint of p^2

    Return:
        xz,wz: Guass-legendre point in [-1,1]
        xp,wp: Guass-legendre point combined with log transform in [xp1,xp2]

    Note:
        x for grid points array; w for weight array
    """
    #gausslegendre points for angular integration
    # xz,wz = np.polynomial.legendre.leggauss(25)    
    xz, wz = roots_legendre(Nz)
    #gausslegendre points for momentum integration
    # xp,wp = np.polynomial.legendre.leggauss(300)
    xp, wp = roots_legendre(Np)
    # transform to [-4,4]
    xp1_log, xp2_log = np.log10(xp1), np.log10(xp2)
    xp = (xp2_log-xp1_log)*xp/2 + (xp1_log+xp2_log)/2
    wp = (xp2_log-xp1_log)/2 * wp
    #log transform to [1e-4,1e4]
    xp = 10**xp
    wp = wp*xp*np.log(10)

    return xz,wz,xp,wp


def solver(Nz,Np,xmin,xmax,max_iter,eps):
    """
    iteratively  solve A,B
    """
    reAi = np.ones(Np)
    reBi = np.full(Np,0.3)
    # reBi = np.zeros(Np)
    reA = np.zeros(Np)
    reB = np.zeros(Np)

    xz, wz, xp, wp = gausslegendreGrid(xmin,xmax,Nz,Np)

    def iterAB(p2,fArn,fBrn):
        fA, fB = 0, 0
        for i in range(Np):
            for j in range(Nz):
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
    
    for iter in range(max_iter):
        fArn = 1.0
        fBrn = m0
        error = 0

        # calculate fA, fB at renormalization point \xi= 361GeV^2
        fArn, fBrn = iterAB(361,fArn,fBrn)

        for i in range(Np):
            reA[i],reB[i] = iterAB(xp[i],fArn,fBrn)
        

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

    z2,z4,reA,reB,xp = solver(Np=200,Nz=25,xmin=p2_min,xmax=p2_max,max_iter=max_iter,eps=eps)

    file = "./abdat.npz"
    M = reB / reA
    np.savez(file=file,A=reA,B=reB,M=M,p2=xp)

