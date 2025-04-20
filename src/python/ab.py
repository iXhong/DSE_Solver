import numpy as np
from scipy.special import roots_legendre


#define parameter
D = 1
Nf = 4
m0 = 0.005
xi = 361 #GeV
gamma_m = 12/25 #12/(33-2*Nf)
tau = np.e**2 - 1
lambda_qcd = 0.234 #GeV
mt = 0.5 #GeV
omega = 0.4 #GeV


def g(k2):
    """
    interaction model,use Qin-chang model
    """
    k2 = max(k2,1e-10)
    part1 = 4*np.pi**2*D/omega**4 * np.exp(-k2/omega**2)
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
    A = np.ones(Np)
    B = np.full(Np,0.3)

    xz, wz, xp, wp = gausslegendreGrid(xmin,xmax,Nz,Np)

    IArn, IBrn = 1,m0
    # for j in range(Np):
    #     IA, IB = 0, 0    
    #     for k in range(Nz):
    #         dxw = xp[j]*wp[j]*np.sqrt(1-xz[k]**2)*wz[k]/(xp[j]*A[j]**2+B[j]**2)
    #         pqz = np.sqrt(xi*xp[j])*xz[k]
    #         k2 = xi +xp[j]-2*pqz
    #         G_k2 = g(k2)
    #         IArn += dxw*A[j]*G_k2*(pqz+2*(xi-pqz)*(pqz-xp[j]))/xp[j]
    #         IBrn += dxw*B[j]*G_k2

    # IArn /= 6*np.pi**3*xi
    # IBrn /= 2*np.pi**3

    for iter in range(max_iter):
        A_old = np.copy(A)
        B_old = np.copy(B)
        for i in range(Np):
            IA,IB = 0,0
            for j in range(Np):
                for k in range(Nz):
                    dxw = xp[j]*wp[j]*np.sqrt(1-xz[k]**2)*wz[k]/(xp[j]*A_old[j]**2+B_old[j]**2)
                    pqz = np.sqrt(xp[i]*xp[j])*xz[k]
                    k2 = xp[i]+xp[j]-2*pqz
                    G_k2 = g(k2)
                    IA += dxw*A[j]*G_k2*(pqz+2*(xp[i]-pqz)*(pqz-xp[j]))/xp[j]
                    IB += dxw*B[j]*G_k2
            A[i] = 1 + IA / (6*np.pi**3 * xp[i]) - IArn
            B[i] = m0 + IB / (2*np.pi**3) -IBrn

        error = np.sum(np.abs(B - B_old))
        print(f"Iteration {iter+1}, error = {error:.6e}")
        if error < eps:
            print(f"Converged after {iter+1} iterations")
            z2 = 1 - IArn
            z4 = 1 - IBrn / m0
            print(f"z2 = {z2:.6e}, z4 = {z4:.6e}, IArn = {IArn:.6e}, IBrn = {IBrn:.6e}")
            break


    file = "../../test/abdat.npz"
    M = B / A
    np.savez(file=file,A=A,B=B,M=M)


if __name__ == "__main__":

    p2_min,p2_max = 1e-4,1e4
    max_iter = 1000
    eps = 1e-4

    solver(Np=100,Nz=25,xmin=p2_min,xmax=p2_max,max_iter=max_iter,eps=eps)











