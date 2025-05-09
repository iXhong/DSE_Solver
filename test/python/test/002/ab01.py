import numpy as np
from scipy.special import roots_legendre


#define parameter
D = (0.8)**3/0.5
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


def solver(Nz, Np, xmin, xmax, max_iter, eps):
    """
    Iteratively solve A(p²), B(p²) with renormalization at mu² = 1.0 GeV²
    """
    A = np.ones(Np)
    B = np.full(Np, 0.3)

    xz, wz, xp, wp = gausslegendreGrid(xmin, xmax, Nz, Np)

    # 设置重整化点 mu^2
    idx_mu = np.argmin(np.abs(xp - xi))  # 离 mu² 最近的格点
    for iter in range(max_iter):
        A_old = np.copy(A)
        B_old = np.copy(B)
        A_new = np.zeros(Np)
        B_new = np.zeros(Np)

        for i in range(Np):
            IA, IB = 0.0, 0.0
            for j in range(Np):
                for k in range(Nz):
                    pqz = np.sqrt(xp[i] * xp[j]) * xz[k]
                    k2 = xp[i] + xp[j] - 2 * pqz
                    G_k2 = g(k2)
                    denom = xp[j]*A_old[j]**2 + B_old[j]**2
                    dxw = xp[j] * wp[j] * np.sqrt(1 - xz[k]**2) * wz[k] / denom
                    IA += dxw * A_old[j] * G_k2 * (pqz + 2*(xp[i]-pqz)*(pqz - xp[j])) / k2
                    IB += dxw * B_old[j] * G_k2

            fA = IA / (6 * np.pi**3 * xp[i])
            fB = IB / (2 * np.pi**3)

            A_new[i] = fA
            B_new[i] = fB

        # 计算 mu² 点上的 loopA, loopB 用于重整化
        loopA_mu, loopB_mu = 0.0, 0.0
        for j in range(Np):
            for k in range(Nz):
                pqz = np.sqrt(xp[idx_mu]*xp[j]) * xz[k]
                k2 = xp[idx_mu] + xp[j] - 2 * pqz
                G_k2 = g(k2)
                denom = xp[j]*A_old[j]**2 + B_old[j]**2
                dxw = xp[j] * wp[j] * np.sqrt(1 - xz[k]**2) * wz[k] / denom
                loopA_mu += dxw * A_old[j] * G_k2 * (pqz + 2*(xp[idx_mu]-pqz)*(pqz - xp[j])) / k2
                loopB_mu += dxw * B_old[j] * G_k2

        loopA_mu /= (6 * np.pi**3 * xp[idx_mu])
        loopB_mu /= (2 * np.pi**3)

        Z2 = 1.0 - loopA_mu
        Z4 = (B_old[idx_mu] - loopB_mu) / m0

        # 更新 A, B
        A = Z2 + A_new
        B = Z4 * m0 + B_new

        error = np.sum(np.abs(B - B_old))
        print(f"Iteration {iter+1}, error = {error:.6e}")
        if error < eps:
            print(f"Converged after {iter+1} iterations")
            print(f"Z2 = {Z2:.6e}, Z4 = {Z4:.6e}")
            break

    return Z2, Z4, A, B, xp

def solver1(Nz, Np, xmin, xmax, max_iter, eps):
    """
    iteratively solve A, B with renormalization at mu^2 = 19 GeV^2
    """
    A = np.ones(Np)
    B = np.full(Np, 0.3)

    xz, wz, xp, wp = gausslegendreGrid(xmin, xmax, Nz, Np)

    # Find the index of renormalization point nearest to mu^2 = 19
    mu2 = 361
    mu_idx = np.argmin(np.abs(xp - mu2))

    for iter in range(max_iter):
        A_old = np.copy(A)
        B_old = np.copy(B)

        fA_mu = 0
        fB_mu = 0
        fA_all = np.zeros(Np)
        fB_all = np.zeros(Np)

        for i in range(Np):
            fA, fB = 0, 0
            for j in range(Np):
                for k in range(Nz):
                    pqz = np.sqrt(xp[i] * xp[j]) * xz[k]
                    k2 = xp[i] + xp[j] - 2 * pqz
                    G_k2 = g(k2)
                    denom = xp[j] * A_old[j] ** 2 + B_old[j] ** 2
                    dxw = xp[j] * wp[j] * np.sqrt(1 - xz[k] ** 2) * wz[k] / denom
                    fA += dxw * A_old[j] * G_k2 * (pqz + 2 * (xp[i] - pqz) * (pqz - xp[j])) / k2
                    fB += dxw * B_old[j] * G_k2
            fA /= 6 * np.pi ** 3 * xp[i]
            fB /= 2 * np.pi ** 3
            fA_all[i] = fA
            fB_all[i] = fB

        # compute Z2 and Z4 using renormalization conditions at mu^2
        Z2 = 1 / (1 + fA_all[mu_idx])
        Z4 = m0 / (m0 + fB_all[mu_idx])

        # update A, B
        A = Z2 * (1 + fA_all)
        B = Z4 * (m0 + fB_all)

        error = np.sum(np.abs(B - B_old))
        print(f"Iteration {iter+1}, error = {error:.6e}")
        if error < eps:
            print(f"Converged after {iter+1} iterations")
            print(f"Z2 = {Z2:.6e}, Z4 = {Z4:.6e}")
            print(f"A(mu^2) = {A[mu_idx]:.6f}, B(mu^2) = {B[mu_idx]:.6f}")
            break

    return Z2, Z4, A, B, xp



if __name__ == "__main__":

    p2_min,p2_max = 1e-4,1e4
    max_iter = 1000
    eps = 1e-4

    z2,z4,A,B,xp = solver1(Np=100,Nz=25,xmin=p2_min,xmax=p2_max,max_iter=max_iter,eps=eps)

    file = "./abdat.npz"
    M = B / A
    np.savez(file=file,A=A,B=B,M=M,p2=xp)









