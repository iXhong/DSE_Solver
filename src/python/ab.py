import numpy as np
from scipy.integrate import quad


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
    part1 = 4*np.pi**2*D/omega**4 * np.exp(-k2/omega**2)
    part2 = 8*np.pi**2*gamma_m/np.log(tau+(1+k2/lambda_qcd**2)**2)
    part3 = (1-np.exp(-k2/(4*mt**2)))/k2

    return part1 + part2 * part3


def integrand(x,p2,q2,A_q2,B_q2):
    """
    Integrand part of A(p2),B(p2)
    """
    k2 = p2 + q2 - 2*np.sqrt(p2*q2)*x
    denom = q2*A_q2**2 + B_q2**2
    termA = np.sqrt(p2*q2)*x + 2*(p2-np.sqrt(p2*q2)*x)*(np.sqrt(p2*q2)*x-q2)
    G_k2 = g(k2)
    
    integrandA = np.sqrt(1-x**2)*A_q2*G_k2 / denom * termA
    integrandB = np.sqrt(1-x**2)*B_q2*G_k2 / denom

    return integrandA, integrandB


def angularInt(p2,q2,A_q2,B_q2):
    """
    Integration of angular part of A,B
    """
    fA = lambda x :integrand(x,p2,q2,A_q2,B_q2)[0]
    fB = lambda x :integrand(x,p2,q2,A_q2,B_q2)[1]
    result_A,_ = quad(fA,-1,1,epsrel=1e-4)
    result_B,_ = quad(fB,-1,1,epsrel=1e-4)

    return result_A, result_B


def logGrid(min,max,N):
    """
    Generate grid in log scale
    """
    lmin, lmax = np.log10(min),np.log10(max)
    x = np.logspace(lmin,lmax,N)
    q2_grid = 10**x
    dx = (lmax-lmin)/(N-1)
    weights = dx*q2_grid*np.log(10)

    return q2_grid, weights


def solver():
    """
    iteratively  solve A,B
    """
    N = 100
    min,max = 1e-4, 1e4
    max_iter = 1000
    eps = 1.0

    #initialization
    A = np.ones(N)
    B = np.full(N,0.3)


    IArn, IBrn = 1,m0
    q2_grid, weights = logGrid(min,max,N)
    p2_grid = np.copy(q2_grid)
    
    for iter in range(max_iter):
        A_old = np.copy(A)
        B_old = np.copy(B)

        for i in range(N):
            IA, IB = 0,0
            for j in range(N):
                q2 = q2_grid[j]
                w = weights[j]
                IA_j, IB_j = angularInt(p2=p2_grid[i],q2=q2,A_q2=A_old[j],B_q2=B_old[j])
                IA += w * q2 * IA_j
                IB += w * q2 * IB_j

        A[i] = 1 + IA / (6*np.pi**3 * p2_grid[i]) - IArn
        B[i] = m0 + IB / (2*np.pi**3) -IBrn

        error = np.sum(np.abs(B - B_old))
        print(f"Iteration {iter+1}, error = {error:.6e}")
        if error < eps:
            print(f"Converged after {iter+1} iterations")
            break
    
    file = "../../test/abdat.npz"
    np.save(A=A,B=B)


solver()










