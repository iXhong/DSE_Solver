import numpy as np
from scipy.integrate import quad


#define parameter
D = (0.74)**2
Nf = 4
m = 0.005
xi = 19 #GeV
gamma_m = 12/25 #12/(33-2*Nf)
tau = np.e**2 - 1
lambda_qcd = 0.234 #GeV
mt = 0.5 #GeV
omega = 0.4 #GeV
# Lambda = 100 #GeV


#define gluon propagator G(k^2) with Maris-Tandy model
def G(k2): 
    k2 = max(k2,1e-10)
    part1 = 4*np.pi**2 / omega**6 * D *k2 * np.exp(k2/omega**2)
    part2 = 8*np.pi**2*gamma_m / (np.log(tau + (1+k2/lambda_qcd**2)**2))
    F = (1-np.exp(-k2/(4*mt**2)))/k2

    return part1*part2*F


#Integration of angular part of A
def IntegralOfAx(p2,q2):
    def integrand(x):
        k2 = p2 + q2- 2*np.sqrt(p2*q2)*x
        k2 = max(k2,1e-10) 
        part1 = 2*(p2*q2+x*(p2+q2)*np.sqrt(p2*q2)-p2*q2*x)
        part2 = np.sqrt(1-x**2)

        return part2*G(k2)/k2 * (np.sqrt(p2*q2)*x + part1/k2)

    integral,_ = quad(integrand,-1,1)
    
    return integral 


#Integration of angular part of B
def IntegralOfBx(p2,q2):
    def integrand(x):
        k2 = p2 + q2- 2*np.sqrt(p2*q2)*x
        k2 = max(k2,1e-10)
        part2 = np.sqrt(1-x**2)

        return part2*G(k2)/k2
    
    integral,_ = quad(integrand,-1,1)

    return integral


#momentum grid in logspace
N = 100
p2_min, p2_max = 1e-4,1e3 #range of p^2
p2_grid = np.logspace(np.log10(p2_min),np.log10(p2_max),N)
q2_grid = np.logspace(np.log10(p2_min),np.log10(p2_max),N)
dq2 = np.diff(q2_grid)  #step length


def IntegrandOfA(p2,q2,A_q2,B_q2,IntegralOfAx):
    part1 = q2*A_q2/(q2*A_q2**2 + B_q2**2)
    part2 = IntegralOfAx(p2,q2)

    return part1*part2


def IntegrandOfB(p2,q2,A_q2,B_q2,IntegralOfBx):
    part1 = q2*B_q2/(q2*A_q2**2+B_q2**2)
    part2 = IntegralOfBx(p2,q2)

    return part1*part2


#iteratively solve A,B //the hardest part
def iterate_AB(Z2,Z4,A,B,tol=1e-6,max_iter=10):

    A_new = np.zeros(N)
    B_new = np.zeros(N)
    A_prime = np.zeros(N)
    B_prime = np.zeros(N)

    for iteration in range(max_iter):

        IA = 0
        IB = 0

        for i,p2 in enumerate(p2_grid):

            for j, q2 in enumerate(q2_grid[:-1]):

                IA += IntegrandOfA(p2,q2,A[j],B[j],IntegralOfAx)*dq2[j]
                
            A_prime[i] = 1/(6*p2*np.pi**3)*IA
            A_new[i] = Z2 + A_prime[i]

            for j,q2 in enumerate(q2_grid[:-1]):

                IB += IntegrandOfB(p2,q2,A[j],B[j],IntegralOfBx)*dq2[j]

            B_prime[i] = 1/(2*np.pi**2)*IB
            B_new[i] = Z4*m + B_prime[i]

        delta_A = np.max(np.abs(A_new - A))
        delta_B = np.max(np.abs(B_new - B))
        print(f"Iteration {iteration},max|A_new - A|={delta_A},max|B_new - B|={delta_B}")

        if delta_A < tol and delta_B < tol :
            print(f"converge in {iteration} times")
            break

        A = A_new.copy()
        B = B_new.copy()

    return A,B,A_prime,B_prime


def findZ2Z4(A,B):
    Z2_init = 1
    Z4_init = 1
    xi_index = np.argmin(np.abs(p2_grid - xi))
    
    A,B,A_prime,B_prime = iterate_AB(Z2_init,Z4_init,A,B)

    Z2 = 1-A_prime[xi_index]
    Z4 = 1-B_prime[xi_index]/m

    return Z2,Z4,A,B


def main():
    #initial value
    A = np.ones(N)
    B = np.zeros(N)

    Z2,Z4,A,B = findZ2Z4(A,B)
    
    print(f"Z2={Z2},Z4={Z4}")

    file = "../../results/python/results.npz"

    np.savez(file,A=A,B=B,Z2=Z2,Z4=Z4)

main()