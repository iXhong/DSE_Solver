import numpy as np
from scipy.integrate import quad


#define parameter
D = 1
Nf = 4
gamma_m = 12/(33-2*Nf)
tau = np.e**2 - 1
lambda_qcd = 0.234 #GeV
mt = 0.5 #GeV
omega = 0.4 #GeV
Lambda = 100 #GeV


#define gluon propagator G(k^2) with Maris-Tandy model
def G(k2):
    k2 = max(k2,1e-10)
    part1 = 4*np.pi**2 / omega**6 * D * k2/omega**2
    part2 = 8*np.pi**2*gamma_m / (np.log(tau + (1+k2/lambda_qcd**2)**2))
    F = (1-np.exp(-k2/(4*mt**2)))/k2

    return part1*part2*F


#Integration of angular part of A
def I_A(p2,q2):
    def integrand(x):
        k2 = max(k2,1e-10)
        k2 = p2 + q2- 2*np.sqrt(p2*q2)*x
        part1 = 2*(p2*q2+x*(p2+q2)*np.sqrt(p2*q2)-p2*q2*x)
        part2 = np.sqrt(1-x**2)

        return part2*G(k2)/k2 * (np.sqrt(p2*q2)*x + part1/k2)

    integral,_ = quad(integrand,-1,1)
    
    return integral 


#Integration of angular of B
def I_B(p2,q2):
    def integrand(x):
        k2 = p2 + q2- 2*np.sqrt(p2*q2)*x
        part2 = np.sqrt(1-x**2)

        return part2*G(k2)/k2
    
    integral,_ = quad(integrand,-1,1)

    return integral


#momentum grid
N = 100
q_max = Lambda
q_grid = np.logspace(-4,np.log10(q_max),N)
dq = np.diff(q_grid)

#initial value
A = np.ones(N)
B = np.zeros(N)


#iteratively solve A,B //the hardest part
def iterate_AB(Z2,Z4,tol=1e-6,max_iter=100):

    A_new = np.zeros(N)
    B_new = np.zeros(N)

    #solve A
    for i, p in enumerate(q_grid):
        #A'
        integrand_A = 0
        for j,q in enumerate(q_grid[:-1]):
            integrand_A = q**3 * A


    return 