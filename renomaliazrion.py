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

    part1 = 4*np.pi**2 / omega**6 * D * k2/omega**2
    part2 = 8*np.pi**2*gamma_m / (np.log(tau + (1+k2/lambda_qcd**2)**2))
    F = (1-np.exp(-k2/(4*mt**2)))/k2

    return part1*part2*F


#Integration of angular part of A
def I_A(p,q):
    def integrand(x):
        k2 = p**2 + q**2- 2*p*q*x
        part1 = 2*(p**2-p*q*x)*(p*q*x-q**2)
        part2 = np.sqrt(1-x**2)

        return part2*G(k2)/k2*(p*q*x + part1/k2)

    integral,_ = quad(integrand,-1,1)
    
    return integral 


#Integration of angular of B
def I_B(p,q):
    def integrand(x):
        k2 = p**2 + q**2- 2*p*q*x
        part2 = np.sqrt(1-x**2)

        return part2*G(k2)/k2
    
    integral,_ = quad(integrand,-1,1)

    return integral


#momentum grid
N = 100
q_max = Lambda
q_grid = np.logspace(-4,np.log10(q_max),N)
dq = np.diff(q_grid)

A = np.ones(N)
B = np.zeros(N)