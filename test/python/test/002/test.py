import numpy as np

# xp,wp,xz,wz = ([],[],[],[])

# def iterAB(p2,Ax,Bx):

#     Np = 100
#     Nz = 25
#     fA, fB = 0, 0

#     for i in range(Np):
#         for j in range(Nz):
#             dxw = xp[i]*wp[i]*np.sqrt(1-xz[j]**2)*wz[j]/(xp[i]*reAi[i]**2+reBi[i]**2)
#             pqz = np.sqrt(p2*xp[i])*xz[j]
#             k2 = p2 +xp[i]-2*pqz
#             G_k2 = g(k2)
#             #integration
#             fA += fA + dxw*reAi[i]*G_k2*(pqz+2*(p2-pqz)*(pqz-xp[i])/k2)
#             fB += fB + dxw*reBi[i]*G_k2

#     fA = 4*fA/(p2*3*8*np.pi**3)
#     fB = 4*fB/(8*np.pi**3)
#     Ax = 1 + fA - fArn 
#     Bx = m0 + fB - fBrn

#     return fA,fB,Ax,Bx

data = np.load("./abdat.npz")
# print(data["z2"])
# print(data["z4"])
print(data["A"][0])
print(data["B"][0])