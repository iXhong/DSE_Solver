import numpy as np
import matplotlib.pyplot as plt

data = np.load("abc_test_2.npz")

B = np.abs(data['B'][:,0])
A = np.abs(data['A'][:,0])
C = np.abs(data['C'][:,0])
p2 = data["p2"]

# M = B/A

# fig,ax = plt.subplots()
# ax.set_xscale('log')
# ax.plot(p2,C,'r',label="C")
# ax.plot(p2,A,'b',label="A")
# ax.legend()
# plt.show()
# # print(data["B"][0,10])


# axs[0].plot(p2,C,label="C(p^2,omega_0)")
# axs[0].plot(p2,A,label="A(p^2,omega_0)")
# axs[1].plot(p2,B,label="B(p^2,omega_0)")



def scatterplt():
    fig = plt.figure(layout="constrained")
    axs = fig.subplots(2,1)
    axs[0].scatter(p2,C,label="C(p^2,omega_0)")
    axs[0].scatter(p2,A,label="A(p^2,omega_0)")
    axs[1].scatter(p2,B,label="B(p^2,omega_0)")
    axs[0].set_xscale("log")
    axs[1].set_xscale("log")
    axs[0].set_title("A,C")
    axs[1].set_title("B")
    axs[0].grid()
    axs[1].grid()
    # plt.savefig("ABC_scatter_002.png",dpi=300)
    plt.show()


def plotofB():
    Nf = 20
    fig = plt.figure()
    axs = fig.subplots()
    for i,omega in enumerate(np.arange(-Nf//2,Nf//2)):
        if i%5 == 0:
            axs.plot(p2,data["B"][:,i],label=f"omega={2*omega+1}")
    axs.set_xscale('log')
    axs.set_xlabel("p2(GeV)")
    axs.set_ylabel("B(GeV)")
    axs.set_title("B of p2 under omegas")
    axs.hlines(0.5,0,1e3,linestyles="dotted",label="0.5GeV")
    axs.grid("minor")
    axs.legend()
    plt.savefig("BofP2.png")
    plt.show()


def plotofA():
    Nf = 20
    fig = plt.figure()
    axs = fig.subplots()
    for i,omega in enumerate(np.arange(-Nf//2,Nf//2)):
        if i%5 == 0:
            axs.plot(p2,data["A"][:,i],label=f"omega={2*omega+1}")
    axs.set_xscale('log')
    axs.set_xlabel("p2(GeV)")
    axs.set_ylabel("A(GeV)")
    axs.set_title("A of p2 under omegas")
    axs.grid("minor")
    axs.legend()
    plt.savefig("A_P2.png")
    plt.show()


def plotofC():
    Nf = 20
    fig = plt.figure()
    axs = fig.subplots()
    for i,omega in enumerate(np.arange(-Nf//2,Nf//2)):
        if i%5 == 0:
            axs.plot(p2,data["C"][:,i],label=f"omega={2*omega+1}")
    axs.set_xscale('log')
    axs.set_xlabel("p2(GeV)")
    axs.set_ylabel("C(GeV)")
    axs.set_title("C of p2 under omegas")
    axs.grid("minor")
    axs.legend()
    plt.savefig("C_P2.png")
    plt.show()


def plotofABC():
    fig = plt.figure()
    axs = fig.subplots()
    B = np.abs(data["B"][:,10])
    A = np.abs(data["A"][:,10])
    C = np.abs(data["C"][:,10])
    axs.plot(p2,A,label="A")
    axs.plot(p2,B,label="B")
    axs.plot(p2,C,label="C")
    axs.set_xscale("log")
    axs.set_ylabel("A/B/C(GeV)")
    axs.set_xlabel("p2(GeV)")
    axs.set_title("A/B/C_p2")
    axs.hlines(0.005,0,1e3,linestyles="dotted",label="0.005GeV")
    axs.grid()
    axs.legend()
    # plt.savefig("ABC_p2.png",dpi=200)
    plt.show()


# plotofABC()
# plotofB()
# scatterplt()
print(data["B"][0,10])
print(data["A"][0,10])