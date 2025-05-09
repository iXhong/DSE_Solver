import numpy as np
import matplotlib.pyplot as plt

data = np.load("abdat.npz")
p2 = data["p2"]
A = data["A"]
B = data["B"]
M = data["M"]


# plt.plot(p2, A, label='A(p^2)')
# plt.plot(p2, B, label='B(p^2)')
# plt.plot(p2,M,label='M(p^2)')
# plt.xscale('log')
# plt.legend()
# plt.savefig("result.png")
# plt.show()
fig = plt.figure(layout="constrained")
axs = fig.subplots(2,1)
axs[0].plot(p2,M,label="M(p^2)")
axs[1].plot(p2,A,label="A(p^2)")
axs[1].plot(p2,B,label="B(p^2)")
axs[0].set_xscale("log")
axs[0].set_xlabel("p2 Gev^2")
axs[0].set_ylabel("M GeV")
axs[0].set_title("Mass")
axs[1].set_xscale("log")
axs[1].set_xlabel("p2 GeV^2")
axs[1].set_ylabel("A/B")
axs[1].set_title("A & B")
axs[0].grid()
axs[1].grid()
axs[0].legend()
axs[1].legend()
plt.savefig("./fig.png")
plt.show()

