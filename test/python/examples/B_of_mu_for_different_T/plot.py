import matplotlib.pyplot as plt
import numpy as np

def plot(folder):
    B_list = []
    mus = np.linspace(0.0,0.2,20)

    for i in range(20):
        data = np.load(f"./{folder}/abc_{i}.npz")
        B = np.abs(data["B"][10,10])
        B_list.append(B)

    plt.scatter(mus,B_list,marker='*',label="T=139MeV")
    plt.xlabel(r"$\mu$ (MeV)")
    plt.ylabel(r"B(0,$\widetilde{\omega}_0$)(MeV)")
    plt.xlim(0.0,0.2)
    plt.ylim(0.0,0.5)
    plt.grid()
    plt.minorticks_on()
    plt.legend()
    plt.show()


def B_list(folder):
    B_list = []

    for i in range(20):
        data = np.load(f"./{folder}/abc_{i}.npz")
        B = np.abs(data["B"][10,10])
        B_list.append(B)

    return B_list


def plotall():
    data = {}
    mus = np.linspace(0.0,0.2,20)
    for folder in ['120','129','139']:
        key = f"{folder}"
        data[key] = B_list(folder)
    
    plt.scatter(mus,data['139'],marker='*',label="T=139MeV")
    plt.scatter(mus,data['129'],marker='.',label="T=129MeV")
    plt.scatter(mus,data['120'],marker='o',label="T=120MeV")
    plt.xlabel(r"$\mu$ (MeV)")
    plt.ylabel(r"B(0,$\widetilde{\omega}_0$)(MeV)")
    plt.xlim(0.0,0.2)
    plt.ylim(0.0,0.5)
    plt.grid()
    plt.minorticks_on()
    plt.legend()
    plt.savefig("B_of_mu_for_T.png")
    plt.show()





def check(i,folder):
    data = np.load(f"./{folder}/abc_{i}.npz")
    B = np.real(data["B"][10,10])
    print(B)

# check(0,120)
# plot(120)
plotall()