from math import *
import numpy as np
from scipy.linalg import expm
from qutip import *
# complex number
j = (-1)**0.5
 
# pauli matrix
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -j], [j, 0]])
sz = np.array([[1, 0], [0, -1]])

# define step size
dt = 1 # 고정
N = 400 # 다양한 값들 넣어보고 가장 빠른 N을 찾아야함
T = np.linspace(0,N,N+1)


# Parameters
d0 = 1.5 #MHz, detuning factor, 나중에는 논문에 나온 식으로 두고 time-dependent하게 풀어야할 수도 있음. 지금은 time-independent
v1 = 5 #Control field의 amplitude
phi_t = pi*(np.sin(2*np.pi*T/N))**2 # phi_t에 대한 특정 조건이 없으므로, 그냥 임의로 잡았음


# Define Rx and Rz gate
def Rx(t):
    w = 2*pi*20 #gate도 결국 pulse의 시간에 의해서 결정 20은 임의의 설정값
    theta = w*t
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])

def Rz(t):
    w = 2*pi*20
    phi = w*t
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])

# Hamiltonian in  the rotating frame
#H0 = 2*np.pi*d0*sz #Time-independent
#Hc_T= np.array(2*np.pi*v1*(np.cos(phi)*sx+np.sin(phi)*sy)) #Time-dependent
#H = H0+Hc_T # Total hamiltonian

# Define hamiltonian
def dH(t):
    H0 = 2*np.pi*d0*sz*0.00001
    Hc_t = 2*np.pi*v1*(np.cos(phi_t[t])*sx+np.sin(phi_t[t])*sy)*0.00001
    return H0+Hc_t






# # Solve hamiltonian
# # 이 파트가 필요한 지 확신이 안섬.
# E = np.zeros(401)

# for i in range(H):
#     eigvals = np.linalg.eigh(i)[0]            # diagonalizing the Hamiltonian 여기서부터 문제 
#     eigvecs = -1*np.linalg.eigh(i)[1]         # eigenvectors
#     E = np.diag(eigvals)                        # exponent of eigenvalues
#     U_H= eigvecs.conj().T                       # unitary matrix formed by eigenvectors

# Define initial wave function
init_wave = np.array([[1],[0]])
# Define initail density matrix
irho_init = np.kron(init_wave,init_wave.conj().T)

# Define target wave fuction
irho_target = rand_dm_ginibre(2, rank=1) #random한 target density matrix
xyz_target = [np.trace(irho_target*sx),np.trace(irho_target*sy),np.trace(irho_target*sz)] #targer density matirix의 bloch sphere 상의 좌표





# #####Hamiltonian에 따른 스핀 축 변화 추이 확인을 위한 파트(실제 코드에는 안들어감)
# def dU(dH) :
#     return expm(-j*dH)
# irho_mid = np.zeros((3,400))
# irho_mid[0][0] = 0
# irho_mid[1][0] = 0
# irho_mid[2][0] = 1
# irho_init = Rx(0.01)@irho_init@Rx(0.01).conj().T
# for k in range(1,5):
#     dHt=dH(k)
#     #print(dHt)
#     dUt=dU(dHt)
#     irho_init = dUt @ irho_init @ dUt.conj().T
#     #print((dUt@dUt.conj().T)) #생성된 dU가 Unitary가 맞는지 확인용
#     irho_mid[0][k] = np.trace(irho_init*sx)
#     irho_mid[1][k] = np.trace(irho_init*sy)
#     irho_mid[2][k] = np.trace(irho_init*sz)
# print(irho_mid[2])






