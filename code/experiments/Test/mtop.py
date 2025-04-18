#%%
# Find the resornance tau
# updated in 07/12/2022 by Paul Junghyun Lee
# 본 코드는 이정현박사님의 매트랩코드를 파이썬으로 옮긴 코드입니다.
from toqito.channels import partial_trace
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import linalg
from math import *
import matplotlib.pyplot as plt

#Generating gate function
def UO(B1,B2,a,D1,D2):
    i   = 1j
    gamma = 2*pi*2.8
    D     = 2870
    UA = [[(B2**2+B1**2*cos(a))/(B1**2+B2**2), -i*B1*(e**(-i*D1))*sin(a)/sqrt(B1**2+B2**2), ((-1+cos(a))*B1*B2*(e**(-i*(D1-D2))))/(B1**2+B2**2)],
            [-i*B1*(e**(i*D1))*sin(a)/sqrt(B1**2+B2**2), cos(a), -i*B2*(e**(i*D2))*sin(a)/sqrt(B1**2+B2**2)],
            [((-1+cos(a))*B1*B2*e**(i*(D1-D2)))/(B1**2+B2**2), -i*B2*(e**(-i*D2))*sin(a)/sqrt(B1**2+B2**2), (B1**2+B2**2*cos(a))/(B1**2+B2**2)]]
    return UA

## Define dimension, pauli matrices
i   = 1j #1j
sx  = 1/sqrt(2)*np.array([[0, 1, 0],[1, 0, 1], [0, 1, 0]])
sy  = 1/sqrt(2)/i*np.array([[0, 1, 0], [-1, 0, 1],[0, -1, 0]])
sz  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
#sz  = [1, 0, 0; 0, -1, 0; 0, 0, 0]
I   = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Rotation matrix projected into 2 level system
Sxp  = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
Sxm  = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
Syp  = 1/i*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
Sym  = 1/i*np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
Szp  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

#Gellman matrix
Sx  = np.array([[0, 0, 1],[0, 0, 0], [1, 0, 0]])
Sy  = np.array([[0, 0, -i],[0, 0, 0], [i, 0, 0]])
Sz  = np.array([[1, 0, 0],[0, 0, 0], [0, 0, -1]])

# Pauli basis for 13C nuclear spin
Ix  = 1/2*np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])   
Iy  = 1/2/i*np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
Iz  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
 

## Define sweep parameters
Sweep = 1001
N = Sweep
B = 403 #[G] magnetic field

# 13C nuclear spin parameters
gammaN = 2*pi*1.071e-3 #[MHz/G]
Al    = 2*pi*0.1 #[MHz] # A_|| hyperfine term
Ap = 2*pi*0.1 #[MHz] # A_per hyperfine term

T = 5; # sweep tau [us]
t = np.linspace(0,T,N)
n = 32; # number of pi pulses

## Define gate operations
# Single Q ms=+1
U090xp = UO(1,0,pi/4,0,0)
U090xmp = UO(1,0,-pi/4,0,0)
U090yp = UO(1,0,pi/4,pi/2,0)
U090ymp = UO(1,0,-pi/4,pi/2,0)
U180xp = UO(1,0,pi/2,0,0)
U180xmp = UO(1,0,-pi/2,0,0)

#Single Q ms=-1
U090xm = UO(0,1,pi/4,0,0)
U090xmm = UO(0,1,-pi/4,0,0)
U180xm = UO(0,1,pi/2,0,0)
U180xmm = UO(0,1,pi/2,0,0)

# Define initial state of the system

irho_p = np.array([[1,0,0],[0,0,0],[0,0,0]]) #;0,0,0;0,0,0]

irho_m = np.array([[0,0,0],[0,0,0],[0,0,1]]) #0,0,0;0,0,1]

irho_z = np.array([[0,0,0],[0,1,0],[0,0,0]]) #0,1,0;0,0,0]

irho_mix = np.array([[1/2,0,0],[0,1/2,0],[0,0,0]])

irho_Z = np.array([[0,0,0],[0,0,0],[0,0,1]])

irho_MIX = np.array([[1/2,0,0],[0,0,0],[0,0,1/2]])

irho = np.kron(irho_z,irho_MIX)

#Initialization
rho_0 = (np.kron(U090xp,I))@irho@((np.kron(U090xp,I)).conj().T) # superposition state on NV

Sa= []

for h in range(N):
    ham = Al*np.kron(sz,Iz) + Ap*np.kron(sz,Ix) + B*gammaN*np.kron(I,Iz)
    
    eigvals = np.linalg.eigh(ham)[0]            # diagonalizing the Hamiltonian 여기서부터 문제 
    eigvecs = -1*np.linalg.eigh(ham)[1]
    E = np.diag(eigvals)                        # exponent of eigenvalues
    U_H= eigvecs.conj().T                       # unitary matrix formed by eigenvectors

    
    #free evolution unitary operator
    U_e2 = (U_H.conj().T)@(linalg.expm(-i*E*t[h]/2)@U_H) # for tau/2
    U_e  = (U_H.conj().T)@(linalg.expm(-i*E*t[h])@U_H)  # for tau
    rho_1 = U_e2 @ rho_0 @ (U_e2.conj().T)                  # first tau/2
    for k in range(n-1):
        rho_1 = U_e @ np.kron(U180xp,I) @ rho_1 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T)
        
    rho_2 = U_e2 @ np.kron(U180xp,I) @ rho_1 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T)
    rho_3 = np.kron(U090xmp,I) @ rho_2 @ ((np.kron(U090xmp,I)).conj().T)    # last pi/2
    res1 = (np.trace(irho_z@partial_trace(rho_3,2))).real                   # NV state 0 population readout
    Sa.append(res1)

#plot graph
plt.subplot(2,1,1)
plt.plot(t,Sa)
index = Sa.index(min(Sa))
tau=t[index]
# 임의 조작 파트
Sa[index] = 1
index = Sa.index(min(Sa))
tau=t[index]
print(tau)

nn = 32
nn_r = 2*np.linspace(1,nn,nn)
irho = np.kron(irho_z,irho_MIX) #initial state

#Initialization
rho_0 = np.kron(U090xp,I)@irho@(np.kron(U090xp,I)).conj().T # superposition state on NV

Sb = []
for p in range(1,nn+1):
    U_e2 = (U_H.conj().T)@(linalg.expm(-i*E*tau/2)@U_H) # for tau/2
    U_e  = (U_H.conj().T)@(linalg.expm(-i*E*tau)@U_H)   # for tau
    rho_1 = U_e2 @ rho_0 @ (U_e2.conj().T)              # first tau/2
    for m in range(1,2*p):
        rho_1 = U_e @ np.kron(U180xp,I) @ rho_1 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T)
    rho_2 = U_e2 @ np.kron(U180xp,I) @ rho_1 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T)
    rho_3 = np.kron(U090xmp,I) @ rho_2 @ (np.kron(U090xmp,I)).conj().T  # last pi/2
    res2 = (np.trace(irho_z@partial_trace(rho_3,2))).real               # NV state 0 population readout 
    Sb.append(res2)
   
#plot graph
plt.subplot(2,1,2)
plt.plot(nn_r,Sb)
plt.show()


# %%
