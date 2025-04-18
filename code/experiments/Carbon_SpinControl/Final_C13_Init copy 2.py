from toqito.channels import partial_trace
from qutip import *
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy import linalg
import math
import matplotlib.pyplot as plt
from scipy import optimize
import random
from math import *
import pandas as pd
from dataclasses import dataclass
import time
from datetime import datetime as dt 
import random
from mpl_toolkits.mplot3d import Axes3D
from sys import stdout
from scipy.linalg import fractional_matrix_power

def UO(B1,B2,a,D1,D2):
    i   = 1j
    gamma = 2*pi*2.8
    D     = 2870
    UA = [[(B2**2+B1**2*cos(a))/(B1**2+B2**2), -i*B1*(e**(-i*D1))*sin(a)/sqrt(B1**2+B2**2), ((-1+cos(a))*B1*B2*(e**(-i*(D1-D2))))/(B1**2+B2**2)],
            [-i*B1*(e**(i*D1))*sin(a)/sqrt(B1**2+B2**2), cos(a), -i*B2*(e**(i*D2))*sin(a)/sqrt(B1**2+B2**2)],
            [((-1+cos(a))*B1*B2*e**(i*(D1-D2)))/(B1**2+B2**2), -i*B2*(e**(-i*D2))*sin(a)/sqrt(B1**2+B2**2), (B1**2+B2**2*cos(a))/(B1**2+B2**2)]]
    return UA

def state_fidelity(rho_1, rho_2): #fidelity
    if np.shape(rho_1) != np.shape(rho_2):
        print("Dimensions of two states do not match.")
        return 0
    else:
        sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
        fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
        return np.real(fidelity)


## Define dimension, pauli matrices
i   = 1j #1j
sx  = 1/sqrt(2)*np.array([[0, 1, 0],[1, 0, 1], [0, 1, 0]])
sy  = 1/sqrt(2)/i*np.array([[0, 1, 0], [-1, 0, 1],[0, -1, 0]])
sz  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
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

irho_p = np.array([[1,0,0],[0,0,0],[0,0,0]]) #[0,0,0;0,0,0]

irho_m = np.array([[0,0,0],[0,0,0],[0,0,1]]) #[0,0,0;0,0,1]

irho_z = np.array([[0,0,0],[0,1,0],[0,0,0]]) #[0,1,0;0,0,0]

irho_mix = np.array([[1/2,0,0],[0,1/2,0],[0,0,0]]) #[1/2,0,0;0,1/2,0;0,0,0]

irho_Z = np.array([[0,0,0],[0,0,0],[0,0,1]]) #target state

irho_MIX = np.array([[1/2,0,0],[0,0,0],[0,0,1/2]])

irho = np.kron(irho_z,irho_MIX) #initial state
trace = [1, 1, 0, 100] # trace of the X, Y, Z, and total density matrices

gammaN = 2*pi*1.071e-3 #[MHz/G]
Al    = 2*pi * random.uniform(0.05, 0.8) #[MHz] # A_|| hyperfine term
Ap = 2*pi* random.uniform(0.05, 0.3) #[MHz] # A_per hyperfine term

rho_0 = (np.kron(U090xp,I))@irho@((np.kron(U090xp,I)).conj().T) 

Sa= []

ham = Al*np.kron(sz,Iz) + Ap*np.kron(sz,Ix) + B*gammaN*np.kron(I,Iz) 
eigvals = np.linalg.eigh(ham)[0]            
eigvecs = -1*np.linalg.eigh(ham)[1]         
E = np.diag(eigvals)                        
U_H= eigvecs.conj().T                       

bbb = [0, 0, 0, 0]
normalxyz = [0, 0, 0]

# global tuples = []
def problem(vari): 
    #for e Ry(pi/2)

    rho1 = np.kron(U090yp,I)@irho@(np.kron(U090yp,I).conj().T)                              # Ry 90도

    #for N Rx(pi/2)
    U_e2=(U_H.conj().T)@(linalg.expm(-i*E* vari[0]/2)@U_H)                                  # for tau/2
    U_e=(U_H.conj().T)@(linalg.expm(-i*E* vari[0])@U_H)                                     # for tau
    rho2=U_e2@rho1@(U_e2.conj().T)                                                          # first tau/2
    for k in range(1,2*math.trunc(vari[1])):                                                # N과 tau를 N개 생성
        rho2 = U_e@np.kron(U180xp,I) @ rho2 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N & tau
    rho3 = U_e2 @ np.kron(U180xp,I) @ rho2 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # last N & tau/2

    #for e Rx(pi/2)
    rho4 = np.kron(U090xp,I)@rho3@(np.kron(U090xp,I).conj().T)                              # Rx 90도

    #for New Gate Rotation
    U_e2=(U_H.conj().T)@(linalg.expm(-i*E*vari[2]/2)@U_H)                                   # for tau/2
    U_e=(U_H.conj().T)@(linalg.expm(-i*E*vari[2])@U_H)                                      # for tau/2
    rho5=U_e2@rho4@(U_e2.conj().T)                                                          # first tau/2
    for k in range(1,2*math.trunc(vari[3])):                                                # N과 tau를 N개 생성
        rho5 = U_e@np.kron(U180xp,I) @ rho5 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N & tau
    rho6 = U_e2 @ np.kron(U180xp,I) @ rho5 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # last N & tau/2

    xx = (np.trace(Ix@partial_trace(rho6,1))).real # for N spin
    yy = (np.trace(Iy@partial_trace(rho6,1))).real
    zz = (np.trace(Iz@partial_trace(rho6,1))).real
    
    cost = 1 - state_fidelity(irho_Z, partial_trace(rho6, 1))

    if(cost < trace[6]):
        trace[6] = cost
        normalxyz[0] = xx
        normalxyz[1] = yy
        normalxyz[2] = zz
        
        bbb[0] = vari[0]
        bbb[1] = vari[1]
        bbb[2] = vari[2]
        bbb[3] = vari[3]
    return cost


aa = []
bb = []
cc = []
dd = []

xx=0
yy=0
zz=0
count = 0

for ccc in (range(20)):
    trace = [1, 1, 0, 100, 100, 1000, 100]
    vvv = [0, 0, 0, 0]
    bbb = [0, 0, 0, 0]
    normalxyz = [0, 0, 0]
    start = time.time()
    #for making 13C nuclear random dataset
    gammaN = 2*pi*1.071e-3 #[MHz/G]
    Al    = 2*pi * random.uniform(0.05, 0.8) #[MHz] # A_|| hyperfine term
    Ap = 2*pi* random.uniform(0.05, 0.3) #[MHz] # A_per hyperfine term

    #Initialization
    rho_0 = (np.kron(U090xp,I))@irho@((np.kron(U090xp,I)).conj().T) # superposition state on NV

    Sa= []

    ham = Al*np.kron(sz,Iz) + Ap*np.kron(sz,Ix) + B*gammaN*np.kron(I,Iz) # Hamiltonian
    eigvals = np.linalg.eigh(ham)[0]            # diagonalizing the Hamiltonian
    eigvecs = -1*np.linalg.eigh(ham)[1]         # eigenvectors
    E = np.diag(eigvals)                        # exponent of eigenvalues
    U_H= eigvecs.conj().T                       # unitary matrix formed by eigenvectors
    
    for h in range(N):

        #free evolution unitary operator
        U_e2 = (U_H.conj().T)@(linalg.expm(-i*E*t[h]/2)@U_H) # for tau/2
        U_e  = (U_H.conj().T)@(linalg.expm(-i*E*t[h])@U_H)  # for tau
        rho_1 = U_e2 @ rho_0 @ (U_e2.conj().T)                  # first tau/2
        for k in range(n-1):                                   # N과 tau를 N개 생성
            rho_1 = U_e @ np.kron(U180xp,I) @ rho_1 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N & tau
            
        rho_2 = U_e2 @ np.kron(U180xp,I) @ rho_1 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # last N & tau/2
        rho_3 = np.kron(U090xmp,I) @ rho_2 @ ((np.kron(U090xmp,I)).conj().T)    # last pi/2
        res1 = (np.trace(irho_z@partial_trace(rho_3,2))).real                   # NV state 0 population readout
        Sa.append(res1)                                                       # append to list
        
    index = Sa.index(min(Sa))
    tau=t[index]

    ham = Al*np.kron(sz,Iz) + Ap*np.kron(sz,Ix) + B*gammaN*np.kron(I,Iz) # Hamiltonian
    eigvals = np.linalg.eigh(ham)[0] # diagonalizing the Hamiltonian 
    eigvecs = -1*np.linalg.eigh(ham)[1] # eigenvectors
    E = np.diag(eigvals)             # exponent of eigenvalues
    U_H= eigvecs.conj().T         # unitary matrix formed by eigenvectors


    tol = 1e-8 
    print(problem([tau,32,0.1*tau,32]))