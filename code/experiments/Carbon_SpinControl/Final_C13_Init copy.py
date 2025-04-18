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
import cmath

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


# Define dimension, pauli matrices
sx  = 1/sqrt(2)*np.array([[0, 1, 0],[1, 0, 1], [0, 1, 0]])
sy  = 1/sqrt(2)/1j*np.array([[0, 1, 0], [-1, 0, 1],[0, -1, 0]])
sz  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
I   = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Rotation matrix projected into 2 level system
Sxp  = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
Sxm  = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
Syp  = 1/1j*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
Sym  = 1/1j*np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
Szp  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
Szm  = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])

#Gellman matrix
Sx  = np.array([[0, 0, 1],[0, 0, 0], [1, 0, 0]])
Sy  = np.array([[0, 0, -1j],[0, 0, 0], [1j, 0, 0]])
Sz  = np.array([[1, 0, 0],[0, 0, 0], [0, 0, -1]])

# Pauli basis for 13C nuclear spin
Ix  = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])   
Iy  = 1/1j*np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
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
U090xp  = UO(1,0,pi/4,0,0)
U090xmp = UO(1,0,-pi/4,0,0)
U090yp  = UO(1,0,pi/4,pi/2,0)
U090ymp = UO(1,0,-pi/4,pi/2,0)
U180xp  = UO(1,0,pi/2,0,0)
U180xmp = UO(1,0,-pi/2,0,0)
U180yp  = UO(1,0,pi/2,pi/2,0)
U180ymp = UO(1,0,-pi/2,pi/2,0)

#Single Q ms=-1
U090xm  = UO(0,1,pi/4,0,0)
U090xmm = UO(0,1,-pi/4,0,0)
U090ym  = UO(0,1,pi/4,0,pi/2)
U090ymm = UO(0,1,-pi/4,0,pi/2)
U180xm  = UO(0,1,pi/2,0,0)
U180xmm = UO(0,1,-pi/2,0,0)
U180ym  = UO(0,1,pi/2,0,pi/2)
U180ymm = UO(0,1,-pi/2,0,pi/2) 

rho_nv_z_zero = np.array([[0,0,0],[0,1,0],[0,0,0]]) # |0>state
rho_nv_z_minusone = np.array([[0,0,0],[0,0,0],[0,0,1]]) # |-1>state
rho_nv_z_spp_px = np.array([[0,0,0],[0,1/2,1/2],[0,1/2,1/2]]) # |-1>state
rho_nv_z_spp_py = np.array([[0,0,0],[0,1/2,-1j/2],[0,1j/2,1/2]]) # |-1>state

rho_c_z_spp = np.array([[0.5,0,0.5],[0,0,0],[0.5,0,0.5]]) # superposition
rho_c_z_zero = np.array([[1,0,0],[0,0,0],[0,0,0]]) # |0>state 
rho_c_z_one = np.array([[0,0,0],[0,0,0],[0,0,1]]) # |0>state 
rho_c_z_mixed = np.array([[0.5,0,0],[0,0,0],[0,0,0.5]]) # mixed state

rho_nv_z = rho_nv_z_zero
rho_c_z = rho_c_z_mixed
irho = np.kron(rho_nv_z,rho_c_z)

gammaN = 2*pi*1.071e-3 #[MHz/G]
Al    = 2*pi*0.1 #[MHz] # A_|| hyperfine term
Ap    = 2*pi*0.1 #[MHz] # A_per hyperfine term


Sa= []

pipulse_list = [U180xm, U180xmm, U180ym, U180ymm]
halfpipulse_list = [U090xm, U090xmm, U090ym, U090ymm]
def make_unitary(tau,choice) :

    ham =  Ap*np.kron(sz,0.5*Ix)  + Al*np.kron(sz,Iz) + B*gammaN*np.kron(I,Iz)
    eigvals = np.linalg.eigh(ham)[0]            # diagonalizing the Hamiltonian 여기서부터 문제 
    eigvecs = 1*np.linalg.eigh(ham)[1]
    E = np.diag(eigvals)                        # exponent of eigenvalues
    U = eigvecs @ linalg.expm(-1j*E*tau/2) @ eigvecs.conj().T # for tau/2
    U_e = U @ np.kron(halfpipulse_list[choice],I) @ U # pi-pulse 한개
    # U_e = U @ U
    return U_e

def find_tau():
    Sa = []
    Sb = []
    rho_0 = (np.kron(U090xm,I)) @ irho @ ((np.kron(U090xm,I)).conj().T) # superposition state on NV
    # rho_0 = rho_ent
    for h in range(round(N)):
        #free evolution unitary operator
        U = make_unitary(t[h],0)
        # if h == 100 :
        #     make_print(U)
        rho_1 = rho_0
        for k in range(n):
            rho_1 = U @ rho_1 @ U.conj().T
        rho_2 = np.kron(U090xmm,I) @ rho_1 @ ((np.kron(U090xmm,I)).conj().T)    # last pi/2
        res1 = (np.trace(rho_nv_z@partial_trace(rho_2,2))).real   # NV state 0 population readout
        res2 = (np.trace(rho_c_z@partial_trace(rho_2,1))).real
        Sa.append(res1)
        Sb.append(res2)

    index = Sa.index(min(Sa))
    tau=t[index]
    #print(tau)
    return tau
# global tuples = []
def problem(vari): 
    #for e Ry(pi/2)

    rho1 = np.kron(U090ym,I)@irho@(np.kron(U090ym,I).conj().T)                              # Ry 90도

    #for N Rx(pi/2)
    CRx = make_unitary(vari[0],0)
    for i in range(vari[1]) :
        rho1 = CRx @ rho1 @ CRx.conj().T
    

    # U_e2=(U_H.conj().T)@(linalg.expm(-i*E* vari[0]/2)@U_H)                                  # for tau/2
    # U_e=(U_H.conj().T)@(linalg.expm(-i*E* vari[0])@U_H)                                     # for tau
    # rho2=U_e2@rho1@(U_e2.conj().T)                                                          # first tau/2
    # for k in range(1,2*math.trunc(vari[1])):                                                # N과 tau를 N개 생성
    #     rho2 = U_e@np.kron(U180xp,I) @ rho2 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N & tau
    # rho3 = U_e2 @ np.kron(U180xp,I) @ rho2 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # last N & tau/2

    #for e Rx(pi/2)
    rho1 = np.kron(U090xm,I)@rho1@(np.kron(U090xm,I).conj().T)                              # Rx 90도

    Rz = make_unitary(vari[2],0)
    for i in range(vari[3]) :
        rho1 = Rz @ rho1 @ Rz.conj().T

    for i in range(vari[1]) :
        rho1 = CRx @ rho1 @ CRx.conj().T
    # #for New Gate Rotation
    # U_e2=(U_H.conj().T)@(linalg.expm(-i*E*vari[2]/2)@U_H)                                   # for tau/2
    # U_e=(U_H.conj().T)@(linalg.expm(-i*E*vari[2])@U_H)                                      # for tau/2
    # rho5=U_e2@rho4@(U_e2.conj().T)                                                          # first tau/2
    # for k in range(1,2*math.trunc(vari[3])):                                                # N과 tau를 N개 생성
    #     rho5 = U_e@np.kron(U180xp,I) @ rho5 @ (np.kron(U180xp,I).conj().T) @ (U_e.conj().T) # N & tau
    # rho6 = U_e2 @ np.kron(U180xp,I) @ rho5 @ (np.kron(U180xp,I).conj().T) @ (U_e2.conj().T) # last N & tau/2

    xx = (np.trace(Ix@partial_trace(rho1,1))).real # for N spin
    yy = (np.trace(Iy@partial_trace(rho1,1))).real
    zz = (np.trace(Iz@partial_trace(rho1,1))).real
    
    return  zz


aa = []
bb = []
cc = []
dd = []

xx=0
yy=0
zz=0
count = 0

for ccc in (range(1)):
    trace = [1, 1, 0, 100, 100, 1000, 100]
    vvv = [0, 0, 0, 0]
    bbb = [0, 0, 0, 0]
    normalxyz = [0, 0, 0]
    start = time.time()

    #Initialization
    rho_0 = (np.kron(U090xp,I))@irho@((np.kron(U090xp,I)).conj().T) # superposition state on NV

    tau = find_tau()
    print(tau)

    print(problem([tau,32,0.01*tau,32]))


    