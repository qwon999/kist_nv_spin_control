from math import *
import numpy as np
from qutip import *
from qutip import bloch
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.linalg import fractional_matrix_power

# complex number
j = (-1)**0.5
 

# pauli matrix
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -j], [j, 0]])
sz = np.array([[1, 0], [0, -1]])
s0 = np.array([[1, 0], [0, 1]])

# Detunning Factor
d0 = 0.15
v0 = 0.02

def Rx(theta):
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])

def Rz(phi): # Rz는 사용하지 않음. 해밀토니안에 의한 회전으로만 컨트롤
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])

def unitary(dt, choice) :
    choice_list = [0,1,-1,1,-1] # x,y-rotiation 방향 선택
    if choice <3 :
        Ham = (d0*sz+v0*choice_list[choice]*sx)
    else :
        Ham = (d0*sz+v0*choice_list[choice]*sy)
    eigvals = np.linalg.eigh(Ham)[0]
    eigvecs = 1*np.linalg.eigh(Ham)[1]
    E = np.diag(eigvals)
    U_H = eigvecs.conj().T
    U_e = U_H.conj().T @ expm(-j*E*dt) @ U_H
    return U_e

# fidelity 계산. 0.99이상이면 일치하는 것으로 간주.
def state_fidelity(rho_1, rho_2): 
    if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
    else:
        sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
        fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
        return np.real(fidelity)
    
    
target_theta, target_phi, dt, combi = pi	, 0,	2.6	, [1]

dt*len(combi)

init_wave = np.array([[1],[0]])
irho_init = np.kron(init_wave,init_wave.conj().T)

target_U = Rz(target_phi) @ Rx(target_theta)
irho_target = target_U @ irho_init @target_U.conj().T


# 중첩에서 시작할때 필요한 코드
# irho_init = Rx(pi/2) @ irho_init @Rx(pi/2).conj().T
k=1
combination = []
for i in combi :
    combination += [i for j in range(k)]
print(combination)

dt = dt/k








irho_1 = np.matrix(irho_init)
irho_2 = np.matrix(irho_init)
irho_3 = np.matrix(irho_init)
irho_4 = np.matrix(irho_init)
rho_list = [irho_1,irho_2,irho_3,irho_4]
for i in range(1,5) :
    U = unitary(dt,i)
    rho_list[i-1] = U @ rho_list[i-1] @ U.conj().T
# print(rho_list)
rho_list[0] = unitary(dt,3) @ rho_list[0] @ unitary(dt,3).conj().T
rho_list[1] = unitary(dt,4) @ rho_list[1] @ unitary(dt,4).conj().T
rho_list[2] = unitary(dt,2) @ rho_list[2] @ unitary(dt,2).conj().T
rho_list[3] = unitary(dt,1) @ rho_list[3] @ unitary(dt,1).conj().T

for i in range(1,5) :
    F = (state_fidelity(rho_list[i-1], irho_target))
    x,y,z = (np.trace(rho_list[i-1]*sx).real),(np.trace(rho_list[i-1]*sy).real),(np.trace(rho_list[i-1]*sz).real)
    print(i, F , x,y,z,)
print('\n')

rho_list[0] = unitary(dt,3) @ rho_list[0] @ unitary(dt,3).conj().T
rho_list[1] = unitary(dt,4) @ rho_list[1] @ unitary(dt,4).conj().T
rho_list[2] = unitary(dt,2) @ rho_list[2] @ unitary(dt,2).conj().T
rho_list[3] = unitary(dt,1) @ rho_list[3] @ unitary(dt,1).conj().T

for i in range(1,5) :
    F = (state_fidelity(rho_list[i-1], irho_target))
    x,y,z = (np.trace(rho_list[i-1]*sx).real),(np.trace(rho_list[i-1]*sy).real),(np.trace(rho_list[i-1]*sz).real)
    print(i, F , x,y,z)
print('\n')

rho_list[0] = unitary(dt,3) @ rho_list[0] @ unitary(dt,3).conj().T
rho_list[1] = unitary(dt,4) @ rho_list[1] @ unitary(dt,4).conj().T
rho_list[2] = unitary(dt,2) @ rho_list[2] @ unitary(dt,2).conj().T
rho_list[3] = unitary(dt,1) @ rho_list[3] @ unitary(dt,1).conj().T

for i in range(1,5) :
    F = (state_fidelity(rho_list[i-1], irho_target))
    x,y,z = (np.trace(rho_list[i-1]*sx).real),(np.trace(rho_list[i-1]*sy).real),(np.trace(rho_list[i-1]*sz).real)
    print(i, F , x,y,z)
print('\n')

rho_list[0] = unitary(dt,3) @ rho_list[0] @ unitary(dt,3).conj().T
rho_list[1] = unitary(dt,4) @ rho_list[1] @ unitary(dt,4).conj().T
rho_list[2] = unitary(dt,2) @ rho_list[2] @ unitary(dt,2).conj().T
rho_list[3] = unitary(dt,1) @ rho_list[3] @ unitary(dt,1).conj().T

for i in range(1,5) :
    F = (state_fidelity(rho_list[i-1], irho_target))
    x,y,z = (np.trace(rho_list[i-1]*sx).real),(np.trace(rho_list[i-1]*sy).real),(np.trace(rho_list[i-1]*sz).real)
    print(i, F , x,y,z)
print('\n')