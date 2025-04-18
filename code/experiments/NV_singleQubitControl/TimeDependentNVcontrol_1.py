# 4/10
# 4/12 초안 완성
# x,y에 대해서 time dependent한 요소를 추가하기 전에,  z축에 대해서만 time dependent한 hamiltonuan 구성먼저 해보자.
from sys import stdout
from math import *
import numpy as np
from scipy.linalg import expm
from qutip import *
import random
from scipy import optimize
from datetime import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
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
    w = 1 #gate도 결국 pulse의 시간에 의해서 결정 20은 임의의 설정값
    theta = w*t
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])

def Rz(t):
    w = 1
    phi = w*t
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])

# Hamiltonian in  the rotating frame
#H0 = 2*np.pi*d0*sz #Time-independent
#Hc_T= np.array(2*np.pi*v1*(np.cos(phi)*sx+np.sin(phi)*sy)) #Time-dependent
#H = H0+Hc_T # Total hamiltonian

# Define hamiltonian
# def dH(t):
#     H = 2*np.pi*d0*np.sin(phi_t[t])*sz*0.1
#     return H






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
# # Define target wave fuction
# # irho_target = rand_dm_ginibre(2, rank=1) #random한 target density matrix
# target_theta = (pi/180)*random.uniform(0,180)
# target_phi = (pi/180)*random.uniform(0,360)
# # target_theta = pi/2
# # target_phi = pi/2
# # print(target_theta,target_phi)
# # target_wave = Rz(target_phi/(2*pi)) * (Rx(target_theta/(2*pi)) * init_wave)
# irho_target= Rz(target_phi/(2*pi)) @ Rx(target_theta/(2*pi)) @ irho_init @ Rx(target_theta/(2*pi)).conj().T @ Rz(target_phi/(2*pi)).conj().T
# print(irho_target)
# print(np.dot(irho_target,irho_target))
# xyz_target = [np.trace(irho_target*sx),np.trace(irho_target*sy),np.trace(irho_target*sz)] #targer density matirix의 bloch sphere 상의 좌표
# print(xyz_target)


# #####Hamiltonian에 따른 스핀 축 변화 추이 확인을 위한 파트(실제 코드에는 안들어감)
# def dU(dH) :
#     return expm(-j*dH)
# irho_mid = np.zeros((3,400))
# irho_mid[0][0] = 0
# irho_mid[1][0] = 0
# irho_mid[2][0] = 1
# irho_init = Rx(0.25)@irho_init@Rx(0.25).conj().T
# print(Rx(0.25))
# for k in range(1,400):
#     dHt=dH(k)
#     #print(dHt)
#     dUt=dU(dHt)

#     irho_init = dUt @ irho_init @ dUt.conj().T
#     #print((dUt@dUt.conj().T)) #생성된 dU가 Unitary가 맞는지 확인용
#     irho_mid[0][k] = np.trace(irho_init*sx)
#     irho_mid[1][k] = np.trace(irho_init*sy)
#     irho_mid[2][k] = np.trace(irho_init*sz)
# #print(irho_mid[0])


def dH(t):
    H = 2*np.pi*d0*np.sin(phi_t[t])*sz*0.01
    return H

def dU(dH) :
    return expm(-j*dH)

output=[]

for x in range(15):

    target_theta = (pi/180)*random.uniform(0,180)
    target_phi = (pi/180)*random.uniform(0,360)
    print(target_theta,target_phi)
    irho_target= Rz(target_phi) @ Rx(target_theta) @ irho_init @ Rx(target_theta).conj().T @ Rz(target_phi).conj().T
    # print(irho_target)
    # print(np.dot(irho_target,irho_target))
    xyz_target = [np.trace(irho_target*sx).real,np.trace(irho_target*sy).real,np.trace(irho_target*sz).real] #targer density matirix의 bloch sphere 상의 좌표
    start = time.time()
    k=0 # T구간에서 몇번째 dH를 선택할 지를 알려주는 parameter
    irho_mid = irho_init 
    # irho_mid = Rx(pi/2) @ irho_init @ Rx(pi/2).conj().T # 중첩상태를 만드는 구간, H의 xy에 대한 영향 파악용.
    z_list=[abs((np.trace(irho_mid*sz)).real-xyz_target[2])+0.0025*k]
    #for y in range(1,round(20*target_theta)+1):
    for y in range(1,N+1):
        dUt=dU(dH(y))
        irho_mid = dUt @ Rx(1/20)  @ irho_mid  @ Rx(1/20).conj().T  @ dUt.conj().T # 단위 dt당 1/20 rad 회전하도록 설정
        # irho_mid = Rx(1/10)  @ irho_mid  @ Rx(1/10).conj().T
        # irho_mid = dUt @ irho_mid @ dUt.conj().T
        # irho_mid = gates @ irho_mid @ gates.conj().T
        k+=1
        z_list.append(abs((np.trace(irho_mid*sz)).real-xyz_target[2])+0.00025*k) # 1차 cost function ; k앞에 곱해지는 상수는 시간과 좌표 모두 fidelity를 만족하기 위한 보정값.
    k = z_list.index(min(z_list))
    print(k)
    irho_mid = irho_init
    for w in range(1,k+1) :
        dUt=dU(dH(w))
        irho_mid = dUt @ Rx(1/20)  @ irho_mid  @ Rx(1/20).conj().T  @ dUt.conj().T # 단위 dt당 1/20 rad 회전하도록 설정
    print(xyz_target[2],(np.trace(irho_mid*sz)).real)
    F = 0
    xyz_fin = [3,3,3]
    for z in range(k+1,N+1):
        k+=1
        dUt=dU(dH(z))
        irho_mid = dUt @ Rz(1/20)  @ irho_mid  @ Rz(1/20).conj().T  @ dUt.conj().T # 단위 dt당 1/20 rad 회전하도록 설정
        xyz_fin = [np.trace(irho_mid*sx).real,np.trace(irho_mid*sy).real,np.trace(irho_mid*sz).real]
        F = 1 - ((xyz_target[0]-xyz_fin[0])**2+(xyz_target[1]-xyz_fin[1])**2+(xyz_target[2]-xyz_fin[2])**2)**0.5
        if F < 0.90:
            continue
        else :
            output.append(["Case" + str(x + 1), target_theta, target_phi, xyz_target[0],xyz_target[1],xyz_target[2],k,F,xyz_fin[0],xyz_fin[1],xyz_fin[2]])
            break 
stdout.write("\n")
date = datetime.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
print(date)
fin1 = pd.DataFrame(output)
fin1.rename(columns={0:"Case", 1:'Theta', 2: 'Phi', 3: 'target_X', 4: 'target_y', 5: 'target_z', 6: 'time', 7: 'fidelity', 8: 'fin_x', 9:'fin_y',10:'fin_z'}, inplace=True)
fin1.to_csv("/Users/qwon/Documents/DataSetForNVSpin/" + printdate + '.csv', index=False)   
#
#  plt.plot(T,z_list)
# plt.show()









