# Ry 선택지 삭제. 
# SHGO 알고리즘 사용

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
from scipy.linalg import fractional_matrix_power
from sklearn.feature_extraction.text import CountVectorizer
from scipy import linalg


time_start = time.time()

# complex number
j = (-1)**0.5
 

# pauli matrix
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -j], [j, 0]])
sz = np.array([[1, 0], [0, -1]])

# define step size
dt = 1 # 고정
N = 1600 # 임의의 시구간
T = np.linspace(0,N,N+1) # N+1구간으로 쪼갬

d0 = 0.15

def Rx(t):
    w = 0.02 #gate도 결국 pulse의 시간에 의해서 결정. 단위시간 당 0.02radian변화하도록 설정
    theta = w*t
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])



def Rxm(t):
    w = 0.02 
    theta = -w*t
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])



def Rz(t): # Rz는 사용하지 않음. 해밀토니안에 의한 회전으로만 컨트롤
    w = 1
    phi = w*t
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])


# z축 로테이션이 시간에 대해 변화하지 않으므로, 현재 문제에서는 함수 선언 필요 X
# def dH(t):
#     H = d0*sz
#     return H
# def dU(dH) :
#     return expm(-j*dH)

def Only_Z(t) :
    return expm(-j*d0*sz)
def Xp_Z(t) :
    Xp = Rx(t)
    Detunning = expm(-j*d0*sz)
    return Detunning@Xp #Xp를 먼저 곱하고, z-rotaion 취함.
def Xm_Z(t) :
    Xm = Rxm(t)
    Detunning = expm(-j*d0*sz)
    return Detunning@Xm
def make_choice_list(t):
    return [Only_Z(t),Xp_Z(t),Xm_Z(t)]
def state_fidelity(rho_1, rho_2): #fidelity
        if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)
        



def problem_by_fidelity(irho,irho_target,k):
    byRz = dU(dH(k))@irho@dU(dH(k)).conj().T
    byRx = Rx(8) @ byRz @ Rx(8).conj().T
    byRxm = Rxm(8) @ byRz @ Rxm(8).conj().T
    choice_list = [byRz,byRx,byRy,byRxm,byRym]
    F_next = 0
    i=0
    for j in choice_list:
        byRz_next = dU(dH(k+1))@j@dU(dH(k+1)).conj().T
        byRx_next = Rx(8) @ byRz_next @ Rx(8).conj().T
        byRxm_next = Rxm(8) @ byRz_next @ Rxm(8).conj().T
        F_next_list = [state_fidelity(irho_target,byRz_next),state_fidelity(irho_target,byRx_next),state_fidelity(irho_target,byRxm_next)]
        if F_next < max(F_next_list):
            F_next =max(F_next_list)
            choice = i
        i+=1
        
    F_for_only_rz = state_fidelity(irho_target,byRz)
    F_by_rx = state_fidelity(irho_target,byRx)
    F_by_rxm = state_fidelity(irho_target,byRxm)
    F = [F_for_only_rz,F_by_rx,F_by_rxm,]     # 각 선택에 대한 cost를 리스트에 임시 저장

    return F[choice], choice_list[choice], choice  # cost, 변환된 density matrix, 선택한 opreation index 저장 

# define initial state

init_wave = np.array([[1],[0]])
irho_init = np.kron(init_wave,init_wave.conj().T)

# 중첩을 만들기위한 pi/2 펄스

Rxp = np.matrix([[cos(pi/2/2),     -1j*sin(pi/2/2)],
                    [-1j*sin(pi/2/2),     cos(pi/2/2)]])


output = []
def find_TOC_gate ():
    target_theta,target_phi = (pi/180)*random.uniform(0,180), (pi/180)*random.uniform(0,360) 

    irho_target= Rz(target_phi/0.02) @ Rx(target_theta/0.02) @ irho_init @ Rx(target_theta/0.02).conj().T @ Rz(target_phi/0.02).conj().T
    xyz_target = [np.trace(irho_target*sx).real,np.trace(irho_target*sy).real,np.trace(irho_target*sz).real]
    k=0
    irho_mid = irho_init
    cost=0
    gate_choice = []
    for i in range(1, N):
        cost, irho_mid, choice = problem_by_fidelity(irho_mid,irho_target,i)
        gate_choice.append(choice)
        if cost > 0.99:
            k=i
            xyz_fin = [np.trace(irho_mid*sx).real,np.trace(irho_mid*sy).real,np.trace(irho_mid*sz).real]
            rlt = (['TOC' ,target_theta, target_phi, xyz_target[0],xyz_target[1],xyz_target[2],k,cost,xyz_fin[0],xyz_fin[1],xyz_fin[2]])
            
            return rlt
        if i == N-1:
            k=i
            xyz_fin = [np.trace(irho_mid*sx).real,np.trace(irho_mid*sy).real,np.trace(irho_mid*sz).real]
            rlt = (['TOC' ,target_theta, target_phi, xyz_target[0],xyz_target[1],xyz_target[2],k,cost,xyz_fin[0],xyz_fin[1],xyz_fin[2]])
    return rlt

 
# output.append(find_TOC_gate())
# output.append(find_TOC_gate())
# output.append(find_TOC_gate())
# output.append(find_TOC_gate())
# output.append(find_TOC_gate())
# print(output)
count = int(2000)
for xx in range (count):
    rlt=find_TOC_gate()
    if xx%(count/20) == 0:
         print(f"{round(xx/count*100)}%")
    if rlt != None:
        output.append(rlt)

#print(output)
date = datetime.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
print(date)
fin1 = pd.DataFrame(output)
fin1.rename(columns={0:"Case", 1:'Theta', 2: 'Phi', 3: 'target_X', 4: 'target_y', 5: 'target_z', 6: 'time', 7: 'fidelity', 8: 'fin_x', 9:'fin_y',10:'fin_z'}, inplace=True)
fin1.to_csv("/Users/qwon/Documents/DataSetForNVSpin/TOC_with_next_dH" + printdate + '.csv', index=False)   

time_end = time.time()

print(f"#success = {len(output)/count*100}%\nTime : " + str(time_end - time_start)) 