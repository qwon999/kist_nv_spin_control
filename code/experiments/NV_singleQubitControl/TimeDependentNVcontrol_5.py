# 4/18  Rxm,Rym 선택지 추가제공

# 모든 문제 해결 완료. 이제 baker formula만 추가하면 됨.
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

phi_t = (np.sin(2*np.pi*T/N))**2 # phi_t에 대한 특정 조건이 없으므로, 그냥 임의로 잡았음
d0 = 0.02

def Rx(t):
    w = 0.02 #gate도 결국 pulse의 시간에 의해서 결정. 단위시간 당 0.02radian변화하도록 설정
    theta = w*t
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])


def Ry(t):
    w = 0.02
    theta = w*t
    return np.matrix([[cos(theta/2),     -sin(theta/2)],
                    [sin(theta/2),     cos(theta/2)]])

def Rxm(t):
    w = 0.02 
    theta = -w*t
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])


def Rym(t):
    w = 0.02
    theta = -w*t
    return np.matrix([[cos(theta/2),     -sin(theta/2)],
                    [sin(theta/2),     cos(theta/2)]])



def Rz(t): # Rz는 사용하지 않음. 해밀토니안에 의한 회전으로만 컨트롤
    w = 1
    phi = w*t
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])

# 특정 dt에서의 hamiltonian에 의한 회전 정함.

def dH(t):
    H = d0*phi_t[t]*sz
    return H

def dU(dH) :
    return expm(-j*dH)

# fielity 계산을 위한 함수(일단은 안 쓸 예정)

def state_fidelity(rho_1, rho_2): #fidelity
        if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)
        
def cost_cal(rho1, rho2):
    rho1_xyz= [np.trace(rho1*sx).real,np.trace(rho1*sy).real,np.trace(rho1*sz).real]
    rho2_xyz= [np.trace(rho2*sx).real,np.trace(rho2*sy).real,np.trace(rho2*sz).real]
    cost = ((rho1_xyz[0]-rho2_xyz[0])**2+(rho1_xyz[1]-rho2_xyz[1])**2+(rho1_xyz[2]-rho2_xyz[2])**2)**0.5
    return cost

# 여기에서, z로테이션과 x,y 로테이션이 commute하지 않으므로 단순히 순차적으로 곱하는 것은 옳지 않다.
# 추후에, baker campbell hausdorff formula로 정확히 구성해야함.

# 4/14 z-rotation과 x,y_rotation이 동시에 적용되는데, commute하지 않으므로 공식으로 근사치로 계산

def problem (irho,irho_target,k): # cost function로 최적화
    
    byRz = dU(dH(k))@irho@dU(dH(k)).conj().T
    byRx = Rx(8) @ byRz @ Rx(8).conj().T
    byRy = Ry(8) @ byRz @ Ry(8).conj().T
    choice_list = [byRz,byRx,byRy]
    
    cost_for_only_rz = state_fidelity(byRz,irho_target)
    cost_by_rx = state_fidelity(byRx,irho_target)
    cost_by_ry = state_fidelity(byRy,irho_target)
    cost = [cost_for_only_rz,cost_by_rx,cost_by_ry]     # 각 선택에 대한 cost를 리스트에 임시 저장

    choice = cost.index(min(cost))                      # 가장 cost가 작은 선택지를 가져옴
    #print(min(cost))
    
    return min(cost), choice_list[choice], choice  # cost, 변환된 density matrix, 선택한 opreation index 저장 

def problem_by_fidelity(irho,irho_target,k):
    byRz = dU(dH(k))@irho@dU(dH(k)).conj().T
    byRx = Rx(8) @ byRz @ Rx(8).conj().T
    byRy = Ry(8) @ byRz @ Ry(8).conj().T
    byRxm = Rxm(8) @ byRz @ Rxm(8).conj().T
    byRym = Rym(8) @ byRz @ Rym(8).conj().T
    choice_list = [byRz,byRx,byRy,byRxm,byRym]
    F_next = 0
    i=0
    for j in choice_list:
        byRz_next = dU(dH(k+1))@j@dU(dH(k+1)).conj().T
        byRx_next = Rx(8) @ byRz_next @ Rx(8).conj().T
        byRy_next = Ry(8) @ byRz_next @ Ry(8).conj().T
        byRxm_next = Rxm(8) @ byRz_next @ Rxm(8).conj().T
        byRym_next = Rym(8) @ byRz_next @ Rym(8).conj().T
        F_next_list = [state_fidelity(irho_target,byRz_next),state_fidelity(irho_target,byRx_next),state_fidelity(irho_target,byRy_next),state_fidelity(irho_target,byRxm_next),state_fidelity(irho_target,byRym_next)]
        if F_next < max(F_next_list):
            F_next =max(F_next_list)
            choice = i
        i+=1
        
    F_for_only_rz = state_fidelity(irho_target,byRz)
    F_by_rx = state_fidelity(irho_target,byRx)
    F_by_ry = state_fidelity(irho_target,byRy)
    F_by_rxm = state_fidelity(irho_target,byRxm)
    F_by_rym = state_fidelity(irho_target,byRym)
    F = [F_for_only_rz,F_by_rx,F_by_ry,F_by_rxm,F_by_rym]     # 각 선택에 대한 cost를 리스트에 임시 저장

    return F[choice], choice_list[choice], choice  # cost, 변환된 density matrix, 선택한 opreation index 저장 

# define initial state

init_wave = np.array([[1],[0]])
irho_init = np.kron(init_wave,init_wave.conj().T)

# 중첩을 만들기위한 pi/2 펄스

Rxp = np.matrix([[cos(pi/2/2),     -1j*sin(pi/2/2)],
                    [-1j*sin(pi/2/2),     cos(pi/2/2)]])


output = []
def find_TOC_gate ():
    target_theta = (pi/180)*random.uniform(0,180)
    target_phi = (pi/180)*random.uniform(0,360)

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