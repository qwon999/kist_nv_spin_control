# 슈고로 랜덤 target에 대한 최적화 값을 얻고 csv로 저장
# 어차피 +-+- 반복이 처적의 솔루션. 그러므로, +-+-를 덧붙이고, 적절한 dt만 shgo로 탐새하는 코드구현

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
from qutip import bloch
import matplotlib.pyplot as plt


time_start = time.time()

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
                     

def instant_hamiltionian(choice):
    choice_list = [0,1,-1] # x-rotiation 방향 선택
    Ham = (d0*sz+v0*choice_list[choice]*sx)
    return Ham

def make_unitary(Ham,dt) : 
    eigvals = np.linalg.eigh(Ham)[0]
    eigvecs = 1*np.linalg.eigh(Ham)[1]
    E = np.diag(eigvals)
    U_H = eigvecs.conj().T
    U_e = U_H.conj().T @ expm(-j*E*dt) @ U_H
    return U_e

def unitary(dt, choice) :
    choice = int(choice)
    choice_list = [0,1,-1] # x-rotiation 방향 선택
    Ham = (d0*sz+v0*choice_list[choice]*sx)
    eigvals = np.linalg.eigh(Ham)[0]
    eigvecs = 1*np.linalg.eigh(Ham)[1]
    E = np.diag(eigvals)
    U_H = eigvecs.conj().T
    U_e = U_H.conj().T @ expm(-j*E*dt) @ U_H
    return U_e


def state_fidelity(rho_1, rho_2): #fidelity
        if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)
        
#어차피 처음 gate choice는 0이 될 수 없으므로, 예외처리 하지않음.

def ten_to_three (gate):
    gate_choice = []

    while gate != 0  :
        gate_choice.append(gate%3)
        gate //= 3

    return gate_choice[::-1]

init_wave = np.array([[1],[0]])
irho_init = np.kron(init_wave,init_wave.conj().T)

cost = [1,[]]
weight = 0.000005 # 시간에 대한 cost 가중치
#gate_combination = [1]
def problem(vari):
    print(vari[0])
    temp_gate_A = [1]
    temp_gate_B = [2]
    for i in range(15) : 
        total_U = s0
        for A in temp_gate_A: 
            instant_U = unitary(vari[0], A)        
            total_U = instant_U @ total_U
        irho_final = total_U @ irho_init @total_U.conj().T
        temp_F = state_fidelity(irho_target,irho_final)
        temp_T = weight*vari[0]*(i+1)  
        temp_cost = 1 - temp_F + temp_T
        if temp_cost<cost[0] :
            cost[0],cost[1] = temp_cost, temp_gate_A.copy()
        if temp_gate_A[i] == 1:
            temp_gate_A.append(2)
        else :
            temp_gate_A.append(1)
    for k in range(15) : 
        total_U = s0
        for B in temp_gate_B: 
            instant_U = unitary(vari[0], B)        
            total_U = instant_U @ total_U
        irho_final = total_U @ irho_init @total_U.conj().T
        temp_F = state_fidelity(irho_target,irho_final)
        temp_T = weight*vari[0]*(k+1)  
        temp_cost = 1 - temp_F + temp_T
        if temp_cost<cost[0] :
            cost[0],cost[1] = temp_cost, temp_gate_B.copy()
        if temp_gate_B[k] == 1:
            temp_gate_B.append(2)
        else :
            temp_gate_B.append(1)      

    print(cost[0],cost[1])
    c=cost[0]
    return c


output = []
count = 1
for t in range(count) : 
    #target_theta,target_phi = (pi/180)*random.uniform(0,180), (pi/180)*random.uniform(0,360)
    target_theta,target_phi = pi/2,pi*0
    target_U = Rz(target_phi/0.02) @ Rx(target_theta/0.02) 
    irho_target = target_U @ irho_init @ target_U.conj().T
    lenn = 1
    action = True
    while action == True:
        cost = [1,[]]
        vari = [10]
        bounds = [(5,30)]
        # integer_constraint = lambda x: np.all(np.mod(x, 1).astype(bool))  # ensure x is integer
        # constraints = [{'type': 'ineq', 'fun': integer_constraint}]
        rlt = optimize.shgo(problem,bounds=bounds,iters=5, options={'xtol':1e-15,'ftol':1e-15})
        output.append(['CASE'+str(t+1),len(cost[1]),target_theta,target_phi,rlt['x'][0],cost[1],rlt['fun'], rlt['nfev']])
        print(rlt['nfev'], cost[0], rlt['fun'],lenn, 'dt : ',rlt['x'][0])
        
        if rlt['fun'] < 0.01 or lenn==1 : 
            action = False
        lenn+=1
        print(action)
    # ₩output.append([])
    print(round((t+1)/count*100),"%")
date = datetime.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
print(date)
fin1 = pd.DataFrame(output)
fin1.rename(columns={0:"Case", 1:'gate lenght', 2:'Theta', 3: 'Phi', 4: 'dt', 5: 'combination', 6: 'cost', 7:'nfev'}, inplace=True)
fin1.to_csv("/Users/qwon/Documents/DataSetForNVSpin/BySHGOtemp4" + printdate + '.csv', index=False)   


time_end = time.time()

print("#success# \nTime : " + str(time_end - time_start)) 