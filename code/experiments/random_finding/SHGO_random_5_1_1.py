# 5_1 : 다음 스텝과 현재 스텝의 z좌표 변화량을 계산하여 가장 많이 내려가는 펄스 선택

# 5_1_1 : 중첩상태에서, 반대로 보내는 케이스에 대해서 5_1 방식이 유효한 지에 대한 확인 코드
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
from scipy import linalg
from sklearn.feature_extraction.text import CountVectorizer
from qutip import bloch
import matplotlib.pyplot as plt
import random





# complex number
j = (-1)**0.5
 

# pauli matrix
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -j], [j, 0]])
sz = np.array([[1, 0], [0, -1]])
s0 = np.array([[1, 0], [0, 1]])

# Detunning Factor
d0 = pi/10
v0 = 0.00

def Rx(theta):
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])

def Rz(phi): # Rz는 사용하지 않음. 해밀토니안에 의한 회전으로만 컨트롤
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])
                     

# dt와 choice를 통한 unitarty gate 만드는 함수
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


def state_fidelity(rho_1, rho_2): #fidelity
        if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)

# d0와 v0로 인해 형성된 rotaiton 축의 vector값 계산        
# norm = (d0**2+v0**2)**0.5-0.001
# choice_0 = np.array([0,0,1])
# choice_1 = np.array([v0/norm,0,d0/norm])
# choice_2 = np.array([-v0/norm,0,d0/norm])
# choice_3 = np.array([0,v0/norm,d0/norm])
# choice_4 = np.array([0,-v0/norm,d0/norm])
# rot_vec = [choice_0,choice_1,choice_2,choice_3,choice_4]

# initial state
init_wave = np.array([[1],[0]])
irho_init = np.kron(init_wave,init_wave.conj().T)






# # QSL 계산을 위한 d0,v0모두 on-off가능한 상황에서 최단경로 찾기.
# for i in range(10) : 
#     # target_theta,target_phi = (pi/180)*random.uniform(0,180), (pi/180)*random.uniform(0,360)
#     target_theta,target_phi = pi/2, (pi/180)*random.uniform(0,360)
#     total = target_theta/v0 + target_phi/d0
#     print(f"theta : {target_theta}\nphi : {target_phi}\ntotal_time : {total}\n\n")

# date = datetime.now()
# printdate = date.strftime('%Y%m%d_%H%M%S')
# filename = "/Users/qwon/Documents/DataSetForNVSpin/BySHGO_random_3_d015" + printdate + '.csv'
# # Create an empty DataFrame and write to CSV file
# df = pd.DataFrame(columns=['v0','time'])
# df.to_csv(filename, index=False)


dt = 0.001
print(dt)
def make_combination() :
    irho_current = Rx(pi/2) @ irho_init @ Rx(pi/2).conj().T
    combination = []
    # choice =np.random.choice([1,2,3,4])
    # instant_U = unitary(dt,choice)
    # irho_current = instant_U @ irho_current @ instant_U.conj().T
    # combination.append(choice)
    # print(choice,end="")
    while target_y>np.trace(irho_current*sy).real:        
        current_y = np.trace(irho_current*sy).real
        temp_y = []
        irho_temp = np.matrix(irho_current)
        for i in range(0,5) :
            instant_U = unitary(dt,i)
            irho_temp = instant_U @ irho_current @ instant_U.conj().T    
            delta = current_y -  np.trace(irho_temp*sy).real
            temp_y.append(delta)
        choice = temp_y.index(min(temp_y))
        print(choice)
        instant_U = unitary(dt,choice)
        irho_current = instant_U @ irho_current @ instant_U.conj().T
        combination.append(choice)
    print(np.trace(irho_current*sy).real)
    return combination

target_theta, target_phi = pi/2, pi
target_U = Rz(target_phi) @ Rx(target_theta) 
irho_target = target_U @ irho_init @ target_U.conj().T
# target_z = np.trace(irho_target*sz)
target_y = (np.trace(irho_target*sy).real)*0.9999

# for i in range(0,1000) :
#     v0 = 1+i/1000
#     result = make_combination()
#     output = [[v0, len(result)*dt]]
#     print(len(result)*dt)
#     df = pd.DataFrame(output, columns=['v0','time'])
#     df.to_csv(filename, mode='a', header=False, index=False)
#     if i%100 == 0 :
#         print(f"{int(i/10)}%")

print(len(make_combination()))