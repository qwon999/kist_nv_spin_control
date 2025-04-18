# 6/2 완전한 랜덤으로 구성. -> 예상보다 너무 뛰어난 성능을 보인다.
# 6/3 기존 best choice에 가중치를 둬서 랜덤탐색
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
import random



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

def Rx(theta):
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])
def Rz(phi): 
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
norm = (d0**2+v0**2)**0.5
choice_0 = np.array([0,0,1])
choice_1 = np.array([v0/norm,0,d0/norm])
choice_2 = np.array([-v0/norm,0,d0/norm])
choice_3 = np.array([0,v0/norm,d0/norm])
choice_4 = np.array([0,-v0/norm,d0/norm])
rot_vec = [choice_0,choice_1,choice_2,choice_3,choice_4]

# initial state
init_wave = np.array([[1],[0]])
irho_init = np.kron(init_wave,init_wave.conj().T)



# 직전 step의 내적값 저장위한 배열
past_inner_product = np.zeros(5)
# trace = [cost,[gate combination],dt]
trace = [1,np.zeros(100000000),0,50000,0]
# 0 : fidelity, 1 : gate_combination, 2 : dt, 3 : total_time, 4 : # of random_choice 

# random에 대한 가중치 주는 함수
def make_weight_list(past_choice) :
    choice_list = [0,1,2,3,4]
    weight = [0.15]*5
    weight[past_choice] = 0.4
    choice = np.random.choice(choice_list, p=weight)
    return choice

# problem 함수
def make_combination(dt,c) :
    F = 1
    irho_current = irho_init
    iter = 0
    combination = []
    if c == 0:
        while (F>0.02):
            choice = random.randint(1,4)

            instant_unitary = unitary(dt, choice)

            irho_current = instant_unitary @ irho_current @ instant_unitary.conj().T

            F = 1 - state_fidelity(irho_target,irho_current)

            combination.append(choice)
            iter+=1  

    while (F>0.01) and (iter<10000000):
        if iter+1 == len(trace[1]):
            print("!")
            return 0
        choice = make_weight_list(iter)
        instant_unitary = unitary(dt, choice)

        irho_current = instant_unitary @ irho_current @ instant_unitary.conj().T

        F = 1 - state_fidelity(irho_target,irho_current)

        combination.append(choice)
        iter+=1  
    print(F)
    
    if len(combination)*dt < trace[3]:
        trace[0] = F
        trace[1] = combination.copy()
        trace[2] = dt
        trace[3] = len(combination)*dt
    return 0


target_theta,target_phi = pi,pi/4
target_U = Rz(target_phi) @ Rx(target_theta) 
irho_target = target_U @ irho_init @ target_U.conj().T

count = 50000
for i in range(count) :
    make_combination(10,i)
print(trace)



# a,b=0,0
# output = []
# count = 1
# for t in range(count) : 
#     # target_theta,target_phi = (pi/180)*random.uniform(0,180), (pi/180)*random.uniform(0,360)
#     # target_theta,target_phi = pi/2,pi
#     target_theta_list = np.linspace(pi/10,pi,10)
#     target_phi_list = np.linspace(0,2*pi,10)
#     target_theta, target_phi = target_theta_list[a], target_phi_list[b]
#     a+=1
#     if a==10:
#         a=0
#         b+=1
#     target_U = Rz(target_phi) @ Rx(target_theta) 
#     irho_target = target_U @ irho_init @ target_U.conj().T
    
#     lenn = 1
#     action = True
#     while action == True:
#         trace = [1,[],0,50000,0]
#         vari = [10]
#         bounds = [(1,15)]
#         rlt = optimize.shgo(make_combination,bounds=bounds,iters=6, options={'xtol':1e-15,'ftol':1e-15})
#         output.append(['CASE'+str(t+1),len(trace[1]),target_theta,target_phi,trace[2],trace[1],rlt['fun'], rlt['nfev'],trace[3],d0, trace[4]])
#         print(rlt['nfev'], trace[0], trace[1],lenn, 'dt : ',trace[2])
#         if rlt['fun'] < 0.01 or lenn==1 : 
#             action = False
#         lenn+=1
#         print(action)
#     print(round((t+1)/count*100),"%")
# date = datetime.now()
# printdate = date.strftime('%Y%m%d_%H%M%S')
# print(date)
# fin1 = pd.DataFrame(output)
# fin1.rename(columns={0:"Case", 1:'gate lenght', 2:'Theta', 3: 'Phi', 4: 'dt', 5: 'combination', 6: 'cost', 7:'nfev', 8: 'Time', 9:'detunning', 10 : '# of random_choice'}, inplace=True)
# fin1.to_csv("/Users/qwon/Documents/DataSetForNVSpin/BySHGO_Final_d015" + printdate + '.csv', index=False)   


# time_end = time.time()

# print("#success# \nTime : " + str(time_end - time_start)) 