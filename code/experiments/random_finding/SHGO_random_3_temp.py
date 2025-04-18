# 1에서 중첩가는 것과 0에서 중첩가는 것에 속도 차이가 있는 지 확인하기 위한 임시 코드
# intial state를 아래쪽으로 설정

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
norm = (d0**2+v0**2)**0.5
choice_0 = np.array([0,0,1])
choice_1 = np.array([v0/norm,0,d0/norm])
choice_2 = np.array([-v0/norm,0,d0/norm])
choice_3 = np.array([0,v0/norm,d0/norm])
choice_4 = np.array([0,-v0/norm,d0/norm])
rot_vec = [choice_0,choice_1,choice_2,choice_3,choice_4]





overall = [1,np.zeros(500),0,50000]
# 0 : fidelity, 1 : gate_combination, 2 : dt, 3 : total_time

# problem 함수
def make_combination(dt) :
    F = 1
    irho_current = irho_init
    iter = 0
    combination = []
    while (F>0.01) and (iter<500):
        if (iter+1)*dt > overall[3]:
            return 0
        choice = random.randint(0,4)

        instant_unitary = unitary(dt, choice)

        irho_current = instant_unitary @ irho_current @ instant_unitary.conj().T

        F = 1 - state_fidelity(irho_target,irho_current)

        combination.append(choice)
        iter+=1  
    if len(combination)*dt < overall[3]:
        overall[0] = F
        overall[1] = combination.copy()
        overall[2] = dt
        overall[3] = len(combination)*dt[0]
    # print(F,len(trace[1]))
    return 0

def problem(dt) :
    # print('\n',dt)
    # start = time.time()
    count = 20000
    for i in range(count) :
        make_combination(dt)
        # if (i+1)%(count/10) == 0 :
        #     print(f"{int((i+1)/count*100)}%")
    # end = time.time()
    # total = end - start
    return overall[3]




# a,b=0,0
# CSV file name setup
date = datetime.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
filename = "/Users/qwon/Documents/DataSetForNVSpin/BySHGO_random_3_d015" + printdate + '.csv'

# Create an empty DataFrame and write to CSV file
df = pd.DataFrame(columns=["Case", 'gate lenght', 'Theta', 'Phi', 'dt', 'combination', 'total time', 'fidelity', 'detunning'])
df.to_csv(filename, index=False)

count=10
for t in range(count) : 
    # initial state
    init_wave = np.array([[1],[0]])
    irho_init = np.kron(init_wave,init_wave.conj().T)
    target_theta,target_phi = pi,0
    target_U = Rz(target_phi) @ Rx(target_theta) 
    irho_target = target_U @ irho_init @ target_U.conj().T
    irho_init = Rx(pi/2) @ irho_init @ Rx(pi/2).conj().T

    overall = [1,np.zeros(500),0,50000]
    
    lenn = 1
    action = True
    
    while action == True:
        
        vari = [10]
        bounds = [(9,10)]
        rlt = optimize.shgo(problem,bounds=bounds,iters=1, options={'xtol':1e-0,'ftol':1e-0})    
        output = [['CASE'+str(t+1),len(overall[1]),target_theta,target_phi,overall[2],overall[1],len(overall[1])*overall[2],overall[0],d0]]
        print(rlt['nfev'], overall[0], overall[1], 'total_time : ',len(overall[1])*overall[2])
        if rlt['fun'] < 0.01 or lenn==1 : 
            action = False

        # Create DataFrame and append to CSV file
        df = pd.DataFrame(output, columns=["Case", 'gate lenght', 'Theta', 'Phi', 'dt', 'combination', 'total time', 'fidelity', 'detunning'])
        df.to_csv(filename, mode='a', header=False, index=False)

    print(round((t+1)/count*100),"%") 



time_end = time.time()

print("#success# \nTime : " + str(time_end - time_start)) 