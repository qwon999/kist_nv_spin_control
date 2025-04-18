# temp 내적값 변화량을 통한 선택 고려
# 1.변화량,2.내적값 모든 조건 같을시 랜덤선택
# 외적값 normailization (외적의 방향만 중요함)
# 일단 1차적으로 코드 마무리 완료
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
d0 = 0.02
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


def state_fidelity(rho_1, rho_2): #fidelity
        if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)
        
norm = (d0**2+v0**2)**0.5
choice_0 = np.array([0,0,1])
choice_1 = np.array([v0/norm,0,d0/norm])
choice_2 = np.array([-v0/norm,0,d0/norm])
choice_3 = np.array([0,v0/norm,d0/norm])
choice_4 = np.array([0,-v0/norm,d0/norm])
rot_vec = [choice_0,choice_1,choice_2,choice_3,choice_4]

init_wave = np.array([[1],[0]])
irho_init = np.kron(init_wave,init_wave.conj().T)

target_theta, target_phi = pi/2, pi/4
target_U = Rz(target_phi) @ Rx(target_theta)
irho_target = target_U @ irho_init @target_U.conj().T

past_inner_product = np.zeros(5)
trace = [1,[],0]
def make_combination(dt) :
    F = 1
    irho_current = irho_init
    vec_t = np.array([np.trace(irho_target*sx).real, np.trace(irho_target*sy).real,np.trace(irho_target*sz).real])
    iter = 0
    combination = []
    while (F>0.01) and (iter<100):
        
        vec_c = np.array([np.trace(irho_current*sx).real,np.trace(irho_current*sy).real,np.trace(irho_current*sz).real])
        cross = np.cross(vec_c,vec_t)
        # print("1",cross)
        len_of_cross = cross[0]**2+cross[1]**2+cross[2]**2
        for k in range(3):
            cross[k] = cross[k]/len_of_cross
        # print("2",cross)
        inner_product = np.zeros(5)
        choice = -1
        for i in range(5):
            inner_product[i] = ((np.inner(rot_vec[i],cross)))

        

        delta = np.zeros(5)
        if iter > 0 :
            for j in range(5) :
                delta[j] = inner_product[j] - past_inner_product[j]
        past_inner_product = inner_product.copy()
        #print(delta)
        max_delta_value = max(delta)
        max_delta_indices = [j for j, v in enumerate(delta) if v==max_delta_value]
        # 첫번째에는 증가량을 따질 수 없으므로 가장 높은 내적값 중 랜덤으로 고른다.
        # 또한, 내적값이 가장 높은 축이 오직 하나라면, 그 값을 고른다.
        if iter == 0 :
            max_value = max(inner_product)
            max_indices = [i for i, v in enumerate(inner_product) if v==max_value]
            choice = random.choice(max_indices)
        elif len(max_delta_indices) == 1 :
            choice = random.choice(max_delta_indices)
        else : 
            for i in range(5):
                if i not in max_delta_indices :
                    inner_product[i] = -3
            max_value = max(inner_product)
            max_indices = [i for i, v in enumerate(inner_product) if v==max_value]
            choice = random.choice(max_indices)


        # 그렇지 않다면, 증가량을 비교한다. 증가량까지 같으면, 그 중에 랜덤선택


        instant_unitary = unitary(dt, choice)
        irho_current = instant_unitary @ irho_current @ instant_unitary.conj().T
        F = 1 - state_fidelity(irho_target,irho_current)
        combination.append(choice)
        # print(combination)
        iter+=1  
        past_inner_product = inner_product.copy()
        #print(past_inner_product)
    #print("\n")

    if trace[0] > F :
        trace[0] = F
        trace[1] = combination.copy()
        trace[2] = dt

    # print(trace[1],trace[2])
    return F


output = []
count = 5
for t in range(count) : 
    #target_theta,target_phi = (pi/180)*random.uniform(0,180), (pi/180)*random.uniform(0,360)
    target_theta,target_phi = pi/2,	0
    target_U = Rz(target_phi) @ Rx(target_theta) 
    irho_target = target_U @ irho_init @ target_U.conj().T
    lenn = 1
    action = True
    while action == True:
        trace = [1,[],0]
        vari = [10]
        bounds = [(1,15)]
        # integer_constraint = lambda x: np.all(np.mod(x, 1).astype(bool))  # ensure x is integer
        # constraints = [{'type': 'ineq', 'fun': integer_constraint}]
        rlt = optimize.shgo(make_combination,bounds=bounds,iters=6, options={'xtol':1e-15,'ftol':1e-15})
        output.append(['CASE'+str(t+1),len(trace[1]),target_theta,target_phi,trace[2],trace[1],rlt['fun'], rlt['nfev']])
        print(rlt['nfev'], trace[0], trace[1],lenn, 'dt : ',trace[2])
        
        if rlt['fun'] < 0.01 or lenn==1 : 
            action = False
        lenn+=1
        print(action)
    # output.append([])
    print(round((t+1)/count*100),"%")
date = datetime.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
print(date)
fin1 = pd.DataFrame(output)
fin1.rename(columns={0:"Case", 1:'gate lenght', 2:'Theta', 3: 'Phi', 4: 'dt', 5: 'combination', 6: 'cost', 7:'nfev'}, inplace=True)
fin1.to_csv("/Users/qwon/Documents/DataSetForNVSpin/BySHGOtemp10_for_Case21_" + printdate + '.csv', index=False)   


time_end = time.time()

print("#success# \nTime : " + str(time_end - time_start)) 