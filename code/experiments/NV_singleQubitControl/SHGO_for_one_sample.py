# 슈고로 원하는 타겟 하나를 얻는 코드


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

target_theta,target_phi = pi/2,0
target_U = Rz(target_phi/0.02) @ Rx(target_theta/0.02) 
irho_target = target_U @ irho_init @ target_U.conj().T


def problem(vari):
    vari[1] = round(vari[1])
    gate=vari[1] 
    gate_choice = ten_to_three(gate)
    

    total_U = s0
    for i in gate_choice: #추후 속도 향상위해서, dt에 따른 instant_U 값 구해놓고 계산.
        instant_U = unitary(vari[0], i)
        total_U = instant_U @ total_U
    irho_final = total_U @ irho_init @total_U.conj().T
    
    F = state_fidelity(irho_target,irho_final)

    #if F > 0.95 : 

    # cost = 1-state_fidelity(target_U,total_U) + 0.0025*vari[0]*len(gate_choice)
    cost = 1 - F 
    print(vari[0],vari[1],gate_choice, cost)
    return cost

# for t in range(1) : 
#     vari = [1, 1]
#     bounds = [(1,10),(1,9999)]
#     integer_idx=[1]
#     integer_bounds = [(1,9999) for _ in integer_idx]
#     options = {'integer_variables': {'indices': integer_idx, 'bounds': integer_bounds},'xtol':1e-15,'ftol':1e-17 }
#     rlt = optimize.shgo(problem,bounds=bounds,iters=4,options=options)
#     print(rlt['x'],rlt['nfev'],rlt['fun'])

for t in range(1) : 
    vari = [1, 1]
    bounds = [(1,10),(1,999990)]
    rlt = optimize.shgo(problem,bounds=bounds,iters=7,options={'xtol':1e-15,'ftol':1e-15})
    rlt['x'][1] = round(rlt['x'][1]) #rounding
    print(rlt['x'],rlt['nfev'],rlt['fun'])