# %%
# NV Single Qubit에서 원하는 target state로 가기위한 theta와 phi를 계산하는 알고리즘

# 본코드는 VSCODE 에디터를 사용하여 Jupyter확장으로 작성하였습니다 By bin 2022.08.31
# 본코드는 VSCODE 에디터를 사용하여 Python으로 작성하였습니다 By Dogyeom 2023.01.26
# 모든 변수들은 함수선언(def)으로 되어있는데 이는 여러가지 알고리즘을 실험하기 위함입니다.
# count 횟수만큼 무작위 Pure density matrix를 생성하여 cost function을 계산합니다.
# idden 변수를 수정하여 target state를 변경할 수 있습니다.
# Powell 최적화 알고리즘을 통해 cost function을 최소화하는 theta와 phi를 구합니다.
# Problem 함수를 수정하여 cost function을 변경할 수 있습니다.
# 본 코드는 시간과 노이즈에 따른 보정을 테스트하기 위해 작성되었습니다. By Dogyeom 2023.04.12

import numpy as np
from qutip import *
from sympy import *
from math import *
import scipy
from scipy import optimize
import random
import sympy as sp
from numpy import ndarray
import pandas as pd
from datetime import datetime as dt                         # 시간을 출력하기 위한 라이브러리  
import math
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm import trange
from scipy.linalg import fractional_matrix_power

# %%
###1 Pauli Matrices(2X2 matrices)               

def I():
    I  = np.array([[ 1, 0],
               [ 0, 1]])
    return I
def Sx():
    Sx = np.array([[ 0, 1],
               [ 1, 0]])
    return Sx
def Sy():
    Sy = np.array([[ 0,-1j],
               [1j, 0]])
    return Sy
def Sz():
    Sz = np.array([[ 1, 0],
               [ 0,-1]])
    return Sz

###2 Rotation operator(X-gate&Z-gate 2X2 matrices)

#X-gate는 Rx(theta)로 표현하고 Z-gate는 Rz(phi)로 표현합니다.
def Rx(theta):
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])

def Rz(phi):
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])


###3 init & density matrix

#initial state |0> 으로 가정한 상태
# |0> = |vector> = |a> = |1> = |up> = |+> = |H> = |R> = |L>

def init():
    init = np.matrix([[1],[0]])
    return init

#state a|0> + b|1> 계수를 괄호안에 입력하여 density 구하는 함수
#density matrix는 2X2 matrix로 표현됩니다.
def todensity (a,b):
    UU=np.array([[a],[b]])
    D = UU@(UU.conj().T)
    return D

idden = []


###4 실행

#problem(cost function)
#cost function은 target state와 계산값의 차이를 계산합니다.
# %%
# def problem(deg):
#     mc = init()*init().T                                        # |vector><vector|
#     gates = np.inner(Rz(deg[1]),Rx(deg[0]))                     # Universal Gate
#     #rho_measure는 계산값(측정값)
#     rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
#     x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
#     y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
#     z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
#     i_m = np.trace(rho_measure*I())                          # Identity projection
#     #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
#     x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
#     y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
#     z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
#     i_id = np.trace(idden*I())                               # target state의 Identity projection
#     # 실험값과 이론값의 비교 costfunction 반환
#     cost = np.abs(x_m-x_id) + np.abs(y_m-y_id) + np.abs(z_m-z_id) + np.abs(i_m-i_id)
#     return cost
# %%

change_weight = 0.5


def state_fidelity(rho_1, rho_2): #fidelity
        if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)


def makeNoise(array):
    arral = [np.trace(array*Sx()), np.trace(array*Sy()), np.trace(array*Sz())]
    # ns = (1 + random.uniform(-0.1, 0.1))
    # np.random.seed(seed=100)
    # ns = np.random.poisson(lam=25, size=1)/25
    ns = (1 + random.uniform(-0.4, 0.4))
    # print(ns)
    arre = np.zeros(3, dtype = 'complex_')
    arre[0] = arral[0] * ns
    sumarr = ((arre[0]) **2 + (arral[1]) **2 + (arral[2]) **2) ** (1/2)
    arre[0] = arre[0] / sumarr
    arre[1] = arral[1] / sumarr
    arre[2] = arral[2] / sumarr
    
    return arre


trace_time = [0, 5]
def problem(deg):
    mc = init()*init().T                                        # |vector><vector|
    timeErr = (deg[0] + deg[1]) * change_weight
    
    gates = np.inner(Rz(deg[1] + timeErr),Rx(deg[0]))                     # Universal Gate
    
    # timeCost = deg[2]                                           # timeCost는 시간에 따른 오차를 보정하기 위한 변수입니다.
    # timeCost = deg[1] + deg[2]
    # timeErr = timeCost * 0.1
    #rho_measure는 계산값(측정값)
    rho_measure = gates*mc*gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    # i_m = np.trace(rho_measure*I())                          # Identity projection
    #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
    # x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    # y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    # z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    x_id = noisy[0]
    y_id = noisy[1]
    z_id = noisy[2]
    # i_id = np.trace(idden*I())                               # target state의 Identity projection
    # 실험값과 이론값의 비교 costfunction 반환
    cost = np.abs(x_m-x_id) + np.abs(y_m-y_id) + np.abs(z_m-z_id)
    if(cost < trace_time[1]):
        trace_time[1] = cost
        trace_time[0] = timeErr
    return cost


bounds = [(0, pi),(0,2*pi)]                                     #theta와 phi의 범위
deg = [(np.pi/180)*random.uniform(0,180),(np.pi/180)*random.uniform(0,360)]


#최적화 정도 측정
#idden과의 비교를 위하여 density matrix로 변환해주는 함수
def degree(theta, phi):
    fx = Rx(theta)
    fz = Rz(phi)
    func = fz * fx
    mc = init()*init().T
    out = func*mc*func.getH()

    return out

###5 결과 출력
#https://docs.scipy.org/doc/scipy/reference/optimize.html

#output: 측정 값은 csv 파일에 저장하기 위한 리스트
#date: 현재 시간을 저장하기 위한 변수 파일 이름 지정에 사용
date = dt.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
output1 = []
datapack = []
output2 = []
fail = 0                                                       #최적화 실패 횟수
success = 0                                                    #최적화 성공 횟수

standard = 0.2                                                 #최적화 정도의 기준 설정
min_stad = 1*e-10                                              #최적화 정도의 최소값 설정
count = 4                                                 #반복 횟수 지정
seccount = 25                                                  #측정 횟수 지정
vastand = 1*e-2                                                #최적화 정도의 기준 설정
repeat = 0                                                     #최적화 정도의 편차가 큰 경우 반복 횟수 지정
allstart = time.time()                                         #시간 측정 시작
for x in range(count):                                       #반복 횟수 지정
    trace_time = [0, 5]
    idden = rand_dm_ginibre(2, rank=1)
    
    ideal = []
    ideal = [np.trace(idden*Sx()),np.trace(idden*Sy()),np.trace(idden*Sz())] #target state의 x,y,z projection을 저장
    tol = 1*e-13
    start = time.time()                                         #시간 측정 시작  
    dat = idden.data.todense().tolist()                         #target state의 density matrix를 리스트로 변환
    tempx = 0
    tempy = 0
    tempz = 0
    tempTheta = 0
    tempPhi = 0
    for y in tqdm(range(seccount)):                                 #측정 횟수 지정 같은 작업을 여러번 진행할 경우를 대비하여 반복문 사용
        trace_time = [0, 5]
        repeat = repeat + 1
        deg = [(np.pi/180)*random.uniform(0,180),(np.pi/180)*random.uniform(0,360)]
                                                                #초기값을 넣는 랜덤변수
        noisy = makeNoise(idden).tolist()
        # result1 = scipy.optimize.minimize(problem,deg,bounds=bounds,method="Powell", options = {'xtol' : tol, 'ftol' : tol })        #Powell 최적화
        result1 = optimize.shgo(problem, bounds = bounds, iters = 8, options={'ftol': tol, 'xtol' : tol})
        fin_phi = result1['x'][1] + trace_time[0]
        deft1 = degree(result1['x'][0], fin_phi)        #최적화된 값으로 density matrix를 생성
        deftl1 = [np.trace(deft1*Sx()),np.trace(deft1*Sy()),np.trace(deft1*Sz())]
                                                                #최적화된 density matrix의 x,y,z projection을 저장
        tempx = tempx + deftl1[0]
        tempy = tempy + deftl1[1]
        tempz = tempz + deftl1[2]
        tempTheta = tempTheta + result1['x'][0]
        if(fin_phi > 2*pi):
            tempPhi = tempPhi + fin_phi - 2*pi
        else:
            tempPhi = tempPhi + fin_phi
        var1 = ((ideal[0] - deftl1[0])**2 + (ideal[1] - deftl1[1])**2 + (ideal[2] - deftl1[2])**2)**(1/2)

        end = time.time()                                   #시간 측정 종료
        final = end - start                                 #측정 시간 저장
        # output1.append(["Case" + str(x + 1), "Powell", result1['x'], ideal, deftl1, var1])                                #측정 값 저장
        temx = tempx / (y+1)
        temy = tempy / (y+1)
        temz = tempz / (y+1)
        temTh = tempTheta / (y+1)
        temPh = tempPhi / (y+1)
        err = np.abs(ideal[0] - temx) + np.abs(ideal[1] - temy) + np.abs(ideal[2] - temz)
        desit = degree(result1['x'][0], fin_phi)
        cost = state_fidelity(idden, desit)
        output1.append(["Case" + str(x + 1), result1['x'][0], result1['x'][1], trace_time[0], ideal[0], ideal[1], ideal[2], deftl1[0], deftl1[1], deftl1[2], noisy[0], noisy[1], noisy[2], temx , temy, temz, err , result1['fun'], temTh, temPh, cost])                                #측정 값 저장
        success = success + 1
    noisy = [temx, temy, temz]
    desit = degree(temTh, temPh)
    cost = state_fidelity(idden, desit)
    result = optimize.shgo(problem, bounds = bounds, iters = 8, options={'ftol': tol, 'xtol' : tol})
    output1.append(["Case" + str(x + 1) + "Fin", result['x'][0], result['x'][1], trace_time[0], ideal[0], ideal[1], ideal[2], deftl1[0], deftl1[1], deftl1[2], noisy[0], noisy[1], noisy[2], temx , temy, temz, err , result['fun'], temTh, temPh, cost]) 
    # print("Case" + str(x + 1) + " clear")                       #측정이 끝난 경우 출력
    
    print(cost)
    # print(1 - cost)

allend = time.time()                                           #시간 측정 종료
print("Success : " + str(success) + "/" + str(count))                                                #측정 성공한 경우 출력
print("repeat : " + str(repeat))                                                                     #측정 반복한 경우 출력
print("Time : " + str(allend - allstart))                                                            #측정 시간 출력
fin1 = pd.DataFrame(output1)
fin1.rename(columns={0:"Case", 1:'Theta', 2: 'Phi', 3: 'timeErr', 4: 'initX', 5: 'initY', 6: 'initZ', 7: 'traceX', 8: 'traceY', 9: 'traceZ', 10: 'noiseX', 11: 'noiseY', 12: 'noiseZ'}, inplace=True)
# fin1.rename(columns={0:"Case", 1:"Used Algorithm", 2:'Theta, Phi', 3: 'time', 4: 'matrix', 5: "degree", 6: "Density Matrix", 7: "Projection", 8: "Projection"}, inplace=True)
fin1.to_csv("C:/Users/Administrator/Dogyeom(2023.01.01)/KIST_intern/Task1/Control_Nuclear_Spins/NVspin_Noise/reData/Result_" + printdate + '.csv', index=false)
print(date)                                                      #측정이 끝난 시간 출력

# %%
###6 결과 분석

#direc = 출발 지점에서의 방향
#fun = x 위치에서의 함수의 값 -> 최적화 정도라고 생각
#message = 메시지 문자열
#nfev = 목적 함수 호출 횟수
#njev = 자코비안 계산 횟수
#nit = x 이동 횟수
#status = 종료 상태, 0이면 최적화 성공
#x = 최적화 해
#xtol = x의 허용 오차
#ftol = func(xopt)에서 허용되는 상대 오류의 수
#https://www.desmos.com/scientific?lang=ko
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
