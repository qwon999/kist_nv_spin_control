# %%
# NV Single Qubit에서 원하는 target state로 가기위한 theta와 phi를 계산하는 알고리즘

# 본코드는 VSCODE 에디터를 사용하여 Jupyter확장으로 작성하였습니다 By bin 2022.08.31
# 모든 변수들은 함수선언(def)으로 되어있는데 이는 여러가지 알고리즘을 실험하기 위함입니다.
# 아래의 주석처리에 따라 target state를 superposition과 random한 state중 한가지를 고를 수 있습니다.

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
from mpl_toolkits.mplot3d import Axes3D
start = time.time()
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
def Splus():
    Splus = np.array([[ 0, 1],
               [ 0, 0]])
    return Splus
def Sminus():
    Sminus = np.array([[ 0, 0],
               [ 1, 0]])
    return Sminus

def SD():
    SD = np.array([[ 1, 0],
               [ 0,-1j]])
    return SD

def SU():
    SU = np.array([[ 1, 0],
               [ 0,1j]])
    return SU

def ST():
    ST = np.array([[ 1, 1],
               [ 1,-1]])
    return ST

def SH():
    SH = np.array([[ 1, 1j],
               [ 1j, 1]])
    return SH

###2 Rotation operator(X-gate&Z-gate 2X2 matrices)

#X-gate는 Rx(theta)로 표현하고 Z-gate는 Rz(phi)로 표현합니다.
def Rx(theta):
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])
#혹시라도 에러가 나면 사용(허수계산이라 날수도 있음)

def Rz(phi):
    return np.matrix([[cos(phi/2)-1j*sin(phi/2), 0],
                     [0, cos(phi/2)+1j*sin(phi/2)]])
                     
def Rz2(phi):
    return np.matrix([[e**(-1j*phi/2),       0],
                     [0,                          e**(1j*phi/2)]])


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
###5 실행6

color = ['r','g','b','y','m','c','k','w']
colort = random.choice(color)


# theta 먼저 구하고 phi 구하는 방법 찾아보기

def problem3(deg):
    mc = init()*init().T                                        # |vector><vector|
    gates = Rz(deg[1])@Rx(deg[0])                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates@mc@gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    # i_m = np.trace(rho_measure*I())                          # Identity projection
    #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    # i_id = np.trace(idden*I())                               # target state의 Identity projection
    print('x_m:',x_m,'y_m:',y_m,'z_m:',z_m)
    print('x_id:',x_id,'y_id:',y_id,'z_id:',z_id)
    cost = (float(x_m-x_id)**2 + float(y_m-y_id)**2 + float(z_m-z_id)**2)**(1/2)    # 실험값과 이론값의 비교 costfunction 반환
    #cost = np.abs(x_m-x_id) + np.abs(y_m-y_id) + np.abs(z_m-z_id)
    #print(rho_measure)
    ax.plot(deg[0],deg[1],cost)
    # ax.tricontour(deg[0],deg[1],cost,c=color[0])
    plt.pause(0.001)
    # plt.scatter(deg[1],deg[0])
    # plt.pause(0.001)
    return cost


def problem4(deg):
    mc = init()*init().T                                        # |vector><vector|
    gates = Rz2(deg[1])@Rx(deg[0])                     # Universal Gate
    #rho_measure는 계산값(측정값)
    rho_measure = gates@mc@gates.getH()                         # Gate|vector><vector|Gate
    x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    # i_m = np.trace(rho_measure*I())                          # Identity projection
    #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
    x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    # i_id = np.trace(idden*I())                               # target state의 Identity projection
    print('x_m:',x_m,'y_m:',y_m,'z_m:',z_m)
    print('x_id:',x_id,'y_id:',y_id,'z_id:',z_id)
    cost = (float(x_m-x_id)**2 + float(y_m-y_id)**2 + float(z_m-z_id)**2)**(1/2)    # 실험값과 이론값의 비교 costfunction 반환
    #cost = np.abs(x_m-x_id) + np.abs(y_m-y_id) + np.abs(z_m-z_id)
    #print(rho_measure)
    ax.plot(deg[0],deg[1],cost,'ro', color = colort)
    # ax.tricontour(deg[0],deg[1],cost,c=color[0])
    plt.pause(0.001)
    # plt.scatter(deg[1],deg[0])
    # plt.pause(0.001)
    return cost

def degree(theta, phi):
    fx = Rx(theta)
    fz = Rz(phi)
    func = fz@fx
    mc = init()*init().T
    out = func@mc@func.getH()

    return out

def degree(theta, phi):
    fx = Rx(theta)
    fz = Rz2(phi)
    func = fz@fx
    mc = init()*init().T
    out = func@mc@func.getH()

    return out

###6 결과 출력
#https://docs.scipy.org/doc/scipy/reference/optimize.html
###아래는 3가지의 옵티마이저를 사용하여 탐색 가능###

#output: 측정 값은 csv 파일에 저장하기 위한 리스트
#date: 현재 시간을 저장하기 위한 변수 파일 이름 지정에 사용
date = dt.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
output1 = []
output2 = []
output3 = []
datapack = []
output4 = []
#pip = [["Density Matrix", nan], [nan, "Matrix"]]
#최적화된 값의 변화가 없을 때 까지 작업을 반복한다.




fail = 0                                                    #최적화 실패 횟수
success = 0                                                 #최적화 성공 횟수

### x, y, z projection 저장

standard = 0.2                                              #최적화 정도의 기준 설정
min_stad = 1*e-10                                           #최적화 정도의 최소값 설정
count = 5
seccount = 3
vastand = 0.33

bounds = [(0, 2 * pi),(0, pi)] 
constraints = ({'type': 'eq', 'fun': lambda x:  np.array([x[0] + x[1] - np.pi])}) #제약조건 설정


# b1 = qutip.Bloch()
# b1.make_sphere()

# b2 = qutip.Bloch()
# b2.make_sphere()

colorset = ['r','g','b','y','c','m','k']

for x in range(count):                                         #반복 횟수 지정
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    idden = rand_dm_ginibre(2, rank=1)
    colort = random.choice(colorset)
    ideal = []
    ideal = [np.trace(idden*Sx()),np.trace(idden*Sy()),np.trace(idden*Sz())] #target state의 x,y,z projection을 저장
    xx = 1e-8
    ff = 1e-8
    minx = 10
    minz = 10
    mincost = 10
    minx2 = 10
    minz2 = 10
    mincost2 = 10
    start = time.time()                                 #시간 측정 시작  
    for m in range(20):
        for n in range(40):
            mc = init()*init().T                                        # |vector><vector|
            gates = np.matmul(Rz(m * 0.16), Rx(n * 0.16))                     # Universal Gate
            #rho_measure는 계산값(측정값)
            rho_measure = gates@mc@gates.getH()                         # Gate|vector><vector|Gate
            x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
            y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
            z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
            # i_m = np.trace(rho_measure*I())                          # Identity projection
            #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
            x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
            y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
            z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
            # i_id = np.trace(idden*I())                               # target state의 Identity projection
            # print('x_m:',x_m,'y_m:',y_m,'z_m:',z_m)
            # print('x_id:',x_id,'y_id:',y_id,'z_id:',z_id)
            # print(rho_measure)
            cost = (float(x_m-x_id)**2 + float(y_m-y_id)**2 + float(z_m-z_id)**2)**(1/2)    # 실험값과 이론값의 비교 costfunction 반환
            if(cost < mincost):
                mincost = cost
                minx = n * 0.2
                minz = m * 0.2
            #cost = np.abs(x_m-x_id) + np.abs(y_m-y_id) + np.abs(z_m-z_id)
            #print(rho_measure)
            ax.plot(n * 0.16 ,m * 0.16,cost,'r', color = colort, markersize=3)
            # ax.tricontour(deg[0],deg[1],cost,c=color[0])
            plt.pause(0.001)
    # ax.set_xlabel('Theta')
    # ax.set_ylabel('Phi')
    # ax.set_zlabel('Error')
    # ax.set_xlim(0, 2*pi)
    # ax.set_ylim(0, pi)
    # plt.show(block=False)
    # plt.pause(2)
    # fignum = x + 1
    # plt.savefig("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Test1/Figure/Figure_" + printdate + "_" + str(fignum) + '.png')
    # plt.close()
    # for m in range(160):
    #     for n in range(160):
    #         mc = init()*init().T                                        # |vector><vector|
    #         gates = Rz2(m * 0.02)@Rx(n * 0.04)                     # Universal Gate
    #         #rho_measure는 계산값(측정값)
    #         rho_measure = gates@mc@gates.getH()                         # Gate|vector><vector|Gate
    #         x_m = np.trace(rho_measure*Sx())                            # Sigma X projection
    #         y_m = np.trace(rho_measure*Sy())                            # Sigma Y projection
    #         z_m = np.trace(rho_measure*Sz())                            # Sigma Z projection
    #         # i_m = np.trace(rho_measure*I())                          # Identity projection
    #         #x_id,y_id,z_id는 주어진 target state를 계산해낸 값(이론값)
    #         x_id = np.trace(idden*Sx())                                 # target state의 Sigma X projection
    #         y_id = np.trace(idden*Sy())                                 # target state의 Sigma Y projection
    #         z_id = np.trace(idden*Sz())                                 # target state의 Sigma Z projection
    #         # i_id = np.trace(idden*I())                               # target state의 Identity projection
    #         # print('x_m:',x_m,'y_m:',y_m,'z_m:',z_m)
    #         # print('x_id:',x_id,'y_id:',y_id,'z_id:',z_id)
    #         cost2 = (float(x_m-x_id)**2 + float(y_m-y_id)**2 + float(z_m-z_id)**2)**(1/2)    # 실험값과 이론값의 비교 costfunction 반환
    #         if(cost2 < mincost2):
    #             mincost2 = cost2
    #             minx2 = n * 0.2
    #             minz2 = m * 0.2
    #         #cost = np.abs(x_m-x_id) + np.abs(y_m-y_id) + np.abs(z_m-z_id)
    #         #print(rho_measure)
    #         # ax.plot(n * 0.2 ,m * 0.2,cost,'ro', color = colort)
    #         # # ax.tricontour(deg[0],deg[1],cost,c=color[0])
    #         # plt.pause(0.001)
    
    # result4 = scipy.optimize.differential_evolution(problem3, bounds, strategy='best1bin', mutation=(0.5, 1), recombination=0.7, init='latinhypercube', atol=0, updating='immediate',workers=1) #differential evolution 최적화
    end = time.time()
    final = end - start
    ax.set_xlabel('Theta')
    ax.set_ylabel('Phi')
    ax.set_zlabel('Error')
    # plt.figure(facecolor='w')
    plt.title(idden)
    plt.show(block=False)
    plt.pause(2)
    fignum = x + 1
    plt.savefig("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Test1/Figure/Figure_" + printdate + "_" + str(fignum) + '.png')
    plt.close()
    print("Case" + str(x + 1) + " Clear   " + str(final))
    print("minx:",minx,"minz:",minz,"mincost:",mincost)
    # print("minx2:",minx2,"minz2:",minz2,"mincost2:",mincost2)
    if(mincost < vastand):
        success = success + 1
    # for y in range(seccount):                                     #측정 횟수 지정 같은 작업을 여러번 진행할 경우를 대비하여 반복문 사용
    #     var = 1
    #     deg = [(np.pi/180)*random.uniform(0,360), (np.pi/180)*random.uniform(0,180)] #초기값을 넣는 랜덤변수
    #     result4 = scipy.optimize.minimize(problem3,deg,bounds=bounds, constraints = constraints, method="Powell", options = {'maxiter' : 10, 'return_all' : True})                        #Powell 최적화
    #     colort = random.choice(colorset)
    #     #result4 = scipy.optimize.differential_evolution(problem3, bounds, strategy='best1bin', mutation=(0.5, 1), recombination=0.7, init='latinhypercube', atol=0, updating='immediate',workers=1) #differential evolution 최적화
    #     #result4 = scipy.optimize.minimize(problem3,deg,bounds=bounds, method="Powell", options = {'maxiter' : 10, 'return_all' : True})                        #Powell 최적화
    #     deft4 = degree(result4['x'][0], result4['x'][1])     #최적화된 값으로 density matrix를 생성
    #     deftl4 = [np.trace(deft4*Sx()),np.trace(deft4*Sy()),np.trace(deft4*Sz())] #최적화된 density matrix의 x,y,z projection을 저장
    #     car4 = ((idden.data[0, 0] - deft4[0, 0])**2 + (idden.data[0, 1] - deft4[0, 1])**2 + (idden.data[1, 1] - deft4[1, 1])**2)**(1/2) #최적화 정도 측정
    #     var4 = ((ideal[0] - deftl4[0])**2 + (ideal[1] - deftl4[1])**2 + (ideal[2] - deftl4[2])**2)**(1/2)
    #     # print(result4)
    #     print("var: " + str(var4))
    #     print("deg: " + str(result4['x']))
    #     if(var4 < var):
    #         theMin = 4
    #         result = result4
    #         deft = deft4
    #         deftl = deftl4
    #         car = car4
    #         var = var4
        
    #     if(var < vastand):
    #         end = time.time()
    #         final = end - start
    #         success = success + 1
    #         output1.append(["Case" + str(x + 1), "Method" + str(theMin), result['x'], final, deft, car, idden.data, ideal, deftl, var])
    #         ax.set_xlabel('Theta')
    #         ax.set_ylabel('Phi')
    #         ax.set_zlabel('Error')
    #         ax.set_xlim(0, 2*pi)
    #         ax.set_ylim(0, pi)
    #         plt.show(block=False)
    #         plt.pause(2)
    #         fignum = x + 1
    #         plt.savefig("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Test1/Figure/Figure_" + printdate + "_" + str(fignum) + '.png')
    #         plt.close()
    #         print("Case" + str(x + 1) + " Clear   " + str(final))


    #         break
    
    #     if(y == seccount - 1):
    #         colort = random.choice(colorset)
    #         # result4 = scipy.optimize.differential_evolution(problem3, bounds, strategy='best1bin', mutation=(0.5, 1), recombination=0.7, init='latinhypercube', atol=0, updating='immediate',workers=1) #differential evolution 최적화
    #         end = time.time()
    #         final = end - start
    #         fail = fail + 1
    #         output1.append(["Case" + str(x + 1), "Fail" + str(theMin), result['x'], final, deft, car, idden.data, ideal, deftl, var])
    #         ax.set_xlabel('Theta')
    #         ax.set_ylabel('Phi')
    #         ax.set_zlabel('Error')
    #         ax.set_xlim(0, 2*pi)
    #         ax.set_ylim(0, pi)
    #         plt.show(block=False)
    #         plt.pause(2)
    #         fignum = x + 1
    #         plt.savefig("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Test1/Figure/Figure_" + printdate + "_" + str(fignum) + '.png')
    #         plt.close()
    #         print("Case" + str(x + 1) + " Clear   " + str(final))


    #         break

                                                   #측정이 끝난 경우 출력
    
print("Success : " + str(success) + "/" + str(count))                                                #측정 실패한 경우 출력
fin1 = pd.DataFrame(output1)
fin1.rename(columns={0:"Case", 1:"Used Algorithm", 2:'Theta, Phi', 3: 'time', 4: 'matrix', 5: "degree", 6: "Density Matrix", 7: "Projection", 8: "Projection"}, inplace=True)
fin1.to_csv("./Result_" + printdate + '.csv', index=false)
print(date)                                                                             #측정이 끝난 시간 출력

#%%

# finalOutput = []
# finalOutput = output1 + output2 + output3
# output1[0][0] = "Nelder-Mead"
# output2[0][0] = "Powell"
# output3[0][0] = "COBYLA"
#print(idden)

#print(date)

# fin1 = pd.DataFrame(output1)
# fin1.rename(columns={0:'Nelder-Mead', 1: 'time', 2: 'matrix', 3: "degree"}, inplace=True)
# fin1.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/Nelder_Mead_result_" + printdate + '.csv', index=false)
# fin2 = pd.DataFrame(output2)
# fin2.rename(columns={0:'Powell', 1: 'time', 2: 'matrix', 3: "degree"}, inplace=True)
# fin2.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/Powell_result_" + printdate + '.csv', index=false)
# fin3 = pd.DataFrame(output3)
# fin3.rename(columns={0:'differential_evolution', 1: 'time', 2: 'matrix', 3: "degree"}, inplace=True)
# fin3.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/differential_evolution_result_" + printdate + '.csv', index=false)
# fin4 = pd.DataFrame(output4)
# fin4.rename(columns={0:'dual_annealing', 1: 'time', 2: 'matrix', 3: "degree"}, inplace=True)
# fin4.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/dual_annealing_result_" + printdate + '.csv', index=false)
# pack = pd.DataFrame(datapack)
# # fin4 = pd.DataFrame(output4)
# pack.rename(columns={0:'Density Matrix'}, inplace=True)

# fin = pd.concat([fin1, fin2, fin3, fin4, pack], axis=1)
# fin.to_csv("C:/Users/Administrator/2023.01.01/KIST_intern/Task1/Control_Nuclear_Spins/NVspin/UNITE/Local/Total_result_" + printdate + '.csv', index=false)

#fin.to_csv('Powell_result_' + printdate + '.csv', index=false)

###7 결과 분석

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


# %%
'''
result1 = scipy.optimize.differential_evolution(problem,bounds,atol=0.00001)
print("result1:")
print(result1) #결과 출력
print("/////////////////////////////////////////////////////////////////")

result11 = scipy.optimize.dual_annealing(problem,bounds)
print("result11:")
print(result11) #결과 출력
print("/////////////////////////////////////////////////////////////////")

result2 = scipy.optimize.dual_annealing(problem,bounds,maxiter=100)
print("result2:")
print(result2) #결과 출력
print("/////////////////////////////////////////////////////////////////")
'''

# %%
'''
result0 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.01,'ftol':0.01})
print("result0:")
print(result0) #결과 출력
output.append(result0)
print("/////////////////////////////////////////////////////////////////")

result1 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.001,'ftol':0.001})
print("result1:")
print(result1) #결과 출력
output.append(result1)
print("/////////////////////////////////////////////////////////////////")

result2 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.001,'ftol':0.001})
print("result2:")
print(result2) #결과 출력
output.append(result2)
print("/////////////////////////////////////////////////////////////////")

result3 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.0001,'ftol':0.0001})
print("result3:")
print(result3) #결과 출력
output.append(result3)
print("/////////////////////////////////////////////////////////////////")

result4 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.00001,'ftol':0.00001})
print("result4:")
print(result4) #결과 출력
output.append(result4)
print("/////////////////////////////////////////////////////////////////")

result5 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.000001,'ftol':0.000001})
print("result5:")
print(result5) #결과 출력
output.append(result5)
print("/////////////////////////////////////////////////////////////////")

result6 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.0000001,'ftol':0.0000001})
print("result6:")
print(result6) #결과 출력
output.append(result6)
print("/////////////////////////////////////////////////////////////////")

result7 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.00000001,'ftol':0.00000001})
print("result7:")
print(result7) #결과 출력
output.append(result7)
print("/////////////////////////////////////////////////////////////////")

result8 = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.000000001,'ftol':0.000000001})
print("result8:")
print(result8) #결과 출력
output.append(result8)
print("/////////////////////////////////////////////////////////////////")
'''

#%%
'''
    idden = rand_dm(2, density=1)
    # data = []
    # data = idden.data
    start = time.time()
    result1 = scipy.optimize.minimize(problem,deg,bounds=bounds,method="Nelder-Mead")
    end = time.time()
    final = end - start
    deft = degree(result1['x'][0], result1['x'][1]) #최적화된 값으로 density matrix를 구한다.
    # print(idden[0][0])
    # print(deft[0][0])
    
    # deft1 = deft[0][0].tolist()
    # deft2 = deft[0][1].tolist()
    # deft3 = deft[1][0].tolist()
    # deft4 = deft[1][1].tolist()
    #find the distance between the target state(idden) and the result state(deft)
    # print(deft1)
    # print(deft2)
    # print(deft3)
    # print(deft4)
    # car = ((deft[0, 0])**2 + (deft[0, 1])**2 + (deft[1, 0])**2 + (deft[1, 1])**2)**(1/2)
    # car2 = ((idden[0][0])**2 + (idden[0][1])**2 + (idden[1][0])**2 + (idden[1][1])**2)**(1/2)
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    #print(car)
    output1.append([result1['x'], final, deft, car])
    
    start = time.time()
    result2 = scipy.optimize.minimize(problem,deg,bounds=bounds,method="Powell")
    end = time.time()
    final = end - start
    deft = degree(result2['x'][0], result2['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    output2.append([result2['x'], final, deft, car])
    
    start = time.time()
    result3 = scipy.optimize.differential_evolution(problem,bounds=bounds)
    end = time.time()
    final = end - start
    deft = degree(result3['x'][0], result3['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    output3.append([result3['x'], final, deft, car])
    
    start = time.time()
    result4 = scipy.optimize.dual_annealing(problem,bounds=bounds)
    end = time.time()
    final = end - start
    deft = degree(result4['x'][0], result4['x'][1])
    car = ((idden.data[0, 0] - deft[0, 0])**2 + (idden.data[0, 1] - deft[0, 1])**2 + (idden.data[1, 1] - deft[1, 1])**2)**(1/2)
    output4.append([result4['x'], final, deft, car])
    #print(idden.data)
    datapack.append(idden.data) 
    
    # result = scipy.optimize.minimize(problem,deg,bounds=bounds,method='Powell',options={'xtol':0.0001,'ftol':0.00001})  
    # output4.append([result['x'], result['fun']])
    
    print("test case" + str(x + 1) + " clear")
'''