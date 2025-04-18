# 현재 문제점 : 4번의 중점적으로 탐색해서 답이 나오면, 연산시간이 빠른데, 거기서 답을 못찾으면 연산시간이 길어지는 것임.
# 처음 선택에 대한 힌트를 주는 것이 현명할 듯함
# openlist의 길이도 문제임
# 아니면, openlist에 무작위성 or 규칙을 부여해야함.
# 위상에 대한 휴리스틱을 세게 걸어주면 해결될 것 같은데, 위험함. -> dt에 민감하고, 오히려 total_time이 길어질 수 있음.
from math import *
import numpy as np
from scipy import linalg
from scipy.linalg import expm
from qutip import *
import random
from datetime import datetime
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power
from sklearn.feature_extraction.text import CountVectorizer
T_start = time.time()
# complex number
j = (-1)**0.5

# pauli matrix
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -j], [j, 0]])
sz = np.array([[1, 0], [0, -1]])
s0 = np.array([[1, 0], [0, 1]])

# parameters
d0 = 0.15
v0 = 0.02
dt = 3

# rotation operater
def Rx(theta):
    return np.matrix([[cos(theta/2),     -1j*sin(theta/2)],
                    [-1j*sin(theta/2),     cos(theta/2)]])

def Rz(phi): 
    return np.matrix([[cos(phi/2)-1j*sin(phi/2),       0],
                     [0,                          cos(phi/2)+1j*sin(phi/2)]])

# dt와 choice를 통한 unitary gate 만드는 함수
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

# fidelity 계산. 0.99이상이면 일치하는 것으로 간주.
def state_fidelity(rho_1, rho_2): 
        if np.shape(rho_1) != np.shape(rho_2):
            print("Dimensions of two states do not match.")
            return 0
        else:
            sqrt_rho_1 = fractional_matrix_power(rho_1, 1 / 2)
            fidelity = np.trace(fractional_matrix_power(sqrt_rho_1 @ rho_2 @ sqrt_rho_1, 1 / 2)) ** 2
            return np.real(fidelity)

# 현재 상태의 phase정보를 계산하는 함수
def phase_cal(x,y) :
    if y >= 0 and x >= 0:
        phase = atan(y/x)
    elif y>0 and x<0 :
        phase = atan(-x/y) + pi/2
    elif y<0 and x<0 :
        phase = atan(y/x) + pi
    else : 
        phase = atan(-x/y) +3/2*pi
    return phase

class Node:
    def __init__(self, parent=None, density_matrix=None, past_pulse = None):
        self.parent = parent
        self.density_matrix = np.matrix(density_matrix)
        self.past_pulse = past_pulse

        self.x = np.trace(self.density_matrix*sx).real
        self.y = np.trace(self.density_matrix*sy).real
        self.z = np.trace(self.density_matrix*sz).real
        self.phase = phase_cal(self.x,self.y)

        self.point = (self.x,self.y,self.z)
        
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.density_matrix == other.density_matrix
    
# QSL을 통한, heuristic 추정
def heuristic(node,target):
    theta = abs((acos(target.z)-acos(node.z)) / 0.0365)
    # phase = target.phase - node.phase+0.1726*theta
    # if phase < 0 :
    #     phase+=2*pi
    # phase%=2*pi
    # print(phase)
    # phi = phase/0.3
    phase = target.phase - node.phase+0.1726*theta
    if phase < 0 :
        phase+=2*pi
    phi = phase*0.1
    return theta + phi

def Astar_Qsearch(init,target) :
    # 불필요한 연산 줄이기 위해 unitary operator 미리 만들어 둠.
    choiceList = []
    for i in range(5) :
        choiceList.append(unitary(dt,i))

    # openList,closedList 초기화
    maximum_len = 50
    openList = []
    closedList = []

    # openList에 initial node 추가
    openList.append(init)

    # fidelity 0.99 만족할 때까지 실행
    count = 0
    while openList :
        
        if count < 10 :
            fid = 0.9999
        else :
            fid = 0.99        
        # current 지정 (openList의 가장 앞에 있는 노드)
        currentNode = openList[0]
        currentIdx = 0
        
        # # 이미 같은 노드가 openList안에 있고, f값이 더 크다면 current node를 openList 안에 있는 것으로 교체한다.
        # # 하지만, 지금 수행하는 길찾기에서는 node가 같은 확률이 0에 수렴한다. 
        # # 따라서 fidelity가 충분히 아주아주 높다면 대체하는 식으로 하는 것이 유리할 듯하다.
        # # 지금은 안쓰는 부분
        for index, item in enumerate(openList):
            if item.f < currentNode.f:
                currentNode = item
                currentIdx = index
        
        # openList에서 제거하고 closedList에 추가
        openList.pop(currentIdx)
        closedList.append(currentNode)
        # print(state_fidelity(currentNode.density_matrix, target.density_matrix),currentNode.f,currentNode.g,currentNode.h)
        if state_fidelity(currentNode.density_matrix, target.density_matrix) > 0.99:
            path = []
            current = currentNode
            while current is not None :
                path.append(current.past_pulse)
                current = current.parent
            path = path[:len(path)-1]
            path = path[::-1]
            return path

        children = []   
        i = 0 
        for rotation in choiceList :
            NewDensityMatrix = rotation @ currentNode.density_matrix @ rotation.conj().T
            new_node = Node(currentNode,NewDensityMatrix,i)
            i+=1
            children.append(new_node)
        
        for child in children : 
            # # 자식이 closedList에 있으면 continue
            # # 그러나 마찬가지로, 구면에서는 겹칠 일이 거의 없다.
            # # 현재는 사용하지 않겠지만, 나중에 아주아주 fidelity가 유사하다면, 대체하는 형태로 코드 수정예정
            # if child in closedList:
            #     continue
            value = True
            for closedNode in closedList :
                if state_fidelity(child.density_matrix,closedNode.density_matrix) > fid :
                    value = False
                    break
            if value == False :
                continue
            child.g = currentNode.g + dt
            child.h = heuristic(child,target)
            child.f = child.g + child.h
        
            # # 자식이 openList에 있고, g값이 더 크면 continue
            # # 이또한 나중에 추가해야할 파트
            if len([openNode for openNode in openList
                    if state_fidelity(child.density_matrix,openNode.density_matrix)>0.999 and child.g > openNode.g]) > 0:
                continue

            openList.append(child)
        # print(len(openList))
        # print(len(closedList))
        count+=1
        if len(openList) > maximum_len :
            openList = openList[:maximum_len]

def main(target_theta,target_phi) :
    start_time = time.time()
    # initial state
    init_wave = np.array([[1],[0]])
    irho_init = np.kron(init_wave,init_wave.conj().T)
    # target state
    target_U = Rz(target_phi) @ Rx(target_theta) 
    irho_target = target_U @ irho_init @ target_U.conj().T
    # path finding
    start = Node(None,irho_init,None)
    end = Node(None,irho_target,None)
    # print(end.x,end.y,end.z)
    # print(end.density_matrix)
    path = Astar_Qsearch(start,end)
    end_time = time.time()
    computing_time = end_time - start_time
    total_time = dt*len(path)
    print(f"""theta = {target_phi}
path : {path}
total_time : {total_time}
computing_time : {computing_time}
""")
    return path, computing_time

# CSV file name setup
date = datetime.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
filename = "/Users/qwon/Documents/DatasetByAstar/ByAstar_" + printdate + '.csv'

# Create an empty DataFrame and write to CSV file
df = pd.DataFrame(columns=["Case", 'gate lenght', 'Theta', 'Phi', 'dt', 'combination', 'total time','computing time'])
df.to_csv(filename, index=False)

for i in range(36) : 
    theta = pi/2
    phi = 2*pi/36*i
    path,computing_time = main(theta,phi)
    output = [['case'+str(i+1),len(path),theta, phi, dt, path, len(path)*dt,computing_time]]
    # Create DataFrame and append to CSV file
    df = pd.DataFrame(output, columns=["Case", 'gate lenght', 'Theta', 'Phi', 'dt', 'combination', 'total time','computing time'])
    df.to_csv(filename, mode='a', header=False, index=False)


T_end = time.time()

print("#success# \nTime : " + str(T_end - T_start)) 