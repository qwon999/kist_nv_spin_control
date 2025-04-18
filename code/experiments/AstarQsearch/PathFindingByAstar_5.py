# cmath를 통한 허수문제 해결 -> 데이터 추출만 하면됨.
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
import heapq
import cmath

T_start = time.time()

# complex number
# j = (-1)**0.5


# pauli matrix 
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
s0 = np.array([[1, 0], [0, 1]])

# parameters
d0 = 0.15 # z축으로 발생하는 회전 속도임. 그런데 사실 이거에 두배인 0.30rad/마이크로초
v0 = 0.02 # 펄스에 의한 회전 속도임. 사실 0.04rad/마이크로초
dt = 2.5 # 펄스 한번당 가하는 시간. 단위 : 마이크로초

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
    # 해밀토니안에 의한 각 사구간에 대한 유니터리 오퍼레이터 만드는 것.
    eigvals = np.linalg.eigh(Ham)[0]
    eigvecs = 1*np.linalg.eigh(Ham)[1]
    E = np.diag(eigvals)
    U_H = eigvecs.conj().T
    U_e = U_H.conj().T @ expm(-1j*E*dt) @ U_H
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

        self.x = round(np.trace(self.density_matrix*sx).real,9)
        self.y = round(np.trace(self.density_matrix*sy).real,9)
        self.z = round(np.trace(self.density_matrix*sz).real,9)
        self.phase = phase_cal(self.x,self.y)

        self.point = (self.x,self.y,self.z)
        
        self.g = 0 # 현재 위치까지 이동한 시간
        self.h = 0 # 앞으로 타겟까지 걸릴 예상 시간
        self.f = 0 # f = g + h
    
    # 이건 필요없음
    def __eq__(self, other):
        return self.density_matrix == other.density_matrix
    
# QSL을 통한, heuristic 추정
def heuristic(node,target):
    delta_z = abs((acos(target.z)-acos(node.z)) / 0.0365)
    phase = target.phase - node.phase+0.1726*theta
    if phase < 0 :
        phase+=2*pi
    phi = phase*0.1
    return delta_z +phi

def Astar_Qsearch(init, target):
    # A unitary operator is created in advance to reduce unnecessary operations.
    choiceList = []
    for i in range(5):
        choiceList.append(unitary(dt, i))

    # Initialize openList, closedList
    maximum_len = 100000
    openList = []
    closedList = []

    # Add initial node to openList
    heapq.heappush(openList, (init.f, init))

    # Run until fidelity 0.99 is satisfied
    iteration = 0
    while openList:


        # 원래 이산적인 node에 대한 A*는 같은 점인지 아닌지를 판단할 수 있는데, 지금 시스템에서는 같은점일 가능성이 거의 제로이다.
        # 비슷한 위치에 있는 점은 같은 점으로 인식해줘야 탐색 효율성을 높일 수 있다.
        if iteration < 10: 
            fid = 0.9999 # 초반에는 로테이션을 해도 많이 움직이지 못한다. 그런 점들을 모두 같은 점으로 인식해서는 안되므로 두 점의 fidelity가 0.9999이상일때만 같다고 인식.
        elif iteration < 500:
            fid = 0.99 # 후반에는 점들 사이의 fidelity가 0.99이상이면 같다고 인식.
        else :
            return [-1] # 10000회의 iteration 내에 정답을 찾지 못하면 포기. 데이터를 많이 획득하기 위한 전략.
        

        # Get the node with the lowest f-score from the openList
        _, currentNode = heapq.heappop(openList)

        # Add currentNode to closedList
        closedList.append(currentNode)
        
        # 그냥 잘 돌아가는지 볼라고 넣음
        # print(currentNode.g,currentNode.h,currentNode.f)
        
        
        if state_fidelity(currentNode.density_matrix, target.density_matrix) > 0.99:
            path = []
            current = currentNode
            while current is not None:
                path.append(current.past_pulse)
                current = current.parent
            path = path[:len(path) - 1]
            path = path[::-1]
            print(iteration)
            return path

        children = []
        i = 0
        for rotation in choiceList:
            NewDensityMatrix = rotation @ currentNode.density_matrix @ rotation.conj().T
            new_node = Node(currentNode, NewDensityMatrix, i)
            i += 1
            children.append(new_node)

        for child in children:
            value = True
            for closedNode in closedList:
                if state_fidelity(child.density_matrix, closedNode.density_matrix) > fid:
                    value = False
                    break
            if not value:
                continue
            child.g = currentNode.g + dt
            child.h = heuristic(child, target)
            child.f = child.g + child.h

            # 자식이 openList에 있고, g값이 더 크면 continue
            if any(child.f < openNode[0] and state_fidelity(child.density_matrix, openNode[1].density_matrix) > 0.999 and child.g > openNode[1].g
                   for openNode in openList):
                continue

            heapq.heappush(openList, (child.f, child))

        iteration += 1
        if len(openList) > maximum_len:
            while len(openList) > maximum_len:
                heapq.heappop(openList)

    return []  # No path found

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
    
    path = Astar_Qsearch(start,end)
    end_time = time.time()
    computing_time = end_time - start_time
    total_time = dt*len(path)

    # 결과 출력
    print(f"""theta = {target_theta}
path : {path}
total_time : {total_time}
computing_time : {computing_time}
""")
    return path, computing_time

# CSV file name setup
date = datetime.now()
printdate = date.strftime('%Y%m%d_%H%M%S')
filename = "/Users/qwon/Documents/ByAstar_dt_"+str(dt)+"_" + printdate + '.csv'

# Create an empty DataFrame and write to CSV file
df = pd.DataFrame(columns=["Case", 'gate lenght', 'Theta', 'Phi', 'dt', 'combination', 'total time','computing time'])
df.to_csv(filename, index=False)

case = 1

while True : 
    theta,phi = random.uniform(0.5,0.75), (pi/180)*random.uniform(0,360)
    path,computing_time = main(theta,phi)
    output = [['case'+str(case),len(path),theta, phi, dt, path, len(path)*dt,computing_time]]
    # Create DataFrame and append to CSV file
    df = pd.DataFrame(output, columns=["Case", 'gate lenght', 'Theta', 'Phi', 'dt', 'combination', 'total time','computing time'])
    df.to_csv(filename, mode='a', header=False, index=False)
    case+=1


T_end = time.time()

print("#success# \nTime : " + str(T_end - T_start)) 