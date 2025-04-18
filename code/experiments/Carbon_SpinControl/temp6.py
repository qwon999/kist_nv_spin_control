# %% [markdown]
# - tau/2,pi_pulse,tau/2 펄스 결합해서 하나의 블럭으로 만듦.
# - (7/14) 펄스 블락이 짝수개일 때, 퓨어한 결과가 나오도록 코드의 문제점을 찾고자 함.
# - (7/19) Pulse block의 instant pulse가 +-x,y바꿔보고 세타도 바꿔가면서 carbon이 어떻게 evolution하는 지 확인하기.
# - (7/20) N=32 case에서, tau값 바꿔가면서 중첩상태에서 어떻게 회전축이 잡히는 지 시각적 확인하는 코드
# - (7/21) pulse block에서, pi rotation이 적용될 때, 짝수번째에서 축에 무관한 이유 : +-x,y어떤 것을 하던지, 대칭적으로 상쇄되기 때문에, 축 방향이 의미가 없음.
# - (7/24) 일단은 pi/2 pulse로 대칭성만 맞추고 중첩
# - (7/24) temp7 : 파이펄스, pi/2펄스 총 8개 중에 랜덤으로 골라서, 중첩으로 가면 stop하는 코드 
# - (7/27) 데아터 파일 저장위한 코드(temp2.py)
# - (7/27) carbon mixed state에서, 랜덤 펄스 줘서 Purity 얼마나 생기는 지 확인하는 코드(temp3.py)
# - (7/28) mixed state에서, 단일 tau로 초기화하는 펄스 찾기(temp4.py)
# - (8/02) fidelity와 sz연산 중 어떤 것이 더 빠른지 확인하는 코드
# - (8/03) 초기화 unitay gate를 step을 추가하며 찾는 코드[temp6.py]
# %%
import numpy as np
from toqito.channels import partial_trace
from qutip import *
from sklearn.feature_extraction.text import CountVectorizer
from scipy import linalg
from math import *
from scipy.linalg import fractional_matrix_power
import cmath
from scipy import optimize
import time
import pandas as pd
from datetime import datetime as dt                         # 시간을 출력하기 위한 라이브러리  
import random

np.set_printoptions(suppress=True, precision=2)

Constant_Ap, Constant_Al = 1,1

# %%
# Define dimension, pauli matrices
sx  = 1/sqrt(2)*np.array([[0, 1, 0],[1, 0, 1], [0, 1, 0]])
sy  = 1/sqrt(2)/1j*np.array([[0, 1, 0], [-1, 0, 1],[0, -1, 0]])
sz  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
I   = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

# Rotation matrix projected into 2 level system
Sxp  = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
Sxm  = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
Syp  = 1/1j*np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
Sym  = 1/1j*np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
Szp  = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
Szm  = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])

#Gellman matrix
Sx  = np.array([[0, 0, 1],[0, 0, 0], [1, 0, 0]])
Sy  = np.array([[0, 0, -1j],[0, 0, 0], [1j, 0, 0]])
Sz  = np.array([[1, 0, 0],[0, 0, 0], [0, 0, -1]])

# Pauli basis for 13C nuclear spin
Ix  = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])   
Iy  = 1/1j*np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
Iz  = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])

# %%
def UO(B1,B2,a,D1,D2):
    i   = 1j
    gamma = 2*pi*2.8
    D     = 2870
    UA = [[(B2**2+B1**2*cos(a))/(B1**2+B2**2), -i*B1*(e**(-i*D1))*sin(a)/sqrt(B1**2+B2**2), ((-1+cos(a))*B1*B2*(e**(-i*(D1-D2))))/(B1**2+B2**2)],
            [-i*B1*(e**(i*D1))*sin(a)/sqrt(B1**2+B2**2), cos(a), -i*B2*(e**(i*D2))*sin(a)/sqrt(B1**2+B2**2)],
            [((-1+cos(a))*B1*B2*e**(i*(D1-D2)))/(B1**2+B2**2), -i*B2*(e**(-i*D2))*sin(a)/sqrt(B1**2+B2**2), (B1**2+B2**2*cos(a))/(B1**2+B2**2)]]
    return UA

# Single Q ms=+1
U090xp  = UO(1,0,pi/4,0,0)
U090xmp = UO(1,0,-pi/4,0,0)
U090yp  = UO(1,0,pi/4,pi/2,0)
U090ymp = UO(1,0,-pi/4,pi/2,0)
U180xp  = UO(1,0,pi/2,0,0)
U180xmp = UO(1,0,-pi/2,0,0)
U180yp  = UO(1,0,pi/2,pi/2,0)
U180ymp = UO(1,0,-pi/2,pi/2,0)

#Single Q ms=-1
U090xm  = UO(0,1,pi/4,0,0)
U090xmm = UO(0,1,-pi/4,0,0)
U090ym  = UO(0,1,pi/4,0,pi/2)
U090ymm = UO(0,1,-pi/4,0,pi/2)
U180xm  = UO(0,1,pi/2,0,0)
U180xmm = UO(0,1,-pi/2,0,0)
U180ym  = UO(0,1,pi/2,0,pi/2)
U180ymm = UO(0,1,-pi/2,0,pi/2) 


# %%
rho_nv_z_zero = np.array([[0,0,0],[0,1,0],[0,0,0]]) # |0>state
rho_nv_z_minusone = np.array([[0,0,0],[0,0,0],[0,0,1]]) # |-1>state
rho_nv_z_spp_px = np.array([[0,0,0],[0,1/2,1/2],[0,1/2,1/2]]) # |-1>state
rho_nv_z_spp_py = np.array([[0,0,0],[0,1/2,-1j/2],[0,1j/2,1/2]]) # |-1>state

rho_c_z_spp = np.array([[0.5,0,0.5],[0,0,0],[0.5,0,0.5]]) # superposition
rho_c_z_zero = np.array([[1,0,0],[0,0,0],[0,0,0]]) # |0>state 
rho_c_z_one = np.array([[0,0,0],[0,0,0],[0,0,1]]) # |0>state 
rho_c_z_mixed = np.array([[0.5,0,0],[0,0,0],[0,0,0.5]]) # mixed state

rho_nv_z = rho_nv_z_zero
rho_c_z = rho_c_z_one
rho_ent = np.kron(rho_nv_z,rho_c_z)


## Define sweep parameters
Sweep = 1001
N = Sweep
B = 403 #[G] magnetic field
# 13C nuclear spin parameters
gammaN = 2*pi*1.071e-3 #[MHz/G]
Al    = 2*pi*0.1*Constant_Al #[MHz] # A_|| hyperfine term
Ap = 2*pi*0.1*2*Constant_Ap  #[MHz] # A_per hyperfine term

# %%
T = 0.78; # sweep tau [us]
t = np.linspace(0,T,N)
n = 32; # number of pi pulses

# %%
pipulse_list = [U180xm, U180xmm, U180ym, U180ymm]
halfpipulse_list = [U090xm, U090xmm, U090ym, U090ymm]
def make_unitary(tau,choice) :
    ham =  Ap*np.kron(sz,0.5*Ix)  + Al*np.kron(sz,Iz) + B*gammaN*np.kron(I,Iz)
    eigvals = np.linalg.eigh(ham)[0]            # diagonalizing the Hamiltonian 여기서부터 문제 
    eigvecs = 1*np.linalg.eigh(ham)[1]
    E = np.diag(eigvals)                        # exponent of eigenvalues
    U = eigvecs @ linalg.expm(-1j*E*tau/2) @ eigvecs.conj().T # for tau/2
    U_e = U @ np.kron(pipulse_list[choice],I) @ U # pi-pulse 한개
    # U_e = U @ U
    return U_e

# %%
Sa = []
Sb = []
rho_0 = (np.kron(U090xm,I)) @ rho_ent @ ((np.kron(U090xm,I)).conj().T) # superposition state on NV
for h in range(round(N)):
    #free evolution unitary operator
    U = make_unitary(t[h],0)
    # if h == 100 :
    #     make_print(U)
    rho_1 = rho_0
    for k in range(n):
        rho_1 = U @ rho_1 @ U.conj().T
    rho_2 = np.kron(U090xmm,I) @ rho_1 @ ((np.kron(U090xmm,I)).conj().T)    # last pi/2
    res1 = (np.trace(rho_nv_z@partial_trace(rho_2,2))).real   # NV state 0 population readout
    res2 = (np.trace(rho_c_z@partial_trace(rho_2,1))).real
    Sa.append(res1)
    Sb.append(res2)

index = Sa.index(min(Sa))
tau=t[index]

# print(Sa.index(min(Sa)))
print(tau)

# %%
pipulse_list = [U180xm, U180xmm, U180ym, U180ymm]
halfpipulse_list = [U090xm, U090xmm, U090ym, U090ymm]
pulse_list = [U180xm, U180xmm, U180ym, U180ymm, U090xm, U090xmm, U090ym, U090ymm]
def make_unitary_2(tau,choice) :

    # B = 403 #[G] magnetic field
    # # 13C nuclear spin parameters
    # gammaN = 2*pi*1.071e-3 #[MHz/G]
    # Al    = 2*pi*0.1 #[MHz] # A_|| hyperfine term
    # Ap = 2*pi*0.1 #[MHz] # A_per hyperfine term
    # # define hamiltonian
    # Ham = Al*np.kron(sz,Iz) + Ap*np.kron(sz,0.5*Ix) + B*gammaN*np.kron(I,Iz)
    ham =  Ap*np.kron(sz,0.5*Ix)  + Al*np.kron(sz,Iz) + B*gammaN*np.kron(I,Iz)
    eigvals = np.linalg.eigh(ham)[0]            # diagonalizing the Hamiltonian 여기서부터 문제 
    eigvecs = 1*np.linalg.eigh(ham)[1]
    E = np.diag(eigvals)                        # exponent of eigenvalues

    U = eigvecs @ linalg.expm(-1j*E*tau/2) @ eigvecs.conj().T # for tau/2
    
    U_e = U @ np.kron(pulse_list[choice],I) @ U # pi-pulse 한개
    # U_e = U @ U
    return U_e
total = []
# %%
trace = [[],10000]
iters = 10000
def problem(vari) :
    
    # trace = [[],1,10000]
    combi =  []
    t = vari[0]
    rho_0 = rho_ent
    # rho_0 = ((np.kron(U090xm,I)) @ rho_ent @ ((np.kron(U090xm,I)).conj().T)) # superposition state on NV
    
    point = [np.trace(Ix@partial_trace(rho_0,1)).real,
             np.trace(Iy@partial_trace(rho_0,1)).real,
             np.trace(Iz@partial_trace(rho_0,1)).real]

    U_0 = make_unitary_2(t,0)
    U_1 = make_unitary_2(t,1)
    U_2 = make_unitary_2(t,2)
    U_3 = make_unitary_2(t,3)
    U_4 = make_unitary_2(t,4)
    U_5 = make_unitary_2(t,5)
    U_6 = make_unitary_2(t,6)
    U_7 = make_unitary_2(t,7)
    U_list = [U_0, U_1, U_2, U_3, U_4, U_5, U_6, U_7]
    
    # rho_0 = U_6 @ rho_0 @ U_6.conj().T

    N = 0
    
    while point[2] < 0.99 :
        temp_z = []
        current_z = point[2]
    
        for i in range(8) :
            U = np.kron(I,I)
            temp_combi = combi + [i,i] + combi[::-1]
    
            for u in temp_combi :
                U = U_list[u] @ U 
            rho_temp = U @ rho_0 @ U.conj().T
            delta = (np.trace(Iz@partial_trace(rho_temp,1)).real - current_z)
            temp_z.append(delta)
    
        max_delta_value = max(temp_z)
        max_delta_indices = [j for j, v in enumerate(temp_z) if v==max_delta_value]
        choice = random.choice(max_delta_indices)
        combi.append(choice)
        temp_combi = combi + combi[::-1]
        U = np.kron(I,I)
    
        for u in temp_combi :
            U = U_list[u] @ U
        rho_temp = U @ rho_0 @ U.conj().T
        point = [np.trace(Ix@partial_trace(rho_temp,1)).real,
                 np.trace(Iy@partial_trace(rho_temp,1)).real,
                 np.trace(Iz@partial_trace(rho_temp,1)).real]
        # print(point[2])
        N+=1
    
        if N >= 100 :
            purity = (sqrt(point[0]**2+point[1]**2+point[2]**2))
            total.append([t,len(combi),combi,10000,point[2],purity])
            return 1000
    
    purity = (sqrt(point[0]**2+point[1]**2+point[2]**2))
    total.append([t,len(combi),combi,len(combi)*t,point[2],purity])
    
    if trace[1] > len(combi)*t :
        trace[1] = len(combi)*t 
        trace[0] = combi
    
    return len(combi)*t

    
    
 
            
for p in range(1): # 1번의 실험을 진행한다.(지역 최적화 알고리즘을 사용할 경우에 수정한다.)
        vari=[(tau)]  #초기값
        bounds = [(0.999*tau,tau*1.001)] #boundary
        res4 = optimize.shgo(problem,bounds=bounds,iters=150,options={'xtol':1e-15,'ftol':1e-3}) #SHGO method
        # res4['x'][0] = trunc(res4['x'][0]) #rounding
        print('\n',res4['x'][0])
        print(trace)

date = dt.now()
printdate = date.strftime('%m%d_%H%M%S')
print(date)

df4 = pd.DataFrame(total) 
df4.rename(columns={0:"tau", 1:"length", 2: "gate", 3: "total_time", 4: "z", 5: "purity"}, inplace=True)
df4.to_csv('/Users/qwon/Documents/data/NuclearSpinData/'+str(Constant_Al)+'Al'+str(Constant_Ap)+'Ap' + printdate +'_temp6.csv' ,index=False)
# %%