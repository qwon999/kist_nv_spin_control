# %% [markdown]
# - tau/2,pi_pulse,tau/2 펄스 결합해서 하나의 블럭으로 만듦.
# - (7/14) 펄스 블락이 짝수개일 때, 퓨어한 결과가 나오도록 코드의 문제점을 찾고자 함.
# - (7/19) Pulse block의 instant pulse가 +-x,y바꿔보고 세타도 바꿔가면서 carbon이 어떻게 evolution하는 지 확인하기.
# - (7/20) N=32 case에서, tau값 바꿔가면서 중첩상태에서 어떻게 회전축이 잡히는 지 시각적 확인하는 코드
# - (7/21) pulse block에서, pi rotation이 적용될 때, 짝수번째에서 축에 무관한 이유 : +-x,y어떤 것을 하던지, 대칭적으로 상쇄되기 때문에, 축 방향이 의미가 없음.
# - (7/24) 일단은 pi/2 pulse로 대칭성만 맞추고 중첩
# - (7/24) temp7 : 파이펄스, pi/2펄스 총 8개 중에 랜덤으로 골라서, 중첩으로 가면 stop하는 코드 

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

np.set_printoptions(suppress=True, precision=2)

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
rho_c_z = rho_c_z_zero
rho_ent = np.kron(rho_nv_z,rho_c_z)


## Define sweep parameters
Sweep = 1001
N = Sweep
B = 403 #[G] magnetic field
# 13C nuclear spin parameters
gammaN = 2*pi*1.071e-3 #[MHz/G]
Al    = 2*pi*0.1 #[MHz] # A_|| hyperfine term
Ap = 2*pi*0.1 #[MHz] # A_per hyperfine term

# %%
T = 5; # sweep tau [us]
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
# print(tau)

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
trace = [[],1,10000]
iters = 10000
def problem(vari) :
    # trace = [[],1,10000]
    
    lenn = trunc(vari[0])
    t = vari[1]
    print(lenn,t)
    U_0 = make_unitary_2(t,0)
    U_1 = make_unitary_2(t,1)
    U_2 = make_unitary_2(t,2)
    U_3 = make_unitary_2(t,3)
    U_4 = make_unitary_2(t,4)
    U_5 = make_unitary_2(t,5)
    U_6 = make_unitary_2(t,6)
    U_7 = make_unitary_2(t,7)
    U_list = [U_0, U_1, U_2, U_3, U_4, U_5, U_6, U_7]
    for i in range(iters) :
        rho_0 = ((np.kron(U090xm,I)) @ rho_ent @ ((np.kron(U090xm,I)).conj().T)) # superposition state on NV
        choice_0 = []
        for i in range(lenn):
            choice_0.append(np.random.choice([0,1,2,3,4,5,6,7]))
            # choice_0.append(3)
        choice_0 = choice_0 + choice_0[::-1]

        for i in range(1,round(lenn*2)+1):
            choice = choice_0[i-1]
            U = U_list[choice]
            rho_0 = U @ rho_0 @ U.conj().T
        
        point = [np.trace(Ix@partial_trace(rho_0,1)).real,
                np.trace(Iy@partial_trace(rho_0,1)).real,
                np.trace(Iz@partial_trace(rho_0,1)).real]
        purity = (sqrt(point[0]**2+point[1]**2+point[2]**2))
        # print(point[2])
        if point[2] > 0.1 or point[2] < -0.1 :
            continue
        else :
            # total.append([choice_0,t,lenn])
            if trace[2] > lenn*t*2 :    
                trace[0] = [choice_0]
                trace[1] = purity
                trace[2] = lenn*t*2
            return lenn*t*2
    return 10000
            
for p in range(1): # 1번의 실험을 진행한다.(지역 최적화 알고리즘을 사용할 경우에 수정한다.)
        vari=[8,tau]  #초기값
        bounds = [(2,16),(tau*0.001,tau*2)] #boundary
        
        res4 = optimize.shgo(problem,bounds=bounds,iters=7,options={'xtol':1e-3,'ftol':1e-3}) #SHGO method
        res4['x'][0] = trunc(res4['x'][0]) #rounding
        print(res4['x'][0],trace[0], res4['x'][1],trace[2])
        print(trace)






