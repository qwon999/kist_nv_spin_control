###강동연 박사님 팀에서 제작한 코드


import numpy as np
import cvxpy as cp
#텐서곱할때 전자꺼가 뒤로
basis_0 = np.array([[1], [0]])
basis_0_H = np.matrix.getH(basis_0)
basis_1 = np.array([[0], [1]])
basis_1_H = np.matrix.getH(basis_1)

basis_00 = basis_0@basis_0_H
basis_10 = basis_1@basis_0_H
basis_01 = basis_0@basis_1_H
basis_11 = basis_1@basis_1_H


basis_H = np.array([[1], [0], [1], [0]])
I = np.array([[1, 0], [0, 1]])
i = 1j
pauli_X = np.array([[0, 1], [1, 0]])
pauli_Y = np.array([[0, -i], [i, 0]])
pauli_Z = np.array([[1, 0], [0, -1]])
theta = np.pi/2
rx = np.array([[np.cos(theta/2), -i*np.sin(theta/2)], [-i*np.sin(theta/2), np.cos(theta/2)]])
rx_m = np.array([[np.cos(-theta/2), -i*np.sin(-theta/2)], [-i*np.sin(-theta/2), np.cos(-theta/2)]])
ry = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
ry_m = np.array([[np.cos(-theta/2), -np.sin(-theta/2)], [np.sin(-theta/2), np.cos(-theta/2)]])
rz = np.array([[np.cos(theta/2) - i*np.sin(theta/2), 0], [0, np.cos(theta/2) + i*np.sin(theta/2)]])
rz_m = np.array([[np.cos(-theta/2) - i*np.sin(-theta/2), 0], [0, np.cos(-theta/2) + i*np.sin(-theta/2)]])
#crx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(theta), -i*np.sin(theta)], [0, 0, -i*np.sin(theta), np.cos(theta)]])
#crx = np.array([[1, 0, 0, 0], [0, np.cos(theta/2), 0, -i*np.sin(theta/2),], [0, 0, 1, 0], [0, -i*np.sin(theta/2), 0, np.cos(theta/2)]])


conditional_rx = np.array([[1, -i, 0, 0], [-i, 1, 0, 0], [0, 0, 1, i], [0, 0, i, 1]])*np.sqrt(1/2)
conditional_ry = np.array([[1, -1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, -1, 1]])*np.sqrt(1/2)
#conditional_ry= np.kron(I, rz)@conditional_rx@np.kron(I, rz_m)
conditional_rx_m = np.kron(np.array([[1, 0], [0, 0]]), rx_m) + np.kron(np.array([[0, 0], [0, 1]]), rx)
conditional_ry_m = np.array([[1, 1, 0, 0], [-1, 1, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]])*np.sqrt(1/2)


XX = conditional_rx_m@np.kron(rx,I)
XY = np.kron(I,rz)@conditional_rx_m@np.kron(rx,I)
XZ = np.kron(I,rx_m)@np.kron(I,rz)@conditional_rx@np.kron(rx,I)@conditional_rx
XI = np.kron(ry,I)

YX = conditional_rx_m@np.kron(ry,I)
YY = np.kron(I,rz)@conditional_rx_m@np.kron(ry,I)
YZ = np.kron(I,rx_m)@np.kron(I,rz)@conditional_rx@np.kron(ry,I)@conditional_rx
YI = np.kron(rx_m,I)

ZX = np.kron(rx,I)@conditional_rx_m@np.kron(ry,I)
ZY = np.kron(rx,rz)@conditional_rx_m@np.kron(ry,I)
ZZ = np.kron(I,rx_m)@np.kron(rx,rz)@conditional_rx@np.kron(ry,I)@conditional_rx
ZI = np.kron(I,I)

## (-Z) 
PX = np.kron(rx_m,I)@conditional_rx_m@np.kron(ry,I)
PY = np.kron(rx_m,rz)@conditional_rx_m@np.kron(ry,I)
PZ = np.kron(I,rx_m)@np.kron(rx_m,rz)@conditional_rx@np.kron(ry,I)@conditional_rx
PI = np.kron(np.array([[np.cos(np.pi/2), -np.sin(np.pi/2)], [np.sin(np.pi/2), np.cos(np.pi/2)]]),I)

IX=  np.kron(I,rz)
#print(IX)

# Compare pauli operator with circuit
#print(np.round(IX@np.kron(pauli_Z, I)@np.matrix.getH(IX),3))
#print(np.kron(-pauli_Z,I))
#target=np.kron(I,pauli_X)
target=np.kron(I,pauli_Z)
##print(np.matrix.getH(XX)@np.kron(pauli_Z, I)@XX) ###
dim=4
eigvals, eigvecs = np.linalg.eig(target)  
print('target IY')
print(target)
##################################### This is for searching the U gate by eigen value problem. U gate make XX to ZI. 
eigvalsbk=np.array(eigvals)
eigvecbk=np.array(eigvecs)                           # diagonalizing the Hamiltnian)
E = np.diag(eigvals)                                                                        # exponent of eigenvalues
U_H = np.matrix.getH(eigvecs)   ## this is gate.

print("dignozalize")

print(np.round(np.matrix.getH(eigvecs)@target@eigvecs,3))

eigvecs_original=np.array(eigvecs)
dia_zed= eigvecs@target@U_H
for k in range(len(eigvalsbk)-1):
      if eigvals[k]<eigvals[k+1]:
          #print(eigvals)
          eigvals[k],eigvals[k+1] = eigvals[k+1], eigvals[k]
          eigvecs[:, [k, k+1]] = eigvecs[:, [k+1, k]] 
          
          #print(eigvals)
for k in range(len(eigvalsbk)-1):
      if eigvals[k]<eigvals[k+1]:
          #print(eigvals)
          eigvals[k],eigvals[k+1] = eigvals[k+1], eigvals[k]
          eigvecs[:, [k, k+1]] = eigvecs[:, [k+1, k]] 
          
U_H = np.matrix.getH(eigvecs)   ## this is gate.


print("dignozalize")
print(np.round(U_H@target@eigvecs,3))
print("U_ZI _ U+")
##print(np.round(U_H@np.kron(pauli_Z,I)@eigvecs,3))  This is incorrect.
print(np.round(eigvecs@np.kron(pauli_Z,I)@U_H,3)) ## This is correct.


print("U_gate")
print(np.round(eigvecs,3))
####################################################################################################################

gates= [ rx,rx_m, ry, ry_m, rz, rz_m]
controlgates= [conditional_rx, conditional_rx_m,  conditional_ry, conditional_ry_m,
               -np.kron(I,I), i*np.kron(I,I), -i*np.kron(I,I) ]
                      
#ontrolgates= [conditional_rx, conditional_rx_m] 
#gate_def=['rxI','Irx', 'rx_mI','Irx_m',  'ry_I', 'Iry',  'ry_mI','Iry_m', 'rzI','Irz', 'rz_mI',  'Irz_m', 'cRx', 'cRxm']
#ate_def=['rxI','Irx', 'rx_mI','Irx_m',  'ry_I',   'ry_mI', 'rzI','Irz', 'rz_mI',  'Irz_m', 'cRx', 'cRxm']
gate_def=['rxI','Irx', 'rx_mI','Irx_m',  'ry_I', 'Iry',  'ry_mI','Iry_m', 'rzI','Irz', 'rz_mI',  'Irz_m', 'cRx', 'cRxm', 'cRy', 'cRym',
          '-I', 'i', '-i' ]
gate_point=[1,10, 2,12, 1, 30,  2,32, 1,10, 2, 12, 10,12, 100,100, 0, 0, 0 ]

gate_list=[]
n_qubits=2
from itertools import product
#from qutip import *
#from qutip.qip.operations import *
#for k,kk in product(gates,repeat=n_qubits):
#    gate_list.append(np.kron(k, kk))
for k in gates:
    gate_list.append(np.kron(k, I))
    
    #if not (np.all(k==ry) or np.all(k==ry_m)):
    #    gate_list.append(np.kron(I, k))        
    gate_list.append(np.kron(I, k))        
for k in controlgates:
    gate_list.append(k)

gate_candi=list(gate_list)
ngate=len(gate_candi)

for k in range(4): # gate+1
    gate_candi0=list(gate_candi)
    if k>0:
        gate_index0=list(gate_index)
        gate_defind0=list(gate_defind)
    gate_candi=[]
    gate_defind=[]
    gate_index=[]
    
    for k1,k2 in product(range(len(gate_candi0)), range(len(gate_list))):
    
        gate_candi.append(gate_list[k2]@gate_candi0[k1])
    
        if k==0:
    
            gate_index.append([k1, k2])
            gate_defind.append([gate_def[k1],gate_def[k2]])
            
        else:
            aaa=[]
            bbb=[]
            for iii in range(len(gate_index0[k1])):
                aaa.append(gate_index0[k1][iii])
                bbb.append(gate_defind0[k1][iii])
            aaa.append(k2)
            bbb.append(gate_def[k2])

            gate_index.append(aaa)  

            gate_defind.append(bbb)  
    minpoint=1000
    for ii in range(len(gate_candi)):
        conv= np.round(gate_candi[ii]@np.kron(pauli_Z, I)@np.matrix.getH(gate_candi[ii]),3)
        #if np.allclose(gate_candi[ii], eigvecs):
        if np.allclose(conv, target):
            #print(gate_candi[ii])
            #print(U_H)
            
            gate_point_list = [gate_point[kkk]   for kkk in gate_index[ii]]
            if np.sum(gate_point_list) <= minpoint:
                minpoint=np.sum(gate_point_list) 
                print(gate_index[ii])
                print(gate_defind[ii])
                print(gate_point_list, np.sum(gate_point_list))
            
# for k in range(5): # gate+1
#     U_est_ind = cp.Variable((k+1,ngate))
#     #Prob_est=[ gate_list[U_est_ind[x]] for x in range(k)]
#     Prob_est=[U_est_ind @ gate_list ]
#     Gate_cv=np.kron(I,I)
#     for x in range(k):
#         Gate_cv=cp.Prob_est[x] @Gate_cv
#     cost = cp.sum([cp.abs(np.subtract(Gate_cv[x][y], eigvecs[x][y]))**2 for x in range(dim) for y in range(dim)] )
#     obj = cp.Minimize(cost)
#     constraints= [ np.allclose(Gate_cv @  np.matrix.getH(Gate_cv), np.kron(I,I))]

# #gate_final=gate_list[14]@gate_list[12]@gate_list[0]
#gate_final=gate_list[14]@gate_list[5]@gate_list[4]@gate_list[12]@gate_list[0]
#print('gate final')
#print(gate_final)
#print('U_ZI_U')
#print(np.round(gate_final@np.kron(pauli_Z, I)@np.matrix.getH(gate_final),3))

#print(U_H)
#print(np.round(np.kron(rx,I),3))
#print(np.round(U_H-np.kron(rx,I),3))
                                                           # unitary matrix formed by eigenvectors
#U_est = cp.Variable((dim,dim), complex =True)
#aaa=(U_est@np.kron(pauli_Z, I)@np.matrix.getH(U_est))
#Prob_est=[aaa[x,y] for x in range(dim) for y in range(dim)]
#Prob_meas=[np.kron(I,pauli_X)[x,y] for x in range(dim) for y in range(dim)]

#cost = cp.sum([cp.abs(Prob_est[k] - Prob_meas[k])**2 for k in range(len(Prob_est))])
#print(len(Prob_est))