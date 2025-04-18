import numpy as np
from scipy.optimize import minimize

def Rx(theta):
    return np.matrix([[np.cos(theta/2), -1j*np.sin(theta/2)],
                      [-1j*np.sin(theta/2), np.cos(theta/2)]])

def Rz(phi):
    return np.matrix([[np.cos(phi/2) - 1j*np.sin(phi/2), 0],
                      [0, np.cos(phi/2) + 1j*np.sin(phi/2)]])

def apply_TOUC(U_target):
    phi = np.angle(np.linalg.det(U_target))
    theta = 2 * np.arccos(np.abs(np.trace(U_target))/2)
    U_decomposed = np.dot(np.dot(Rz(-phi), Rx(-theta/2)), np.dot(U_target, Rx(theta/2)))
    return U_decomposed

def random_state_on_bloch_sphere():
    u = np.random.rand(3)
    theta = 2*np.arccos(u[0]-1)
    phi = 2*np.pi*u[1]
    rho = np.sqrt(u[2])
    x = rho * np.sin(theta) * np.cos(phi)
    y = rho * np.sin(theta) * np.sin(phi)
    z = rho * np.cos(theta)
    return np.array([x, y, z])

def initialize_NV_spin():
    target_unitary = np.matrix([[0, 1], [1, 0]])
    initial_state = random_state_on_bloch_sphere()
    def objective(x):
        U_init = np.dot(Rx(x[0]), Rz(x[1]))
        U_decomposed = apply_TOUC(U_init)
        F = np.linalg.norm(target_unitary - np.dot(U_decomposed, np.dot(np.diag([1, np.exp(1j * x[2])]), np.linalg.inv(U_decomposed))))
        return F
    result = minimize(objective, [0, 0, 0], method='Powell')
    angles = result.x
    U_init = np.dot(Rx(angles[0]), Rz(angles[1]))
    NV_spin = np.dot(U_init, initial_state)
    return NV_spin

# example usage:
NV_spin = initialize_NV_spin()
print("Initial NV spin state:", NV_spin)
