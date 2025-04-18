import numpy as np
from qutip import *
import krotov
# Define the NV electron spin Hamiltonian
D = 2*np.pi*2.87e9
Ez = 2*np.pi*2.8e6
Ex = 2*np.pi*10e6
Sz = tensor(sigmaz(), qeye(2))
Sx = tensor(sigmax(), qeye(2))
H0 = D*Sz**2
H1 = [Ez*Sz, Ex*Sx]
H = [H0, H1]

# Define the target state
psi_target = (tensor(basis(2, 0), basis(2, 1)) + tensor(basis(2, 1), basis(2, 0))).unit()

# Define the initial state and controls
psi0 = tensor(basis(2, 0), basis(2, 0))
eps0 = lambda t, args: np.random.rand() * 2*np.pi
eps1 = lambda t, args: np.random.rand() * 2*np.pi
eps  = [eps0, eps1]

# Define the object function
def obj_func(psi, t, eps):
    return 1.0 - abs((psi.dag() * psi_target)[0][0])**2

# Define the Krotov optimization
opt_result = krotov.optimize_pulses(
    objectives=[krotov.Objective(initial_state=psi0, target=psi_target, 
                                 H=H)],
    pulse_generators=[eps],
    tlist=np.linspace(0, 2e-6, 100),
    propagator=krotov.propagators.expm,
    chi_constructor=krotov.functionals.chis_ss,
    info_hook=krotov.info_hooks.print_table(np.inf),
    check_convergence=krotov.convergence.Or(
        krotov.convergence.value_below(1e-3, name='J_T'),
        krotov.convergence.check_monotonic_error,
    ),
    store_all_pulses=True,
)

# Plot the optimized controls
krotov.plot_controls(opt_result)
