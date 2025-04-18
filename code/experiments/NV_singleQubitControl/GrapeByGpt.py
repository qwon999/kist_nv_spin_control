import numpy as np
import qutip as qt
import qutip.control as qc

# Define the Hamiltonian and initial and target states
H0 = qt.Qobj(np.diag([1, -1]))
Hc = qt.Qobj(np.array([[0, 1], [1, 0]]))
H = [H0, [Hc, lambda t, args: control_amplitude[t]]]
initial_state = qt.basis(2, 0)
target_state = qt.basis(2, 1)

# Define the optimization parameters
control_duration = 1.0
n_time_steps = 100
control_amplitude = np.zeros(n_time_steps)

# Define the GRAPE optimizer
result = qc.grape_pulse_optim(
    H, initial_state, target_state, control_duration, n_time_steps,
    amp_lbound=-10.0, amp_ubound=10.0, phase_option='SU', 
    method_params={'max_iterations': 500}
)

# Extract the optimized pulse
optimized_pulse = result.evo_full_ops[0].coeff

# Apply the optimized pulse to the system
tlist = np.linspace(0, control_duration, n_time_steps)
propagator = qt.propagator(H, tlist, args={}, parallel=False, options=None)
final_state = propagator * initial_state
fidelity = (final_state.dag() * target_state).norm() ** 2

# Print the fidelity of the final state
print(f"Fidelity: {fidelity:.4f}")
