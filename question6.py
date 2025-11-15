"""
QUESTION 6: Anode Break Excitation
A) Simulate what happens during the first 10 ms with initial Vm = -105 mV
B) Study the membrane conductance and explain why this excitation occurs
"""

import numpy as np
import matplotlib.pyplot as plt
from hh_model import hh_model, calculate_steady_state_gating

# Parameters from table
params = {
    'gNa': 120,      # mS/cm²
    'gK': 36,        # mS/cm²
    'gL': 0.3,       # mS/cm²
    'Cm': 1.0,       # μF/cm²
    'ENa': 52.4,     # mV
    'EK': -72.1,     # mV
    'EL': -49.2,     # mV
    'T': 6.3         # °C
}

# Initial conditions - CHANGED for anode break
V0 = -105             # mV (hyperpolarized)
# Calculate steady-state gating variables at -105 mV
m0, h0, n0 = calculate_steady_state_gating(V0, params)
print(f'Steady-state gating variables at V = {V0} mV:')
print(f'  m0 = {m0:.6f}')
print(f'  h0 = {h0:.6f}')
print(f'  n0 = {n0:.6f}')
print()

# Simulation parameters
dt = 0.01             # ms
t_end = 10            # ms

# No external stimulus - just observe the response from hyperpolarized state
def I_stim(t):
    return 0

print('Question 6: Anode Break Excitation')
print('=' * 42)
print(f'Initial voltage: {V0} mV')
print()

# Run simulation
t, Vm, I_Na, I_K, I_L, m, h, n = hh_model(I_stim, dt, t_end, V0, m0, h0, n0, params)

# Calculate conductances
g_Na = params['gNa'] * m**3 * h
g_K = params['gK'] * n**4
g_L = params['gL'] * np.ones_like(t)
g_total = g_Na + g_K + g_L

# Calculate reversal potentials weighted by conductances
V_eq = (g_Na * params['ENa'] + g_K * params['EK'] + g_L * params['EL']) / g_total

# Plot results
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot A1: Membrane Voltage
ax = axes[0, 0]
ax.plot(t, Vm, 'b-', linewidth=2)
ax.axhline(-60, color='r', linestyle='--', linewidth=1, label='Resting potential')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Membrane Voltage (mV)')
ax.set_title('A) Membrane Voltage vs Time')
ax.grid(True)
ax.legend()
ax.set_xlim([0, t_end])

# Plot A2: Gating Variables
ax = axes[0, 1]
ax.plot(t, m, 'r-', linewidth=1.5, label='m')
ax.plot(t, h, 'b-', linewidth=1.5, label='h')
ax.plot(t, n, 'g-', linewidth=1.5, label='n')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Gating Variable')
ax.set_title('Gating Variables')
ax.legend()
ax.grid(True)
ax.set_xlim([0, t_end])
ax.set_ylim([0, 1])

# Plot A3: Ionic Currents
ax = axes[0, 2]
ax.plot(t, I_Na, 'r-', linewidth=1.5, label='I$_{Na}$')
ax.plot(t, I_K, 'b-', linewidth=1.5, label='I$_K$')
ax.plot(t, I_L, 'g-', linewidth=1.5, label='I$_L$')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Current (μA/cm²)')
ax.set_title('Ionic Currents')
ax.legend()
ax.grid(True)
ax.set_xlim([0, t_end])

# Plot B1: Conductances
ax = axes[1, 0]
ax.plot(t, g_Na, 'r-', linewidth=1.5, label='g$_{Na}$')
ax.plot(t, g_K, 'b-', linewidth=1.5, label='g$_K$')
ax.plot(t, g_L, 'g-', linewidth=1.5, label='g$_L$')
ax.plot(t, g_total, 'k--', linewidth=1.5, label='g$_{total}$')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Conductance (mS/cm²)')
ax.set_title('B) Membrane Conductances')
ax.legend()
ax.grid(True)
ax.set_xlim([0, t_end])

# Plot B2: Conductance Ratios
ax = axes[1, 1]
ax.plot(t, g_Na / g_total, 'r-', linewidth=1.5, label='g$_{Na}$/g$_{total}$')
ax.plot(t, g_K / g_total, 'b-', linewidth=1.5, label='g$_K$/g$_{total}$')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Conductance Ratio')
ax.set_title('Conductance Ratios')
ax.legend()
ax.grid(True)
ax.set_xlim([0, t_end])

# Plot B3: Voltage vs Equilibrium Potential
ax = axes[1, 2]
ax.plot(t, Vm, 'b-', linewidth=2, label='V$_m$')
ax.plot(t, V_eq, 'r--', linewidth=1.5, label='Equilibrium potential')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Voltage (mV)')
ax.set_title('Voltage vs Equilibrium Potential')
ax.legend()
ax.grid(True)
ax.set_xlim([0, t_end])

plt.suptitle('Question 6: Anode Break Excitation (Initial V$_m$ = -105 mV)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
plt.savefig('question6_results.png', dpi=300, bbox_inches='tight')

# Print analysis
print('=== Analysis ===')
print(f'Initial voltage: {V0} mV')
print(f'Final voltage: {Vm[-1]:.2f} mV')
print(f'Peak voltage: {np.max(Vm):.2f} mV')
if np.max(Vm) > 0:
    print('Action potential occurred!')
    print(f'Time to peak: {t[np.argmax(Vm)]:.2f} ms')
else:
    print('No action potential occurred')

print('\n=== Conductance Analysis ===')
print('At t=0:')
print(f'  g_Na: {g_Na[0]:.4f} mS/cm²')
print(f'  g_K: {g_K[0]:.4f} mS/cm²')
print(f'  g_L: {g_L[0]:.4f} mS/cm²')
print(f'  g_total: {g_total[0]:.4f} mS/cm²')

peak_idx = np.argmax(Vm)
print(f'\nAt peak voltage (t={t[peak_idx]:.2f} ms):')
print(f'  g_Na: {g_Na[peak_idx]:.4f} mS/cm²')
print(f'  g_K: {g_K[peak_idx]:.4f} mS/cm²')
print(f'  g_total: {g_total[peak_idx]:.4f} mS/cm²')

plt.show()

