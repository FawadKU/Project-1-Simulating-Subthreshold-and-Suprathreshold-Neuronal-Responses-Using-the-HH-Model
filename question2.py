"""
QUESTION 2: Step Current Injection
Generate plots of membrane voltage, ionic currents, and gating variables
for a step current injection: I = 53 μA/cm² for 0 ≤ t < 0.2 ms
"""

import numpy as np
import matplotlib.pyplot as plt
from hh_model import hh_model

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

# Initial conditions
V0 = -60              # mV (resting potential)
m0 = 0.0393
h0 = 0.6798
n0 = 0.2803

# Simulation parameters
dt = 0.01             # ms
t_end = 5             # ms (enough to see the action potential)

# Step current: I = 53 μA/cm² for 0 ≤ t < 0.2 ms, otherwise 0
def I_stim(t):
    return 53 if (t >= 0 and t < 0.2) else 0

# Run simulation
t, Vm, I_Na, I_K, I_L, m, h, n = hh_model(I_stim, dt, t_end, V0, m0, h0, n0, params)

# Calculate total ionic current
I_ion = I_Na + I_K + I_L
I_stim_vec = np.array([I_stim(ti) for ti in t])

# Plot results
plt.figure(figsize=(14, 8))

# Subplot 1: Membrane Voltage
plt.subplot(2, 3, 1)
plt.plot(t, Vm, 'b-', linewidth=1.5)
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Voltage (mV)')
plt.title('Membrane Voltage vs Time')
plt.grid(True)
plt.xlim([0, t_end])

# Subplot 2: Stimulus Current
plt.subplot(2, 3, 2)
plt.plot(t, I_stim_vec, 'r-', linewidth=1.5)
plt.xlabel('Time (ms)')
plt.ylabel('Stimulus Current (μA/cm²)')
plt.title('Stimulus Current')
plt.grid(True)
plt.xlim([0, t_end])

# Subplot 3: Ionic Currents
plt.subplot(2, 3, 3)
plt.plot(t, I_Na, 'r-', linewidth=1.5, label='I$_{Na}$')
plt.plot(t, I_K, 'b-', linewidth=1.5, label='I$_K$')
plt.plot(t, I_L, 'g-', linewidth=1.5, label='I$_L$')
plt.plot(t, I_ion, 'k--', linewidth=1.5, label='I$_{ion}$')
plt.xlabel('Time (ms)')
plt.ylabel('Current (μA/cm²)')
plt.title('Ionic Currents')
plt.legend(loc='best')
plt.grid(True)
plt.xlim([0, t_end])

# Subplot 4: Gating Variables
plt.subplot(2, 3, 4)
plt.plot(t, m, 'r-', linewidth=1.5, label='m')
plt.plot(t, h, 'b-', linewidth=1.5, label='h')
plt.plot(t, n, 'g-', linewidth=1.5, label='n')
plt.xlabel('Time (ms)')
plt.ylabel('Gating Variable')
plt.title('Gating Variables (m, h, n)')
plt.legend(loc='best')
plt.grid(True)
plt.xlim([0, t_end])
plt.ylim([0, 1])

# Subplot 5: Sodium Conductance
plt.subplot(2, 3, 5)
g_Na = params['gNa'] * m**3 * h
plt.plot(t, g_Na, 'r-', linewidth=1.5)
plt.xlabel('Time (ms)')
plt.ylabel('Conductance (mS/cm²)')
plt.title('Sodium Conductance (g$_{Na}$)')
plt.grid(True)
plt.xlim([0, t_end])

# Subplot 6: Potassium Conductance
plt.subplot(2, 3, 6)
g_K = params['gK'] * n**4
plt.plot(t, g_K, 'b-', linewidth=1.5)
plt.xlabel('Time (ms)')
plt.ylabel('Conductance (mS/cm²)')
plt.title('Potassium Conductance (g$_K$)')
plt.grid(True)
plt.xlim([0, t_end])

plt.suptitle('Question 2: Step Current Injection (I = 53 μA/cm², 0-0.2 ms)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
plt.savefig('question2_results.png', dpi=300, bbox_inches='tight')
print('Question 2 completed. Results saved to question2_results.png')
print(f'Peak voltage: {np.max(Vm):.2f} mV')
print(f'Time to peak: {t[np.argmax(Vm)]:.2f} ms')

plt.show()

