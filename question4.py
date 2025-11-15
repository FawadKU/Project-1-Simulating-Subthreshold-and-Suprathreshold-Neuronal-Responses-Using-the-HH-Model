"""
QUESTION 4: Time to Action Potential Peak
Find time interval from stimulus start to peak of action potential
for different stimulus amplitudes: 50, 200, 500 μA/cm²
Stimulus duration: 0.15 ms
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
V0 = -60              # mV
m0 = 0.0393
h0 = 0.6798
n0 = 0.2803

# Simulation parameters
dt = 0.01             # ms
t_end = 10            # ms

# Stimulus parameters
stim_duration = 0.15  # ms
amplitudes = [50, 200, 500]  # μA/cm²

time_to_peak = []
peak_voltages = []

print('Question 4: Time to Action Potential Peak')
print('=' * 42)
print()

plt.figure(figsize=(16, 5))

for i, I_amp in enumerate(amplitudes):
    # Square pulse: I = I_amp for 0 ≤ t < stim_duration, otherwise 0
    def I_stim(t):
        return I_amp if (t >= 0 and t < stim_duration) else 0
    
    # Run simulation
    t, Vm, _, _, _, _, _, _ = hh_model(I_stim, dt, t_end, V0, m0, h0, n0, params)
    
    # Find peak voltage and time
    peak_idx = np.argmax(Vm)
    peak_voltage = Vm[peak_idx]
    peak_time = t[peak_idx]
    time_to_peak.append(peak_time)
    peak_voltages.append(peak_voltage)
    
    # Plot
    plt.subplot(1, 3, i+1)
    plt.plot(t, Vm, 'b-', linewidth=1.5, label='V$_m$')
    plt.plot([0, stim_duration], [V0, V0], 'r-', linewidth=3, label='Stimulus')
    plt.plot(peak_time, peak_voltage, 'ro', markersize=10, linewidth=2, label='Peak')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Voltage (mV)')
    plt.title(f'I = {I_amp} μA/cm²\nTime to peak: {peak_time:.3f} ms')
    plt.grid(True)
    plt.xlim([0, t_end])
    plt.legend(loc='best')
    
    print(f'Amplitude: {I_amp} μA/cm²')
    print(f'  Peak voltage: {peak_voltage:.2f} mV')
    print(f'  Time to peak: {peak_time:.3f} ms')
    print(f'  Time from stimulus end: {peak_time - stim_duration:.3f} ms')
    print()

plt.suptitle('Question 4: Time to Action Potential Peak (Stimulus: 0.15 ms)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
plt.savefig('question4_results.png', dpi=300, bbox_inches='tight')

# Summary table
print('\n=== Summary ===')
print('Amplitude (μA/cm²)\tTime to Peak (ms)\tPeak Voltage (mV)')
for amp, ttp, pv in zip(amplitudes, time_to_peak, peak_voltages):
    print(f'{amp}\t\t\t{ttp:.3f}\t\t\t{pv:.2f}')

plt.show()

