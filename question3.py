"""
QUESTION 3: Sinusoidal Stimulation
Generate spike count vs stimulus frequency plot
Peak amplitude: 15 μA/cm², frequencies: 1, 2, 5, 10, 20, 50, 100 Hz
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
t_end = 1000          # ms (1 second)

# Frequencies to test (Hz)
frequencies = np.array([1, 2, 5, 10, 20, 50, 100])
spike_counts = np.zeros(len(frequencies))
I_peak = 15           # μA/cm²

# Function to detect spikes (threshold: 0 mV, crossing from below)
def detect_spikes(Vm):
    """Count spikes by detecting upward crossings of 0 mV"""
    crossings = np.diff((Vm > 0).astype(int))
    return np.sum(crossings == 1)

# Store results for plotting
Vm_results = []
I_stim_results = []

print('Running sinusoidal stimulation simulations...')

for i, f in enumerate(frequencies):
    print(f'  Frequency: {f} Hz')
    
    # Sinusoidal current: I = I_peak * sin(2*pi*f*t)
    # Convert frequency from Hz to cycles per ms
    def I_stim(t):
        return I_peak * np.sin(2 * np.pi * f * t / 1000)  # f in Hz, t in ms
    
    # Run simulation
    t, Vm, _, _, _, _, _, _ = hh_model(I_stim, dt, t_end, V0, m0, h0, n0, params)
    
    # Detect spikes
    spike_count = detect_spikes(Vm)
    spike_counts[i] = spike_count
    
    # Store for plotting (first 100 ms for visualization)
    idx_plot = t <= 100
    Vm_results.append(Vm[idx_plot])
    I_stim_results.append(np.array([I_stim(ti) for ti in t[idx_plot]]))
    
    print(f'    Spike count: {spike_count}')

# Create comprehensive plot
plt.figure(figsize=(16, 10))

# Plot 1: Spike count vs frequency
plt.subplot(2, 3, (1, 2))
plt.semilogx(frequencies, spike_counts, 'o-', linewidth=2, markersize=8)
plt.xlabel('Stimulus Frequency (Hz)')
plt.ylabel('Spike Count (spikes/second)')
plt.title('Spike Count vs Stimulus Frequency')
plt.grid(True)
plt.xlim([0.5, 150])

# Plot input and spiking patterns for selected frequencies
selected_freqs = [1, 10, 50, 100]
selected_idx = [0, 3, 5, 6]  # Indices in frequencies array

for i, (freq, idx) in enumerate(zip(selected_freqs, selected_idx)):
    plt.subplot(2, 3, i+3)
    t_plot = t[t <= 100]
    
    ax1 = plt.gca()
    ax1.plot(t_plot, I_stim_results[idx], 'r-', linewidth=1.5, label='Stimulus')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Stimulus Current (μA/cm²)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_ylim([-20, 20])
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.plot(t_plot, Vm_results[idx], 'b-', linewidth=1.5, label='Voltage')
    ax2.set_ylabel('Membrane Voltage (mV)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim([-80, 60])
    
    plt.title(f'f = {frequencies[idx]} Hz (Spikes: {int(spike_counts[idx])})')

plt.suptitle('Question 3: Sinusoidal Stimulation Response (I$_{peak}$ = 15 μA/cm²)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
plt.savefig('question3_results.png', dpi=300, bbox_inches='tight')

# Print summary
print('\n=== Question 3 Results ===')
print('Frequency (Hz)\tSpike Count')
for freq, count in zip(frequencies, spike_counts):
    print(f'{freq}\t\t{int(count)}')

plt.show()

