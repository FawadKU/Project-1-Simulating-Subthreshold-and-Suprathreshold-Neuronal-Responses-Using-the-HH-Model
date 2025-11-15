"""
QUESTION 5: Refractory Period Analysis
Find the earliest time after an initial square pulse stimulus to give a second 
stimulus that leads to an action potential.
Stimulus duration: 0.15 ms, amplitudes: 50, 200, 500 μA/cm²
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
t_end = 50            # ms (enough to see both spikes)

# Stimulus parameters
stim_duration = 0.15  # ms
amplitudes = [50, 200, 500]  # μA/cm²

def detect_spike(Vm, threshold=0):
    """Detect if a spike occurs (voltage crosses threshold from below)"""
    return np.any(np.diff((Vm > threshold).astype(int)) == 1)

def find_refractory_period(I_amp, dt, t_end, V0, m0, h0, n0, params):
    """
    Find the earliest time for a second stimulus to produce an action potential
    """
    stim_duration = 0.15
    
    # First stimulus at t=0
    def I_stim1(t):
        return I_amp if (t >= 0 and t < stim_duration) else 0
    
    # Test different delay times for second stimulus
    delay_times = np.arange(0.5, 30, 0.1)  # Start testing from 0.5 ms
    
    for delay in delay_times:
        # Combined stimulus: first at t=0, second at t=delay
        def I_stim_combined(t):
            stim1 = I_amp if (t >= 0 and t < stim_duration) else 0
            stim2 = I_amp if (t >= delay and t < delay + stim_duration) else 0
            return stim1 + stim2
        
        # Run simulation
        t, Vm, _, _, _, _, _, _ = hh_model(I_stim_combined, dt, t_end, V0, m0, h0, n0, params)
        
        # Check if second spike occurred (after delay)
        Vm_after_delay = Vm[t >= delay]
        if detect_spike(Vm_after_delay):
            return delay
    
    return None  # No second spike found within tested range

print('Question 5: Refractory Period Analysis')
print('=' * 42)
print()

refractory_periods = []
plt.figure(figsize=(16, 10))

for i, I_amp in enumerate(amplitudes):
    print(f'Testing amplitude: {I_amp} μA/cm²')
    
    # Find refractory period
    ref_period = find_refractory_period(I_amp, dt, t_end, V0, m0, h0, n0, params)
    refractory_periods.append(ref_period)
    
    if ref_period is not None:
        print(f'  Earliest time for 2nd stimulus: {ref_period:.2f} ms')
        
        # Create stimulus with second pulse at refractory period
        def I_stim(t):
            stim1 = I_amp if (t >= 0 and t < stim_duration) else 0
            stim2 = I_amp if (t >= ref_period and t < ref_period + stim_duration) else 0
            return stim1 + stim2
        
        # Run simulation
        t, Vm, _, _, _, _, _, _ = hh_model(I_stim, dt, t_end, V0, m0, h0, n0, params)
        
        # Plot
        plt.subplot(2, 3, i+1)
        plt.plot(t, Vm, 'b-', linewidth=1.5, label='V$_m$')
        I_stim_vec = np.array([I_stim(ti) for ti in t])
        plt.plot(t, I_stim_vec / 10 - 80, 'r-', linewidth=2, label='Stimulus (scaled)')
        plt.axvline(ref_period, color='g', linestyle='--', linewidth=1.5, label='2nd stimulus start')
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane Voltage (mV)')
        plt.title(f'I = {I_amp} μA/cm²\nRefractory Period: {ref_period:.2f} ms')
        plt.grid(True)
        plt.xlim([0, t_end])
        plt.legend(loc='best')
        
        # Also test with slightly earlier time (should not produce spike)
        if ref_period > 0.1:
            test_time = ref_period - 0.1
            def I_stim_test(t):
                stim1 = I_amp if (t >= 0 and t < stim_duration) else 0
                stim2 = I_amp if (t >= test_time and t < test_time + stim_duration) else 0
                return stim1 + stim2
            
            t_test, Vm_test, _, _, _, _, _, _ = hh_model(I_stim_test, dt, t_end, V0, m0, h0, n0, params)
            
            plt.subplot(2, 3, i+4)
            plt.plot(t_test, Vm_test, 'b-', linewidth=1.5, label='V$_m$')
            I_stim_vec_test = np.array([I_stim_test(ti) for ti in t_test])
            plt.plot(t_test, I_stim_vec_test / 10 - 80, 'r-', linewidth=2, label='Stimulus (scaled)')
            plt.axvline(test_time, color='g', linestyle='--', linewidth=1.5, label='2nd stimulus start')
            plt.xlabel('Time (ms)')
            plt.ylabel('Membrane Voltage (mV)')
            plt.title(f'I = {I_amp} μA/cm²\nTest: {test_time:.2f} ms (No 2nd spike)')
            plt.grid(True)
            plt.xlim([0, t_end])
            plt.legend(loc='best')
    else:
        print('  No second spike found within tested range')
    print()

plt.suptitle('Question 5: Refractory Period Analysis (Stimulus: 0.15 ms)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# Save figure
plt.savefig('question5_results.png', dpi=300, bbox_inches='tight')

# Summary table
print('\n=== Summary ===')
print('Amplitude (μA/cm²)\tRefractory Period (ms)')
for amp, ref_per in zip(amplitudes, refractory_periods):
    if ref_per is not None:
        print(f'{amp}\t\t\t{ref_per:.2f}')
    else:
        print(f'{amp}\t\t\tNot found')

plt.show()

