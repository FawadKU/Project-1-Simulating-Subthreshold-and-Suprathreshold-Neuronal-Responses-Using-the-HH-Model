"""
Hodgkin-Huxley Model Implementation
Following the algorithm from Bioelectricity: A Quantitative Approach, Chapter 5
Pages 131-140
"""

import numpy as np


def calculate_steady_state_gating(V, params):
    """
    Calculate steady-state gating variables at a given voltage V.
    
    Parameters:
    -----------
    V : float
        Membrane potential (mV)
    params : dict
        Dictionary with model parameters (must include 'T')
    
    Returns:
    --------
    m_inf : float
        Steady-state sodium activation
    h_inf : float
        Steady-state sodium inactivation
    n_inf : float
        Steady-state potassium activation
    """
    T = params['T']
    Q10 = 3
    phi = Q10 ** ((T - 6.3) / 10)
    
    # Calculate alpha and beta
    # Sodium activation (m)
    if abs(V + 40) < 1e-6:
        alpha_m = phi * 0.1
    else:
        alpha_m = phi * 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta_m = phi * 4 * np.exp(-(V + 65) / 18)
    
    # Sodium inactivation (h)
    alpha_h = phi * 0.07 * np.exp(-(V + 65) / 20)
    beta_h = phi * 1 / (1 + np.exp(-(V + 35) / 10))
    
    # Potassium activation (n)
    if abs(V + 55) < 1e-6:
        alpha_n = phi * 0.01
    else:
        alpha_n = phi * 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    beta_n = phi * 0.125 * np.exp(-(V + 65) / 80)
    
    # Steady-state values
    m_inf = alpha_m / (alpha_m + beta_m)
    h_inf = alpha_h / (alpha_h + beta_h)
    n_inf = alpha_n / (alpha_n + beta_n)
    
    return m_inf, h_inf, n_inf


def hh_model(I_stim, dt, t_end, V0, m0, h0, n0, params):
    """
    Hodgkin-Huxley Model Simulation
    Following the 4-step algorithm from the textbook:
    1. Set Im (stimulus current)
    2. Estimate ΔVm (equation 5.53)
    3. Estimate Δn, Δm, Δh (equations 5.56-5.58)
    4. Advance to next time (equations 5.59-5.62)
    
    Parameters:
    -----------
    I_stim : callable or array-like
        Stimulus current density (μA/cm²) as function of time or array
    dt : float
        Time step (ms) - corresponds to Δt in textbook
    t_end : float
        End time (ms)
    V0 : float
        Initial membrane potential (mV)
    m0, h0, n0 : float
        Initial gating variable values
    params : dict
        Dictionary with model parameters:
        - gNa: maximum Na+ conductivity (mS/cm²) - corresponds to ḡNa
        - gK: maximum K+ conductivity (mS/cm²) - corresponds to ḡK
        - gL: leakage conductivity (mS/cm²)
        - Cm: membrane capacitance (μF/cm²)
        - ENa: Na+ Nernst potential (mV)
        - EK: K+ Nernst potential (mV)
        - EL: leakage Nernst potential (mV)
        - T: temperature (°C)
    
    Returns:
    --------
    t : ndarray
        Time vector (ms)
    Vm : ndarray
        Membrane voltage (mV) vs time
    I_Na : ndarray
        Sodium current (μA/cm²)
    I_K : ndarray
        Potassium current (μA/cm²)
    I_L : ndarray
        Leakage current (μA/cm²)
    m : ndarray
        Sodium activation gating variable
    h : ndarray
        Sodium inactivation gating variable
    n : ndarray
        Potassium activation gating variable
    """
    
    # Extract parameters
    gNa_max = params['gNa']  # ḡNa in textbook
    gK_max = params['gK']     # ḡK in textbook
    gL = params['gL']
    Cm = params['Cm']
    ENa = params['ENa']
    EK = params['EK']
    EL = params['EL']
    T = params['T']
    
    # Temperature scaling factor (Q10 = 3)
    # From equation (5.64): Q = 3^P where P = (T - 6.3)/10
    Q10 = 3
    phi = Q10 ** ((T - 6.3) / 10)  # This is Q in the textbook
    
    # Initialize time vector
    t = np.arange(0, t_end + dt, dt)
    N = len(t)
    
    # Initialize state variables
    Vm = np.zeros(N)
    Vm[0] = V0
    m = np.zeros(N)
    m[0] = m0
    h = np.zeros(N)
    h[0] = h0
    n = np.zeros(N)
    n[0] = n0
    
    # Initialize currents
    I_Na = np.zeros(N)
    I_K = np.zeros(N)
    I_L = np.zeros(N)
    
    # STEP 1: Set Im (membrane current = stimulus current)
    # Convert I_stim to array (Im in textbook)
    if callable(I_stim):
        Im = np.array([I_stim(ti) for ti in t])
    elif np.isscalar(I_stim):
        Im = I_stim * np.ones(N)
    else:
        Im = np.array(I_stim)
        if len(Im) != N:
            raise ValueError(f"I_stim length ({len(Im)}) must match time vector length ({N})")
    
    # Main simulation loop
    for i in range(N - 1):
        # All calculations at time i (before advancing)
        V = Vm[i]
        m_i = m[i]
        h_i = h[i]
        n_i = n[i]
        
        # Calculate alpha and beta rate constants at time i
        # These use equations from earlier in Chapter 5
        
        # Sodium activation (m) - equations 5.36
        if abs(V + 40) < 1e-6:
            alpha_m = phi * 0.1  # Handle singularity at V = -40 mV
        else:
            alpha_m = phi * 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta_m = phi * 4 * np.exp(-(V + 65) / 18)
        
        # Sodium inactivation (h) - equations 5.37
        alpha_h = phi * 0.07 * np.exp(-(V + 65) / 20)
        beta_h = phi * 1 / (1 + np.exp(-(V + 35) / 10))
        
        # Potassium activation (n) - equations 5.24, 5.25
        if abs(V + 55) < 1e-6:
            alpha_n = phi * 0.01  # Handle singularity at V = -55 mV
        else:
            alpha_n = phi * 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta_n = phi * 0.125 * np.exp(-(V + 65) / 80)
        
        # STEP 2: Estimate ΔVm (equation 5.53)
        # First calculate conductances at time i (equation 5.55)
        g_Na_i = gNa_max * m_i**3 * h_i
        g_K_i = gK_max * n_i**4
        
        # Calculate ionic currents at time i (equations 5.54)
        I_Na_i = g_Na_i * (V - ENa)
        I_K_i = g_K_i * (V - EK)
        I_L_i = gL * (V - EL)
        
        # Total ionic current at time i
        I_ion_i = I_Na_i + I_K_i + I_L_i
        
        # Store currents for output
        I_Na[i] = I_Na_i
        I_K[i] = I_K_i
        I_L[i] = I_L_i
        
        # Calculate ΔVm (equation 5.53)
        delta_Vm = (dt / Cm) * (Im[i] - I_ion_i)
        
        # STEP 3: Estimate Δn, Δm, Δh (equations 5.56, 5.57, 5.58)
        # Using explicit form from textbook
        delta_n = dt * (alpha_n * (1 - n_i) - beta_n * n_i)
        delta_m = dt * (alpha_m * (1 - m_i) - beta_m * m_i)
        delta_h = dt * (alpha_h * (1 - h_i) - beta_h * h_i)
        
        # STEP 4: Advance to next time (equations 5.59-5.62)
        Vm[i+1] = Vm[i] + delta_Vm
        n[i+1] = n[i] + delta_n
        m[i+1] = m[i] + delta_m
        h[i+1] = h[i] + delta_h
    
    # Calculate final currents at last time point
    V = Vm[N-1]
    m_final = m[N-1]
    h_final = h[N-1]
    n_final = n[N-1]
    
    g_Na_final = gNa_max * m_final**3 * h_final
    g_K_final = gK_max * n_final**4
    
    I_Na[N-1] = g_Na_final * (V - ENa)
    I_K[N-1] = g_K_final * (V - EK)
    I_L[N-1] = gL * (V - EL)
    
    return t, Vm, I_Na, I_K, I_L, m, h, n

