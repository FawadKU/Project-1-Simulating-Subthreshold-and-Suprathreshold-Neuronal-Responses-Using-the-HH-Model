# Hodgkin-Huxley Model Simulation - Project 1

This project implements the Hodgkin-Huxley (HH) model for simulating neuronal membrane dynamics.

## Requirements

Install the required packages: 
```bash
pip install -r requirements.txt
```

## Files

- `hh_model.py`: Core HH model implementation
- `question2.py`: Step current injection (53 μA/cm² for 0.2 ms)
- `question3.py`: Sinusoidal stimulation with frequency analysis
- `question4.py`: Time to action potential peak analysis
- `question5.py`: Refractory period analysis
- `question6.py`: Anode break excitation simulation

## Usage

Run each question script individually:

```bash
python question2.py
python question3.py
python question4.py
python question5.py
python question6.py
```

Each script will:
1. Run the simulation
2. Generate plots
3. Save results as PNG files
4. Print analysis results to the console

## Parameters

The model uses the following default parameters (from Table 13.2):
- gNa = 120 mS/cm²
- gK = 36 mS/cm²
- gL = 0.3 mS/cm²
- Cm = 1.0 μF/cm²
- ENa = 52.4 mV
- EK = -72.1 mV
- EL = -49.2 mV
- Temperature = 6.3 °C
- Initial conditions: m₀ = 0.0393, h₀ = 0.6798, n₀ = 0.2803

## Results

Each script generates plots saved as:
- `question2_results.png`
- `question3_results.png`
- `question4_results.png`
- `question5_results.png`
- `question6_results.png`

Thank You
