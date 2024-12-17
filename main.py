import numpy as np
from neuron_models import Brain_Neuronals_Models, Graph_Neuronal_Models

Brain_Models = Brain_Neuronals_Models()
Graph = Graph_Neuronal_Models()
# QIF Neuron Parameters
a = 1
V_peak = 100                # Max Peak Value 
V_reset = - V_peak/a        # Min Peak Value after Triggering

I_ext = 1                   # External Current
tau_m = 10                  # Membrane’s Time Constant

# QIF Model Parameters
Neurons = 10000             # Number of Neurons
g = 2.5                     # Electrical Coupling

# Voltage initialization 
u_0 = 1
r_0 = 0.015

# Generation of heterogeneous external currents
MMX = Neurons               # Number of Neurons
n_0 = 1
delta = 1
j = np.arange(MMX)
inc = (2 * (j + 1) - MMX - 1) / (MMX + 1)
cmat = n_0 + delta * np.tan(np.pi / 2 * inc)

# Initialization of neuron voltages
u = u_0 + r_0 * np.pi * tau_m * np.tan(np.pi / 2 * inc)

#Verification Population Model
I_t = 0
n_ = n_0 + I_t
J = 0

# Simulation Time
start_time = 0              # Start of simulation time (ms)
stop_time = 120              # Stop of simulation time (ms)
time_steps = 10e-4          # Time step of simulation (ms)
num_steps = int((stop_time - start_time) / time_steps)
time_vector = np.arange(start_time, stop_time, time_steps)

# QIF Neurons
V_QIF_Neuron = Brain_Models.QIF_Neuron(num_steps, V_peak, V_reset, I_ext, tau_m, time_steps)

# Population QIF Neurons
V_mean_QIF, r_t = Brain_Models.Population_QIF_Neurons(Neurons, num_steps, u, cmat, g, time_steps,  tau_m , V_peak, V_reset)

# Model Firing Rate Ecuations
r_fre, u_fre = Brain_Models.Firing_Rate_Model(num_steps, r_0, u_0, delta, tau_m, g , n_, J, a ,time_steps )

# Bins for average firing rate
binInterval = 0.1  # Bin interval in seconds
r_binned_qif = Brain_Models.timeSeries2bins(r_t, time_steps, binInterval, np.sum)
r_binned_fre = Brain_Models.timeSeries2bins(r_fre, time_steps, binInterval, np.sum)

Graph.Graph_QIF_Neuron( time_vector , V_QIF_Neuron, a)
Graph.Graph_Models(r_binned_qif, r_binned_fre, binInterval, a, V_peak)






