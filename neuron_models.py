import numpy as np
from matplotlib import pyplot as plt

class Brain_Neuronals_Models:
    def __init__(self) -> None:
        pass
    
    def QIF_Neuron(self, num_steps, V_peak, V_reset, I_ext, tau_m, time_steps): 
        # Initialize membrane potentials
        u = np.zeros( num_steps )
        u[0] = V_reset
        # Simulation
        for t in range(1,num_steps):
                du_dt = u[t - 1]**2 + I_ext 
                u[t]  = u[t - 1] + du_dt * time_steps / tau_m 
                if u[t] >= V_peak:
                    u[t-1] = V_peak     # Peak of membrane potential
                    u[t] = V_reset      # Reset of membrane potential
        return u

    def Population_QIF_Neurons(self,num_neurons, num_steps, u, I_ext, g, time_steps,  tau_m , V_peak, V_reset):
        # Initialize average potentials
        u_mean= np.zeros(num_steps)
        r_t= np.zeros(num_steps)
        
        for t in range(num_steps):
            # Calculate the average of the voltages
            v = np.mean(u)
            
            # Differential equation for vector u at each time interval
            du_dt = u**2 + I_ext + g * (v - u)
            u = u + du_dt * time_steps  / tau_m 
            
            # Identify the shots (when u > V_peak)
            spikes = u >= V_peak    # Very high threshold to simulate escape to infinity
            u[spikes] = V_reset     # Voltage reset to V_reset
            
            u_mean[t] = v
            r_t[t] = np.sum(spikes) / (num_neurons * time_steps)  # Average firing rate
        
        return u_mean, r_t
    def Firing_Rate_Model(self,num_steps, r_0, u_0, delta, tau_m, g , n_, J, a, time_steps  ):
        # Variable initialization
        r = np.zeros(num_steps)
        u = np.zeros(num_steps)
        r[0] = r_0
        u[0] = u_0
        
        for t in range(1, num_steps ):
            dr_dt = (delta / (np.pi * tau_m) + 2 * r[t - 1] * u[t - 1] - g * r[t - 1]) 
            du_dt = ( u[t - 1]**2 + n_ - (np.pi * tau_m * r[t - 1]) ** 2 + (J + g * np.log(a))*tau_m * r[t - 1])
            
            r[t] = r[t - 1] + dr_dt * time_steps / tau_m
            u[t] = u[t - 1] + du_dt * time_steps / tau_m
        
        return r, u

    def timeSeries2bins(self,ts, dt, binInterval, op):
        count = len(ts)
        numBins, binSize = np.divmod(count, binInterval / dt)
        sorted_ts = ts.reshape((int(numBins), int(binInterval / dt)))
        bins = op(sorted_ts, axis=1)
        return bins
    
    
class Graph_Neuronal_Models:
    def __init__(self) -> None:
        pass
    
    def Graph_QIF_Neuron(self, time_vector , V_QIF_Neuron, a):
        
        x_position_text = int(0.1 * time_vector.max() )         # Text position on x-axes
        y_position_text = int(0.75 * V_QIF_Neuron.max() )       # Text position on y-axes
        
        pos_peak_max = np.argmax(V_QIF_Neuron)                  # Position of the first maximum value of the QIF_Neuron vector
        
        y_max = int(V_QIF_Neuron.max() )                        # Maximum value of the QIF_Neuron vector
        y_min= int(V_QIF_Neuron.min() )                         # Minimum value of the QIF_Neuron vector
        
        y_lim_max = 1.3* y_max                                  # Maximum value of y-axes
        y_lim_min = 1.3*y_min                                   # Minimum value of y-axes
        
        x_lim_max = int(time_vector.max())                      # Maximum value of x-axes
        x_lim_min = 0                                           # Minimum value of x-axes
        
        x_peak = time_vector[pos_peak_max]                      # Position of the first maximum value of the QIF_Neuron vector in x-axes
        
        # Plotting results
        fig, ax = plt.subplots(figsize=(7 , 3))
        plt.rcParams['text.usetex'] = True                      # For use of latex in the graphic
        
        ax.set_title(r'Quadratic Integrate-and-Fire Neuron')   
        ax.plot(time_vector, V_QIF_Neuron,color = 'blue', label = r'QIF Neuron')
        ax.set_xlabel(r'Time (ms)')
        ax.set_ylabel(r'Voltage [mV]')
        ax.set_ylim(y_lim_min, y_lim_max)
        ax.set_xlim(x_lim_min, x_lim_max)
        
        ax.text(x_position_text, y_position_text , r'$a = (\frac{V_p}{V_r})$ = '+ fr'${a}$', fontsize = 12)
        
        ax.annotate(fr'$V_p = {y_max}$', xy=( x_peak, y_max), xytext=(1.25 * x_peak,  y_max), arrowprops=dict(facecolor='black', shrink=0.005, width=0.5, headwidth=3), fontweight='bold')
        ax.annotate(fr'$V_r = {y_min}$', xy=( x_peak , y_min), xytext=(1.25 * x_peak,  y_min), arrowprops=dict(facecolor='black', shrink=0.005, width=0.5, headwidth=3), fontweight='bold')
        
        plt.subplots_adjust(top=0.88)
        plt.tight_layout()
        plt.legend()

        
    def Graph_Models(self,r_qif, r_fre, binInterval,a, V_peak):
        
        x = np.arange(len(r_qif)) * binInterval                     # X array 
        x_position_text = 0.8 * len(r_qif)* binInterval             # Text position on x-axes
        y_position_text = int(r_fre.max())                          # Text position on y-axes
        y_lim_max = y_position_text * 1.2                           # Maximum value of y-axes
        
        # Plotting results
        fig, ax = plt.subplots(figsize=(7 , 3))
        plt.rcParams['text.usetex'] = True 
        
        ax.set_title(r'Verification Population Model')
        ax.plot(x, r_qif, color = 'black',  label = r'QIF Network' )
        ax.plot(x, r_fre, color = 'red', label = r'Firing Rate Model')
        ax.set_xlabel(r'Time (ms)')
        ax.set_ylabel(r'r (Hz)')
        ax.set_ylim(0,y_lim_max)
        ax.set_ylim(0,y_lim_max)
        ax.text(x_position_text, y_position_text , fr'$a = {a} , V_p = {V_peak} $', fontsize = 12)
        
        plt.subplots_adjust(top=0.88)
        plt.tight_layout()
        plt.legend()
        plt.show()