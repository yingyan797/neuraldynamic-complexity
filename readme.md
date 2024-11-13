# Izhikevich network dynamical complexity experiment

SWMNetwork is the a Python class for simulating spiking neurons in a small-world modu-
lar network. This class uses the IzNetwork class from the iznetwork package to simulate the
dynamics of a Izhikevich neuron model. Experiments are performed in the main function of
swm_network.py.

## Denpendancy
iznetwork: Contains the IzNetwork class, which simulates neural dynamics.
numpy: Used for matrix operations and random sampling.
PIL: For image manipulation and saving.
matplotlib: Used to create raster plots and firing rate plots.

## Class Structure
1. init () Method
Initializes the network with the following parameters:
EE module neurons (int): The number of neurons in one excitatory-excitatory module.
EE module edges (int): The number of edges in one excitatory-excitatory module.
EE module num (int): The number of excitatory-excitatory modules in the network.
i neuron num (int): The number of inhibitory neurons in the network.
p (float): The probability of rewiring connections to introduce small-world properties.
dmax (int): The maximum delay allowed while updating the network.

2. Key Methods
plot weight matrix(fn): Visualizes the weight matrix of synaptic connections, saving the plot
to a specified file.
simulate(period, step size): Runs the network simulation for a given time period, gener-
ating raster and mean firing rate plots.
rewire(p): Rewires connections with probability p to introduce small-world characteristics.
mean firing rate(spike counts, step size): Calculates mean firing rates for each module
over the simulation period.

3. Visualization Output
Connectivity Matrix: Scaled weights for each value of rewiring probability p.
Raster Plot: Plots showing spike times for neurons in the network.
Mean Firing Rate Plot: Plots showing the mean firing rate over time for each module.