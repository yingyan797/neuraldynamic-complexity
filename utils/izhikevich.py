from matplotlib import pyplot as plt
import numpy as np

''' Runge Kutta 4th order incremental step
'''
def _rk4_increment(grad, state, dt):
    k1 = grad(state)
    k2 = grad(state+0.5*dt*k1)
    k3 = grad(state+0.5*dt*k2)
    k4 = grad(state+dt*k3)
    return (k1+k2+k3+k4)*dt/6

''' Izhikevich neuron
'''
class Neuron:
    def __init__(self, excitatory:bool):
        self.excitatory = excitatory
        self.a, self.c = 0.02, -65
        if excitatory:
            self.b, self.d = 0.2, 8
        else:
            self.b, self.d = 0.25, 2

        self.state = np.array([-65, -1], dtype=float) # 2D vector representing V and U potentials

    def time_step(self, in_current, dt):
        def gradient(state:np.ndarray):     # [dv/dt, du/dt]
            return np.array([np.power(state[0], 2)/25 + 5*state[0] + 140 -state[1] + in_current,
                        self.a * (self.b * state[0] - state[1])])
        
        self.state += _rk4_increment(gradient, self.state, dt)
        if self.state[0] >= 30:     
            self.state[0] = self.c
            self.state[1] += self.d
            return True     # Firing
        return False        # No firing
        

    


