from iznetwork import IzNetwork
import numpy as np
from PIL import Imag

def create_EE_block(n_neurons=100, n_edges=1000, weight=1):
    full_edges = n_neurons*(n_neurons-1)
    block = [0 for _ in range(full_edges)]
    edges = np.random.choice(full_edges, n_edges, replace=False)
    for i in edges:
        block[i] = weight
    i = 0
    while i < n_neurons*n_neurons:
        block.insert(i, 0)
        i += n_neurons+1
    # 100 by 100 matrix
    return np.array(block).reshape((n_neurons, n_neurons))

def create_EI_block(shift, n_rows=100, n_cols=200):
    # return (100, 200) matrix
    block = np.zeros((n_rows, n_cols))
    for i, r in enumerate(range(0, 100, 4)):
        block[r: r+4, shift+i: shift+i+1] = np.random.random(size=(4, 1))
    return block

def plot_weight_matrix(weight, fn="static/weight.png"):
    weight = np.array(255 * (weight-(-1)) / 2, dtype=np.uint8)
    Image.fromarray(weight, mode="L").save(fn)

class SWMNetwork:
    def __init__(self, EE_module_neurons = 100, EE_module_edges = 1000, EE_module_num = 8, i_neuron_num = 200, dmax=1):
        self.ee_m_neurons = EE_module_neurons
        self.ee_m_edges = EE_module_edges
        self.modules_num = EE_module_num
        self.i_neurons = i_neuron_num

        self.N = self.ee_m_neurons * self.modules_num + self.i_neurons
        self.net = IzNetwork(self.N, dmax)

        a = 0.02*np.ones(self.N)
        b = np.concatenate((0.2*np.ones(self.ee_m_neurons * self.modules_num), 0.25 * np.ones(self.i_neurons)))
        c = -65*np.ones(self.N)
        d = np.concatenate((8 * np.ones(self.ee_m_neurons * self.modules_num), 2 * np.ones(self.i_neurons)))

        F_EE = 17
        F_EI = 50
        F_IE = 2
        F_II = 1

        zero_block = np.zeros((self.ee_m_neurons, self.ee_m_neurons))
        II_block = -1 * np.random.random(size=(self.i_neurons, self.i_neurons))
        np.fill_diagonal(II_block, 0)
        IE_blocks = [-1 * np.random.random(size=(self.i_neurons, self.ee_m_neurons)) for _ in range(self.modules_num)]
        EI_blocks = [create_EI_block(i, self.ee_m_neurons, self.i_neurons) for i in range(0, self.i_neurons, 25)]

        """
        [   EE      zero    zero    zero    zero    zero    zero    E-I
            zero    EE      zero    zero    zero    zero    zero    E-I
            ...
            zero    zero    zero    zero    zero    zero    EE      E-I
            I-E     I-E     I-E     I-E     I-E     I-E     I-E     I-I          
        
        ]
        """
        ## TODO generate W matrix
        W = np.bmat([
            [zero_block for _ in range(0, i)] + [create_EE_block(self.ee_m_neurons, self.ee_m_edges)] + [zero_block for _ in range(i+1, self.modules_num)] + [EI_blocks[i]] for i in range(self.modules_num)
        ] + [IE_blocks + [II_block]])

        plot_weight_matrix(W)

        # TODO generate D matrix
        D = dmax*np.ones((self.N, self.N), dtype=int)
        self.rewire()

        self.net.setParameters(a, b, c, d)
        self.net.setDelays(D)
        self.net.setWeights(W)    
   
    ## TODO rewireing
    def rewire(self):
        pass

    def update(self):
        pass

if __name__ == "__main__":
    swm = SWMNetwork()
