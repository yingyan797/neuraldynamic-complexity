from iznetwork import IzNetwork
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import networkx as nx

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
    l, h = np.min(weight), np.max(weight)
    imarr = np.zeros((weight.shape[0], weight.shape[1], 3))
    imarr[:,:, 0] = weight
    def set_color(w):
        if w[0] == 0:
            return np.array([0,0,0], dtype=np.uint8)
        if w[0] < 0:
            return np.array([0, 0, 55+200*w[0]/l], dtype=np.uint8)
        return np.array([200+55*w[0]/h, 0, 0], dtype=np.uint8)
    imarr = np.apply_along_axis(set_color, 2, imarr)
    Image.fromarray(imarr, mode="RGB").save(fn)

class SWMNetwork:
    def __init__(self, EE_module_neurons=100, EE_module_edges=1000, EE_module_num=8, i_neuron_num=200, p=0.1, dmax=20):
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
        II_block = -1 * np.random.random(size=(self.i_neurons, self.i_neurons)) * F_II
        np.fill_diagonal(II_block, 0)
        IE_blocks = [-1 * np.random.random(size=(self.i_neurons, self.ee_m_neurons)) * F_IE for _ in range(self.modules_num)]
        EI_blocks = [create_EI_block(i, self.ee_m_neurons, self.i_neurons) * F_EI for i in range(0, self.i_neurons, 25)]

        """
        [   EE      zero    zero    zero    zero    zero    zero    E-I
            zero    EE      zero    zero    zero    zero    zero    E-I
            ...
            zero    zero    zero    zero    zero    zero    EE      E-I
            I-E     I-E     I-E     I-E     I-E     I-E     I-E     I-I          
        
        ]
        """
        ## generate W matrix
        W = np.bmat([
            [zero_block for _ in range(0, i)] + [create_EE_block(self.ee_m_neurons, self.ee_m_edges) * F_EE] + [zero_block for _ in range(i+1, self.modules_num)] + [EI_blocks[i]] for i in range(self.modules_num)
        ] + [IE_blocks + [II_block]])

        plot_weight_matrix(W)

        # generate D matrix
        D = dmax*np.ones((self.N, self.N), dtype=int)
        ee_matrix = self.ee_m_neurons * self.modules_num
        D[:ee_matrix, :ee_matrix] = 1 + np.random.random(size=(ee_matrix, ee_matrix)) * 19 # random delay between 1ms and 20ms

        self._rewire(W, p)
        plot_weight_matrix(W, "static/weight_rewired.png")

        self.net.setParameters(a, b, c, d)
        self.net.setDelays(D)
        self.net.setWeights(W)    
   
    ## TODO rewireing
    def _rewire(self, W, p):
        n_candidates = self.ee_m_neurons*(self.modules_num-1)
        for i in range(self.modules_num):
            # Work out bounds of the current block
            block_start = i * self.ee_m_neurons
            block_end = (i + 1) * self.ee_m_neurons
            for j in range(block_start, block_end):
                n_avail = n_candidates
                for k in range(block_start, block_end):
                    # Record current weight, only process if there is an edge
                    current_weight = W[j, k]
                    if current_weight != 0 and np.random.choice([True, False], p=[p, 1 - p]):
                        # New target must be from other block
                        new_target = int(n_avail * np.random.random())+1
                        t, acc = 0, 0
                        while t < self.modules_num * self.ee_m_neurons:
                            if t == block_start:
                                t += self.ee_m_neurons
                                continue
                            if W[j,t] == 0:
                                acc += 1
                            if acc == new_target:
                                break
                            t += 1

                        # Process the rewiring
                        n_avail -= 1
                        W[j, k] = 0
                        W[j, t] = current_weight

    def simulate(self, period=1000):
        ntot_neurons = self.ee_m_neurons * self.modules_num + self.i_neurons
        fire_time = []
        fire_num = []
        ps_dtbn = np.random.poisson(lam=0.01, size=(ntot_neurons, period))
        for t in range(period):
            I = 15 * ps_dtbn[:, t]
            self.net.setCurrent(I)
            for i in filter(lambda n: n < self.modules_num*self.ee_m_neurons, self.net.update()):
                fire_time.append(t)
                fire_num.append(i)
            
        plt.title("Scatter plot")
        plt.figure(figsize=(10, 5))
        plt.xlabel("Time (ms)")
        plt.ylabel("Neuron ID")
        plt.scatter(fire_time, fire_num, s=1)
        plt.savefig("static/raster.png")        

if __name__ == "__main__":
    swm = SWMNetwork()
    swm.simulate()
