from iznetwork import IzNetwork
import numpy as np

class SWMNetwork:
    def __init__(self, dmax):
        self.N = 1000
        net = IzNetwork(self.N, dmax)

        a = 0.02*np.ones(self.N)
        b = np.concatenate((0.2*np.ones(800), 0.25 * np.ones(200)))
        c = -65*np.ones(self.N)
        d = np.concatenate((8 * np.ones(800), 2 * np.ones(200)))

        F_EE = 17
        F_EI = 50
        F_IE = 2
        F_II = 1

        EE_block = np.zeros((100, 100)) # function generate_random_EE_block
        zero_block = np.zeros((100, 100))
        II_block = -1 * np.random.rand((200, 200))
        np.fill_diagonal(II_block, 0)
        IE_block = -1 * np.random.rand((200, 100))

        """
        [   EE      zero    zero    zero    zero    zero    zero    E-I
            zero    EE      zero    zero    zero    zero    zero    E-I
            ...
            zero    zero    zero    zero    zero    zero    EE      E-I
            I-E     I-E     I-E     I-E     I-E     I-E     I-E     I-I          
        
        ]
        """
        ## TODO generate W matrix
        W = np.bmat([])

        # TODO generate D matrix
        D = dmax*np.ones((self.N, self.N), dtype=int)
        self.rewire()

        net.setParameters(a, b, c, d)
        net.setDelays(D)
        net.setWeights(W)

        return net
    
    def generate_EI_block(self, shift):
        # return (100, 200) matrix
        shift *= 25
        block = np.zeros((100, 200))
        for i, r in enumerate(range(0, 100, step=4)):
            block[r: r+4,shift + i:shift+i+1] = np.random.rand((4, 1))
        return block

    
    ## TODO rewireing
    def rewire(self):
        pass

    def update(self):
        pass