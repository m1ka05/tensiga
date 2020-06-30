import numpy as np

class Quadrature:
    
    def __init__(self, ip, weights, W):
        self.ip = ip
        self.weights = weights
        self.W = W

        self.nip = W.size
        self.dim = W.ndim



    
