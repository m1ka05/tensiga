import numpy as np

class Quadrature:
    
    def __init__(self, deg, ip, weights, W):
        self.deg = deg
        self.ip = ip
        self.weights = weights
        self.W = W

        self.nip = W.size
        self.dim = W.ndim



    
