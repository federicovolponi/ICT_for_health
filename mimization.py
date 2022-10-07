import numpy as np
import matplotlib.pyplot as plt
class SolveMinProbl:
    def __init__(self, y=np.ones((3,)), A=np.eye(3)):
        self.matr = A
        self.Np = y.shape[0]
        self.Nf = A.shape[1]
        self.vect = y
        self.sol = np.zeros((self.Nf,),dtype=float)
        return
    
    
        