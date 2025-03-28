import numpy as np
class ANN:
    def __init__(self):
        self.x = None
        self.w = None
        self.y = 0
        self.bias = 0
    
    def ReLu(self,a):
        return np.maximum(0, a)
   
    def summation(self):
        a = np.dot(self.x, self.w)+self.bias
        return a
    def output(self):
        a = self.summation()
        output = self.ReLu(a)
        return output



