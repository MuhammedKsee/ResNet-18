import numpy as np
from ArtificalNeuralNetwork import ANN

class ResNet18:
    def __init__(self):
        pass

    def Conv(self,inputx,weight,bias):
        a = inputx*weight+bias
        return a

    def predict(self,inputx,weight1,bias1,weight2,bias2):
        relu = ANN()
        a = self.Conv(inputx,weight1,bias1)
        h1 = relu.ReLu(a)
        a2 = self.Conv(h1,weight2,bias2)
        return a2 + inputx

model = ResNet18()

x = np.array([1.0, -0.5, 2.0])
w1 = np.random.randn(3)
b1 = np.random.randn()
w2 = np.random.randn(3)
b2 = np.random.randn()

output = model.predict(x, w1, b1, w2, b2)
print("Sonuc:", output)





