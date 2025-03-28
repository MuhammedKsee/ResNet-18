import numpy as np
from ArtificalNeuralNetwork import ANN

class ResNet18:
    def __init__(self):
        pass

    def Conv(self,inputx,weight,bias):
        a = inputx*weight+bias
        return a

    def predict(self,inputx,weight,bias):
        relu = ANN()
        a = self.Conv(inputx,weight[0],bias[0])
        h1 = relu.ReLu(a)
        a2 = self.Conv(h1,weight[1],bias[1])
        h2 = relu.ReLu(a2)
        a3 = self.Conv(h2,weight[2],bias[2])
        h3 = relu.ReLu(a3)
        a4 = self.Conv(h3,weight[3],bias[3])
        h4 = relu.ReLu(a4)
        a5 = self.Conv(h4,weight[4],bias[4])
        return a5 + inputx

model = ResNet18()

x = np.array([1.0, -0.5, 2.0])
weights = [np.random.randn(3) for _ in range(5)]
biases = [np.random.randn() for _ in range(5)]

output = model.predict(x, weights, biases)
print("Sonuc:", output)

