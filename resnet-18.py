import numpy as np
from ArtificalNeuralNetwork import ANN

class ResNet18:
    def __init__(self):
        pass

    def gap(self,x):
        return np.mean(x)

    def fit(self,x,y,b_fc,w_fc,learningRate = 0.01 ,epochs = 50):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(x)):
                x_i = x[i]
                y_true = y[i]

                # forward
                out = self.predict(x_i,weights,biases)
                pooled = self.gap(out)
                y_pred = pooled * w_fc + b_fc

                # loss (MSE)
                loss = (y_pred - y_true) ** 2
                total_loss +=loss

                # gradients (MSE derivate)
                dW = 2* (y_pred - y_true) * pooled
                db = 2* (y_pred - y_true)

                w_fc -= learningRate * dW
                b_fc -= learningRate * db

            avg_loss = total_loss / len(x)            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        return w_fc,b_fc

    def Conv(self,x,weight,bias):
        a = np.dot(x, weight)+bias
        return a
    
    def Conv2_x(self,x,weight,bias):
        relu = ANN()
        # first block
        h = self.Conv(x,weight[0],bias[0])
        h1 = relu.ReLu(h)
        h2 = self.Conv(h1,weight[1],bias[1])
        y = relu.ReLu(h2+x)

        # second block
        h3 = self.Conv(y,weight[2],bias[2])
        h4 = relu.ReLu(h3)
        h5 = self.Conv(h4,weight[3],bias[3])
        y2 = relu.ReLu(h5)
        return y2 + y

    def Conv3_x(self, x, weight, bias):
        relu = ANN()
        # first block
        h = self.Conv(x,weight[0],bias[0])
        h1 = relu.ReLu(h)
        h2 = self.Conv(h1,weight[1],bias[1])
        shortcut = self.Conv(x, weight[2], bias[2])
        y = relu.ReLu(h2 + shortcut)

        # second block
        h3 = self.Conv(y,weight[3],bias[3])
        h4 = relu.ReLu(h3)
        h5 = self.Conv(h4,weight[4],bias[4])
        return relu.ReLu(h5+y)
    
    def Conv4_x(self, x, weight, bias):
        relu = ANN()
        # first block
        h = self.Conv(x,weight[0],bias[0])
        h1 = relu.ReLu(h)
        h2 = self.Conv(h1,weight[1],bias[1])
        shortcut = self.Conv(x, weight[2], bias[2])
        y = relu.ReLu(h2 + shortcut)

        # second block
        h3 = self.Conv(y,weight[3],bias[3])
        h4 = relu.ReLu(h3)
        h5 = self.Conv(h4,weight[4],bias[4])
        return relu.ReLu(h5+y)
    
    def Conv5_x(self, x, weight, bias):
        relu = ANN()
        # first block
        h = self.Conv(x,weight[0],bias[0])
        h1 = relu.ReLu(h)
        h2 = self.Conv(h1,weight[1],bias[1])
        shortcut = self.Conv(x, weight[2], bias[2])
        y = relu.ReLu(h2 + shortcut)

        # second block
        h3 = self.Conv(y,weight[3],bias[3])
        h4 = relu.ReLu(h3)
        h5 = self.Conv(h4,weight[4],bias[4])
        return relu.ReLu(h5+y)

    def predict(self, x, weights, biases):
        # Her aşama için 5 weight ve 5 bias gerekiyor
        out = self.Conv2_x(x, weights[0], biases[0])
        out = self.Conv3_x(out, weights[1], biases[1])
        out = self.Conv4_x(out, weights[2], biases[2])
        out = self.Conv5_x(out, weights[3], biases[3])
        return out

model = ResNet18()

x = np.array([1.0, -0.5, 2.0])
weights = [[np.random.randn(3) for _ in range(5)] for _ in range(4)]
biases = [[np.random.randn() for _ in range(5)] for _ in range(4)]

output = model.predict(x, weights, biases)
gap_output = model.gap(output)

print("GAP Çıktısı:", gap_output)


# Global Average Pooling
pooled = np.mean(output)

# FC layer simülasyonu
w_fc = np.random.randn()
b_fc = np.random.randn()
fc_output = w_fc * pooled + b_fc

# Softmax simülasyonu (tek sınıf olsa da gösterim)
softmax_output = 1 / (1 + np.exp(-fc_output))

print("Pooled:", pooled)
print("FC Output:", fc_output)
print("Softmax:", softmax_output)


X = [np.random.rand(3) for _ in range(5)]  # 5 örnek
y = [1, 0, 1, 0, 1]  # hedefler (binary sınıflar)

weights = [[np.random.randn(3) for _ in range(5)] for _ in range(4)]
biases = [[np.random.randn() for _ in range(5)] for _ in range(4)]

w_fc,b_fc = model.fit(X, y,b_fc,w_fc,learningRate=0.1, epochs=50 )
