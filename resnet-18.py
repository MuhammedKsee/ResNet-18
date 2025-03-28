import numpy as np
from ArtificalNeuralNetwork import ANN

class ResNet18:
    def __init__(self):
        pass

    def gap(self,x):
        return np.mean(x)
    
    def backprop_conv(self, x, weight, bias, y_true, lr = 0.01):
        # Forward
        z = np.dot(x, weight) + bias
        y_pred = np.maximum(0, z)  # ReLU

        # Loss
        loss = (y_pred - y_true) ** 2

        # Backward
        dL_dy = 2 * (y_pred - y_true)
        dy_dz = (z > 0).astype(float)  # ReLU türevi
        dL_dz = dL_dy * dy_dz

        # Gradients
        dL_dw = dL_dz * x
        dL_db = dL_dz

        # Update
        weight -= lr * dL_dw
        bias -= lr * dL_db

        return weight, bias, loss


    def fit(self, x, y, b_fc, w_fc, weights, biases, learningRate=0.01, epochs=50, loss_list=None):
        if loss_list is None:
            loss_list = []
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(weights)):
                for j in range(len(weights[i])):
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
                    weights[i][j], biases[i][j], loss = self.backprop_conv(x_i, weights[i][j], biases[i][j], y_true, learningRate)
            avg_loss = total_loss / len(x) 
            loss_list.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        return w_fc,b_fc,weights,biases,loss_list
    
    def evaluate_accuracy(self, X, y, w_fc, b_fc, weights, biases):
        correct = 0
        for i in range(len(X)):
            x_i = X[i]
            y_true = y[i]

            out = self.predict(x_i, weights, biases)
            pooled = self.gap(out)
            y_pred = pooled * w_fc + b_fc

            # Softmax gibi threshold
            prediction = 1 if y_pred >= 0.5 else 0

            if prediction == y_true:
                correct += 1

        accuracy = correct / len(X)
        print(f"Accuracy: {accuracy:.2%}")
        return accuracy



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
weights = [[np.random.randn(3,3) for _ in range(5)] for _ in range(4)]
biases = [[np.random.randn(3) for _ in range(5)] for _ in range(4)]

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

w_fc,b_fc,weights,biases,loss_list = model.fit(X, y,b_fc,w_fc,weights,biases,learningRate=0.1, epochs=50)

acc = model.evaluate_accuracy(X, y, w_fc, b_fc, weights, biases)

print(f"Accuracy: {acc:.4f}")


import matplotlib.pyplot as plt


# Eğitim bitince:
plt.plot(loss_list)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.show()
