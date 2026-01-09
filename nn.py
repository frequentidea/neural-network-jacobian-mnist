import os
import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import colors


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train, X_test = X_train.reshape(-1, 784), X_test.reshape(-1, 784)

y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Activation functions
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
    return (x > 0).astype(float)

def cross_entropy(y_pred, y_true, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


class DeepNeuralNetwork:
    def __init__(self, layers, lr=0.01, save_file="mnist_dnn.pkl"):
        self.layers = layers
        self.lr = lr
        self.save_file = save_file
        self.W, self.b = [], []

        if os.path.exists(save_file):
            self.load()
        else:
            for i in range(len(layers) - 1):
                self.W.append(np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]))
                self.b.append(np.zeros((1, layers[i+1])))

    def forward(self, x):
        self.a, self.z = [x], []
        for i in range(len(self.W) - 1):
            z = self.a[-1] @ self.W[i] + self.b[i]
            self.z.append(z)
            self.a.append(ReLU(z))
        # Output layer (softmax)
        z = self.a[-1] @ self.W[-1] + self.b[-1]
        self.z.append(z)
        self.a.append(softmax(z))
        return self.a[-1]

    def backward(self, y_true):
        m = y_true.shape[0]
        dW, db = [], []

        # Output layer gradient
        dz = self.a[-1] - y_true
        dw = self.a[-2].T @ dz / m
        dbb = np.sum(dz, axis=0, keepdims=True) / m
        dW.insert(0, dw)
        db.insert(0, dbb)

        # Hidden layers
        da = dz
        for i in range(len(self.W) - 2, -1, -1):
            dz = (da @ self.W[i+1].T) * dReLU(self.z[i])
            dw = self.a[i].T @ dz / m
            dbb = np.sum(dz, axis=0, keepdims=True) / m
            dW.insert(0, dw)
            db.insert(0, dbb)
            da = dz

        # Update weights
        for i in range(len(self.W)):
            self.W[i] -= self.lr * dW[i]
            self.b[i] -= self.lr * db[i]

    def train(self, X, Y, epochs=10, batch_size=64):
        for epoch in range(epochs):
            idx = np.random.permutation(len(X))
            X, Y = X[idx], Y[idx]

            for i in range(0, len(X), batch_size):
                x_batch = X[i:i+batch_size]
                y_batch = Y[i:i+batch_size]
                self.forward(x_batch)
                self.backward(y_batch)

            y_pred = self.forward(X)
            loss = cross_entropy(y_pred, Y)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

        self.save()

    def evaluate(self, X, Y):
        y_pred = self.forward(X)
        acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(Y, axis=1))
        return acc

    def save(self):
        with open(self.save_file, "wb") as f:
            pickle.dump((self.W, self.b), f)
        print(f"Model saved to {self.save_file}")

    def load(self):
        with open(self.save_file, "rb") as f:
            self.W, self.b = pickle.load(f)
        print(f"Model loaded from {self.save_file}")



mnist_dnn = DeepNeuralNetwork([784, 128, 64, 10], lr=0.01, save_file="mnist_dnn.pkl")


if not os.path.exists("mnist_dnn.pkl"):
    mnist_dnn.train(X_train, y_train, epochs=10, batch_size=128)


def compute_jacobian(model, x):
    """
    Compute Jacobian of model outputs wrt input x.
    Returns array of shape (num_classes, input_dim).
    """
    x = x.reshape(1, -1)
    y = model.forward(x)
    num_classes = y.shape[1]
    input_dim = x.shape[1]
    jacobian = np.zeros((num_classes, input_dim))

    for c in range(num_classes):

        dy_dz = np.zeros_like(y)
        for k in range(num_classes):
            if k == c:
                dy_dz[0, k] = y[0, c] * (1 - y[0, c])
            else:
                dy_dz[0, k] = -y[0, c] * y[0, k]

        # Backpropagate gradient to the input
        delta = dy_dz
        for i in range(len(model.W) - 1, -1, -1):
            delta = delta @ model.W[i].T
            if i > 0:
                delta *= dReLU(model.z[i - 1])

        jacobian[c] = delta.flatten()

    return jacobian



sample = X_test[0]
J = compute_jacobian(mnist_dnn, sample)

heatmap = J[7].reshape(28, 28)

heatmap = heatmap / np.max(np.abs(heatmap))

plt.figure(figsize=(4, 4))


plt.imshow((1-sample).reshape(28, 28), cmap="gray")


cmap = plt.cm.coolwarm
norm = colors.Normalize(vmin=-1, vmax=1)

im = plt.imshow(heatmap, cmap=cmap, norm=norm, alpha=0.6)

plt.colorbar(im, fraction=0.046, pad=0.04, label="Sensitivity")

plt.title("Class 7 Jacobian Overlay")
plt.axis("off")
plt.axis("off")
plt.show()
