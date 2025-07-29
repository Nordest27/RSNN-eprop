from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ann import MLP, Layer
from activation_functions import Sigmoid, Lineal, ReLU, Tanh, LeakyReLU
from loss_functions import SigmoidCrossEntropy
import numpy as np 

n_classes = 2
epsilon = 1e-4

X, y = make_classification(n_samples=1000, n_features=20, n_classes=n_classes, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

layers = [
    Layer(20, 10, Lineal()),
    Layer(10, 10, Tanh()),
    Layer(10, 10, LeakyReLU()),
    Layer(10, 10, Sigmoid()),
    Layer(10, 1, ReLU()),
    Layer(1, 0, Lineal())
]
loss_function = SigmoidCrossEntropy()
mlp = MLP(layers, loss_function, learning_rate=0.01)


x = X_train[0]
y_true = np.array([y_train[0]])
mlp.backprop(mlp.forward(x), y_true)

weights_gradient = np.array([None for _ in layers])
biases_gradient = np.array([None for _ in layers])
for i, layer in enumerate(layers):
    weights_gradient[i] = layer.outweights_gradient
    biases_gradient[i] = layer.biases_gradient

numerical_weights_gradient = np.array([None for _ in layers])
numerical_biases_gradient = np.array([None for _ in layers])
for l, layer in enumerate(layers):
    numerical_biases_gradient[l] = np.zeros_like(layer.biases)
    
    for i in range(len(layer.biases)):
        layer.biases[i] += epsilon
        plus_loss = loss_function.get_loss(mlp.forward(x), y_true)

        layer.biases[i] -= 2*epsilon
        minus_loss = loss_function.get_loss(mlp.forward(x), y_true)

        numerical_biases_gradient[l][i] = (plus_loss - minus_loss)/(2*epsilon)
        
        layer.biases[i] += epsilon

    if layer.outweights is None:
        continue

    numerical_weights_gradient[l] = np.zeros_like(layer.outweights)
    for i in range(len(layer.outweights)):
        for j in range(len(layer.outweights[0])):
            layer.outweights[i][j] += epsilon
            plus_loss = loss_function.get_loss(mlp.forward(x), y_true)
            
            layer.outweights[i][j] -= 2*epsilon
            minus_loss = loss_function.get_loss(mlp.forward(x), y_true)
            
            numerical_weights_gradient[l][i][j] = (plus_loss - minus_loss)/(2*epsilon)
            layer.outweights[i][j] += epsilon

biases_numerator = 0
biases_denominator = 0

weights_numerator = 0
weights_denominator = 0
for i in range(len(layers)):
    biases_numerator += sum(abs(biases_gradient[i] - numerical_biases_gradient[i]))
    biases_denominator += sum(abs(biases_gradient[i]) + abs(numerical_biases_gradient[i]))

    if layers[i].outweights is None:
        continue
    weights_numerator += sum(sum(abs(weights_gradient[i] - numerical_weights_gradient[i])))
    weights_denominator += sum(sum(abs(weights_gradient[i]) + abs(numerical_weights_gradient[i])))

print("Biases Diff:", biases_numerator/biases_denominator)
print("Weights Diff:", weights_numerator/weights_denominator)

assert biases_numerator/biases_denominator < 5e-7, "Bias gradients check failed"
assert weights_numerator/weights_denominator < 5e-7, "Weight gradients check failed"