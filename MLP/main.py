import numpy as np


class MLP:

    def __init__(self, num_inputs=3, num_hidden=None, num_outputs=2):
        if num_hidden is None:
            num_hidden = [3, 5]
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hidden = num_hidden

        layers = [self.num_inputs] + self.num_hidden + [self.num_outputs]

        # initiate weights
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            self.weights.append(w)

        # initiate activations
        activations = []
        for a in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        # initiate activations
        derivatives = []
        for d in range(len(layers)):
            d = np.zeros(layers[i], layers[i + 1])
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):

        activations = inputs
        self.activations[0] = inputs
        for i, w in enumerate(self.weights):
            # find net input
            net_inputs = np.dot(activations, w)
            # find activation
            activations = self._sigmoid(net_inputs)
            self.activations[i + 1] = activations
        return activations

    def back_propagate(self, error):

        # dE/dW_i = (y - a_[i+1]) s'(h_[i+1]) a_i
        # s'(h_[i+1]) = s(h_[i+1])(1 - s(h_[i+1]))
        # s(h_[i+1]) = a_[i+1]

        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i + 1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta)

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    mlp = MLP()

    inputs = np.random.rand(mlp.num_inputs)
    outputs = mlp.forward_propagate(inputs)
    print("The inputs are {}".format(inputs))
    print("The outputs are {}".format(outputs))
