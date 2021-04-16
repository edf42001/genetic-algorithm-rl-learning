import numpy as np


class AdamOptimizer:
    """
    Implementation of the often used Adam backpropagation optimizer
    """
    def __init__(self, weights_shape, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1.0E-7):
        self.mean_buffer = []
        self.var_buffer = []

        for shape in weights_shape:
            self.mean_buffer.append(np.zeros(shape))
            self.var_buffer.append(np.zeros(shape))

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def get_gradients(self, grad_in, epochs):

        mean_buffer_hat = []
        var_buffer_hat = []
        grad_out = []
        for i in range(len(grad_in)):
            # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
            self.mean_buffer[i] = self.beta1 * self.mean_buffer[i] + (1.0 - self.beta1) * grad_in[i]

            # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
            self.var_buffer[i] = self.beta2 * self.var_buffer[i] + (1.0 - self.beta2) * grad_in[i] ** 2

            # Bias correction
            # mhat(t) = m(t) / (1 - beta1(t))
            # vhat(t) = v(t) / (1 - beta2(t))
            mean_buffer_hat.append(self.mean_buffer[i] / (1.0 - self.beta1**(epochs+1)))
            var_buffer_hat.append(self.var_buffer[i] / (1.0 - self.beta2**(epochs+1)))

            # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + eps)
            grad_out.append(self.learning_rate * mean_buffer_hat[i] / (np.sqrt(var_buffer_hat[i]) + self.epsilon))

        return grad_out
