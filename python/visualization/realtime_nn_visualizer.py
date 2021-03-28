#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np


class RTNNVisualizer:
    def __init__(self, shape):
        # Size and number of layers in network
        self.shape = shape

        self.xs = None
        self.ys = []

        self.init_xs_and_ys()

        plt.ion()
        self.fig = plt.figure()
        self.axes = self.fig.add_subplot(111)
        self.scatter = self.axes.scatter([0], [0])

    def init_xs_and_ys(self):
        """
        Finds x and y coordinates of the circles in each layer
        """
        num_layers = len(self.shape)
        width = 0.7
        self.xs = (np.arange(num_layers)) / (num_layers - 1) * width + (1 - width) / 2

        for i in range(num_layers):
            num_nodes = self.shape[i]
            # Min spacing between circles, or squeeze if there are too many to fit
            height = np.fmin(0.9, 0.1 * num_nodes)
            self.ys.append((np.arange(num_nodes)) / (num_nodes - 1) * height + (1 - height) / 2)

    def draw(self, values):
        self.axes.clear()
        for i in range(len(self.shape)):
            num_nodes = self.shape[i]
            c = values[i][::-1]  # Color based on value

            # Last layer is softmaxed, convert to between [-2, 2]
            if i == len(self.shape) - 1:
                c = self.softmax(c) * 4 - 2

            xs = np.ones(num_nodes) * self.xs[i]
            self.scatter = self.axes.scatter(xs, self.ys[i], s=500, c=c, cmap="Greys_r",
                                             edgecolor="black", vmin=-2, vmax=2)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    @staticmethod
    def close_plots():
        print("Closing  Visualizer")
        plt.close()

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


if __name__ == "__main__":
    import time
    network_shape = [16, 8, 5]

    drawer = RTNNVisualizer(network_shape)
    for _ in range(100):

        data = []
        for s in network_shape:
            data.append(np.random.normal(0, 3, size=s))
        drawer.draw(data)

        time.sleep(0.01)

