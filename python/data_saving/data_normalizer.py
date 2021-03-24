import numpy as np


class DataNormalizer:
    """
    Keeps a running total of the mean and std deviation of observed state variables
    Normalize input data to a neural network is common practice
    """

    def __init__(self):
        self.u = None
        self.var = None
        self.iterations = 0

        self.clip = 5
        self.noop_value = -1000000

    def record_data(self, x):
        """
        Record running mean and std deviation of each variable in the state
        x: array of state variables
        """
        if self.u is None:
            # Assume the first one will not have any noop values in it
            self.u = x
            self.var = np.zeros_like(x)
        else:
            # Update values, but not ones that don't have values
            mask = x != self.noop_value
            self.u[mask] = (self.u[mask] * self.iterations + x[mask]) / (self.iterations + 1)
            self.var[mask] = (self.var[mask] * self.iterations + np.square(x[mask] - self.u[mask])) / (self.iterations + 1)

        self.iterations += 1

    def normalize_data(self, x):
        # Normalize data, set values of dead units (indicated with -1E6) to 0
        # Clip to ensure all values are reasonable
        normalized = np.divide((x - self.u), np.sqrt(self.var), where=(self.var != 0))
        normalized[x == self.noop_value] = 0
        normalized = np.clip(normalized, -self.clip, self.clip)
        return normalized


