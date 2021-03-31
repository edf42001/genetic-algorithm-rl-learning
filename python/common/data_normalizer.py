import numpy as np


class DataNormalizer:
    """
    Keeps a running total of the mean and std deviation of observed state variables
    Normalize input data to a neural network is common practice
    """
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype='float64')
        self.var = np.ones(shape, dtype='float64')
        self.count = 0.01  # This initial value makes var not 0 if the case of constant values in first few obs

        self.clip = 5
        self.noop_value = -1000000

    def record_data(self, x):
        # Replace filled noop values with the mean so as to not influence results too much
        for obs in x:
            empty_values = (obs == self.noop_value)
            obs[empty_values] = self.mean[empty_values]

        batch_count = x.shape[0]

        tot_count = self.count + batch_count

        # Running batch update
        self.mean = (self.mean * self.count + np.sum(x, axis=0)) / tot_count
        self.var = (self.var * self.count + np.sum(np.square(x - self.mean), axis=0)) / tot_count
        self.count = tot_count

    def normalize_data(self, x):
        # Normalize data, set values of dead units (indicated with -1E6) to 0
        # Clip to ensure all values are reasonable
        normalized = np.divide(x - self.mean, np.sqrt(self.var), where=(self.var != 0))
        normalized[x == self.noop_value] = 0
        normalized[normalized > self.clip] = self.clip
        normalized[normalized < -self.clip] = -self.clip

        return normalized
