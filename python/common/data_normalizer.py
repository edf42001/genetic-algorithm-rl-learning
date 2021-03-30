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
        """
        Record running mean and std deviation of each variable in the state
        x: batch of state variables from many timesteps
        """
        for obs in x:
            empty_values = (obs == self.noop_value)
            obs[empty_values] = self.mean[empty_values]

        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        # Update values, but not ones that don't have values
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def normalize_data(self, x):
        # Normalize data, set values of dead units (indicated with -1E6) to 0
        # Clip to ensure all values are reasonable
        normalized = np.divide(x - self.mean, np.sqrt(self.var), where=(self.var != 0))
        normalized[x == self.noop_value] = 0
        normalized[normalized > self.clip] = self.clip
        normalized[normalized < -self.clip] = -self.clip

        return normalized


# I'm defining this function outside of the class (so like a static method)
# Because I saw OpenAI do this in their baselines, and I'm trying out new ideas
# to see if there's any benefit to them
def update_mean_var_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = m2 / tot_count

    return new_mean, new_var, tot_count

# mask = x != self.noop_value
# -            self.u[mask] = (self.u[mask] * self.iterations + x[mask]) / (self.iterations + 1)
# -            self.var[mask] = (self.var[mask] * self.iterations + np.square(x[mask] - self.u[mask])) / (s
#                                                                                                         elf.iterations + 1)


# new_mean = (mean * count + np.sum(batch)) / tot_count
# new_var = (var * count + np.sum(np.square(x - new_mean)) / tot_count