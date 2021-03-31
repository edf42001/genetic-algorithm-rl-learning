import numpy as np
import random
import time
import json

from agents import Agent
from common.data_normalizer import DataNormalizer


class CrossEntropyNNAgent(Agent):
    def __init__(self):
        super().__init__()

        # Number of agents in population
        self.pop_size = 100

        # Number of the best to select to average for the next population
        self.elite_percent = 0.2

        # The agent currently being tested
        self.active_agent = 0

        # How many runs have been done with current agent this episode
        self.trials = 0

        # How many runs to do per agent per epoch (larger sample size per agent)
        self.num_trials = 1

        # Layer sizes in the neural network. Must contain at least two layers, for input and output size
        self.layers = [16, 8, 5]

        # Weight given to entropy of action probabilities bonus
        self.entropy_weight = 0.01

        # Parameters that control adding noise to std dev of policy params
        self.std_noise = 0.01
        self.std_noise_decay = 5E-5

        # Parameters of distribution of weights for each layer. Will be learned to converge on good values
        self.u = []
        self.std = []
        self.initialize_u_std()

        # Randomly generated population of weights matrix. Multiplies state vectors to get values for each of 5 actions.
        # Each item in the list is a (pop_size, in_size, out_size), matrix, which can be indexed along the first axis
        # to get that layer's weight matrix for each agent in the population.
        self.population = []
        self.generate_new_population()

        # Scores for each member of population
        self.population_scores = np.zeros(self.pop_size)

        # Helps us normalize input data to network
        self.data_normalizer = DataNormalizer(shape=self.layers[0])

        # The agent only takes actions in eval mode, and doesn't train
        self.eval_mode = False

        # How many iterations this agent has been training for
        self.iterations = 0

        # How many epochs we have been training for, an epoch being every agent going num_trials times
        self.epochs = 0

        # Keep track of the total elapsed time
        self.start_time = time.time()

        # How often to save, in epochs. Because winner callback is called multiple times per epoch, need to check if
        # we saved that epoch already
        self.save_freq = 15
        self.should_save = False

        # List of all observations vectors seen in an epoch
        self.obs = []

        # Print arrays nicer
        np.set_printoptions(precision=3, floatmode='fixed', linewidth=1000, suppress=True)

    def env_callback(self, request):
        self.iterations += 1

        # Actions to be returned
        actions = []

        # Add reward to this agent's total
        self.population_scores[self.active_agent] += np.sum([d.last_action_reward for d in request.agent_data])

        # An empty state indicates the episode as ended
        # We still need to use the last reward to update
        episode_over = len(request.agent_data[0].state) == 0

        if episode_over:
            return None

        # Update observation list with obs from all units
        for data in request.agent_data:
            self.obs.append(data.state)

        # Request contains, for each agent:
        # the current state of the world as a result of the last action
        # and the reward for the last action
        for data in request.agent_data:
            state = np.array(data.state, dtype='float64')
            reward = data.last_action_reward
            unit_id = data.unit_id

            # print(state)
            state = self.data_normalizer.normalize_data(state)
            # print(state)

            # Forward pass state through network:
            result = state
            for layers in self.population:
                result = layers[self.active_agent].dot(result)

            # Softmax activation and random sample
            softmax = self.softmax(result)
            action = self.random_weighted_index(softmax)

            # print(result)
            # print(softmax)
            # print()

            # Add entropy bonus to agent's fitness
            self.population_scores[self.active_agent] += self.entropy_of_probabilities(softmax) * self.entropy_weight

            actions += [action]

        return actions

    def winner_callback(self, request):
        """
        Indicates the end of an episode, someone won (or tied), not necessarily us
        Switch to the next agent for testing
        :param request: Data from sepia about the win state
        :return: None
        """

        # If in eval mode, no need to update. Just run
        if self.eval_mode:
            return

        # Because winner callback is called multiple times per epoch, only save upon the transition to new epoch
        self.should_save = False

        # Increment trial
        self.trials += 1

        # Update running mean and std of observations, for normalizing
        self.data_normalizer.record_data(np.array(self.obs))
        self.obs = []

        # If we aren't done with trials, nothing left to do in this function
        if self.trials != self.num_trials:
            return

        # If we are, move to next agent and reset trials
        self.active_agent += 1
        self.trials = 0

        # If all agents have been tested
        if self.active_agent == self.pop_size:
            self.epochs += 1

            # Set should save upon transition to new epoch
            self.should_save = (self.epochs % self.save_freq == 0)

            print("Epoch %d: Generating new population" % self.epochs)

            # Select a percentage of best agents to reproduce
            best_agents = self.population_scores.argsort()[::-1][:max(1, int(self.pop_size * self.elite_percent))]

            # print(self.population_scores)
            print(self.population_scores[best_agents])

            # Find new param distribution from elite agents
            self.update_u_std(best_agents)

            # Generate new population
            self.generate_new_population()

            # Reset scores
            self.population_scores = np.zeros(self.pop_size)

            # Reset active agent
            self.active_agent = 0

            print(self.u)
            print(self.std)

    def initialize_u_std(self):
        """Initialize parameter's mu and std matrices"""

        for i in range(1, len(self.layers)):
            # Due to method of array multiplication, shape is (out_size, in_size)
            shape = (self.layers[i], self.layers[i-1])
            self.u.append(np.zeros(shape=shape))  # Start with 0 mean
            self.std.append(np.ones(shape=shape) * 0.7)  # Start with 0.7 std

    def generate_new_population(self):
        """Generates a new random population from the parameter's mu and std"""

        self.population = []  # Empty current population

        # Add a layer for every layer in the network
        for i in range(1, len(self.layers)):
            shape = (self.layers[i], self.layers[i-1])

            # For this layer, create a random population of weights with the given u and std
            self.population.append(np.random.normal(self.u[i-1], self.std[i-1], (self.pop_size, *shape)))

    def update_u_std(self, best_agents):
        """Updates the parameter distribution mu and standard deviation by averaging the fittest agents"""
        # Reset u and std
        self.u = []
        self.std = []

        # Fill each layer's param back in by averaging the best agent's weights
        for i in range(len(self.layers) - 1):
            # Update mu and std based on the elite population
            self.u.append(np.mean(self.population[i][best_agents], axis=0))
            self.std.append(np.std(self.population[i][best_agents], axis=0))

            # Add a bit to std to prevent early convergence
            self.std[i] += np.fmax(0, self.std_noise - self.std_noise_decay * self.epochs)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x)
        return e_x / np.sum(e_x)

    def random_weighted_index(self, x):
        """Picks an action at random, weighted by value"""
        return random.choices(np.arange(x.size), x)[0]

    def entropy_of_probabilities(self, x):
        return -np.sum(x * np.log2(x))

    def set_eval_mode(self, eval_mode):
        self.eval_mode = eval_mode

        # When eval mode is set, remove randomness from parameters
        # Zero out std dev and just use mu, then regenerate population
        for std in self.std:
            std *= 0

        self.generate_new_population()

    def save_to_file(self, folder):
        config_file = folder + "/config.txt"
        params_file = folder + "/agent.npy"

        config = dict()
        config["name"] = type(self).__name__
        config["iterations"] = self.iterations
        config["elapsed_time"] = time.time() - self.start_time
        config["std_noise"] = 0
        config["num_trials"] = self.num_trials
        config["epochs"] = self.epochs
        config["layers"] = self.layers

        # Create a layers dict to save each layer's params to a numpy savez
        layers = dict()
        for i in range(len(self.layers) - 1):
            layers["u_" + str(i)] = self.u[i]
            layers["std_" + str(i)] = self.std[i]

        # Save arrays
        with open(params_file, 'wb') as f:
            np.savez(f, **layers, data_u=self.data_normalizer.mean, data_var=self.data_normalizer.var)

        # Save other info
        with open(config_file, 'w') as f:
            json.dump(config, f)

    def load_from_folder(self, folder):
        config_file = folder + "/config.txt"
        params_file = folder + "/agent.npy"

        with open(config_file, 'r') as f:
            config = json.load(f)
            self.iterations = config["iterations"]
            self.num_trials = config["num_trials"]
            self.epochs = config["epochs"]
            self.layers = config["layers"]
            # self.std_noise = config["std_noise"]

        with np.load(params_file, 'rb') as data:
            # Reset u and std
            self.u = []
            self.std = []

            # Load each layer's u and std params
            for i in range(len(self.layers) - 1):
                self.u.append(data["u_" + str(i)])
                self.std.append(data["std_" + str(i)])

            # Load our data normalizer's info
            self.data_normalizer.mean = data["data_u"]
            self.data_normalizer.var = data["data_var"]
            self.data_normalizer.count = self.iterations

        # After loading, generate a random population from the data
        self.generate_new_population()

    def should_save_to_folder(self):
        return self.should_save
