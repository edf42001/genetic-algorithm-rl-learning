import numpy as np
import time
import json

from agents.agent import Agent
from data_saving.data_normalizer import DataNormalizer


class CrossEntropyNNAgent(Agent):
    def __init__(self):
        super().__init__()

        # Number of agents in population
        self.pop_size = 100

        # Number of the best to select to average for the next population
        self.elite_percent = 0.2

        # The agent currently being tested
        self.active_agent = 0

        # Parameters of distribution of weights. Will be learned to converge on good values
        self.u = np.zeros(shape=(5, 16))
        self.std = np.ones(shape=(5, 16)) * 0.35
        # Randomly generated population of weights matrix. Multiplies state vectors to get values for each of 5 actions
        self.population = np.random.normal(self.u, self.std, (self.pop_size, 5, 16))

        # Scores for each member of population
        self.population_scores = np.zeros(self.pop_size)

        # Helps us normalize input data to network
        self.data_normalizer = DataNormalizer()

        # The agent only takes actions in eval mode, and doesn't train
        self.eval_mode = False

        # How many iterations this agent has been training for
        self.iterations = 0

        # Keep track of the total elapsed time
        self.start_time = time.time()

        np.set_printoptions(precision=3, floatmode='fixed', linewidth=1000)

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

        # Update data normalizer before we do any processing
        for data in request.agent_data:
            self.data_normalizer.record_data(np.array(data.state, dtype='float64'))
            # print(np.array(data.state, dtype='float64'))
            # print(self.data_normalizer.u)
            # print(self.data_normalizer.var)
            # print()

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
            result = self.population[self.active_agent].dot(state)
            # print(result)
            softmax = self.softmax(result)
            # print(softmax)
            # print()

            action = self.random_weighted_index(softmax)

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

        self.active_agent += 1

        # If all agents have been tested
        if self.active_agent == self.pop_size:
            print("Generating new population")

            # Select a percentage of best agents to reproduce
            best_agents = self.population_scores.argsort()[::-1][:int(self.pop_size * self.elite_percent)]
            # print(best_agents)
            # print(self.population[best_agents].shape)

            # Update mu and std based on the elite population
            self.u = np.mean(self.population[best_agents], axis=0)
            self.std = np.std(self.population[best_agents], axis=0)

            # print(self.population_scores)
            print(self.population_scores[best_agents])
            # Add a bit to std to prevent early convergence
            self.std += 0.0

            # Generate new population
            self.population = np.random.normal(self.u, self.std, (self.pop_size, 5, 16))

            # Reset scores
            self.population_scores = np.zeros(self.pop_size)

            # Reset active agent
            self.active_agent = 0

            print(self.u)
            print(self.std)

    def save_to_file(self, folder):
        config_file = folder + "/config.txt"
        params_file = folder + "/agent.npy"

        config = dict()
        config["name"] = type(self).__name__
        config["iterations"] = self.iterations
        config["elapsed_time"] = time.time() - self.start_time
        config["std_noise"] = 0

        with open(params_file, 'wb') as f:
            np.savez(f, u=self.u, std=self.std, data_u=self.data_normalizer.u, data_var=self.data_normalizer.var)

        with open(config_file, 'w') as f:
            json.dump(config, f)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def random_weighted_index(self, x):
        return np.random.choice(np.arange(x.size), 1, p=x)[0]

    def load_from_folder(self, folder):
        config_file = folder + "/config.txt"
        params_file = folder + "/agent.npy"

        with np.load(params_file, 'rb') as data:
            self.u = data["u"]
            self.std = data["std"]
            self.data_normalizer.u = data["data_u"]
            self.data_normalizer.var = data["data_var"]

        with open(config_file, 'r') as f:
            config = json.load(f)
            self.iterations = config["iterations"]
            self.data_normalizer.iterations = self.iterations

        # After loading, generate a random population from the data
        self.population = np.random.normal(self.u, self.std, (self.pop_size, 5, 16))
