import numpy as np
import random
import time
import json

from agents import Agent
from common.data_normalizer import DataNormalizer
from common.backpropagation import policy_feed_forward, model_gradients

EP_END = -100  # Number in rewards that indicates game end. Rewards in one game should not interact with others


class DeepPolicyNNAgent(Agent):
    def __init__(self):
        super().__init__()

        self.discount_rate = 0.95  # Discount future rewards. Time horizon = 1 / (1 - rate)
        self.learning_rate = 1.0E-3  # NN gradient learning rate
        self.learning_rate_decay = 0.96

        self.mini_batch_size = 3  # After how many games to do a network update
        self.batch_size = 15  # After how many games to do a rmsprop param update

        self.entropy_weight = 0.01  # Weight given to entropy of action probabilities bonus

        # Layer sizes in the neural network. Must contain at least two layers, for input and output size
        self.layers = [16, 16, 5]

        self.model = self.create_model(self.layers)

        # Helps us normalize input data to network
        self.data_normalizer = DataNormalizer(shape=self.layers[0])

        # Bookkeeping for backprop
        self.obs = []  # List of all observations vectors seen in an epoch
        self.rewards = dict()  # List of all rewards seen in an epoch
        self.actions = dict()  # List of all actions taken in an epoch
        self.nn_outputs = dict()  # List of softmax outputs each timestep
        self.nn_hidden = dict()  # List of hidden layer values each timestep
        self.nn_inputs = dict()  # List of the inputs to the nn

        self.grad_buffer = [np.zeros_like(layer) for layer in self.model]

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
        self.save_freq = 1500
        self.should_save = False

        self.ep_start = True  # True if this turn is the start of the episode

        # Print arrays nicer
        np.set_printoptions(precision=3, floatmode='fixed', linewidth=1000, suppress=True)

    def env_callback(self, request):
        self.iterations += 1

        # Actions to be returned
        actions = []

        # print([d.last_action_reward for d in request.agent_data])

        # An empty state indicates the episode as ended
        # We still need to use the last reward to update
        episode_over = len(request.agent_data[0].state) == 0

        # Record reward, if this is not the first turn
        for state in request.agent_data:
            if state.unit_id not in self.rewards:
                # If empty, initialize. The first reward is not real because we haven't yet taken an action
                self.rewards[state.unit_id] = []
            elif not self.ep_start:
                self.rewards[state.unit_id].append(state.last_action_reward)

        # If over, next turn will be first turn, otherwise do nothing
        if episode_over:
            self.ep_start = True
            return None

        # Request contains, for each agent:
        # the current state of the world as a result of the last action
        # and the reward for the last action
        for data in request.agent_data:

            # Do not process agents who are just telling us their last reward
            if len(data.state) == 0:
                continue

            # Update observation list with obs from all units
            self.obs.append(data.state)

            # Convert state to np for processing
            state = np.array(data.state, dtype='float64')
            reward = data.last_action_reward
            unit_id = data.unit_id

            # print(state)
            # Normalize state for NN
            state = self.data_normalizer.normalize_data(state)
            # print(state)

            # Forward pass state through network:
            result, hidden = policy_feed_forward(self.model, state)

            # Softmax activation and random sample
            softmax = self.softmax(result)
            action = self.random_weighted_index(softmax)

            # 1-hot encoded action
            action_1_hot = np.zeros(self.layers[-1])
            action_1_hot[action] = 1

            # Check if the lists are empty and initialize it they are
            # Record actual network output, to be compared to the action taken
            # To 'encourage' it to take this action again in the future if this action gives big reward
            if unit_id not in self.actions:
                self.actions[unit_id] = [action_1_hot]
                self.nn_outputs[unit_id] = [softmax]
                self.nn_hidden[unit_id] = [hidden]
                self.nn_inputs[unit_id] = [state]
            else:
                self.actions[unit_id].append(action_1_hot)
                self.nn_outputs[unit_id].append(softmax)
                self.nn_hidden[unit_id].append(hidden)
                self.nn_inputs[unit_id].append(state)

            # print(result)
            # print(softmax)
            # print()
            actions += [action]

        # Once this executes this is no longer the first turn
        if self.ep_start:
            self.ep_start = False

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

        # Update running mean and std of observations, for normalizing
        self.data_normalizer.record_data(np.array(self.obs))
        self.obs = []

        self.epochs += 1

        if self.epochs % self.mini_batch_size == 0:
            # print("Epoch %d: finding grads" % self.epochs)
            for unit_id in self.rewards.keys():
                rewards = np.array(self.rewards[unit_id], dtype='float64')
                actions_taken = np.vstack(self.actions[unit_id])
                nn_inputs = np.vstack(self.nn_inputs[unit_id])
                nn_hidden = np.vstack(self.nn_hidden[unit_id])
                nn_outputs = np.vstack(self.nn_outputs[unit_id])
                discounted_rewards = np.vstack(self.discount_rewards(rewards))

                # print("Reward sum: %.2f" % sum(rewards))
                # print([float("%.2f" % x) for x in rewards[rewards != EP_END]])
                # Normalize rewards. Makes backprop work better
                # print([float("%.2f" % x) for x in discounted_rewards])
                discounted_rewards -= np.mean(discounted_rewards)
                discounted_rewards /= np.std(discounted_rewards)
                # print([float("%.2f" % x) for x in discounted_rewards])
                # print()

                # Difference between softmax output and 1 hot encoded action
                # Multiply it by rewards (advantage) to encourage those actions which led
                # to high rewards, and discourage those that didn't
                y_loss = nn_outputs - actions_taken
                y_loss *= discounted_rewards

                grad = model_gradients(nn_inputs, nn_hidden, y_loss, self.model[1])

                # Store grads in buffer for eventual update
                for i in range(len(self.grad_buffer)):
                    self.grad_buffer[i] += grad[i]

            # Reset storage
            self.actions, self.rewards, self.nn_outputs = dict(), dict(), dict()

            # Do model update
            if self.epochs % self.batch_size == 0:
                # TODO make it so these don't have to be mulitplies of each other

                # Implement learning rate decay
                learning_rate = self.learning_rate * np.power(self.learning_rate_decay, self.iterations / 10000.0)

                print("Epoch %d: doing model update %.6f" % (self.epochs, learning_rate))
                for i in range(len(self.grad_buffer)):
                    self.model[i] += self.grad_buffer[i] * learning_rate

                    # Reset buffer
                    self.grad_buffer[i] = np.zeros_like(self.grad_buffer[i])
        else:
            # If we are not doing an update, record this as the end of an episode in the rewards list:
            for unit_id in self.rewards:
                self.rewards[unit_id].append(EP_END)  # Indicates end of episode

        # Set should save upon transition to new epoch
        self.should_save = (self.epochs % self.save_freq == 0)

    def policy_feed_forward(self, input):
        result = input
        for layers in self.model:
            result = layers.dot(result)

        return result

    def create_model(self, layers):
        """Creates network matrices and randomly initializes starting model parameters"""

        model = []  # Empty current population

        # Add a layer for every layer in the network
        for i in range(1, len(layers)):
            shape = (layers[i], layers[i-1])
            u = 0
            std = 1 / np.sqrt(layers[i-1])  # "Xavier" initialization

            # For this layer, create a random weight matrix with the given u and std
            model.append(np.random.normal(u, std, shape))

        return model

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x)
        return e_x / np.sum(e_x)

    def discount_rewards(self, r):
        """Take 1D float array of rewards and compute discounted reward"""
        length = len(r)
        shape = length - self.mini_batch_size + 1  # Reduce space because the end of episode markers don't count
        discounted_r = np.zeros(shape=shape)
        running_add = 0

        i = shape - length  # Index counter, to keep track of boundary offset
        for t in range(length - 1, -1, -1):
            if r[t] == EP_END:
                running_add = 0  # reset the sum, since this was a game boundary
                i += 1  # 1 more boundary offset has been introduced
            else:
                running_add = running_add * self.discount_rate + r[t]
                discounted_r[t + i] = running_add
        return discounted_r

    def random_weighted_index(self, x):
        """Picks an action at random, weighted by value"""
        return random.choices(np.arange(x.size), x)[0]

    def entropy_of_probabilities(self, x):
        return -np.sum(x * np.log2(x))

    def set_eval_mode(self, eval_mode):
        self.eval_mode = eval_mode

    def save_to_file(self, folder):
        config_file = folder + "/config.txt"
        params_file = folder + "/agent.npy"

        config = dict()
        config["name"] = type(self).__name__
        config["iterations"] = self.iterations
        config["epochs"] = self.epochs
        config["elapsed_time"] = time.time() - self.start_time
        config["discount_rate"] = self.discount_rate
        config["learning_rate"] = self.learning_rate
        config["learning_rate"] = self.learning_rate_decay
        config["mini_batch_size"] = self.mini_batch_size
        config["batch_size"] = self.batch_size
        config["entropy_weight"] = self.entropy_weight
        config["layers"] = self.layers

        # Create a weights dict to save each layer's params to a numpy savez
        weights = dict()
        for i in range(len(self.model)):
            weights["weights_" + str(i)] = self.model[i]

        # Save arrays
        with open(params_file, 'wb') as f:
            np.savez(f, **weights, data_u=self.data_normalizer.mean, data_var=self.data_normalizer.var)

        # Save config info
        with open(config_file, 'w') as f:
            json.dump(config, f)

    def load_from_folder(self, folder):
        config_file = folder + "/config.txt"
        params_file = folder + "/agent.npy"

        with open(config_file, 'r') as f:
            config = json.load(f)
            self.iterations = config["iterations"]
            self.epochs = config["epochs"]
            self.discount_rate = config["discount_rate"]
            self.learning_rate = config["learning_rate"]
            self.learning_rate_decay = config["learning_rate"]
            self.mini_batch_size = config["mini_batch_size"]
            self.batch_size = config["batch_size"]
            self.entropy_weight = config["entropy_weight"]
            self.layers = config["layers"]

        with np.load(params_file, 'rb') as data:
            # Reset model
            self.model = []

            # Load each layer's u and std params
            for i in range(len(self.layers) - 1):
                self.model.append(data["weights_" + str(i)])

            # Load our data normalizer's info
            self.data_normalizer.mean = data["data_u"]
            self.data_normalizer.var = data["data_var"]
            self.data_normalizer.count = self.iterations

    def should_save_to_folder(self):
        return self.should_save
