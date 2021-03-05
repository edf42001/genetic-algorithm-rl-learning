#!/usr/bin/env python3

from environment_service_server import EnvironmentServiceImpl

import numpy as np


class ReinforcementAgent:
    def __init__(self):
        self.server = EnvironmentServiceImpl(self.callback)

        self.q_table = np.zeros((7, 7, 5))
        self.last_action = -1
        self.last_state = [-1, -1]

        self.epsilon = 1.0
        self.discount_rate = 0.1
        self.learning_rate = 0.9

    def callback(self, request):
        # Request contains the current state of the world
        # as a result of the last action and the reward
        # for the last action
        state = request.state
        reward = request.lastActionReward

        # If we have not yet done an action we can not
        # do a q table update so just return a action
        if self.last_action == -1:
            action = self.select_epsilon_action(state)

            self.last_action = action
            self.last_state = state

            return action

        # An empty state means the episode is over
        episode_over = (len(state) == 0)
        if episode_over:
            # No future if episode is over
            # Just use end of episode reward
            max_future_q = 0

            self.last_action = -1
        else:
            # Get the max reward from the last state and last action
            # by looking at the max reward of
            # all actions for  the current state
            max_future_q = np.max(self.q_table[state[0], state[1]])

        # current q value of state, action
        current_q = self.q_table[self.last_state[0], self.last_state[1], self.last_action]

        # Do the q update with the Bellman equation
        new_q = current_q + self.learning_rate * (reward + self.discount_rate * max_future_q - current_q)
        # print("%.5f, %.5f, %.5f, %.5f" % (current_q, max_future_q, reward, new_q))

        self.q_table[self.last_state[0], self.last_state[1], self.last_action] = new_q

        # # print q table for debugging
        # for i in range(self.q_table.shape[2]):
        #     print(self.q_table[:, :, i])
        # print("-------------")

        # Print current state's q values
        # If episode over current state is undefined
        if not episode_over:
            print("Current state q values:")
            print(self.q_table[state[0], state[1]])
        print("Last state q values:")
        print(self.q_table[self.last_state[0], self.last_state[1]])
        print()

        if episode_over:
            # Set last action to -1 to reset
            action = -1
            self.last_action = action
            print("Episode over, restarting")
            print(self.epsilon)
        else:
            action = self.select_epsilon_action(state)
            self.last_action = action
            self.last_state = state

        if self.epsilon > 0.05:
            self.epsilon *= 0.99

        return action

    def select_epsilon_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(5)
            print("Taking random action " + str(action))
        else:
            action = np.argmax(self.q_table[state[0], state[1]])

        return action


if __name__ == "__main__":
    agent = ReinforcementAgent()
    agent.server.serve()
