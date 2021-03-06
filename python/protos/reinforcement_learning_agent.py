#!/usr/bin/env python3
import sys

from environment_service_server import EnvironmentServiceImpl

import numpy as np


class ReinforcementAgent:
    def __init__(self):
        self.server = EnvironmentServiceImpl(self.callback)

        self.q_table = np.zeros((7, 7, 7, 7, 7, 7, 5))
        self.last_actions = dict()
        self.last_states = dict()

        self.epsilon = 1.0
        self.discount_rate = 0.3
        self.learning_rate = 0.9

    def callback(self, request):
        # Request contains the current state of the world
        # as a result of the last action and the reward
        # for the last action
        state = list(request.state)
        reward = request.last_action_reward
        unit_id = request.unit_id

        # If we have not yet done an action we can not
        # do a q table update so just return a action
        if unit_id not in self.last_actions:
            action = self.select_epsilon_action(state)

            self.last_actions[unit_id] = action
            self.last_states[unit_id] = state

            return action

        # print(self.last_actions)
        # print(self.last_states)

        # An empty state means the episode is over
        episode_over = (len(state) == 0)
        if episode_over:
            # No future if episode is over
            # Just use end of episode reward
            max_future_q = 0
        else:
            # Get the max reward from the last state and last action
            # by looking at the max reward of
            # all actions for the current state
            max_future_q = np.max(self.q_table[tuple(state)])

        # current q value of state, action pair
        # Use tuple to index high dimensional q table
        indices = tuple(self.last_states[unit_id] + [self.last_actions[unit_id]])
        current_q = self.q_table[indices]

        # Do the q update with the Bellman equation
        new_q = current_q + self.learning_rate * (reward + self.discount_rate * max_future_q - current_q)
        # print("%.5f, %.5f, %.5f, %.5f" % (current_q, max_future_q, reward, new_q))

        self.q_table[indices] = new_q

        # # print q table for debugging
        # for i in range(self.q_table.shape[2]):
        #     print(self.q_table[:, :, i])
        # print("-------------")

        # Print current state's q values
        # If episode over current state is undefined
        if not episode_over:
            print("Unit %d Current state q values:" % unit_id)
            print(self.q_table[tuple(state)])
        print("Unit %d Last state q values:" % unit_id)
        print(self.q_table[tuple(self.last_states[unit_id])])
        print()

        if episode_over:
            # Reset last actions and return a noop action
            action = -1
            self.last_actions = dict()
            print("Episode over, restarting")
            print("Epsilon: %.3f" % self.epsilon)
        else:
            action = self.select_epsilon_action(state)
            self.last_actions[unit_id] = action
            self.last_states[unit_id] = state

        if self.epsilon > 0.05:
            self.epsilon *= 0.9995

        return action

    def select_epsilon_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(5)
            print("Taking random action " + str(action))
        else:
            action = np.argmax(self.q_table[tuple(state)])

        return action


if __name__ == "__main__":
    agent = ReinforcementAgent()
    agent.server.serve()
