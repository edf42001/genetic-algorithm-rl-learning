import numpy as np
import pickle


class QTableAgent:
    def __init__(self):
        self.q_table = np.zeros((7, 7, 7, 7, 7, 7, 5))
        self.last_actions = dict()
        self.last_states = dict()

        self.epsilon = 1.0  # Random action chance
        self.epsilon_decay = 0.99995
        self.discount_rate = 0.95  # Discount future rewards. Time horizon = 1 / (1 - rate)
        self.learning_rate = 0.05  # Learning rate
        self.team_spirit = 0  # How much rewards are shared

        # The agent only takes actions in eval mode, and doesn't train
        self.eval_mode = False

        self.total_epoch_reward = 0

    def eval_mode_update(self, request):
        # Actions to be returned
        actions = []

        # An empty state indicates the episode as ended
        episode_over = len(request.agent_data[0].state) == 0

        if episode_over:
            return None

        # Because we are in eval mode, we don't care about the reward
        # just return the best action for the state
        for data in request.agent_data:
            state = list(data.state)
            action = self.select_epsilon_action(state, 0)
            actions += [action]

        return actions

    def learning_mode_update(self, request):
        """
        This function does the Q table update and returns
        the next actions to take
        :param request: Data from the SEPIA environment
        :return: Actions for units in SEPIA to take. Is passed to gRPC server
        """
        # Get total reward for team spirit and logging
        total_reward = sum([a.last_action_reward for a in request.agent_data])
        self.total_epoch_reward += total_reward

        # Actions to be returned
        actions = []

        # An empty state indicates the episode as ended
        # We still need to use the last reward to update
        episode_over = len(request.agent_data[0].state) == 0

        # Request contains, for each agent:
        # the current state of the world as a result of the last action
        # and the reward for the last action
        for data in request.agent_data:
            state = list(data.state)
            reward = data.last_action_reward
            unit_id = data.unit_id

            # If we have not yet done an action we can not
            # do a q table update so just choose an action
            # and move on with loop
            if unit_id not in self.last_actions:
                action = self.select_epsilon_action(state, self.epsilon)

                self.last_actions[unit_id] = action
                self.last_states[unit_id] = state

                actions += [action]
                continue

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

            # Print current state's q values
            # If episode over current state is undefined
            # if not episode_over:
            #     print("Unit %d Current state q values:" % unit_id)
            #     print(self.q_table[tuple(state)])
            # print("Unit %d Last state q values:" % unit_id)
            # print(self.q_table[tuple(self.last_states[unit_id])])
            # print()

            if not episode_over:
                action = self.select_epsilon_action(state, self.epsilon)
                actions += [action]
                self.last_actions[unit_id] = action
                self.last_states[unit_id] = state

        if self.epsilon > 0.05:
            self.epsilon *= self.epsilon_decay

        if episode_over:
            # Reset last actions and return a noop action
            actions = None
            self.last_actions = dict()
            self.last_states = dict()

            # Save total epoch reward and reset
            # self.data_saver.write_line_to_rewards_file([self.total_epoch_reward])
            self.total_epoch_reward = 0

            print("Episode over, restarting")
            print("Last total reward: %.3f" % total_reward)
            print("Epsilon: %.3f" % self.epsilon)

        return actions

    def callback(self, request):
        """Simply pass the request to the corresponding handler"""
        if self.eval_mode:
            return self.eval_mode_update(request)
        else:
            return self.learning_mode_update(request)

    def select_epsilon_action(self, state, epsilon):
        if np.random.uniform() < epsilon:
            action = np.random.randint(5)
            # print("Taking random action " + str(action))
        else:
            action = np.argmax(self.q_table[tuple(state)])

        return action

    def save_to_file(self, file):
        data = dict()
        data["name"] = "QTableAgent"
        data["q_table"] = self.q_table

        with open(file, 'wb') as f:
            pickle.dump(data, f)

    def load_from_file(self, file):
        with open(file, 'rb') as f:
            data = pickle.load(f)

            self.q_table = data["q_table"]

    def set_eval_mode(self, eval_mode):
        self.eval_mode = eval_mode
