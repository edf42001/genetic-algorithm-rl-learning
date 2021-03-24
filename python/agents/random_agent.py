import numpy as np
import json

from agents import Agent


class RandomAgent(Agent):
    def __init__(self):
        # How many iterations this agent has been training for
        super().__init__()
        self.iterations = 0

        # The agent only takes actions in eval mode, and doesn't train
        self.eval_mode = False

    def env_callback(self, request):
        """Do a random action if this isn't the terminal step"""
        # An empty state indicates the episode as ended
        episode_over = len(request.agent_data[0].state) == 0

        if episode_over:
            return None
        else:
            # Randomly choose an action, including moving and attacking
            # actions are from 0 - 4. 1 action per agent
            return np.random.randint(5, size=len(request.agent_data))

    def winner_callback(self, request):
        """Noop"""
        pass

    def save_to_file(self, folder):
        config_file = folder + "/config.txt"
        config = dict()
        config["name"] = type(self).__name__

        with open(config_file, 'w') as f:
            json.dump(config, f)

    def load_from_folder(self, folder):
        """Nothing to do"""
        pass

    def set_eval_mode(self, eval_mode):
        self.eval_mode = eval_mode
