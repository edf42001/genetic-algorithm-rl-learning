#!/usr/bin/env python3
import sys

from protos.environment_service_server import EnvironmentServiceImpl

from agents.q_table_agent import QTableAgent
from agents.random_agent import RandomAgent
from agents.q_table_exploration_agent import QTableExplorationAgent

import numpy as np
import os.path

from data_saving.data_saver import DataSaver
import time


class RLTrainingSelfPlay:
    def __init__(self):
        self.server = EnvironmentServiceImpl(self.env_callback, self.winner_callback)

        # Create the main agent and a slot for the other agent they will be playing against
        self.main_agent = QTableAgent()
        self.other_agent = QTableAgent()

        # Create object to handle data saving
        self.data_saver = DataSaver("saved_runs")

        # Create a new folder for the current time
        self.data_saver.open_data_files()

        # How many timesteps have occurred
        self.iterations = 0

        # Stop after this many iterations
        self.num_iterations = 180010

        # Used to make sure both players have sent their final message before closing the server
        self.winner_pid_seen = 0

        self.start_time = time.time()

    def on_shutdown(self):
        print("Closing data files")
        self.data_saver.close_data_files()
        print("Iterations, time, iters/s")
        print(self.iterations)
        print(time.time() - self.start_time)
        print(str(int(self.iterations / (time.time() - self.start_time))))

    def env_callback(self, request):
        """
        Called once per agent, so twice per timestep
        Do a learning update and return the requested action
        """

        # Save every couple of steps, only once for player0, don't double save
        should_save = (self.iterations < 100000 and (self.iterations + 1) % 15000 == 0) or \
                      (self.iterations + 1) % 30000 == 0
        if should_save and request.player_id == 0:
            print("Saving agent to file")
            self.data_saver.save_agent_to_file(self.main_agent)

        # Pass the data to the agent, and return the actions returned
        # Make sure to use the right agent
        if request.player_id == 0:
            self.iterations += 1  # Both player's iterations are the same so only count one
            return self.main_agent.callback(request)
        else:
            return self.other_agent.callback(request)

    def winner_callback(self, request):

        # Record win history
        self.data_saver.write_line_to_wins_file(request.winner)

        # Check for doneness of our session, close server
        if self.iterations > self.num_iterations:
            # This winner_callback will be called twice, once for each agent
            # Only close on the second time
            if self.winner_pid_seen == 0:
                self.winner_pid_seen += 1
            else:
                print("Done, stopping server")
                agent.server.stop()

        return None


if __name__ == "__main__":
    agent = RLTrainingSelfPlay()
    agent.server.serve()
    agent.on_shutdown()
