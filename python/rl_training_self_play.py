#!/usr/bin/env python3
import sys

from protos.environment_service_server import EnvironmentServiceImpl

from agents import QTableAgent
from agents import RandomAgent
from agents import QTableExplorationAgent

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

        # How many games have been played
        self.epochs = 0

        # Used to make sure both players have sent their final message before closing the server
        self.winner_pid_seen = 0

        # What percent of the time to play against past agents
        self.past_play_ratio = 0.2

        # Learning rate for determining quality of previous agents
        self.past_play_lr = 0.01

        # Quality of past agents. Store folder name, and quality
        self.past_agents_quality = np.array([], dtype='float64')
        self.past_agents_folders = []
        self.current_past_agent = -1  # The index of the agent that is currently active, -1 for ourselves

        # Start time
        self.start_time = time.time()

    def on_shutdown(self):
        print("Closing data files")
        self.data_saver.close_data_files()
        print("Iterations, epochs, time, iters/s")
        dt = time.time() - self.start_time
        print("%d, %d, %.2f, %d" % (self.iterations, self.epochs, dt, int(self.iterations / dt)))

    def env_callback(self, request):
        """
        Called once per agent, so twice per timestep
        Do a learning update and return the requested action
        """

        # Save every couple of steps, only once for player0, don't double save
        should_save = (self.iterations < 100000 and (self.iterations + 1) % 15000 == 0) or \
                      (self.iterations + 1) % 30000 == 0
        if should_save and request.player_id == 0:
            # print("Saving agent to file")
            # self.data_saver.save_agent_to_file(self.main_agent)
            pass

        # Pass the data to the agent, and return the actions returned
        # Make sure to use the right agent
        if request.player_id == 0:
            self.iterations += 1  # Both player's iterations are the same so only count one
            return self.main_agent.env_callback(request)
        else:
            return self.other_agent.env_callback(request)

    def winner_callback(self, request):
        # This is called twice, only do things for one of the times
        if request.player_id == 0:
            self.epochs += 1

            # If we won, update the past agent's quality
            if request.winner == 0:
                self.past_agent_quality_update()

            # Save a new agent every 200 games
            if self.epochs % 200 == 0:
                # Save current agent to a file
                folder = self.data_saver.save_agent_to_file(self.main_agent)

                # Store agent's folder name and quality
                self.past_agents_folders.append(folder)
                if self.past_agents_quality.size == 0:
                    self.past_agents_quality = np.array([1.0])
                else:
                    self.past_agents_quality = np.append(self.past_agents_quality, np.max(self.past_agents_quality))

            # Switch to a new agent every 50 games
            if self.epochs % 50 == 0:
                print(self.past_agents_quality)
                print(self.softmax(self.past_agents_quality))
                print()
                # Get a new agent
                self.select_new_agent()

            # Record win history
            self.data_saver.write_line_to_wins_file(request.winner)
            # print("Episode over, winner " + str(request.winner))

        # Check for doneness of our session, close server
        if self.iterations > self.num_iterations:
            # This winner_callback will be called twice, once for each agent
            # Only close on the second time (order of calling not known)
            if self.winner_pid_seen == 0:
                self.winner_pid_seen += 1
            else:
                print("Done, stopping server")
                agent.server.stop()

        return None

    def select_new_agent(self):
        # Play against self most of the time
        if np.random.uniform() > self.past_play_ratio:
            # Load latest version of ourself
            index = -1
        else:
            # Pick random past agent according to quality
            index = self.random_weighted_index(self.softmax(self.past_agents_quality))

        self.current_past_agent = index
        self.other_agent = QTableAgent()
        print("Selected agent " + str(index) + " to play against")
        self.other_agent.load_from_folder(self.past_agents_folders[index])
        self.other_agent.set_eval_mode(True)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def random_weighted_index(self, x):
        return np.random.choice(np.arange(x.size), 1, p=x)[0]

    def past_agent_quality_update(self):
        """Update past agent qualities, called when current agent beats a past agent"""
        prob = self.softmax(self.past_agents_quality)[self.current_past_agent]

        # Update quality of agent i = qi - lr / (N * pi)
        self.past_agents_quality[self.current_past_agent] -= self.past_play_lr / (len(self.past_agents_quality) * prob)


if __name__ == "__main__":
    agent = RLTrainingSelfPlay()
    agent.server.serve()
    agent.on_shutdown()
