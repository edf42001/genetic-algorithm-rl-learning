#!/usr/bin/env python3

import os
import datetime


class DataSaver:
    def __init__(self, folder):
        # The folder where all runs are stored
        self.top_folder = folder

        # The folder where this particular run will be stored
        self.run_folder = self.top_folder + "/" + DataSaver.get_time_string()

        # The folder where the saved agents will be stored
        self.agents_folder = self.run_folder + "/agents"

        self.stats_folder = self.run_folder + "/stats"

        # Create all folders if they don't exist
        self.create_if_not_exist(self.top_folder)
        self.create_if_not_exist(self.run_folder)
        self.create_if_not_exist(self.agents_folder)
        self.create_if_not_exist(self.stats_folder)

        self.rewards_file = None
        self.wins_file = None

    @staticmethod
    def create_if_not_exist(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def open_data_files(self):
        rewards_path = self.stats_folder + "/" + "rewards.txt"
        self.rewards_file = open(rewards_path, 'a')

        wins_path = self.stats_folder + "/wins.txt"
        self.wins_file = open(wins_path, 'a')

    def write_line_to_rewards_file(self, data):
        self.rewards_file.write(" ".join([("%.3f" % num) for num in data]) + "\n")
        self.rewards_file.flush()

    def write_line_to_wins_file(self, winner):
        """Used to record win loss tie record"""
        self.wins_file.write(str(winner) + "\n")
        self.wins_file.flush()

    def close_data_files(self):
        self.rewards_file.close()
        self.wins_file.close()

    def save_agent_to_file(self, agent):
        # Create a new folder for this particular snapshot of our agent
        folder = self.agents_folder + "/" + DataSaver.get_time_string()
        folder = DataSaver.get_unique_folder_name(folder)

        DataSaver.create_if_not_exist(folder)

        agent.save_to_file(folder)

        return folder

    @staticmethod
    def get_time_string():
        # Formatted for use in file names
        return datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

    @staticmethod
    def get_unique_folder_name(folder):
        if not os.path.exists(folder):
            return folder

        i = 1
        while os.path.exists(folder + "-" + str(i)):
            i += 1

        return folder + "-" + str(i)
