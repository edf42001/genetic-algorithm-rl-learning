#!/usr/bin/env python3

import os
import datetime


class DataSaver:
    def __init__(self, folder):
        self.folder = folder
        self.data_folder = ""

        # Create the data directory if it doesn't exit
        self.create_if_not_exist(folder)

        self.rewards_file = None

    @staticmethod
    def create_if_not_exist(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def create_new_date_folder(self):
        now = DataSaver.get_time_string()
        path = self.folder + "/run_statistics/" + now
        DataSaver.create_if_not_exist(path)

        self.data_folder = path

    def open_rewards_file(self):
        path = self.data_folder + "/" + "rewards.txt"
        self.rewards_file = open(path, 'a')

    def write_line_to_rewards_file(self, data):
        self.rewards_file.write(" ".join([("%.5f" % num) for num in data]) + "\n")

    def close_rewards_file(self):
        self.rewards_file.close()

    def save_agent_to_file(self, agent):
        now = DataSaver.get_time_string()
        folder = self.folder + "/trueskill/reference_agents/" + now

        DataSaver.create_if_not_exist(folder)
        path = folder + "/" + "agent.p"

        agent.save_to_file(path)

    @staticmethod
    def get_time_string():
        # Formatted for use in file names
        return datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S")




