#!/usr/bin/env python3
import sys

from environment_service_server import EnvironmentServiceImpl

from agents.q_table_agent import QTableAgent

import numpy as np
import os.path

from data_saving.data_saver import DataSaver
import time


class RLTraining:
    def __init__(self):
        self.server = EnvironmentServiceImpl(self.callback)

        self.agent = QTableAgent()

        # Create object to handle data saving
        self.data_saver = DataSaver("saved_data")

        # Create a new folder for the current time
        self.data_saver.create_new_date_folder()
        self.data_saver.open_rewards_file()

        self.iterations = 0

        self.start_time = time.time()

    def on_shutdown(self):
        print("Closing rewards file")
        self.data_saver.close_rewards_file()
        print("Iterations, time")
        print(self.iterations)
        print(time.time() - self.start_time)

    def callback(self, request):
        self.iterations += 1

        if self.iterations % 10000 == 0:
            print("Saving agent to file")
            self.data_saver.save_agent_to_file(self.agent)

        # Pass the data to the agent, and return the actions returned
        return self.agent.callback(request)


if __name__ == "__main__":
    agent = RLTraining()
    agent.server.serve()
    agent.on_shutdown()