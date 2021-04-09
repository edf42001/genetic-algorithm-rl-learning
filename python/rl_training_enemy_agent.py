#!/usr/bin/env python3

import time

from protos.environment_service_server import EnvironmentServiceImpl

from agents import QTableAgent
from agents import RandomAgent
from agents import QTableExplorationAgent
from agents import CrossEntropyNNAgent
from agents import DeepPolicyNNAgent

from data_saving.data_saver import DataSaver


class RLTrainingEnemyAgent:
    def __init__(self):
        self.server = EnvironmentServiceImpl(self.env_callback, self.winner_callback)

        self.agent = DeepPolicyNNAgent()

        # Create object to handle data saving
        self.data_saver = DataSaver("saved_runs")

        # Create a new folder for the current time
        self.data_saver.open_data_files()

        # How many timesteps have occurred
        self.iterations = 0

        # Stop after this many iterations
        self.num_iterations = 680010

        self.start_time = time.time()

        self.win_stats = [0, 0]

    def on_shutdown(self):
        print("Closing data files")
        self.data_saver.close_data_files()
        print("Iterations, time, iters/s")
        dt = time.time() - self.start_time
        print("%d, %.2f, %d" % (self.iterations, dt, int(self.iterations / dt)))

    def env_callback(self, request):
        self.iterations += 1

        if self.iterations % 10000 == 0:
            self.data_saver.write_line_to_wins_file(self.win_stats[0] / sum(self.win_stats))
            self.win_stats = [0, 0]

        # Pass the data to the agent, and return the actions returned
        return self.agent.env_callback(request)

    def winner_callback(self, request):
        self.win_stats[request.winner] += 1
        # Record win history
        print("Episode over, winner " + str(request.winner))

        # Tell the agent the episode has ended
        self.agent.winner_callback(request)

        if self.agent.should_save_to_folder():
            print("Saving agent to file")
            self.data_saver.save_agent_to_file(self.agent)

        if self.iterations > self.num_iterations:
            print("Done, stopping server")
            self.server.stop()

        return None


if __name__ == "__main__":
    trainer = RLTrainingEnemyAgent()
    trainer.server.serve()
    trainer.on_shutdown()
