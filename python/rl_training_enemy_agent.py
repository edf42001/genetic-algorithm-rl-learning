#!/usr/bin/env python3

import time
import argparse
import sys
import glob
import os

from protos.environment_service_server import EnvironmentServiceImpl

from agents import QTableAgent
from agents import RandomAgent
from agents import QTableExplorationAgent
from agents import CrossEntropyNNAgent
from agents import load_agent

from data_saving.data_saver import DataSaver


def get_restore_point(run_folder):
    """
    Gets the last saved agent folder from a run folder
    Assumes agent folders are in alphabetical order of date
    """
    agents_folder = "saved_runs/" + run_folder + "/agents"

    if not os.path.exists(agents_folder):
        sys.exit("Error: folder not found " + os.path.abspath(agents_folder))

    return sorted(glob.glob(agents_folder + "/*"))[-1]


class RLTrainingEnemyAgent:
    """
    Receives the game callbacks and passes them to the agent
    Handles saving and stopping the training
    """
    def __init__(self, agent):
        self.server = EnvironmentServiceImpl(self.env_callback, self.winner_callback)

        self.agent = agent

        # Create object to handle data saving
        self.data_saver = DataSaver("saved_runs")

        # Create a new folder for the current time
        self.data_saver.open_data_files()

        # How many timesteps have occurred
        self.iterations = 0

        # Stop after this many iterations
        self.num_iterations = 180010

        self.start_time = time.time()

    def on_shutdown(self):
        print("Closing data files")
        self.data_saver.close_data_files()
        print("Iterations, time, iters/s")
        dt = time.time() - self.start_time
        print("%d, %.2f, %d" % (self.iterations, dt, int(self.iterations / dt)))

    def env_callback(self, request):
        self.iterations += 1

        # Pass the data to the agent, and return the actions returned
        return self.agent.env_callback(request)

    def winner_callback(self, request):
        # Record win history
        self.data_saver.write_line_to_wins_file(request.winner)
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


def main(args):
    if args.restore_from:
        agent = load_agent(get_restore_point(args.restore_from))
    else:
        agent = CrossEntropyNNAgent(args.trials, args.hidden_dim,
                                    args.entropy_bonus, args.save_freq)

    trainer = RLTrainingEnemyAgent(agent)
    trainer.server.serve()
    trainer.on_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training Enemy Agent")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--save-freq", type=int, default=15)
    parser.add_argument("--hidden-dim", type=int, nargs="*", default=[8])
    parser.add_argument("--entropy-bonus", type=float, default=0.01)
    parser.add_argument("--restore-from", type=str, default=None)

    known_args, unknown_args = parser.parse_known_args()

    print(known_args)
    print(unknown_args)

    main(known_args)
