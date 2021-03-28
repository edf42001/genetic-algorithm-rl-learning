#!/usr/bin/env python3

import sys
import time
import argparse
import os.path

from protos.environment_service_server import EnvironmentServiceImpl
from agents import load_agent


class IndividualAgentTest:
    def __init__(self, run_folder, agent_folder):
        # Folders where agent can be found
        self.run_folder = run_folder
        self.agent_folder = agent_folder

        # Server from SEPIA
        self.server = EnvironmentServiceImpl(self.env_callback, self.winner_callback)

        # How many runs the agent has done
        self.trials = 0

        # How many total trials to do before we are done
        self.num_trials = 400

        # How many game steps have occurred
        self.iterations = 0

        # Start time of run
        self.start_time = time.time()

        # Load agent from disk
        self.agent = self.load_agent()

        # Wins, losses, ties
        self.win_stats = [0, 0, 0]

    def winner_callback(self, request):
        # Who won?
        winner = request.winner

        print("Episode over, winner " + str(winner))

        # Record the win, loss, or draw
        self.win_stats[winner] += 1

        self.trials += 1

        # Stop if we have reached the trial limit
        if self.trials >= self.num_trials:
            self.server.stop()

    def env_callback(self, request):
        self.iterations += 1

        # An empty state indicates the episode as ended
        episode_over = len(request.agent_data[0].state) == 0

        # Return actions if the episode is not over
        if not episode_over:
            actions = self.agent.env_callback(request)
            return actions
        else:
            return None

    def load_agent(self):
        agent_folder = self.run_folder + "/agents/" + self.agent_folder

        if not os.path.exists(agent_folder):
            sys.exit("Error: folder not found " + os.path.abspath(agent_folder))

        agent = load_agent(agent_folder)
        # Set agents to eval mode so they don't learn, just run
        agent.set_eval_mode(True)

        return agent

    def on_shutdown(self):
        print("Win stats")
        print(self.win_stats)
        # Print win rate
        print(" ".join([("%.2f" % (x / sum(self.win_stats))) for x in self.win_stats]))
        print("Iterations, , time, iters/s")
        dt = time.time() - self.start_time
        print("%d, %.2f, %d" % (self.iterations, dt, int(self.iterations / dt)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Runs an individual agent')
    parser.add_argument('run_folder', help="The data folder of the run")
    parser.add_argument('agent_folder', help="The agent folder containing the agent")

    args = parser.parse_args()

    # Important: folder is relative to the folder storing all run folders
    run_folder = args.run_folder
    run_folder = "saved_runs/" + run_folder

    agent_folder = args.agent_folder

    # For now, by default, we will do tournaments against the enemy sepia agent, not ourselves
    tester = IndividualAgentTest(run_folder, agent_folder)
    tester.server.serve()
    print("Server stopped, shutting down")
    tester.on_shutdown()

