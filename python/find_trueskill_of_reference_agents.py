#!/usr/bin/env python3
import glob
import sys
import time
import argparse
import os.path
import numpy as np

import trueskill

from protos.environment_service_server import EnvironmentServiceImpl
from agents import QTableAgent
from agents import QTableExplorationAgent
from agents import CrossEntropyNNAgent
from data_saving.data_saver import DataSaver


class TrueSkillTournament:
    def __init__(self, run_folder, self_play):
        self.run_folder = run_folder

        # Server from SEPIA
        self.server = EnvironmentServiceImpl(self.env_callback, self.winner_callback)

        # Which two agents are currently being played against each other
        self.active_agents = [0, 0]

        # List of agents in tournament. Will be ~50MB total
        self.agents = []

        # List of agent associated data, such as names, used for saving later
        self.agents_data = []

        # Trueskill ratings for each agent
        self.ratings = []

        # How many times each agent as been played against each other
        self.trials = 0

        # How many total trials to do before we are done
        self.num_trials = 40

        # True if player 1 is a sepia agent, not another one of our agents
        self.playing_against_sepia = not self_play

        # How many game steps have occurred
        self.iterations = 0

        # Start time of run
        self.start_time = time.time()

        # Load agents from disk
        self.load_agents()

        # Wins, losses, ties
        self.win_stats = np.zeros((len(self.agents), 3))

        # Set up the trueskill environment and make it a global env
        env = trueskill.TrueSkill(mu=25.0, sigma=8.33, beta=4.17, tau=0, draw_probability=0.05)
        env.make_as_global()

        # Initialize ratings to be all the same
        self.ratings = [trueskill.Rating() for _ in range(len(self.agents))]
        self.sepia_rating = trueskill.Rating(30, 0.00000000001)

    def winner_callback(self, request):
        # Who won?
        winner = request.winner

        # Which player sent this data
        player_id = request.player_id

        # When the episode is over, switch to the next agent to train
        # We will get two episode over notices, from each agent
        # so only update on one of them
        if player_id == 0:
            # Agents that are not trained tend to tie, because they don't approach and the enemy doesn't approach them
            # Then the episode hits the step limit and ends, trueskill thinks a tie means you are on even footing,
            # but it just means you didn't bother attacking (with this current setup, different agents might behave)
            # Differently in the future. So for now we say that a tie is a loss for us
            draw = (winner == -1)
            agent0_wins = (winner == 0)

            # Record the win, loss, or draw
            self.win_stats[self.active_agents[0], winner] += 1

            # Update the ratings based on the match result
            self.update_ratings(agent0_wins=agent0_wins, draw=False)

            if self.playing_against_sepia:
                if self.active_agents[0] == len(self.agents) - 1:
                    self.active_agents[0] = 0
                    self.trials += 1
                    print("Trials elapsed: " + str(self.trials))
                else:
                    self.active_agents[0] += 1
            # If agent[0] has played against all other agents
            elif self.active_agents[1] == len(self.agents) - 1:
                # and agent[0] is the second to last, because the last agent would have noone to play
                if self.active_agents[0] == len(self.agents) - 2:
                    # Reset agents to the first for next trial runs
                    self.active_agents = [0, 1]
                    self.trials += 1
                else:
                    # Otherwise, set agent[0] to the next one and keep going
                    self.active_agents[0] += 1
                    self.active_agents[1] = self.active_agents[0] + 1
            else:
                # Otherwise, increment agent[1] to the next agent
                self.active_agents[1] += 1

            # Stop if we have reached the trial limit
            if self.trials >= self.num_trials:
                self.save_results()
                self.server.stop()

            # Print who is playing right now
            # print("Now playing version %d vs %d" % (self.active_agents[0], self.active_agents[1]))

    def env_callback(self, request):
        self.iterations += 1

        # Which player sent this data
        player_id = request.player_id

        # An empty state indicates the episode as ended
        episode_over = len(request.agent_data[0].state) == 0

        # Return actions if the episode is not over
        if not episode_over:
            # Get the current active agent controlled by the player whose data was sent
            agent = self.agents[self.active_agents[player_id]]
            actions = agent.env_callback(request)
            return actions

    def update_ratings(self, agent0_wins, draw):
        # Update the ratings based on the result of the match
        # Need to pass winner as first argument to rate_1vs1
        if self.playing_against_sepia:
            # print("Before ratings " + str(self.ratings[self.active_agents[0]]) + " " + str(self.sepia_rating))
            # print("Agent0 won " + str(agent0_wins))
            if agent0_wins:
                rating0, rating1 = trueskill.rate_1vs1(self.ratings[self.active_agents[0]], self.sepia_rating, drawn=draw)
                self.ratings[self.active_agents[0]] = rating0
                self.sepia_rating = rating1
            else:
                rating0, rating1 = trueskill.rate_1vs1(self.sepia_rating, self.ratings[self.active_agents[0]], drawn=draw)
                self.sepia_rating = rating0
                self.ratings[self.active_agents[0]] = rating1
            # print("After ratings " + str(self.ratings[self.active_agents[0]]) + " " + str(self.sepia_rating))

        else:
            if agent0_wins:
                winner_idx = self.active_agents[0]
                loser_idx = self.active_agents[1]
            else:
                winner_idx = self.active_agents[1]
                loser_idx = self.active_agents[0]

            rating0, rating1 = trueskill.rate_1vs1(self.ratings[winner_idx], self.ratings[loser_idx], drawn=draw)
            self.ratings[winner_idx] = rating0
            self.ratings[loser_idx] = rating1

    def load_agents(self):
        agents_folder = self.run_folder + "/agents"

        if not os.path.exists(agents_folder):
            sys.exit("Error: folder not found " + os.path.abspath(agents_folder))

        for agent_folder in sorted(glob.glob(agents_folder + "/*")):
            agent = CrossEntropyNNAgent()
            agent.load_from_folder(agent_folder)

            # Set agents to eval mode so they don't learn, just run
            agent.set_eval_mode(True)

            # Store agent and agent's file name
            self.agents.append(agent)
            self.agents_data.append(os.path.basename(agent_folder))

        if len(self.agents) == 0:
            sys.exit("Error: No agents found in " + os.path.abspath(agents_folder))

    def save_results(self):
        print("Final ratings")
        print(" ".join(["%.2f" % rating.mu for rating in self.ratings]))
        print("Sepia bot rating")
        print(self.sepia_rating)
        print("Win stats")
        print(self.win_stats)
        print("Iterations, time")
        print(self.iterations)
        print(time.time() - self.start_time)

        results_folder = self.run_folder + "/results"
        DataSaver.create_if_not_exist(results_folder)
        results_file = results_folder + "/trueskill.txt"

        # Fix the sepia bot with a skill of 30 to compare
        offset = 30 - self.sepia_rating.mu
        with open(results_file, 'w') as f:
            for i in range(len(self.ratings)):
                name = self.agents_data[i]
                iters = self.agents[i].iterations
                rating = self.ratings[i].mu + offset

                f.write("%s %d %.2f\n" % (name, iters, rating))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the trueskill of agents')
    parser.add_argument('run_folder', help="The data folder for the run to be analyzed")
    parser.add_argument('--self-play', action="store_true", help="Flag to have agents play among themselves, "
                                                                 "instead of fighting the sepia enemy agent")

    args = parser.parse_args()

    # Important: folder is relative to the folder storing all run folders
    run_folder = args.run_folder
    run_folder = "saved_runs/" + run_folder

    self_play = args.self_play

    # For now, by default, we will do tournaments against the enemy sepia agent, not ourselves
    tournament = TrueSkillTournament(run_folder, self_play)
    tournament.server.serve()
    print("Server stopped, shutting down")
