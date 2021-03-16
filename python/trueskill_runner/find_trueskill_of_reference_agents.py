#!/usr/bin/env python3
import glob
import sys

import trueskill

from protos.environment_service_server import EnvironmentServiceImpl
from agents.q_table_agent import QTableAgent


class TrueSkillTournament:
    def __init__(self):
        # Server from SEPIA
        self.server = EnvironmentServiceImpl(self.callback)

        # Which two agents are currently being played against each other
        self.active_agents = [0, 1]

        # List of agents in tournament. Will be ~50MB total
        self.agents = []

        # Trueskill ratings for each agent
        self.ratings = []

        # How many times each agent as been played against each other
        self.trials = 0

        # How many total trials to do before we are done
        self.num_trials = 5

        # Whether or not we are finished with the testing
        self.done = False

        # Load agents from disk
        self.load_agents("../saved_data/trueskill/reference_agents")

        # Set up the trueskill environment and make it a global env
        env = trueskill.TrueSkill(mu=25.0, sigma=8.33, beta=4.17, tau=0.0833, draw_probability=0.1)
        env.make_as_global()

        # Initialize ratings to be all the same
        self.ratings = [trueskill.Rating() for _ in range(len(self.agents))]

    def callback(self, request):
        # An empty state indicates the episode as ended
        episode_over = len(request.agent_data[0].state) == 0

        # When the episode is over, switch to the next agent to train
        if episode_over:
            # If agent[0] has played against all other agents
            if self.active_agents[1] == len(self.agents) - 1:
                # and agent[0] is the last agent in the list
                if self.active_agents[0] == len(self.agents) - 1:
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
            if self.trials == self.num_trials:
                self.done = True
                print("Final ratings")
                print(self.ratings)
                sys.exit(0)

            # Update the ratings based on the match result
            self.update_ratings(agent0_wins=True, draw=False)

        agent0 = self.agents[self.active_agents[0]]
        agent1 = self.active_agents[0]

        actions = agent0.callback(request)
        return actions

    def update_ratings(self, agent0_wins, draw):
        # Need to pass winner as first argument to rate_1vs1
        if agent0_wins:
            winner_idx = self.active_agents[0]
            loser_idx = self.active_agents[1]
        else:
            winner_idx = self.active_agents[1]
            loser_idx = self.active_agents[0]

        # Update the ratings based on the result of the match
        rating0, rating1 = trueskill.rate_1vs1(self.ratings[winner_idx], self.ratings[loser_idx], drawn=draw)
        self.ratings[winner_idx] = rating0
        self.ratings[loser_idx] = rating1

    def load_agents(self, agents_folder):
        for agent_folder in glob.glob(agents_folder + "/*"):
            file = agent_folder + "/agent.p"
            agent = QTableAgent()
            agent.load_from_file(file)

            # Set agents to eval mode so they don't learn, just run
            agent.set_eval_mode(True)
            self.agents.append(agent)


if __name__ == "__main__":
    tournament = TrueSkillTournament()
    tournament.server.serve()


