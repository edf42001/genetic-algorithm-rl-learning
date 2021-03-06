# Reinforcement Learning and Genetic Algorithms

This project is an experimental dive into machine learning to create an agent for the SEPIA environment.
[SEPIA](https://github.com/rail-cwru/Sepia) (Strategy Engine for Programming Intelligent Agents) is a Java environment given to those in CWRU's CSDS 391 (Intro to AI) course.
Agents in this environment control units which can perform tasks such as gathering resources, building buildings, and fighting. I have chosen to focus on the combat aspect as I find it the most exciting.

This project uses gRPC to allow the neural network code (written in Python) to communicate with SEPIA (written in Java).

Implementation is heavily inspired by the [paper](https://arxiv.org/abs/1912.06680) written by OpenAI on their DOTA 2 bot,
a reinforcement learning algorithm that learns to play the video game DOTA 2. 

## Table of Contents
* [Reinforcement Learning and Genetic Algorithms](#reinforcement-learning-and-genetic-algorithms)
    * [Table of Contents](#table-of-contents)
    * [SEPIA](#sepia)
    * [Methods](#methods)
      * [Genetic Algorithms](#genetic-algorithms)
      * [Reinforcement Learning](#reinforcement-learning)
    * [Genetic Algorithm Implementation](#genetic-algorithm-implementation)
      * [Network architecture](#network-architecture)
      * [Observation space](#observation-space)
      * [Action space](#action-space)
      * [Fitness](#fitness)
      * [Results](#results)
    * [Reinforcement Learning Implementation](#reinforcement-learning-implementation)
    * [Q Table Agent](#q-table-agent)
      * [Observation Space](#observation-space-1)
      * [Action Space](#action-space-1)
      * [Results](#results-1)
    * [Cross Entropy Agent](#cross-entropy-agent)
      * [Policy](#policy)
      * [Observation Space](#observation-space-2)
      * [Action Space](#action-space-2)
      * [Rewards](#rewards)
      * [Results and Future Work](#results-and-future-work)
      * [Code Efficiency Analysis](#code-efficiency-analysis)
    * [Deep Policy Learning Agent](#deep-policy-learning-agent)
      * [Policy](#policy-1)
      * [Observation Space](#observation-space-3)
      * [Action Space](#action-space-3)
      * [Rewards](#rewards-1)
      * [Results and Future Work](#results-and-future-work-1)
      * [Appendix A: Parameter Values](#appendix-a-parameter-values)
      * [Appendix B: Entropy Bonus &amp; Randomization](#appendix-b-entropy-bonus--randomization)
      * [Appendix C: Code Efficiency Analysis](#appendix-c-code-efficiency-analysis)
    * [gRPC Implementation](#grpc-implementation)



## SEPIA

The SEPIA environment consists of a grid of squares that can be occupied by a unit, a resource, or be empty.
Units can move, gather resources, build buildings, and attack other units. Each unit can carry a different amount of resources,
has a different attack range, attack damage, and health. Resources consist of gold and wood. These can be used to construct new buildings,
and gold can be used to make more units.

The game is turn based, one player does their actions, then the other. Every unit on a team can do one action per timestep.
Actions can be configured to take more than one time step, it this setup, every actions takes only one step.

The game can be configured to have different end states. For example, when a certain amount of gold has been collected,
or when all the units on a team have been killed.

When the Fog of War is on, units can only "see" a certain distance. When it is off, all units have perfect information
of the game state. By default, the Fog of War is off. 

The following picture shows an example SEPIA game state. The green units are on Team 0, and the red units are on Team 1.
The "f" stands for Footman, a simple unit with an attack range of 1 square.

![SEPIA environment](readme_images/sepia_basic.png "SEPIA environment")

## Methods

I have looked into two machine learning methods for this project. These were genetic algorithms and Reinforcement learning.
In all methods used, every unit on the board is controlled by the same policy. For example, all units would use a copy
of the same neural network weights. Independent and different actions arise because each unit observes different parts
of the state from its location, and has a different internal state (for example, health, or the unit's attack range)


#### Genetic Algorithms

This was the first method investigated. A genetic algorithm (GA) works on a population of agents.
Agents use a neural network to observe their environment at each time step and choose an action to take.
Agents are then run in the environment and those with higher fitness (a measure of success) are selected to reproduce.
Reproduction consists of taking two agents and combining parts of their neural networks together to produce a child.
The child is then randomly mutated. Over time, the fitness of the population increases.

Due to the large observation space and random nature of the algorithm, these agents quickly hit a skill ceiling.
I abandoned the genetic algorithm in favor of reinforcement learning. 

#### Reinforcement Learning

In reinforcement learning (RL), the agent observes its environment each time step and produces an action.
It then receives a reward or penalty for the action. For example, in combat, killing an enemy unit would have a high reward.
Over time, the agent learns to maximize the expected rewards by choosing the best actions. There are different methods of
implementing an observation->actions policy for RL agents. Two common ones are a lookup table (known as a Q table) or a
neural network.

## Genetic Algorithm Implementation

As the GA was the first method investigated, gRPC was not integrated into the project yet. Thus, instead of being able to use python's
many machine learning packages, a very basic neural network implementation was written from scratch in Java.
There were only three different layer types, a Dense layer, Recurrent layer, and LSTM layer.

An agent in SEPIA controls the entire team of units. Each unit, has a copy of the neural network that is being evolved.
Units will take different actions because each unit observes different the world relative to where they are.

#### Network architecture

The brain of an agent consisted of a 3 layer neural network: two recurrent layers of size 16 followed by a fully connected (dense)
layer of size 16.

#### Observation space

The inputs to this network consist of: The unit's current health, then, for every friendly unit, the x and y
displacement to us, and their health. And the same for enemy units. This gives an input size of 10, when playing a 2v2 match.

#### Action space

The action space is 8. The action taken is the output neuron with the highest value. The action space is: move north, south,
east, or west, or attack one of the enemy units. If no neuron has an activation of > 0.5, no action is taken.

#### Fitness

Fitness was based on average distance to the enemy (to encourage encounters where damage could be dealt), amount of damage dealt,
and amount of enemy units killed.

#### Results

A population size of around 400 was used, and training for 500 epochs would take around 3 minutes.
However, agents were usually only able to kill only one of the enemy units, no more. It was at this time that development
pivoted to reinforcement learning. Thus, the rest of this paper shall discuss the reinforcement learning implementation.


## Reinforcement Learning Implementation

A few RL algorithms were tried. Details are listed below.


## Q Table Agent

The first agent created was an agent that used a Q table. A Q Table stores the learned utility of every state action pair.
An optimal agent then simply takes the action with the maximum utility in the current state. This agent was tried first
for its simplicity, however the number of discrete states increases exponentially with the complexity of the environment,
making the Q table agent usable only in the smallest of base cases.

#### Observation Space

The Q table agent plays only in 2 vs 2 matches. The state consists of 6 numbers, these are, the relative x and y coordinates
to our fellow friendly agent and the two enemy agents. However, on a 19x13 board, there are ~(19*13) ^ 3 discrete states,
or 15,000,000. Thus, units can only see the exact position of another unit in a 5x5 grid around them, or, if the other unit
is not in that grid, a value stating which direction the unit is in. This reduces the state size to ~28^3 = 22,000.

#### Action Space
The agent can pick one of five actions: move up, down, left or right, and attack.
Attacking always attacks the closest enemy. This may be changed later.

#### Results

The Q Table agent was succesfully able to coordinate both units to attack the same enemy agent at the same time,
(which human observations have determined leads to a high chance of success in the 2v2 setup). However, this was not
tested with random unit starting locations. It is possible the Q Table agent would perform worse in this scenario.


## Cross Entropy Agent

The cross entropy agent is a policy based RL algorithm similar to a genetic algorithm. The policy is represented by some
parameters, θ. The policy acts on the observation space to produce some actions. We define the distribution of possible
parameters with a mean and standard deviation. Generate a population of policies from this distribution.
Then, run episodes to measure the performance of each policy. Take the top policies, and find the average and standard
deviation of their policy parameters. In this way, the distribution of θ evolves to produce policies that get better
results.

In addition, when averaging the elite's parameters, we add a small, decaying value to the standard deviation,
which helps prevent early convergence. 

#### Policy

The policy of choice was chosen to be a fully connected neural network. The number and size of hidden layers can be
varied, but the input and output sizes have to be consistent with the observation and action space. The CE algorithm
thus learns the optimal weights of the network.

#### Observation Space

Because the CE agent uses a neural network architecture instead of a Q table, we can have many more observation variables.

| **Each unit observes** |
| :--- |
| Current health |
| **For every other unit** |
| x and y distance to unit |
| total distance to unit |
| unit's health |
| Is unit in attack range? |

Integers and floats are interpreted as is, boolean values are converted to 0 or 1. 

The agent keeps a running total of the mean and standard deviation of every input variable, which it uses to normalize
the input data. The input data is then clipped to +- 5 after normalization.

#### Action Space

A unit can take five actions: moving in the 4 cardinal directions, or attacking the nearest enemy.
The action to take is selected by a weighted probability of the softmax output of the network.

#### Rewards

Agents are given rewards for actions they perform. The fitness of an agent is the sum of the rewards each of its units
receives at every time step for one game. Rewards were chosen to encourage getting near and quickly killing the enemy.
The rewards are not perfectly symmetric because it was believed this could cause agents to simply not attack, for fear
of taking damage. If dealing damage is weighted more heavily than taking damage, an agent that attacks will on average
gain a positive reward. The following table lists actions and their associated rewards.

| Action | Reward |
|   ---  |   ---  |
| Every step | -0.03 |
| 1 / (distance to enemy) | 0.009 |
| Damage dealt | 0.05 |
| Damage taken | -0.03 |
| Win | 1.0 |

#### Results and Future Work
A CE agent with 1 hidden layer of size 8 was trained for around 900,000 iterations. It achieved a win rate of
90% in the 2v2 against the default CombatAgentEnemy, which corresponds to a trueskill of 38. Training takes ~7 minutes.

This chart shows the win rate over time of 5 identical training runs of the CE agent. Due to the random nature of the CE algorithm,
although the agent often achieves the 90% win rate, it can sometimes converge to less optimal behavior.

![CE Agent 5 Runs Win Rate](readme_images/ce_agent_5_runs_win_rate.png "CE Agent 5 Runs Win Rate")

The agent employ a few neat strategies. Notably, because the CombatAgentEnemy is very basic, and always tries to move
closer and attack, the CE agent simply waits in place for the enemy to approach it. This allows it to get an attack in
when the enemy moves into range, giving it an advantage in the battle. The friendly units often wait for each other
to catch up and move in a group to the right, which allows them to gang up on the enemy units. However, the waiting for
the enemy strategy can fail and cause a unit to wait right out of range of an enemy unit, because that enemy unit is
attacking its partner and will not move towards it. Simply moving one space would allow the unit to attack, greatly
increasing their chance of winning. Possible remedies to help encourage this behavior in the future would be different
input observations. For example, a state history, which would allow the agent to know if enemy units had moved last turn.
A different network architecture, such as an LSTM, could also resolve this problem, but may be difficult to work with the
CE algorithm.

#### Code Efficiency Analysis

During initial testing of the CE agent, the code was seen to run notably slower than the Q table agent, at only ~1200
game steps / s. Using [cProfile](https://docs.python.org/3/library/profile.html#module-cProfile) and
[gprof2dot](https://github.com/jrfonseca/gprof2dot),
the agent's control code was profiled and a graph of the performance statistics generated.
A cropped portion of this graph is show below. This chart shows the cumulative execution time of a function and the
functions it calls, as well as the time spent in that function alone. Using this graph, the functions
`random_weighted_index`, `normalize_data`, `record_data`, and `softmax`
were selected as potential efficiency-improving targets (The function being profiled, `env_callback`, trivially has the
largest cumulative time spent executing, thus its warning red color can be ignored).

![CE Agent Code Profiling](readme_images/ce_agent_code_profiling.png "CE Agent Code Profiling")

`record_data`, which calculates the running mean and std dev of the agent's observations, was improved by doing
the calculations in batches, so that the function did not need to be called on every timestep.
The other functions were improved by putting them into a test bed where their execution speed could be timed directly,
and by finding small code tweaks that improved their efficiency.
For example, using the python module `random` to select a random value is more effecient than the initially used
using `numpy.random`. After these changes were made, code efficiency increased 60% to ~1900 game steps / s

## Deep Policy Learning Agent

The Deep Policy agent marks the start of the RL algorithms that use a form of _policy gradient_ to learn. In essence,
the agent will sample many actions during the game, and recieve a reward for each. The policy gradient then changes the
weights of the policy so that actions with high rewards will tend to be chosen again when the agent encounters that state
in the future. This agent is inspired by Andrej Karpathy's excellent blog post,
[_Pong From Pixels_](http://karpathy.github.io/2016/05/31/rl/).


#### Policy

The policy network consists of a neural network of size 16x16x5. Each agent observes 16 input variables, has a hidden
layer of size 16, and can choose between 5 output actions. The observation and action spaces are discussed in more
detail below. Currently, because the backpropagation method was programmed by hand as an academic exercise, the network
is limited to one hidden layer. Future work will likely switch to [keras](https://keras.io/), which can do backpropagation
on networks of any size. This will allow for more complex policy networks, such as the use of more hidden layers or LSTM
layers. 

#### Observation Space

A Deep Policy unit observes the following environment data:

| **Each unit observes** |
| :--- |
| Current health |
| **For every other unit** |
| x and y distance to unit |
| total distance to unit |
| unit's health |
| Is unit in attack range? |

Integers and floats are interpreted as is, boolean values are converted to 0 or 1.

The agent keeps a running total of the mean and standard deviation of every input variable, which it uses to normalize
the input data. The input data is then clipped to +- 5 after normalization.

Future work will expand upon this to give the agent more information about its and the enemies' behavior. This will
include observations such as: "Is this enemy attacking me?", "The last action I took", and more.

#### Action Space

A unit can take five actions: moving in the 4 cardinal directions, or attacking the nearest enemy.
The action to take is selected by a weighted probability of the softmax output of the network.

#### Rewards

Agents are given rewards for actions they perform. In order to solve the credit assignment problem, rewards are discounted
over time with a discount factor. Thus, agents are much more likely to take an action that leads to an immediate reward,
but still also likely to take actions that lead to future rewards. The closes the discount factor is towards 1, the farther
into the future the agent looks for rewards. The goal of the agent is to win the game, which requires a certain level of
lookahead. See Appendix A for a list of parameter values used during training of the Deep Policy Agent. 

Rewards were chosen to encourage getting near and quickly killing the enemy.
The rewards are not perfectly symmetric because it was believed this could cause agents to simply not attack, for fear
of taking damage. If dealing damage is weighted more heavily than taking damage, an agent that attacks will on average
gain a positive reward. The following table lists actions and their associated rewards.

| Action | Reward |
|   ---  |   ---  |
| 1 / (distance to enemy) | 0.009 |
| Damage dealt | 0.05 |
| Damage taken | -0.03 |
| Win | 1.0 |

#### Results and Future Work

The Deep Policy agent was able to achieve a 95% win rate, in only 300,000 iterations. This is approximately 3x faster
than the CE agent, and it looses half as often. This speed increase is likely due to the CE agent needing to run
100 trials before doing a parameter update. The following graph shows the results of 5 training runs. Some runs do not
get as good as a start as others, however all tend to end up at the same win rate. Some runs are lucky and encounter
enemies right away and learn that attacking them is good. Others take longer to do so. Possible solutions to this
conundrum are discussed in Appendix B. 

![Deep Policy Win Rate](readme_images/deep_policy_agent_win_rate.png "Deep Policy Win Rate")

#### Appendix A: Parameter Values

These are the parameters that gave the best results (~95% win rate) of the Deep Policy agent

| Parameter | Value |
| --- | --- |
| Discount rate | 0.95 |
| Team spirit | 0.00 |
| Entropy bonus | 0.04 |
| Network size | 16x16x5 |
| Learning rate | 0.01 |
| Batch size | 16 |
| Adam optimizer beta1 | 0.99 |
| Adam optimizer beta2 | 0.999 |

#### Appendix B: Entropy Bonus & Randomization

A common issue with any neural network based learning algorithm is to get stuck in a local minimum.The first solution to
this  is to randomize the initial state of the game. Currently, this consists of randomizing unit's starting locations,
but future work may also add health randomization and more. If an event such as killing an enemy agent takes a long,
specific sequence of actions to achieve, it is unlikely to occur, and the agent will never learn that that event gives
a large reward. By randomizing agent's starting health, this event is more easily happened upon,
and the agent will learn to attack. 

The second way to encourage exploration is with an entropy bonus. The entropy of the policy is the amount of randomness
in the softmax outputs of the network. Without the entropy bonus, the advantage backpropagation will tend to push one
of the action outputs towards 1 and the rest towards 0, which as a low entropy, and the agent will not explore other
actions that may lead to higher rewards. To solve this we add the entropy bonus loss gradient to the advantage gradient.
This makes the agent take slightly more random actions.

The results of this are shown below. For each
entropy bonus value, 3 training runs were done and the results are averaged together. It can be seen that high entropy
bonuses do not do well, because the actions the agents take are too random. The sample size of 3 is somewhat small,
as there is often large variability in training speeds, but it can be generally seen that an entropy bonus of 0.04 learns
the fastest and achieved the best results. The effect of this may become more pronounced as the observation and action
spaces are increased in complexity.

![Deep Policy Entropy Bonus](readme_images/deep_policy_entropy_bonus_win_rate.png "Deep Policy Entropy Bonus")

#### Appendix C: Code Efficiency Analysis
Due to the forward and backward pass operations, code efficiency slowed to 1750 iterations / s. The entropy bonus
backpropigation is especially inefficient, as it uses direct Jacobian matrix multiplication to find the gradient.
Unlike for the regular cross-entropy loss gradient, I am currently unaware of any tricks to simplify the calculations.
This speed is still acceptable.


## gRPC Implementation

gRPC is used to allow SEPIA, a Java program, to communicate with the control code, in Python.
Currently, the gRPC server is in Python, and custom Java code that interfaces with SEPIA is the client.
This may seem backwards, as SEPIA is where the environment is actually located.
However, attempts to implement the server in Java and client in Python ran into issues
with synchronizing, as SEPIA runs on a separate timer and did not wait for the Python client to make requests.

The gRPC service is very simple. A message is sent to the control code containing the current state of
the SEPIA environment, and the response contains the actions each unit should perform.
During self play, both agents in the environment interact with the same Python server. The Python code
differentiates between requests using the player_id field.
