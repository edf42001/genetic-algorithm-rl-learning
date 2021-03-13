# Reinforcement Learning and Genetic Algorithms

This project is an experimental dive into machine learning to create an agent for the SEPIA environment.
[SEPIA] (https://github.com/timernsberger/sepia) (Strategy Engine for Programming Intelligent Agents) is a Java environment given to those in CWRU's CSDS 391 (Intro to AI) course.
Agents in this environment control units which can perform tasks such as gathering resources, building buildings, and fighting. I have chosen to focus on the combat aspect as I find it the most exciting.

This project uses gRPC to allow the neural network code (written in Python) to communicate with SEPIA (written in Java).

I was heavily inspired by the [paper] (https://arxiv.org/abs/1912.06680) written by OpenAI on their DOTA 2 bot, a reinforcement learning algorithm that learned
to play the video game DOTA 2. 

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

![SEPIA environment] (readme_images/sepia_basic.png "SEPIA environment")


## Methods

I have looked into two machine learning methods for this project. These were genetic algorithms and Reinforcement learning.

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
As the GA was the first method investigated, gRPC was not integrated into the project yet. Instead of being able to use python's
many machine learning packages, a very basic neural network implementation was written from scratch in Java.
There were only three different layer types, a Dense layer, Recurrent layer, and LSTM layer. 

