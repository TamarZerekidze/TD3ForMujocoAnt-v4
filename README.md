# TD3-Ant-v4

This project implements the Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm for training an agent to control an Ant robot in the OpenAI Gym environment. TD3 is a reinforcement learning algorithm designed for continuous action spaces and addresses several limitations of the Deep Deterministic Policy Gradient (DDPG) algorithm.

## Overview

The TD3-Ant project includes the following key components:

- **Replay Buffer**: Stores experience tuples to facilitate off-policy learning.
- **Critic Network**: Evaluates the quality of state-action pairs by predicting Q-values.
- **Actor Network**: Determines the optimal action to take given a state.
- **Agent**: Manages the training process, including environment interaction, network updates, and model saving/loading.

## Requirements

To run this project, you need to have the following Python libraries installed:
- `numpy 1.26.3`
- `torch 2.2.0`
- `gym 0.26.2`

You can install these dependencies using pip

## Notes

- The TD3 algorithm incorporates techniques such as "smoothing" and "clamping" to stabilize training. Smoothing involves adding noise to the target actions to prevent the overestimation of Q-values, while clamping ensures that the actions remain within the valid action space.
- The `Agent` class coordinates the interaction with the environment and performs updates to the neural networks based on the experiences stored in the replay buffer. It handles exploration and exploitation, updates the actor and critic networks, and manages the soft updates of the target networks.
- The Jupyter Notebook file provides a detailed explanation and visualization of the results. It includes comprehensive insights into the algorithm's performance, the training process, and the final outcomes.
