# Rubik RL
**Rubik-RL** is a reinforcement learning project focused on solving the 2x2 Rubik's Cube.  

The project includes:
- Reward systems with multiple modes (e.g., "basic", "bottom_layer_corners", "bottom_face", "bottom_layer").  
- Custom environment for the Rubik2x2 cube.  
- Scripts for training, evaluating, and running experiments.
- MLflow integration for tracking experiments and metrics.  
- Implementations of RL DQN agent for cube solving.  

## Project Structure

```yaml
Rubik-RL
├── agents
│   ├── __init__.py
│   ├── dqn_agent.py                     # Implementation of DQN agent and training loop
│   ├── q_network.py                     # Q-network neural network architecture
│   └── replay_buffer.py                 # Replay buffer for experience replay
├── envs
│   ├── rewards
│   │   ├── reward_helpers.py            # Helper functions for reward calculations
│   │   └── reward_interface.py          # Reward computation interface
│   ├── __init__.py
│   ├── cube_state.py                    # Cube state representation and utilities
│   ├── render_utils.py                  # Rendering utilities for cube visualization
│   └── rubik2x2_env.py                  # Gym environment for the 2x2 Rubik's Cube
├── experiments
│   └── experiment_results.csv           # CSV logs of experiment results
├── models
│   └── rl_agent.pth                     # Saved trained RL agent
├── scripts
│   └── evaluate_model.py                # Scripts to evaluate a trained agent
├── training
│   ├── __init__.py
│   ├── rl_experiment_planner.py         # Planner for running RL experiments
│   └── train_rl.py                      # Training script for RL agent
├── .gitignore
├── LICENSE
└── README.md
└── requirements.txt                     # Python dependencies
```

## Run locally
Run RL experiments as follows:
```bash
PYTHONDONTWRITEBYTECODE=1 python -m training.rl_experiment_planner
```

## License

Rubik-RL is released under the MIT license.

## Author

Sebastian Brzustowicz &lt;Se.Brzustowicz@gmail.com&gt;