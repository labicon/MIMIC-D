# Multi-Agent Diffusion Models

This repository contains code for multi-agent diffusion models used in path planning around obstacles. The project is organized into two main folders: `base` and `random`.

## Folder Structure

### old/base
The `base` folder contains the original working 2-agent diffusion model path planning code. This model is designed to navigate around a central obstacle using similar training data. The key components of this folder include:

- **data/**: Includes the training data used for the model.
- **figs/**: Includes the results of the generated trajectory and trained model.
- **diffusion_data_generation.py**: Julia script that generates expert demonstrations for two agents seeking to avoid one another to swap positions.
- **traj_from_diffusion_transformer.py**: Generates a trajectory from a trained 2-agent diffusion model.
- **diffusion_transformer.py**: Uses expert demonstrations to train the 2-agent diffusion model.
- **diffusion_data_generation_obstacle.py**: Julia script that generates expert demonstrations for two agents seeking to avoid one another and a central obstacle to swap positions.
- **traj_from_diffusion_transformer_obstacle.py**: Generates a trajectory from a trained 2-agent diffusion model while avoiding and obstacle.
- **diffusion_transformer_obstacle.py**: Uses expert demonstrations to train the 2-agent diffusion model that avoid an obstacle.


### old/decentralized
The `decentralized` folder contains the 2-agent diffusion model path planning code with the goal of training two agents' policies against another so that they can be executed in a decentralized manner. This approach tries to use the attribute function provided by the AlignDiff code. The key components of this folder include:

- **checking/**: Code to test and verify that multiple modes are able to appear in the generated data when they appear in the expert demonstrations.
- **multi_mode/**: Original code for generating trajectories that are able to reflect multiple modes in the training data.
- **multi_mode_alternating/**: Similar to multi_mode but the training procedure trains each agent for a set number of iterations before training the second agent during every epoch.
- **same_policy/**: Each agent does not have a separate denoiser and instead generates trajectories based on the same denoiser model.
- **single_mode/**: The training data for each agent only contains a single mode and thus each agent has only one possible mode to generate.
- **vary_init/**: Varying the initial and final conditions of the expert demonstrations by adding more noise to the training data effectively having demonstrations start and end at different positions.

### old/random
The `random` folder contains work that utilizes expert demonstration data that includes a random array of initial and final positions for the agents. The goal of this work is to find optimal trajectories for two agents starting at given initial and final conditions. The key components of this folder include:

- **data/**: Includes the expert demonstration data.
- **figs/**: Includes the results of the generated trajectory and trained model.
- **traj_from_diffusion_transformer_cond.py**: Generates a trajectory from a trained diffusion model with random initial and final positions.
- **diffusion_transformer_cond.py**: Uses expert demonstrations to train the diffusion model with random initial and final positions.

### old/conditional
The `conditional` folder is for performing the 2-agent swap task with a conditional diffusion model.
- **data/**: Includes the expert demonstration data.
- **figs/**: Includes the results of the generated trajectory and trained model.
- **conditional_training.py**: Uses expert demonstrations to train the diffusion model conditioned on initial and final positions.

### ctde_conditional
The `ctde_conditional` folder extends the 'conditional' folder's work in the 2-agent swap envionrment by training a separate model for each agent with a shared loss and then executing the models separately during trajectory generation. See the README.md file in this directory for more information.

### arm
The `arm` folder is for the robosuite environment tasks. See the README.md file in this directory for more information.

### madiff
The `madiff` folder is for running the MaDiff code for our 2-agent swap environment.

### old/herding
The `herding` folder is the beginning of creating our own herding environment.

### dependencies
The `dependencies` folder contains the yaml and requirements files for the environment.