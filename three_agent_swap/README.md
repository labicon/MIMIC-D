# Three Agent Road Crossing

This directory contains the implementation of a conditional diffusion model for multi-agent systems using the Centralized Training with Decentralized Execution (CTDE) paradigm for the three agent road crossing scenario. Only relevant files are included.

## Subdirectories

- **`data/`**: Contains datasets used directly for training
- **`init_final_pose/`**: Contains the lists of randomly sampled initial and final positions for the three agents to use.
- **`loss/`**: Recorded data related to the loss function where "npy/" stores the data and "plots/" stores the final plots.
- **`sampled_trajs/`**: Contains the sampled trajectories for the various implementations(see below).
- **`trained_models/`**: Contains the trained models for the various implementations (see below).


## trained_models/ Descriptions
P#E# means that it is trained for a certain number of planning timesteps and a certain number of execution timesteps.
- **`P25E1_400demos_06noise_hierarchicalfinalposcond`**: MPC planning for 25 timesteps, execute 1 timesteps wtih 400 expert demonstration, 0.6 demo noise, and conditioned on each others' current positions.
- **`P25E1_400demos_06noise_vanillaCTDE`**: MPC planning for 25 timesteps, execute 1 timesteps wtih 400 expert demonstration, 0.6 demo noise, and only conditioned on own inital and final positions.


## sampled_trajs/ Descriptions
P#E# means that it is trained for a certain number of planning timesteps and a certain number of execution timesteps.
- **`mpc_P25E1_400demos_06demonoise_06samplenoise_200N_vettedinitfinal_vanillaCTDE`**: MPC planning for 25 timesteps, execute 1 timesteps wtih 400 expert demonstration, 0.6 demo noise, 0.6 sample noise, 200 denoising steps, vetted intial and final positions to ensure 0.75 apart, and only conditioned on own inital and final positions.
- **`mpc_P25E1_400demos_06demonoise_finalposcond_06samplenoise_200N_vettedinitfinal`**: MPC planning for 25 timesteps, execute 1 timesteps wtih 400 expert demonstration, 0.6 demo noise, 0.6 sample noise, 200 denoising steps, vetted intial and final positions to ensure 0.75 apart, and conditioned on each others' initial and final positions.


## Main Files
- **`analyze_expertdata.ipynb`**: Visualizing the expert demonstrations.
- **`analyze_sampled.ipynb`**: Visualizing the sampled trajectories and analyzing collisions.
- **`three_agent_data_generation_highway.ipynb`**: Generating the expert demonstrations.
- **`training_mpc_finalposcond_nolf.py`**: Training and sampling code to the full conditioning with no leader follower.
- **`training_mpc_vanillaCTDE.py`**: Training and sampling code with the simplified vanilla CTDE conditioning with no leader follower.
