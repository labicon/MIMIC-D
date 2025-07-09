# CTDE Conditional

This directory contains the implementation of a conditional diffusion model for multi-agent systems using the Centralized Training with Decentralized Execution (CTDE) paradigm.

## Subdirectories

- **`data/`**: Contains expert demonstrations and stored means and variances.
- **`figs/`**: Resulting plots of sampled trajectories.
- **`sampled_trajs/`**: Stores of sampled trajectories for plotting.
- **`trained_models/`**: Contains the trained models for the various implementations (see below).

## figs/ Descriptions
Some of the overlap between categories in the 'splice/' and not is due to those in the 'splice/' directory planning for the whole trajectory where as those not in the directory are only planning for a single 10 timestep horizon.
- **`splice/baseline`**: Simple plan 10 timesteps, execute 10 timesteps
- **`splice/reactive`**: Plan 10 timesteps, execute 10 timesteps with conditioning on both agents' positions
- **`splice/reactive_collision`**: Plan 10 timesteps, execute 10 timesteps with collision avoidance (bad)
- **`splice/reactive_mode`**: Plan 10 timesteps, execute 10 timesteps with conditioning on mode
- **`splice/reactive_safety`**: Plan 10 timesteps, execute 10 timesteps with safety filter based on how likely pair of positions are to appear in the expert demonstration distribution
- **`T10`**: Planning only 10 timesteps after splicing up expert demonstrations
- **`T10_reactive`**: Planning only 10 timesteps and conditioning on own state and other agent's state
- **`T10_reactivemode`**: Planning only 10 timesteps and conditioning on both agents' state as well as the mode (labeled data)
- **`T100`**: Full horizon (100 timestpes) planning
- **`T100_noise0.05`**: Full horizon (100 timestpes) planning with noised sampled atat 0.05 noise level

## sampled_trajs
- **`mpc_P25E1_nolf_revisedsampling`**: Plan 25, execute 1 conditioned on each others' current initial positions and final positions with correct sampling; model: _P25E1_nolf.pt
- **`mpc_P25E1_vanillaCTDE`**: Plan 25, execute 1 only conditioned on own initial and final positions; model: _P25E1_vanillaCTDE.pt


## sampled_trajs/deprec
- **`mpc_guidance_P10E1`**: Plan 10, execute 1 with old sampling guidance; model: _T10_mpc_guidance.pt
- **`mpc_latent_P10E1`**: Plan 10, execute 1 with latent state conditioning; model: _P10E1_lf_latent.pt
- **`mpc_latent_P10E3`**: Plan 10, execute 3 with latent state conditioning; model: missing
- **`mpc_latenttrain_P10E3`**: Plan 10, execute 3 with latent state conditioning where the latent state MLP is simultaneously trained; model: _P10E3_lf_latenttrain.pt
- **`mpc_latenttrain_P25E3`**: Plan 25, execute 3 with latent state conditioning where the latent state MLP is simultaneously trained; model: _P25E3_lf_latenttrain.pt
- **`mpc_latenttrain_P25E3_bigger`**: Plan 25, execute 3 with latent state conditioning where the latent state MLP is simultaneously trained with a larger model; model: _P25E3_lf_latenttrain_bigger.pt
- **`mpc_latenttrain_P25E5`**: Plan 25, execute 5 with latent state conditioning where the latent state MLP is simultaneously trained; model: _P25E5_lf_latenttrain.pt
- **`mpc_P10E1`**: Plan 10, execute 1 with latent state conditioning; model: _P10E1_lf_latent.pt
- **`mpc_P10E5`**: Plan 10, execute 5; model: _P10_E5_reactive.pt
- **`mpc_P20E10`**: Plan 20, execute 10; model: _P20E10_lf.pt
- **`mpc_P25E1_nolf`**: Plan 25, execute 1 with no leader follower structure but incorrect sampling; model: _P25E1_nolf.pt
- **`mpc_P25E3`**: Plan 25, execute 3; model: _P25E3.pt
- **`mpc_P25E3_guidesample1`**: Plan 25, execute 3 with sampling guidance; model: _P25E3.pt
- **`mpc_P25E3_guidesample2`**: Plan 25, execute 3 with sampling guidance; model: _P25E3_2.pt
- **`mpc_P25E3_mpcdata1`**: Plan 25, execute 3 using MPC generated expert data; model: _P25E3_mpcdata.pt
- **`mpc_P25E3_mpcdata2`**: Plan 25, execute 3 using MPC generated expert data and sampling guidance; model: _P25E3_mpcdata.pt
- **`mpc_P25E3_noguidance1`**: Plan 25, execute 3; model: _P25E3.pt
- **`mpc_P25E3_noguidance2`**: Plan 25, execute 3; model: _P25E3_2.pt
- **`mpc_P25E3_smallcond`**: Plan 25, execute 3 with smaller conditional vector; model: _P25E3_smallcond.pt
- **`mpc_P25E3_smallcond_bigger`**: Plan 25, execute 3 with smaller conditional vector and bigger model; model: _P25E3_smallcond_bigger.pt
- **`mpc_P25E3_smallcond_guidesample`**: Plan 25, execute 3 with smaller conditional vector and guidance during sampling; model: _P25E3_smallcond.pt
- **`mpc_P25E3_smallcond_guidesample2`**: Plan 25, execute 3 with smaller conditional vector and guidance during sampling; model: _P25E3_smallcond.pt
- **`mpc_P25E5`**: Plan 25, execute 5; model: _P25E5.pt
- **`mpc_P25E5_guidesample`**: Plan 25, execute 5 with sampling guidance; model: _P25E5.pt
- **`mpc_P25E5_noguidance`**: Plan 25, execute 5; model: _P25E5.pt
- **`mpc_P25E5_smallcond`**: Plan 25, execute 5 with smaller conditional vector; model: _P25E5_smallcond.pt
- **`mpc_P25E5_smallcond_guidesample`**: Plan 25, execute 5 with smaller conditional vector and guidance during sampling; model: _P25E5_smallcond.pt
- **`mpc_P25E5_vanillaCTDE`**: Plan 25, execute 5 with only conditioning on own initial and final position; model: _P25E5_vanillaCTDE.pt

## Main Files
- **`training_mpc_guidance.py`**: Training and sampling of MPC approach (10 timestep planning, 1 timestep execution) with guidance function in denoising process to avoid collision (old)
- **`training_mpc_guidesample.py`**: Training and sampling of MPC approach with guidance function during sampling for collision avoidance
- **`training_mpc_lf_latent.py`**: Training and sampling of mpc approach(varying timestep planning, varying timestep execution) while conditioned as a leader/follower approach on the latent representation of the leader's history
- **`training_mpc_lf_restore.py`**: Training and sampling of mpc approach(varying timestep planning, varying timestep execution) while conditioned as a leader/follower approach (basically the same as training_mpc_lf.py but just haven't deleted that file yet)
- **`training_mpc_lf.py`**: Training and sampling of mpc approach(varying timestep planning, varying timestep execution) while conditioned as a leader/follower approach
- **`training_mpc.py`**: Training and sampling of MPC approach (10 timestep planning, 1 timestep execution)
- **`training_reactive.py`**: Training and sampling of splice approach(10 timestep planning, 10 timestep execution) while conditioned on additional elements (other agent's state and/or expert demonstration mode)
- **`training_splice.py`**: Training and sampling of splice approach (10 timestep planning, 10 timestep execution)
- **`training.py`**: Training and sampling of full horizon planning (original)
- **`splice_baseline.py`**: Sampling of splice approach to generate a full 100 timestep trajectory
- **`splice_reactive_mode.py`**: Sampling of splice approach with additional conditioning on other agent's state and expert demonstration mode to generate a full 100 timestep trajectory
- **`splice_reactive.py`**: Sampling of splice approach with additional conditioning on other agent's state to generate a full 100 timestep trajectory