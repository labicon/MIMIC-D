# This script trains the Conditional ODE model for the Two Arm Lift task,
# using separate image latent vectors for each arm's on-board camera.

import torch
import numpy as np
from conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
from discrete import *
import sys
import pdb

def create_mpc_dataset(expert_data, planning_horizon=25):
    n_traj, horizon, state_dim = expert_data.shape
    n_subtraj = horizon  # we'll create one sub-trajectory starting at each time step

    # Resulting array shape: (n_traj * n_subtraj, planning_horizon, state_dim)
    result = []

    for traj in expert_data:
        for start_idx in range(n_subtraj):
            # If not enough steps, pad with the last step
            end_idx = start_idx + planning_horizon
            if end_idx <= horizon:
                sub_traj = traj[start_idx:end_idx]
            else:
                # Need padding
                sub_traj = traj[start_idx:]
                padding = np.repeat(traj[-1][np.newaxis, :], end_idx - horizon, axis=0)
                sub_traj = np.concatenate([sub_traj, padding], axis=0)
            result.append(sub_traj)

    result = np.stack(result, axis=0)
    return result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Parameters
n_gradient_steps = 50_000
batch_size = 32
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 25 # horizon, length of each trajectory
T = 700 # total time steps

# Load expert data
expert_data = np.load("data/expert_actions_newslower_20.npy")
expert_data1 = expert_data[:, :, :7]
expert_data2 = expert_data[:, :, 7:14]

# Load and process image data for each arm
# Assuming the image latents are 128-dimensional for each arm
expert_images_latents_arm1 = np.load("data/arm1_images_latents.npy")
expert_images_latents_arm2 = np.load("data/arm2_images_latents.npy")
print(f"Loaded arm 1 image latents with shape: {expert_images_latents_arm1.shape}")
print(f"Loaded arm 2 image latents with shape: {expert_images_latents_arm2.shape}")

# Create MPC datasets for actions and images
expert_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)
expert_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)
expert_images_latents_arm1 = create_mpc_dataset(expert_images_latents_arm1, planning_horizon=H)
expert_images_latents_arm2 = create_mpc_dataset(expert_images_latents_arm2, planning_horizon=H)

# Compute mean and standard deviation
combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
mean = np.mean(combined_data, axis=(0,1))
std = np.std(combined_data, axis=(0,1))
mean_arm1 = np.mean(expert_data1, axis=(0,1))
std_arm1 = np.std(expert_data1, axis=(0,1))
mean_arm2 = np.mean(expert_data2, axis=(0,1))
std_arm2 = np.std(expert_data2, axis=(0,1))

# Normalize data
expert_data1 = (expert_data1 - mean_arm1) / std_arm1
expert_data2 = (expert_data2 - mean_arm2) / std_arm2

# Define an enviornment objcet which has attrubutess like name, state_size, action_size etc
class TwoArmLift():
    def __init__(self, state_size=7, action_size=7):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoArmLift"
env = TwoArmLift()

# Preparing expert data for training
actions1 = expert_data1[:, :H, :]
actions2 = expert_data2[:, :H, :]
actions1 = torch.FloatTensor(actions1).to(device)
actions2 = torch.FloatTensor(actions2).to(device)
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()

# Prepare conditional vectors with separate image information
with open("data/pot_start_newslower_20.npy", "rb") as f:
    obs = np.load(f)
obs_init1 = expert_data1[:, 0, :]
obs_init2 = expert_data2[:, 0, :]
obs = np.repeat(obs, repeats=T, axis=0)

# Use the initial image latent for each sub-trajectory
image_latents_initial_arm1 = expert_images_latents_arm1[:, 0, :]
image_latents_initial_arm2 = expert_images_latents_arm2[:, 0, :]

# Stack all conditions together
# Arm 1 condition: [initial state of arm 1, initial pot grasp, initial image latent of arm 1]
attr1 = np.hstack([obs_init1, obs_init2, obs, image_latents_initial_arm1])
# Arm 2 condition: [initial state of arm 2, initial state of arm 1, initial pot grasp, initial image latent of arm 2]
attr2 = np.hstack([obs_init2, obs_init1, obs, image_latents_initial_arm2])

attr1 = torch.FloatTensor(attr1).to(device)
attr2 = torch.FloatTensor(attr2).to(device)
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]

# Training
end="_lift_mpc_P25E1_crosscond_nofinalpos_rotvec_separatenorm_dual_camera"
action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
action_cond_ode.train([actions1, actions2], [attr1, attr2], int(5*n_gradient_steps), batch_size, extra=end, endpoint_loss=False)
action_cond_ode.save(extra=end)
action_cond_ode.load(extra=end)