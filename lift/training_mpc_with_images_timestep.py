# This script trains the Conditional ODE model for the Two Arm Lift task,
# using separate image latent vectors for each arm's on-board camera.

import torch
import numpy as np
from conditional_Action_DiT2 import Conditional_ODE
import matplotlib.pyplot as plt
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
n_gradient_steps = 1_000
batch_size = 32
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 25 # horizon, length of each trajectory
T = 700 # total time steps

# Load expert data
expert_data = np.load("data/models/VAE_models_ICON/expert_actions_newslower_20.npy")
expert_data1 = expert_data[:, :, :7]
expert_data2 = expert_data[:, :, 7:14]

# Load and process image data for each arm
# Assuming the image latents are 128-dimensional for each arm
expert_images_latents_arm1 = np.load("data/models/VAE_models_ICON/arm1_images_latents.npy")
expert_images_latents_arm2 = np.load("data/models/VAE_models_ICON/arm2_images_latents.npy")
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
with open("data/models/VAE_models_ICON/pot_start_newslower_20.npy", "rb") as f:
    obs = np.load(f)
obs_init1 = expert_data1[:, 0, :]
obs_init2 = expert_data2[:, 0, :]
obs = np.repeat(obs, repeats=T, axis=0)

# Use the initial image latent for each sub-trajectory
# ---------------- B1 per-timestep conditioning ---------------- #
# Combine per-timestep states and image latents: [state_t, image_latent_t]

# If you have saved per-timestep states, load here:
state_file = "data/models/VAE_models_ICON/expert_states_newslower_20.npy"
try:
    state_seq_all = np.load(state_file)  # shape: (N, H, state_dim) or (N, T, state_dim)
    # If longer than H, create MPC windows
    if state_seq_all.shape[1] > H:
        state_seq = create_mpc_dataset(state_seq_all, planning_horizon=H)
    else:
        state_seq = state_seq_all[:, :H, :]
    print(f"[info] Loaded state sequence from {state_file}, shape {state_seq.shape}")
except Exception as e:
    print(f"[warning] Could not load state sequence: {e}. Falling back to expert_data as proxy")
    state_seq_arm1 = expert_data1.copy()
    state_seq_arm2 = expert_data2.copy()
else:
    # split into arms if needed (common case: 14-dim total state)
    if state_seq.shape[2] == 14:
        state_seq_arm1 = state_seq[:, :, :7]
        state_seq_arm2 = state_seq[:, :, 7:14]
    else:
        state_seq_arm1 = state_seq.copy()
        state_seq_arm2 = state_seq.copy()

# If fallback used
if 'state_seq_arm1' not in locals() or state_seq_arm1 is None:
    state_seq_arm1 = expert_data1.copy()
    state_seq_arm2 = expert_data2.copy()

# Make sure sequences align with image latents
def ensure_len_H(arr, H):
    if arr.shape[1] > H:
        return arr[:, :H, :]
    elif arr.shape[1] < H:
        pad_len = H - arr.shape[1]
        pad_block = np.repeat(arr[:, -1:, :], pad_len, axis=1)
        return np.concatenate([arr, pad_block], axis=1)
    return arr

state_seq_arm1 = ensure_len_H(state_seq_arm1, H)
state_seq_arm2 = ensure_len_H(state_seq_arm2, H)
img_latents_arm1 = ensure_len_H(expert_images_latents_arm1, H)
img_latents_arm2 = ensure_len_H(expert_images_latents_arm2, H)

# Build per-timestep conditioning vectors
attr1_np = np.concatenate([state_seq_arm1, img_latents_arm1], axis=2)  # (N, H, state+latent)
attr2_np = np.concatenate([state_seq_arm2, img_latents_arm2], axis=2)

attr1 = torch.FloatTensor(attr1_np).to(device)
attr2 = torch.FloatTensor(attr2_np).to(device)
attr_dim1 = attr1.shape[2]
attr_dim2 = attr2.shape[2]

print(f"[info] attr1 shape: {attr1.shape}, attr2 shape: {attr2.shape}")
# --------------------------------------------------------------


# Training
end="_lift_mpc_P25E1_crosscond_nofinalpos_rotvec_separatenorm_dual_camera2"
action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
action_cond_ode.train([actions1, actions2], [attr1, attr2], int(5*n_gradient_steps), batch_size, extra=end, endpoint_loss=False)
action_cond_ode.save(extra=end)
action_cond_ode.load(extra=end)