# This script is used to train the Conditional ODE model for the Two Arm Lift task.
# It uses the 3-dimensional rotation vector of the arm's state and action.
# The model is conditioned on the initial grasp position of the two pot handles.

import torch
import numpy as np
from conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
from discrete import *
import sys
import pdb
import random
random.seed(0)
np.random.seed(0)
torch.manual_seed
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
n_gradient_steps = 100_000
batch_size = 32
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
# model_size = {
#     "d_model": 512,      # twice the transformer width
#     "n_heads": 8,        # more attention heads
#     "depth":   6,        # twice the number of layers
# }
H = 200 # horizon, length of each trajectory
T = 1100 # total time steps

# Load expert data
expert_data = np.load("data/expert_actions_rotmat_sparse_1100.npy")
expert_data1_temp = expert_data[:, :, :10]
expert_data2_temp = expert_data[:, :, 10:20]
expert_data1 = create_mpc_dataset(expert_data1_temp, planning_horizon=H)
expert_data2 = create_mpc_dataset(expert_data2_temp, planning_horizon=H)
# breakpoint()

# Compute mean and standard deviation
combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
# breakpoint()
try:
    kinova_mean = np.load("data/mean_kinova_rotmat_1100.npy")
    kinova_std = np.load("data/std_kinova_rotmat_1100.npy")
    xarm_mean = np.load("data/mean_xarm_rotmat_1100.npy")
    xarm_std = np.load("data/std_xarm_rotmat_1100.npy")
except FileNotFoundError:
    kinova_mean = np.mean(expert_data1, axis=(0, 1))
    kinova_std = np.std(expert_data1, axis=(0, 1))
    xarm_mean = np.mean(expert_data2, axis=(0, 1))
    xarm_std = np.std(expert_data2, axis=(0, 1))
    np.save("data/mean_kinova_rotmat_1100.npy", kinova_mean)
    np.save("data/std_kinova_rotmat_1100.npy", kinova_std)
    np.save("data/mean_xarm_rotmat_1100.npy", xarm_mean)
    np.save("data/std_xarm_rotmat_1100.npy", xarm_std)

# Normalize data
expert_data1_temp = (expert_data1_temp - kinova_mean) / kinova_std
expert_data2_temp = (expert_data2_temp - xarm_mean) / xarm_std
expert_data1 = (expert_data1 - kinova_mean) / kinova_std
expert_data2 = (expert_data2 - xarm_mean) / xarm_std

# Define an enviornment objcet which has attrubutess like name, state_size, action_size etc
class TwoArmLift():
    def __init__(self, state_size=10, action_size=10):
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

# Prepare conditional vectors for training
obs_init1 = expert_data1[:, 0, :]
obs_init2 = expert_data2[:, 0, :]
obs1 = np.hstack([obs_init1, obs_init2])
obs2 = np.hstack([obs_init2, obs_init1])
obs1 = torch.FloatTensor(obs1).to(device)
obs2 = torch.FloatTensor(obs2).to(device)
attr1 = obs1
attr2 = obs2
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]

# Training
end = "_lift_mpc_P200E1_1100T_fullstate_nofinalpos_nopot_rotmat_fixkinova_separatenorm"
action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=10, n_models = 2, **model_size)
# action_cond_ode.train([actions1, actions2], [attr1, attr2], int(5*n_gradient_steps), batch_size, extra=end, endpoint_loss=False)
# action_cond_ode.save(extra=end)
action_cond_ode.load(extra=end)

# Sampling
def reactive_mpc_plan(ode_model, initial_states, segment_length=100, total_steps=2000, n_implement=1):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep.
    
    Parameters:
    - ode_model: the Conditional_ODE (diffusion model) instance.
    - env: your environment, which must implement reset_to() and step().
    - initial_states: a numpy array of shape (n_agents, state_size) that represent the starting states for the robots.
    - obs: a numpy array of shape (6,) representing the eef positions of the two pot handles.
    - total_steps: total length of the planned trajectory.
    - n_implement: number of steps to implement at each iteration.
    
    Returns:
    - full_traj: a numpy array of shape (n_agents, total_steps, state_size)
    """
    full_traj = []
    # initial_states = [np.array([-0.83898655,  1.39477865,  0.24218132,
    #                             2.08176365,  0.7110728 , -0.95662697, -0.59402534]), np.array([-0.37961092,  1.40740813, -2.30580383, -0.8449774 , -0.28674541,
    #                             1.43522927,  1.72807314])]
    current_states = initial_states.copy()      # shape: (n_agents, state_size)
    n_agents = len(current_states)
    # breakpoint()

    for seg in range(total_steps // n_implement):

        base_states = current_states.copy()     
        segments = []

        for i in range(n_agents):
            cond = [base_states[i]]  # start with the current state and the final state for this agent
            for j in range(n_agents):
                if j != i:
                    cond.append(base_states[j])
            # cond.append(obs)
            cond = np.hstack(cond)
    #         cond = np.array([-0.83896989,  1.39478993,  0.12084441,  2.08174162,  0.71107765,
    #    -0.95662494, -0.59402534, -0.37961092,  1.40740813, -2.30580383,
    #    -0.8449774 , -0.28674541,  1.43522927,  1.72807314,  0.47816622,
    #    -0.14264476,  0.09862956,  0.20981865,  0.0048536 ,  0.02984436])
            cond_tensor = torch.tensor(cond, dtype=torch.float32, device=ode_model.device).unsqueeze(0)

            sampled = ode_model.sample(
                attr=cond_tensor,
                traj_len=segment_length,
                n_samples=1,
                w=1.0,
                model_index=i
            )
            seg_i = sampled.cpu().numpy()[0]  # (segment_length, action_size)

            if seg == 0:
                take = seg_i[0:n_implement, :]
                new_state = seg_i[n_implement-1, :]
            else:
                take = seg_i[1:n_implement+1, :]
                new_state = seg_i[n_implement, :]
            segments.append(take)
            current_states[i] = new_state

        full_traj.append(np.stack(segments, axis=0))  # (n_agents, n_implement, action_size)

    # concat all segments along the time dimension
    full_traj = np.concatenate(full_traj, axis=1)     # (n_agents, total_steps, action_size)
    return full_traj

for i in range(10):
    seed = i * 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for i in range(16):
        cond_idx = i
        # breakpoint()
        planned_trajs = reactive_mpc_plan(action_cond_ode, [expert_data1_temp[cond_idx][0], expert_data2_temp[cond_idx][0]], segment_length=H, total_steps=T, n_implement=10)
        planned_traj1 =  planned_trajs[0] * kinova_std + kinova_mean
        np.save("samples/testseed/planned_traj1_seed%s_%s_new.npy" % (seed, cond_idx), planned_traj1)
        planned_traj2 = planned_trajs[1] * xarm_std + xarm_mean
        np.save("samples/testseed/planned_traj2_seed%s_%s_new.npy" % (seed, cond_idx), planned_traj2)