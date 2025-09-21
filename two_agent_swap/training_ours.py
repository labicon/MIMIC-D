import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from utils.discrete import *
import sys
import pdb
import csv
from utils.conditional_Action_DiT import Conditional_ODE, count_parameters
seed = 40
np.random.seed(seed)
torch.manual_seed(seed)

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
# model_size = {
#     "d_model": 512,      # twice the transformer width
#     "n_heads": 8,        # more attention heads
#     "depth":   6,        # twice the number of layers
#     "lin_scale": 256,    # larger conditional embedder
# }
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 25 # horizon, length of each trajectory
T = 100 # total time steps

# Define initial and final points, and a single central obstacle
initial_point_up = np.array([0.0, 0.0])
final_point_up = np.array([20.0, 0.0])
final_point_down = np.array([0.0, 0.0])
initial_point_down = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0) 

# Loading training trajectories
expert_data_1 = np.load('data/expert_data1_100_traj.npy')
expert_data_2 = np.load('data/expert_data2_100_traj.npy')
expert_data1 = create_mpc_dataset(expert_data_1, planning_horizon=H)
expert_data2 = create_mpc_dataset(expert_data_2, planning_horizon=H)

combined_data1 = np.concatenate((expert_data1, expert_data2), axis=0)
mean1 = np.mean(expert_data1, axis=(0,1))
std1 = np.std(expert_data1, axis=(0,1))
mean2 = np.mean(expert_data2, axis=(0,1))
std2 = np.std(expert_data2, axis=(0,1))
expert_data1 = (expert_data1 - mean1) / std1
expert_data2 = (expert_data2 - mean2) / std2

# Define environment
class TwoUnicycle():
    def __init__(self, state_size=2, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoUnicycle"
env = TwoUnicycle()

# Setting up training data
obs_init1 = expert_data1[:, 0, :]
obs_init2 = expert_data2[:, 0, :]
obs1 = np.hstack([obs_init1, obs_init2])
obs2 = np.hstack([obs_init2, obs_init1])
obs_temp1 = obs1
obs_temp2 = obs2
actions1 = expert_data1[:, :H-1, :]
actions2 = expert_data2[:, :H-1, :]
obs1 = torch.FloatTensor(obs1).to(device)
obs2 = torch.FloatTensor(obs2).to(device)

attr1 = obs1
attr2 = obs2
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]

actions1 = torch.FloatTensor(actions1).to(device)
actions2 = torch.FloatTensor(actions2).to(device)
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()
sig = np.array([sigma_data1, sigma_data2])

# Training
end = "_P25E1_nolf_nofinalpos_matchtrain_50k"
action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=10, n_models = 2, **model_size)
diff_pair_params = sum(count_parameters(F) for F in action_cond_ode.F_list)
print(f"Diffusion pair params: {diff_pair_params:,}")
# action_cond_ode.train([actions1, actions2], [attr1, attr2], int(n_gradient_steps), batch_size, extra=end, subdirect="mpc/")
# action_cond_ode.save(subdirect="mpc/", extra=end)
action_cond_ode.load(subdirect="mpc/", extra=end)

# Sampling

def reactive_mpc_plan(
        ode_model,
        initial_states,
        segment_length=25,
        total_steps=100,
        n_implement=5):
    """
    Plans a full trajectory (total_steps long) by iteratively planning
    segment_length-steps using the diffusion model and replanning at every timestep,
    but ensures each agent conditions on its peers at the same timestep.
    
    Parameters:
      - ode_model: the Conditional_ODE instance.
      - initial_states: list or array of shape (n_agents, state_size).
      - fixed_goals:    list or array of shape (n_agents, state_size).
      - segment_length: number of timesteps to plan in each segment.
      - total_steps:    total length of the planned trajectory.
      - n_implement:    number of steps to execute before replanning each segment.
    
    Returns:
      - full_traj: np.ndarray of shape (n_agents, total_steps, action_size)
    """
    full_traj = []
    current_states = initial_states.copy()      # shape: (n_agents, state_size)
    n_agents = len(current_states)

    for seg in range(total_steps // n_implement):
        # snapshot everyone's state at the start of this segment
        base_states = current_states.copy()     

        segments = []
        for i in range(n_agents):
            # build conditioning vector from base_states
            cond = [ base_states[i]]
            for j in range(n_agents):
                if j != i:
                    cond.append(base_states[j])
            cond = np.hstack(cond)  # shape: (attr_dim,)
            cond_tensor = torch.tensor(cond, dtype=torch.float32,
                                       device=ode_model.device).unsqueeze(0)

            # sample this agent’s segment
            sampled = ode_model.sample(
                attr=cond_tensor,
                traj_len=segment_length,
                n_samples=1,
                w=1.0,
                model_index=i
            )
            seg_i = sampled.cpu().detach().numpy()[0]  # (segment_length, action_size)

            # select which slice to execute and update current_states
            if seg == 0:
                to_take   = seg_i[0:n_implement]
                new_state = seg_i[n_implement-1]
            else:
                to_take   = seg_i[1:n_implement+1]
                new_state = seg_i[n_implement]
            segments.append(to_take)
            current_states[i] = new_state

        # stack across agents: shape (n_agents, n_implement, action_size)
        full_traj.append(np.stack(segments, axis=0))

    # concat all segments along time: shape (n_agents, total_steps, action_size)
    full_traj = np.concatenate(full_traj, axis=1)
    return full_traj

for s in range(10):
    seed = s * 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    path = f"sampled_trajs/ours_seed{seed}"
    os.makedirs(path, exist_ok=True)
    for i in range(100):
        print("Planning Sample %s" % i)
        noise_std = 0.4
        initial1 = initial_point_up + noise_std * np.random.randn(*np.shape(initial_point_up))
        initial1 = (initial1 - mean1) / std1
        initial2 = initial_point_down + noise_std * np.random.randn(*np.shape(initial_point_down))
        initial2 = (initial2 - mean2) / std2

        planned_trajs = reactive_mpc_plan(action_cond_ode, [initial1, initial2], segment_length=H, total_steps=T, n_implement=1)

        planned_traj1 =  planned_trajs[0] * std1 + mean1
        np.save(os.path.join(path, f"mpc_traj1_{i}.npy"), planned_traj1)

        planned_traj2 = planned_trajs[1] * std2 + mean2
        np.save(os.path.join(path, f"mpc_traj2_{i}.npy"), planned_traj2)