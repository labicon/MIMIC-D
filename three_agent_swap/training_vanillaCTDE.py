import torch
import numpy as np
from utils.conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
from utils.discrete import *
import sys
import pdb
import csv
from utils.mpc_util import reactive_mpc_plan_vanillaCTDE

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
batch_size = 64
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
initial_point_1 = np.array([0.0, 2.0])
final_point_1 = np.array([2.0, 0.0])
initial_point_2 = np.array([0.75, -2.0])
final_point_2 = np.array([0.75, 2.0])
initial_point_3 = np.array([-0.25, 0.75])
final_point_3 = np.array([1.75, 0.75])

# Loading training trajectories
expert_data_1 = np.load('data/expert_data1_400_traj_06_noise.npy')
expert_data_2 = np.load('data/expert_data2_400_traj_06_noise.npy')
expert_data_3 = np.load('data/expert_data3_400_traj_06_noise.npy')

orig1 = expert_data_1
orig2 = expert_data_2
orig3 = expert_data_3
print(expert_data_1.shape)
print(expert_data_2.shape)
print(expert_data_3.shape)

orig1 = np.array(orig1)
orig2 = np.array(orig2)
orig3 = np.array(orig3)
print(orig1.shape)
print(orig2.shape)
print(orig3.shape)

expert_data1 = create_mpc_dataset(expert_data_1, planning_horizon=H)
expert_data2 = create_mpc_dataset(expert_data_2, planning_horizon=H)
expert_data3 = create_mpc_dataset(expert_data_3, planning_horizon=H)
print(expert_data1.shape)
print(expert_data2.shape)
print(expert_data3.shape)


combined_data = np.concatenate((expert_data1, expert_data2, expert_data3), axis=0)
# mean = np.mean(combined_data, axis=(0,1))
# std = np.std(combined_data, axis=(0,1))
# np.save("data/mean_400demos_06noise.npy", mean)
# np.save("data/std_400demos_06noise.npy", std)
mean = np.load("data/mean_400demos_06noise.npy")
std = np.load("data/std_400demos_06noise.npy")
expert_data1 = (expert_data1 - mean) / std
expert_data2 = (expert_data2 - mean) / std
expert_data3 = (expert_data3 - mean) / std
orig1 = (orig1 - mean) / std
orig2 = (orig2 - mean) / std
orig3 = (orig3 - mean) / std

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
obs_init3 = expert_data3[:, 0, :]
obs_final1 = np.repeat(orig1[:, -1, :], repeats=100, axis=0)
obs_final2 = np.repeat(orig2[:, -1, :], repeats=100, axis=0)
obs_final3 = np.repeat(orig3[:, -1, :], repeats=100, axis=0)
obs1 = np.hstack([obs_init1, obs_final1])
obs2 = np.hstack([obs_init2, obs_final2])
obs3 = np.hstack([obs_init3, obs_final3])
obs_temp1 = obs1
obs_temp2 = obs2
obs_temp3 = obs3
actions1 = expert_data1[:, :H-1, :]
actions2 = expert_data2[:, :H-1, :]
actions3 = expert_data3[:, :H-1, :]
obs1 = torch.FloatTensor(obs1).to(device)
obs2 = torch.FloatTensor(obs2).to(device)
obs3 = torch.FloatTensor(obs3).to(device)

attr1 = obs1
attr2 = obs2
attr3 = obs3
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]
attr_dim3 = attr3.shape[1]

actions1 = torch.FloatTensor(actions1).to(device)
actions2 = torch.FloatTensor(actions2).to(device)
actions3 = torch.FloatTensor(actions3).to(device)
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()
sigma_data3 = actions3.std().item()
sig = np.array([sigma_data1, sigma_data2, sigma_data3])

# Training
end = "_P25E1_vanillaCTDE"
action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2, attr_dim3], [sigma_data1, sigma_data2, sigma_data3], device=device, N=200, n_models = 3, **model_size)
# action_cond_ode.train([actions1, actions2, actions3], [attr1, attr2, attr3], int(5*n_gradient_steps), batch_size, extra=end)
# action_cond_ode.save(extra=end)
action_cond_ode.load(extra=end)

# Sampling
try:
    init1_list = np.load("init_final_pos/init1_list.npy")
    init2_list = np.load("init_final_pos/init2_list.npy")
    init3_list = np.load("init_final_pos/init3_list.npy")
    final1_list = np.load("init_final_pos/final1_list.npy")
    final2_list = np.load("init_final_pos/final2_list.npy")
    final3_list = np.load("init_final_pos/final3_list.npy")
except FileNotFoundError:
    init1_list = None

for i in range(100):
    print("Planning Sample %s" % i)
    noise_std = 0.6
    threshold = 0.75

    while True:
        if init1_list is not None:
            initial1 = init1_list[i]
            final1 = final1_list[i]
            initial2 = init2_list[i]
            final2 = final2_list[i]
            initial3 = init3_list[i]
            final3 = final3_list[i]
            break
        else:
            initial1 = initial_point_1 + np.random.uniform(-noise_std, noise_std, size=(2,))    
            final1 = final_point_1 + np.random.uniform(-noise_std, noise_std, size=(2,))
            initial2 = initial_point_2 + np.random.uniform(-noise_std, noise_std, size=(2,))
            final2 = final_point_2 + np.random.uniform(-noise_std, noise_std, size=(2,))
            initial3 = initial_point_3 + np.random.uniform(-noise_std, noise_std, size=(2,))
            final3 = final_point_3 + np.random.uniform(-noise_std, noise_std, size=(2,))

            d_init12 = np.linalg.norm(initial1 - initial2)
            d_init13 = np.linalg.norm(initial1 - initial3)
            d_init23 = np.linalg.norm(initial2 - initial3)
            d_fin12 = np.linalg.norm(final1 - final2)
            d_fin13 = np.linalg.norm(final1 - final3)
            d_fin23 = np.linalg.norm(final2 - final3)

            if (d_init12 > threshold and d_init13 > threshold and d_init23 > threshold and
                d_fin12  > threshold and d_fin13  > threshold and d_fin23  > threshold):
                break

    initial1 = (initial1 - mean) / std
    final1 = (final1 - mean) / std
    initial2 = (initial2 - mean) / std
    final2 = (final2 - mean) / std
    initial3 = (initial3 - mean) / std
    final3 = (final3 - mean) / std

    planned_trajs = reactive_mpc_plan_vanillaCTDE(action_cond_ode, [initial1, initial2, initial3], [final1, final2, final3], segment_length=H, total_steps=T, n_implement=1)

    planned_traj1 =  planned_trajs[0] * std + mean
    np.save("sampled_trajs/mpc_P25E1_vanillaCTDE/traj1_%s.npy" % i, planned_traj1)

    planned_traj2 = planned_trajs[1] * std + mean
    np.save("sampled_trajs/mpc_P25E1_vanillaCTDE/traj2_%s.npy" % i, planned_traj2)

    planned_traj3 = planned_trajs[2] * std + mean
    np.save("sampled_trajs/mpc_P25E1_vanillaCTDE/traj3_%s.npy" % i, planned_traj3)