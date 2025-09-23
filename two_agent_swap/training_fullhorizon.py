import torch
import numpy as np
from utils.conditional_Action_DiT import Conditional_ODE
import matplotlib.pyplot as plt
from utils.discrete import *
import sys
import csv
import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
n_gradient_steps = 100_000
batch_size = 64
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 100 # horizon, length of each trajectory

# Define initial and final points, and a single central obstacle
initial_point_up = np.array([0.0, 0.0])
final_point_up = np.array([20.0, 0.0])
final_point_down = np.array([0.0, 0.0])
initial_point_down = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)

# Loading the trajectories
all_points1 = []
all_points2 = []
with open('data/trajs_noise1.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        x1, y1 = float(row[4]), float(row[5])
        x2, y2 = float(row[7]), float(row[8])
        all_points1.append([x1, y1])
        all_points2.append([x2, y2])

num_trajectories = 1000
points_per_trajectory = 100

expert_data1 = [
    all_points1[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory1 = expert_data1[0]
x1 = [point[0] for point in first_trajectory1]
y1 = [point[1] for point in first_trajectory1]

expert_data2 = [
    all_points2[i * points_per_trajectory:(i + 1) * points_per_trajectory]
    for i in range(num_trajectories)
]
first_trajectory2 = expert_data2[0]
x2 = [point[0] for point in first_trajectory2]
y2 = [point[1] for point in first_trajectory2]


expert_data1 = np.array(expert_data1)
expert_data2 = np.array(expert_data2)


# Compute mean and standard deviation
combined_data = np.concatenate((expert_data1, expert_data2), axis=0)
mean = np.mean(combined_data, axis=(0,1))
std = np.std(combined_data, axis=(0,1))

# Normalize data
expert_data1 = (expert_data1 - mean) / std
expert_data2 = (expert_data2 - mean) / std

# Prepare Data for Training
X_train1 = []
Y_train1 = []

for traj in expert_data1:
    for i in range(len(traj) - 1):
        X_train1.append(np.hstack([traj[i], final_point_up]))  # Current state + goal
        Y_train1.append(traj[i + 1])  # Next state

X_train1 = torch.tensor(np.array(X_train1), dtype=torch.float32)  # Shape: (N, 4)
Y_train1 = torch.tensor(np.array(Y_train1), dtype=torch.float32)  # Shape: (N, 2)

X_train2 = []
Y_train2 = []

for traj in expert_data2:
    for i in range(len(traj) - 1):
        X_train2.append(np.hstack([traj[i], final_point_down]))  # Current state + goal
        Y_train2.append(traj[i + 1])  # Next state

X_train2 = torch.tensor(np.array(X_train2), dtype=torch.float32)  # Shape: (N, 4)
Y_train2 = torch.tensor(np.array(Y_train2), dtype=torch.float32)  # Shape: (N, 2)

# define an enviornment objcet which has attrubutess like name, state_size, action_size etc
class TwoUnicycle():
    def __init__(self, state_size=2, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoUnicycle"

env = TwoUnicycle()

# Setting up conditional vectors
obs_init1 = expert_data1[:, 0, :]
obs_init2 = expert_data2[:, 0, :]
obs_final1 = expert_data1[:, -1, :]
obs_final2 = expert_data2[:, -1, :]
obs1 = np.hstack([obs_init1, obs_final1])
obs2 = np.hstack([obs_init2, obs_final2])
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
assert attr_dim1 == env.state_size * 2
assert attr_dim2 == env.state_size * 2

actions1 = torch.FloatTensor(actions1).to(device)
actions2 = torch.FloatTensor(actions2).to(device)
sigma_data1 = actions1.std().item()
sigma_data2 = actions2.std().item()


# Training
action_cond_ode = Conditional_ODE(env, [attr_dim1, attr_dim2], [sigma_data1, sigma_data2], device=device, N=100, n_models = 2, **model_size)
# action_cond_ode.train([actions1, actions2], [attr1, attr2], int(5*n_gradient_steps), batch_size, extra="_T100")
# action_cond_ode.save(extra="_T100")
action_cond_ode.load(extra="_T100")

noise_std = 0.05
noise = np.ones(np.shape(obs_temp1))
obs_temp1 = obs_temp1 + noise_std * noise
obs_temp2 = obs_temp2 + noise_std * noise
obs_temp_tensor1 = torch.FloatTensor(obs_temp1).to(device)
obs_temp_tensor2 = torch.FloatTensor(obs_temp2).to(device)
attr_test1 = obs_temp_tensor1
attr_test2 = obs_temp_tensor2
expert_data1 = expert_data1 * std + mean
expert_data2 = expert_data2 * std + mean
ref1 = np.mean(expert_data1, axis=0)
ref2 = np.mean(expert_data2, axis=0)
ref_agent1 = ref1[:, :]
ref_agent2 = ref2[:, :]

# Sampling (no plotting)
for i in range(100):
    noise_std = 0.4
    initial1 = initial_point_up + noise_std * np.random.randn(*np.shape(initial_point_up))
    initial1 = (initial1 - mean) / std
    final1 = final_point_up + noise_std * np.random.randn(*np.shape(final_point_up))
    final1 = (final1 - mean) / std
    initial2 = initial_point_down + noise_std * np.random.randn(*np.shape(initial_point_down))
    initial2 = (initial2 - mean) / std
    final2 = final_point_down + noise_std * np.random.randn(*np.shape(final_point_down))
    final2 = (final2 - mean) / std

    cond1 = np.hstack([initial1, final1])
    cond2 = np.hstack([initial2, final2])
    cond_tensor1 = torch.tensor(cond1, dtype=torch.float32, device=device).unsqueeze(0)
    cond_tensor2 = torch.tensor(cond2, dtype=torch.float32, device=device).unsqueeze(0)

    traj_len = 100
    n_samples = 1

    sampled1 = action_cond_ode.sample(cond_tensor1, traj_len, n_samples, w=1., model_index = 0).cpu().detach().numpy()[0]
    sampled2 = action_cond_ode.sample(cond_tensor2, traj_len, n_samples, w=1., model_index = 1).cpu().detach().numpy()[0]
    sampled1 = sampled1 * std + mean
    sampled2 = sampled2 * std + mean

    np.save("data/T100/traj1_%s.npy" % i, sampled1)
    np.save("data/T100/traj2_%s.npy" % i, sampled2)

# # Sampling (with plotting)
# for i in range(10):
#     attr_t1 = attr_test1[i*10].unsqueeze(0)
#     attr_t2 = attr_test2[i*10].unsqueeze(0)
#     attr_n1 = attr_t1.cpu().detach().numpy()[0]
#     attr_n2 = attr_t2.cpu().detach().numpy()[0]

#     traj_len = 100
#     n_samples = 1

#     sampled1 = action_cond_ode.sample(attr_t1, traj_len, n_samples, w=1., model_index = 0)
#     sampled2 = action_cond_ode.sample(attr_t2, traj_len, n_samples, w=1., model_index = 1)

#     sampled1 = sampled1.cpu().detach().numpy()
#     sampled2 = sampled2.cpu().detach().numpy()
#     sampled1 = sampled1 * std + mean
#     sampled2 = sampled2 * std + mean
#     test1 = np.mean(sampled1, axis=0)
#     test2 = np.mean(sampled2, axis=0)
#     test_agent1 = test1[:, :]
#     test_agent2 = test2[:, :]

#     sys.setrecursionlimit(10000)
#     fast_frechet = FastDiscreteFrechetMatrix(euclidean)
#     frechet1 = fast_frechet.distance(ref_agent1,test_agent1)
#     frechet2 = fast_frechet.distance(ref_agent2,test_agent2)
#     print(frechet1, frechet2)

#     init_state1 = attr_n1[:2]
#     final_state1 = attr_n1[2:]
#     init_state2 = attr_n2[:2]
#     final_state2 = attr_n2[2:]

#     init_state1 = init_state1 * std + mean
#     final_state1 = final_state1 * std + mean
#     init_state2 = init_state2 * std + mean
#     final_state2 = final_state2 * std + mean

#     attr_n1 = np.concatenate([init_state1, final_state1])
#     attr_n2 = np.concatenate([init_state2, final_state2])

#     plt.figure(figsize=(20, 8))
#     plt.scatter(expert_data1[:, 0, 0], expert_data1[:, 0, 1], color='green')
#     plt.scatter(expert_data2[:, 0, 0], expert_data2[:, 0, 1], color='green')
#     plt.scatter(expert_data1[:, -1, 0], expert_data1[:, -1, 1], color='green')
#     plt.scatter(expert_data2[:, -1, 0], expert_data2[:, -1, 1], color='green')
#     plt.plot(attr_n1[0], attr_n1[1], 'bo')
#     plt.plot(attr_n2[0], attr_n2[1], 'o', color='orange')
#     plt.plot(attr_n1[2], attr_n1[3], 'bo')
#     plt.plot(attr_n2[2], attr_n2[3], 'o', color='orange')
#     plt.plot(sampled1[0, :, 0], sampled1[0, :, 1], color='blue', label=f"Agent 1 Traj (Frechet: {frechet1:.2f})")
#     plt.plot(sampled2[0, :, 0], sampled2[0, :, 1], color='orange', label=f"Agent 2 Traj (Frechet: {frechet2:.2f})")
#     plt.legend(loc="upper right", fontsize=14)
#     plt.savefig("figs/temp_T2/plot%s.png" % i)


