# This script is used to train the Conditional ODE model for the Two Arm Lift task.
# It uses the 3-dimensional rotation vector of the arm's state and action.
# The model is conditioned on the initial grasp position of the two pot handles.

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.discrete import *
import sys
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Behavioral Cloning Network
class ImitationNet(nn.Module):
    def __init__(self, input_size, hidden_size=256, output_size=7):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)
   
class MLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1, expansion=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * expansion, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.fc2(h)
        h = self.drop(h)
        return x + h

class BigImitationNet(nn.Module):
    def __init__(self, input_size, hidden_size=1024, output_size=7, horizon=10,
                 num_layers=8, dropout=0.1, expansion=4):
        super().__init__()
        self.horizon = horizon

        self.input = nn.Linear(input_size, hidden_size)

        # Learnable horizon/position embeddings (0..horizon-1)
        self.step_embed = nn.Embedding(horizon, hidden_size)

        self.blocks = nn.ModuleList([
            MLPBlock(hidden_size, dropout=dropout, expansion=expansion)
            for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # x: (batch_size, input_size)
        B = x.size(0)
        device = x.device

        # Encode input -> (B, hidden)
        h = self.input(x)  # (B, hidden_size)

        # Expand across horizon and add learnable step embeddings
        h = h.unsqueeze(1).expand(B, self.horizon, -1)              # (B, H, hidden)
        steps = torch.arange(self.horizon, device=device)           # (H,)
        h = h + self.step_embed(steps).unsqueeze(0)                 # (B, H, hidden)

        # Per-step MLP blocks
        for blk in self.blocks:
            h = blk(h)                                              # (B, H, hidden)

        # Project to outputs per step
        out = self.head(h)                                          # (B, H, output_size)
        return out

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

# Parameters
n_gradient_steps = 100_000
batch_size = 32
model_size = {"d_model": 256, "n_heads": 4, "depth": 3}
H = 25 # horizon, length of each trajectory
T = 950 # total time steps

# Load expert data
expert_data = np.load("data/expert_actions_newslowerer_100.npy")
expert_data1 = expert_data[:, :, :7]
expert_data2 = expert_data[:, :, 7:14]
expert_data1 = create_mpc_dataset(expert_data1, planning_horizon=H)
expert_data2 = create_mpc_dataset(expert_data2, planning_horizon=H)

# Compute mean and standard deviation
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

with open("data/pot_start_newslowerer_100.npy", "rb") as f:
    pot = np.load(f)

pot_mean = np.mean(pot, axis=0)
pot_std = np.std(pot, axis=0)
pot = (pot - pot_mean) / pot_std
pot_dim = pot.shape[1]

obs = np.repeat(pot, repeats=T, axis=0)

obs_init1 = expert_data1[:, 0, :]
obs_init2 = expert_data2[:, 0, :]
obs1 = np.hstack([obs_init1, obs_init2, obs])
obs2 = np.hstack([obs_init2, obs_init1, obs])
obs1 = torch.FloatTensor(obs1).to(device)
obs2 = torch.FloatTensor(obs2).to(device)
attr1 = obs1
attr2 = obs2
attr_dim1 = attr1.shape[1]
attr_dim2 = attr2.shape[1]

# # Build BC training data
# def build_bc_data(expert_data, pot):
#     X, Y = [], []
#     for i in range(len(expert_data)):
#         traj = expert_data[i]
#         pot_start = pot[i]
#         for t in range(len(traj) - 1):
#             current = traj[t]
#             X.append(np.hstack([current, pot_start]))  # [x_t, x_0]
#             Y.append(traj[t + 1])                 # x_{t+1}
#     return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

# X1, Y1 = build_bc_data(expert_data1, pot)
# X2, Y2 = build_bc_data(expert_data2, pot)
# X1, Y1 = torch.from_numpy(X1).to(device), torch.from_numpy(Y1).to(device)
# X2, Y2 = torch.from_numpy(X2).to(device), torch.from_numpy(Y2).to(device)

input_size = attr1.shape[1]
# model1 = ImitationNet(input_size, hidden_size=256, output_size=7).to(device)
# model2 = ImitationNet(input_size, hidden_size=256, output_size=7).to(device)

model1 = BigImitationNet(input_size, hidden_size=256, output_size=7, horizon=actions1.shape[1], num_layers=8, dropout=0.1).to(device)
model2 = BigImitationNet(input_size, hidden_size=256, output_size=7, horizon=actions1.shape[1], num_layers=8, dropout=0.1).to(device)

# print the number of parameter for model1 and model 2
print(f"Total parameters: {sum(p.numel() for p in model1.parameters()) + sum(p.numel() for p in model2.parameters())}")

# import ipdb; ipdb.set_trace()

params = list(model1.parameters()) + list(model2.parameters())
optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

# Joint training for the agents
def joint_train(models, optimizer, criterion, datasets, num_epochs=5000):
    actions, attrs = zip(*datasets)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = 0.0
        for model, action, attr in zip(models, actions, attrs):
            # randomly sample batch_size indices
            idx = np.random.randint(0, action.shape[0], batch_size)
            true_actions = action[idx]
            input_attrs = attr[idx]
            pred_action = model(input_attrs)
            loss += criterion(pred_action, true_actions)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 5000 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    return models

model1, model2 = joint_train(
    [model1, model2], optimizer, criterion,
    [(actions1, attr1), (actions2, attr2)], num_epochs=35000
)

# Save trained models
save_path1 = "trained_models/bc/model1_big_newslowerer_100traj.pth"
save_path2 = "trained_models/bc/model2_big_newslowerer_100traj.pth"

torch.save(model1.state_dict(), save_path1)
torch.save(model2.state_dict(), save_path2)