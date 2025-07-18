import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class ImitationNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, output_size=2):
        super(ImitationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def create_mpc_dataset(expert_data, planning_horizon=25):
    n_traj, horizon, state_dim = expert_data.shape
    n_subtraj = horizon

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
n_epochs      = 500
learning_rate = 1e-3

# Define initial and final points, and a single central obstacle
initial_point_up = np.array([0.0, 0.0])
final_point_up = np.array([20.0, 0.0])
final_point_down = np.array([0.0, 0.0])
initial_point_down = np.array([20.0, 0.0])
obstacle = (10, 0, 4.0) 

# Loading training trajectories
expert_data1 = np.load('data/expert_data1_100_traj.npy')
expert_data2 = np.load('data/expert_data2_100_traj.npy')

combined_data1 = np.concatenate((expert_data1, expert_data2), axis=0)
mean = np.mean(combined_data1, axis=(0,1))
std = np.std(combined_data1, axis=(0,1))
expert_data1 = (expert_data1 - mean) / std
expert_data2 = (expert_data2 - mean) / std

# Define environment
class TwoUnicycle():
    def __init__(self, state_size=2, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.name = "TwoUnicycle"
env = TwoUnicycle()

# Prepare Data for Training
# Create input-output pairs (state + goal -> next state)
X_train1 = []
Y_train1 = []
X_train2 = []
Y_train2 = []

for traj in expert_data1:
    for i in range(len(traj) - 1):
        X_train1.append(np.hstack([traj[i], final_point_up]))  # Current state + goal
        Y_train1.append(traj[i + 1])  # Next state

for traj in expert_data2:
    for i in range(len(traj) - 1):
        X_train2.append(np.hstack([traj[i], final_point_down]))  # Current state + goal
        Y_train2.append(traj[i + 1])  # Next state

X_train1 = torch.tensor(np.array(X_train1), dtype=torch.float32).to(device)  # Shape: (N, 4)
Y_train1 = torch.tensor(np.array(Y_train1), dtype=torch.float32).to(device)  # Shape: (N, 2)
X_train2 = torch.tensor(np.array(X_train2), dtype=torch.float32).to(device)  # Shape: (N, 4)
Y_train2 = torch.tensor(np.array(Y_train2), dtype=torch.float32).to(device)  # Shape: (N, 2)

# Initialize Model, Loss Function, and Optimizers
model1 = ImitationNet(input_size=4, hidden_size=64, output_size=2).to(device)
model2 = ImitationNet(input_size=4, hidden_size=64, output_size=2).to(device)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate)

# Train the Model
losses1 = []
losses2 = []

for epoch in range(1, n_epochs+1):
    model1.train()
    model2.train()
    total1 = 0.0
    total2 = 0.0

    # Single pass for both models
    # (you could also interleave but the key is: no double backward)
    preds1 = model1(X_train1)
    loss1  = criterion(preds1, Y_train1)
    optimizer1.zero_grad()
    loss1.backward()
    optimizer1.step()

    preds2 = model2(X_train2)
    loss2  = criterion(preds2, Y_train2)
    optimizer2.zero_grad()
    loss2.backward()
    optimizer2.step()

    total1 += loss1.item()
    total2 += loss2.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss1: {total1:.4f} | Loss2: {total2:.4f}")


torch.save(model1.state_dict(), "trained_models/bc/bc_agent1.pth")
torch.save(model2.state_dict(), "trained_models/bc/bc_agent2.pth")

model1.eval()
model2.eval()

num_samples = 100
T = 100

for i in range(num_samples):
    noise_std = 0.4
    initial1 = initial_point_up + noise_std * np.random.randn(*np.shape(initial_point_up))
    initial1 = (initial1 - mean) / std
    final1 = final_point_up + noise_std * np.random.randn(*np.shape(final_point_up))
    final1 = (final1 - mean) / std
    initial2 = initial_point_down + noise_std * np.random.randn(*np.shape(initial_point_down))
    initial2 = (initial2 - mean) / std
    final2 = final_point_down + noise_std * np.random.randn(*np.shape(final_point_down))
    final2 = (final2 - mean) / std

    traj1 = [initial1.copy()]
    traj2 = [initial2.copy()]

    for t in range(T):

        inp1 = torch.tensor(np.hstack([initial1, final1]), dtype=torch.float32, device=device)
        inp2 = torch.tensor(np.hstack([initial2, final2]), dtype=torch.float32, device=device)

        with torch.no_grad():
            s1_next_norm = model1(inp1).cpu().numpy()
            s2_next_norm = model2(inp2).cpu().numpy()

        s1 = s1_next_norm * std + mean
        s2 = s2_next_norm * std + mean

        traj1.append(s1.copy())
        traj2.append(s2.copy())

    traj1 = np.stack(traj1, axis=0)
    traj2 = np.stack(traj2, axis=0)

    # save each rollout
    np.save(f"sampled_trajs/bc/traj1_{i:03d}.npy", traj1)
    np.save(f"sampled_trajs/bc/traj2_{i:03d}.npy", traj2)