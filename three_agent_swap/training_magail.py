# training_magail_ctde_v2.py

import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 1) Device & performance tweak
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 2) Hyperparameters
state_dim   = 4       # e.g. [x, y, goal_x, goal_y]
action_dim  = 2       # e.g. next [x, y]
hidden_size = 64
batch_size  = 64
n_epochs    = 2000
lr_G        = 1e-4
lr_D        = 1e-4

# 3) Models
class GenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,   hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DiscNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(hidden_size,             hidden_size), nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1)).squeeze(1)

# Individual generators
G1 = GenNet().to(device)
G2 = GenNet().to(device)
G3 = GenNet().to(device)
# Shared discriminator
D  = DiscNet().to(device)

optG = optim.Adam(list(G1.parameters()) + list(G2.parameters()) + list(G3.parameters()), lr=lr_G)
optD = optim.Adam(D.parameters(), lr=lr_D)
bce_logits = nn.BCEWithLogitsLoss()

# 4) Load expert trajectories for normalization & conditioning
expert_data1 = np.load('data/expert_data1_400_traj_06_noise.npy')  # (n_traj, horizon, 2)
expert_data2 = np.load('data/expert_data2_400_traj_06_noise.npy')
expert_data3 = np.load('data/expert_data3_400_traj_06_noise.npy')

def build_sa(expert):
    S, A = [], []
    for traj in expert:
        goal = traj[-1]
        for t in range(len(traj)-1):
            s = np.hstack([traj[t], goal])  # (4,)
            a = traj[t+1]                   # (2,)
            S.append(s); A.append(a)
    return np.stack(S), np.stack(A)

S1, A1 = build_sa(expert_data1)
S2, A2 = build_sa(expert_data2)
S3, A3 = build_sa(expert_data3)

# 5) Joint normalization for states & actions
state_data  = np.concatenate([S1, S2, S3], axis=0)
action_data = np.concatenate([A1, A2, A3], axis=0)
state_mean, state_std   = state_data.mean(0),  state_data.std(0)  + 1e-6
action_mean, action_std = action_data.mean(0), action_data.std(0) + 1e-6
S1 = (S1 - state_mean) / state_std
S2 = (S2 - state_mean) / state_std
S3 = (S3 - state_mean) / state_std
A1 = (A1 - action_mean) / action_std
A2 = (A2 - action_mean) / action_std
A3 = (A3 - action_mean) / action_std

# 6) Torch datasets & loaders
tS1, tA1 = torch.tensor(S1, dtype=torch.float32), torch.tensor(A1, dtype=torch.float32)
tS2, tA2 = torch.tensor(S2, dtype=torch.float32), torch.tensor(A2, dtype=torch.float32)
tS3, tA3 = torch.tensor(S3, dtype=torch.float32), torch.tensor(A3, dtype=torch.float32)
loader1 = DataLoader(TensorDataset(tS1, tA1), batch_size=batch_size, shuffle=True, drop_last=True)
loader2 = DataLoader(TensorDataset(tS2, tA2), batch_size=batch_size, shuffle=True, drop_last=True)
loader3 = DataLoader(TensorDataset(tS3, tA3), batch_size=batch_size, shuffle=True, drop_last=True)

# 7) Training loop (centralized training structure)
for epoch in range(1, n_epochs + 1):
    G1.train(); G2.train(); G3.train(); D.train()
    lossD_sum = 0.0
    lossG_sum = 0.0

    for (x1, a1), (x2, a2), (x3, a3) in zip(loader1, loader2, loader3):
        x1, a1 = x1.to(device), a1.to(device)
        x2, a2 = x2.to(device), a2.to(device)
        x3, a3 = x3.to(device), a3.to(device)

        # —— Discriminator update across all agents ——
        with torch.no_grad():
            fake1 = G1(x1)
            fake2 = G2(x2)
            fake3 = G3(x3)
        real_logit1 = D(x1, a1)
        fake_logit1 = D(x1, fake1)
        real_logit2 = D(x2, a2)
        fake_logit2 = D(x2, fake2)
        real_logit3 = D(x3, a3)
        fake_logit3 = D(x3, fake3)
        lossD1 = 0.5*(bce_logits(real_logit1, torch.ones_like(real_logit1)) + bce_logits(fake_logit1, torch.zeros_like(fake_logit1)))
        lossD2 = 0.5*(bce_logits(real_logit2, torch.ones_like(real_logit2)) + bce_logits(fake_logit2, torch.zeros_like(fake_logit2)))
        lossD3 = 0.5*(bce_logits(real_logit3, torch.ones_like(real_logit3)) + bce_logits(fake_logit3, torch.zeros_like(fake_logit3)))
        lossD = lossD1 + lossD2 + lossD3
        optD.zero_grad()
        lossD.backward()
        optD.step()

        # —— Generator update for each agent ——
        fake1 = G1(x1)
        fake2 = G2(x2)
        fake3 = G3(x3)
        lossG1 = bce_logits(D(x1, fake1), torch.ones_like(fake1[:, 0]))
        lossG2 = bce_logits(D(x2, fake2), torch.ones_like(fake2[:, 0]))
        lossG3 = bce_logits(D(x3, fake3), torch.ones_like(fake3[:, 0]))
        lossG = lossG1 + lossG2 + lossG3
        optG.zero_grad()
        lossG.backward()
        optG.step()

        lossD_sum += lossD.item()
        lossG_sum += lossG.item()

    if epoch % 100 == 0:
        avgD = lossD_sum / len(loader1)
        avgG = lossG_sum / len(loader1)
        print(f"Epoch {epoch:4d} | lossD: {avgD:.4f} | lossG: {avgG:.4f}")

# 8) Save models
torch.save(G1.state_dict(), "trained_models/magail/G1.pth")
torch.save(G2.state_dict(), "trained_models/magail/G2.pth")
torch.save(G3.state_dict(), "trained_models/magail/G3.pth")
torch.save(D.state_dict(),  "trained_models/magail/D.pth")
print("CTDE MAGAIL training complete; models saved under trained_models/magail/")


# 9) Sampling
def sample_trajectory(G, init_pos, goal, horizon):
    pos = init_pos.copy()
    traj = [pos.copy()]
    for _ in range(horizon-1):
        s = np.hstack([pos, goal])
        s_norm = (s - state_mean) / state_std
        s_t = torch.tensor(s_norm, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a_norm = G(s_t).cpu().numpy()[0]
        a = a_norm * action_std + action_mean
        pos = a.copy()
        traj.append(pos.copy())
    return np.stack(traj)

num_samples = 100
T = expert_data1.shape[1]

plt.figure(figsize=(8, 8))
for i in range(num_samples):
    idx = np.random.randint(len(expert_data1))
    init1, goal1 = expert_data1[idx, 0], expert_data1[idx, -1]
    init2, goal2 = expert_data2[idx, 0], expert_data2[idx, -1]
    init3, goal3 = expert_data3[idx, 0], expert_data3[idx, -1]

    traj1 = sample_trajectory(G1, init1, goal1, T)
    np.save(f"sampled_trajs/magail/vary_init/traj1_{i}.npy", traj1)
    traj2 = sample_trajectory(G2, init2, goal2, T)
    np.save(f"sampled_trajs/magail/vary_init/traj2_{i}.npy", traj2)
    traj3 = sample_trajectory(G3, init3, goal3, T)
    np.save(f"sampled_trajs/magail/vary_init/traj3_{i}.npy", traj3)

    plt.plot(traj1[:,0], traj1[:,1], color='blue', alpha=0.7)
    plt.plot(traj2[:,0], traj2[:,1], color='orange', alpha=0.7)
    plt.plot(traj3[:,0], traj3[:,1], color='green', alpha=0.7)

# Overlay start & goal points for clarity
# (optional: comment out if not needed)
plt.scatter(expert_data1[:,0,0], expert_data1[:,0,1], marker='o', color='blue', s=20, label='Agent 1 start')
plt.scatter(expert_data2[:,0,0], expert_data2[:,0,1], marker='o', color='orange', s=20, label='Agent 2 start')
plt.scatter(expert_data1[:,-1,0], expert_data1[:,-1,1], marker='x', color='blue', s=40, label='Agent 1 goal')
plt.scatter(expert_data2[:,-1,0], expert_data2[:,-1,1], marker='x', color='orange', s=40, label='Agent 2 goal')
plt.scatter(expert_data3[:,0,0], expert_data3[:,0,1], marker='o', color='green', s=20, label='Agent 3 start')
plt.scatter(expert_data3[:,-1,0], expert_data3[:,-1,1], marker='x', color='green', s=40, label='Agent 3 goal')

plt.title("MAGAIL‐Generated Trajectories")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- pick one fixed context ---------------------
idx      = 0                                # choose any trajectory index
init1    = expert_data1[idx, 0];  goal1 = expert_data1[idx, -1]
init2    = expert_data2[idx, 0];  goal2 = expert_data2[idx, -1]
init3    = expert_data3[idx, 0];  goal3 = expert_data3[idx, -1]
T        = expert_data1.shape[1]

# --- sample N rollouts for each agent ----------
N = 100
rollouts1 = [sample_trajectory(G1, init1, goal1, T) for _ in range(N)]
rollouts2 = [sample_trajectory(G2, init2, goal2, T) for _ in range(N)]
rollouts3 = [sample_trajectory(G3, init3, goal3, T) for _ in range(N)]
for i in range(N):
    np.save(f"sampled_trajs/magail/static_init/traj1_{i}.npy", rollouts1[i])
    np.save(f"sampled_trajs/magail/static_init/traj2_{i}.npy", rollouts2[i])
    np.save(f"sampled_trajs/magail/static_init/traj3_{i}.npy", rollouts3[i])

plt.figure(figsize=(6,6))
for r in rollouts1:
    plt.plot(r[:,0], r[:,1], color='blue', alpha=0.3)
for r in rollouts2:
    plt.plot(r[:,0], r[:,1], color='orange', alpha=0.3)
for r in rollouts3:
    plt.plot(r[:,0], r[:,1], color='green', alpha=0.3)
plt.title("Mode Collapse Check (fixed start→goal)")
plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal'); plt.grid()
plt.show()
