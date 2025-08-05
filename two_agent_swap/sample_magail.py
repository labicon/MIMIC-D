import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1) Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Generator network (must match training script)
class GenNet(nn.Module):
    def __init__(self, state_dim=4, hidden_size=64, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,   hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    def forward(self, x):
        return self.net(x)

# 3) Instantiate & load weights
G1 = GenNet().to(device)
G2 = GenNet().to(device)
G1.load_state_dict(torch.load("trained_models/magail/G1.pth", map_location=device))
G2.load_state_dict(torch.load("trained_models/magail/G2.pth", map_location=device))
G1.eval(); G2.eval()

# 4) Load expert trajectories for normalization & conditioning
expert1 = np.load("data/expert_data1_100_traj.npy")  # shape (n_traj, T, 2)
expert2 = np.load("data/expert_data2_100_traj.npy")

def build_sa(expert):
    S, A = [], []
    for traj in expert:
        goal = traj[-1]
        for t in range(len(traj)-1):
            s = np.hstack([traj[t], goal])  # (4,)
            a = traj[t+1]                   # (2,)
            S.append(s); A.append(a)
    return np.stack(S), np.stack(A)

S1, A1 = build_sa(expert1)
S2, A2 = build_sa(expert2)

# 5) Compute normalization stats
state_data  = np.concatenate([S1, S2], axis=0)
action_data = np.concatenate([A1, A2], axis=0)
state_mean, state_std   = state_data.mean(0),  state_data.std(0)  + 1e-6
action_mean, action_std = action_data.mean(0), action_data.std(0) + 1e-6

# 6) Sampling function
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

# 7) Sample & plot on single axes
num_samples = 100
T = expert1.shape[1]

plt.figure(figsize=(8, 8))
for _ in range(num_samples):
    idx = np.random.randint(len(expert1))
    init1, goal1 = expert1[idx, 0], expert1[idx, -1]
    init2, goal2 = expert2[idx, 0], expert2[idx, -1]

    traj1 = sample_trajectory(G1, init1, goal1, T)
    traj2 = sample_trajectory(G2, init2, goal2, T)

    plt.plot(traj1[:,0], traj1[:,1], color='blue', alpha=0.7)
    plt.plot(traj2[:,0], traj2[:,1], color='orange', alpha=0.7)

# Overlay start & goal points for clarity
# (optional: comment out if not needed)
plt.scatter(expert1[:,0,0], expert1[:,0,1], marker='o', color='blue', s=20, label='Agent 1 start')
plt.scatter(expert2[:,0,0], expert2[:,0,1], marker='o', color='orange', s=20, label='Agent 2 start')
plt.scatter(expert1[:,-1,0], expert1[:,-1,1], marker='x', color='blue', s=40, label='Agent 1 goal')
plt.scatter(expert2[:,-1,0], expert2[:,-1,1], marker='x', color='orange', s=40, label='Agent 2 goal')

plt.title("MAGAIL‐Generated Trajectories")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

# --- pick one fixed context ---------------------
idx      = 0                                 # choose any trajectory index
init1    = expert1[idx, 0];  goal1 = expert1[idx, -1]
init2    = expert2[idx, 0];  goal2 = expert2[idx, -1]
T        = expert1.shape[1]

# --- sample N rollouts for each agent ----------
N = 100
rollouts1 = [sample_trajectory(G1, init1, goal1, T) for _ in range(N)]
rollouts2 = [sample_trajectory(G2, init2, goal2, T) for _ in range(N)]

# --- compute diversity metric -------------------
# pairwise L2 distance between trajectories
def traj_dist(a, b):
    return np.linalg.norm(a - b)

dists1 = [traj_dist(r1, r2) 
          for i, r1 in enumerate(rollouts1) 
          for r2 in rollouts1[i+1:]]
dists2 = [traj_dist(r1, r2) 
          for i, r1 in enumerate(rollouts2) 
          for r2 in rollouts2[i+1:]]

print(f"Agent 1 pairwise mean‒std‒dev: {np.std(dists1):.3e}")
print(f"Agent 2 pairwise mean‒std‒dev: {np.std(dists2):.3e}")

# --- plot all rollouts --------------------------------
plt.figure(figsize=(6,6))
for r in rollouts1:
    plt.plot(r[:,0], r[:,1], color='blue', alpha=0.3)
for r in rollouts2:
    plt.plot(r[:,0], r[:,1], color='orange', alpha=0.3)
plt.title("Mode Collapse Check (fixed start→goal)")
plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal'); plt.grid()
plt.show()
