# training_magail_ctde.py

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
seed = 10
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 2) Hyperparameters
state_dim   = 6       # e.g. [x, y, goal_x, goal_y]
action_dim  = 2       # e.g. next [x, y]
hidden_size = 64
batch_size  = 64
n_epochs    = 2000
lr_G        = 1e-4
lr_D        = 1e-4
GEN_WIDTH  = 2000   # or 1996 for ~8.0M total
DISC_WIDTH, DISC_DEPTH = 768, 8
def count_params(m): return sum(p.numel() for p in m.parameters())


# Define initial and final points, and a single central obstacle
initial_point1 = np.array([0.0, 0.0])
final_point1 = np.array([20.0, 0.0])
initial_point2 = np.array([20.0, 0.0])
final_point2 = np.array([0.0, 0.0])
obstacle = (10, 0, 4.0)  # Single central obstacle: (x, y, radius)

# 3) Models
class GenNet(nn.Module):
    def __init__(self, width=GEN_WIDTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim,   width), nn.ReLU(inplace=True),
            nn.Linear(width,       width), nn.ReLU(inplace=True),
            nn.Linear(width,       action_dim)
        )
    def forward(self, x): return self.net(x)

class DiscNet(nn.Module):
    def __init__(self, width=DISC_WIDTH, depth=DISC_DEPTH, spectral=True):
        super().__init__()
        in_dim = state_dim + action_dim  # 4 + 2 = 6
        def lin(i, o):
            layer = nn.Linear(i, o)
            return nn.utils.spectral_norm(layer) if spectral else layer
        layers = [lin(in_dim, width), nn.LeakyReLU(0.2, inplace=True)]
        for _ in range(depth - 1):
            layers += [lin(width, width), nn.LeakyReLU(0.2, inplace=True)]
        layers += [lin(width, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=1)).squeeze(1)

G1 = GenNet(width=GEN_WIDTH).to(device)
G2 = GenNet(width=GEN_WIDTH).to(device)
D  = DiscNet(width=DISC_WIDTH, depth=DISC_DEPTH, spectral=True).to(device)
print(f"G total params: {count_params(G1)+count_params(G2):,}")
print(f"D params:       {count_params(D):,}")

# optG = optim.Adam(list(G1.parameters()) + list(G2.parameters()), lr=lr_G)
# optD = optim.Adam(D.parameters(), lr=lr_D)
optG = optim.Adam(list(G1.parameters()) + list(G2.parameters()), lr=1e-4, betas=(0.5, 0.999))
optD = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))  # D a bit faster than G

bce_logits = nn.BCEWithLogitsLoss()

# 4) Load expert data & build (s,a) pairs
expert1 = np.load('data/expert_data1_100_traj.npy')  # shape (n_traj, T, 2)
expert2 = np.load('data/expert_data2_100_traj.npy')

def build_sa(expert, other):
    S, A = [], []
    for i in range(len(expert)):
        traj = expert[i]
        other_traj = other[i]
        for t in range(len(traj)-1):
            s = np.hstack([traj[t], traj[-1], other_traj[t]])  # (6,)
            a = traj[t+1]                   # (2,)
            S.append(s); A.append(a)
    return np.stack(S), np.stack(A)

S1, A1 = build_sa(expert1, expert2)
S2, A2 = build_sa(expert2, expert1)

# 5) Joint normalization for states & actions
state_data  = np.concatenate([S1, S2], axis=0)
action_data = np.concatenate([A1, A2], axis=0)

state_mean, state_std   = state_data.mean(0),  state_data.std(0)  + 1e-6
action_mean, action_std = action_data.mean(0), action_data.std(0) + 1e-6

S1 = (S1 - state_mean) / state_std
S2 = (S2 - state_mean) / state_std
A1 = (A1 - action_mean) / action_std
A2 = (A2 - action_mean) / action_std

# 6) Torch datasets & loaders for each agent
tS1 = torch.tensor(S1, dtype=torch.float32)
tA1 = torch.tensor(A1, dtype=torch.float32)
tS2 = torch.tensor(S2, dtype=torch.float32)
tA2 = torch.tensor(A2, dtype=torch.float32)

loader1 = DataLoader(TensorDataset(tS1, tA1),
                     batch_size=batch_size, shuffle=True, drop_last=True)
loader2 = DataLoader(TensorDataset(tS2, tA2),
                     batch_size=batch_size, shuffle=True, drop_last=True)

# 7) Training loop with centralized training structure
# for epoch in range(1, n_epochs+1):
#     G1.train(); G2.train(); D.train()
#     lossD_sum = 0.0
#     lossG_sum = 0.0

#     for (x1, a1), (x2, a2) in zip(loader1, loader2):
#         x1, a1 = x1.to(device), a1.to(device)
#         x2, a2 = x2.to(device), a2.to(device)

#         # ——— Discriminator update using both agents' data ———
#         with torch.no_grad():
#             fake1 = G1(x1)
#             fake2 = G2(x2)

#         real_logit1 = D(x1, a1)
#         fake_logit1 = D(x1, fake1)
#         real_logit2 = D(x2, a2)
#         fake_logit2 = D(x2, fake2)

#         lossD1 = 0.5*(bce_logits(real_logit1, torch.ones_like(real_logit1)) +
#                       bce_logits(fake_logit1, torch.zeros_like(fake_logit1)))
#         lossD2 = 0.5*(bce_logits(real_logit2, torch.ones_like(real_logit2)) +
#                       bce_logits(fake_logit2, torch.zeros_like(fake_logit2)))

#         lossD = lossD1 + lossD2
#         optD.zero_grad()
#         lossD.backward()
#         optD.step()

#         # ——— Generator update for each agent ———
#         fake1 = G1(x1)
#         fake2 = G2(x2)

#         lossG1 = bce_logits(D(x1, fake1), torch.ones_like(fake1[:,0]))
#         lossG2 = bce_logits(D(x2, fake2), torch.ones_like(fake2[:,0]))

#         lossG = lossG1 + lossG2
#         optG.zero_grad()
#         lossG.backward()
#         optG.step()

#         lossD_sum += lossD.item()
#         lossG_sum += lossG.item()

#     if epoch % 100 == 0:
#         avgD = lossD_sum / len(loader1)
#         avgG = lossG_sum / len(loader1)
#         print(f"Epoch {epoch:4d} | lossD: {avgD:.4f} | lossG: {avgG:.4f}")

# 8) Save models
save_path_G1 = "trained_models/magail/G1_extrainfo.pth"
save_path_G2 = "trained_models/magail/G2_extrainfo.pth"
save_path_D  = "trained_models/magail/D_extrainfo.pth"
# torch.save(G1.state_dict(), save_path_G1)
# torch.save(G2.state_dict(), save_path_G2)
# torch.save(D.state_dict(),  save_path_D)
print("CTDE MAGAIL training complete; models saved under trained_models/magail/")

# --------------------- reactive MPC sampling ----------------
# Generators for each agent
G1 = GenNet(width=GEN_WIDTH).to(device)
G2 = GenNet(width=GEN_WIDTH).to(device)
G1.load_state_dict(torch.load(save_path_G1, map_location=device))
G2.load_state_dict(torch.load(save_path_G2, map_location=device))
G1.eval(); G2.eval()

# 6) Sampling function
# def sample_trajectory(G, init_pos, init_pos2, horizon):
#     pos = init_pos.copy()
#     pos2 = init_pos2.copy()
#     traj = [pos.copy()]
#     for _ in range(horizon-1):
#         s = np.hstack([pos, pos2])
#         s_norm = (s - state_mean) / state_std
#         s_t = torch.tensor(s_norm, dtype=torch.float32, device=device).unsqueeze(0)
#         with torch.no_grad():
#             a_norm = G(s_t).cpu().numpy()[0]
#         a = a_norm * action_std + action_mean
#         pos = a.copy()
#         traj.append(pos.copy())
#     return np.stack(traj)
@torch.no_grad()
def rollout_two_agents(G1, G2, init1, init2, final1, final2, T, state_mean, state_std, action_mean, action_std, device):
    cur1 = init1.copy(); cur2 = init2.copy()
    traj1 = [cur1.copy()]; traj2 = [cur2.copy()]
    for _ in range(T-1):
        s1 = np.hstack([cur1, final1, cur2])  # other = current, not initial
        s2 = np.hstack([cur2, final2, cur1])
        s1n = (s1 - state_mean) / state_std
        s2n = (s2 - state_mean) / state_std
        a1n = G1(torch.tensor(s1n, dtype=torch.float32, device=device).unsqueeze(0)).cpu().numpy()[0]
        a2n = G2(torch.tensor(s2n, dtype=torch.float32, device=device).unsqueeze(0)).cpu().numpy()[0]
        next1 = a1n * action_std + action_mean     # absolute next state
        next2 = a2n * action_std + action_mean
        traj1.append(next1.copy()); traj2.append(next2.copy())
        cur1, cur2 = next1, next2
    return np.stack(traj1), np.stack(traj2)


# 7) Sample & plot on single axes
num_samples = 100
T = expert1.shape[1]
noise_std = 0.4

# plt.figure(figsize=(8, 8))
for s in range(10):
    seed = s * 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    path_vary = f"sampled_trajs/magail_extrainfo_seed{seed}/vary_init"
    path_static = f"sampled_trajs/magail_extrainfo_seed{seed}/static_init"
    os.makedirs(path_vary, exist_ok=True)
    os.makedirs(path_static, exist_ok=True)

    for i in range(num_samples):
        idx = np.random.randint(len(expert1))
        # init1, goal1 = expert1[idx, 0], expert1[idx, -1]
        # init2, goal2 = expert2[idx, 0], expert2[idx, -1]
        init1 = initial_point1 + noise_std * np.random.randn(2)
        final1 = final_point1 + noise_std * np.random.randn(2)
        init2 = initial_point2 + noise_std * np.random.randn(2)
        final2 = final_point2 + noise_std * np.random.randn(2)

        traj1, traj2 = rollout_two_agents(G1, G2, init1, init2, final1, final2, T, state_mean, state_std, action_mean, action_std, device)

        np.save(os.path.join(path_vary, f"mpc_traj1_{i}.npy"), traj1)
        np.save(os.path.join(path_vary, f"mpc_traj2_{i}.npy"), traj2)

        # np.save(f"sampled_trajs/magail_extrainfo_seed{seed}/vary_init/mpc_traj1_{i}.npy", traj1)
        # np.save(f"sampled_trajs/magail_extrainfo_seed{seed}/vary_init/mpc_traj2_{i}.npy", traj2)

        # plt.plot(traj1[:,0], traj1[:,1], color='blue', alpha=0.7)
        # plt.plot(traj2[:,0], traj2[:,1], color='orange', alpha=0.7)

    # plt.title("MAGAIL‐Generated Trajectories")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # plt.legend()
    # plt.axis('equal')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # --- pick one fixed context ---------------------
    idx      = 0                                 # choose any trajectory index
    init1    = expert1[idx, 0];  goal1 = expert1[idx, -1]
    init2    = expert2[idx, 0];  goal2 = expert2[idx, -1]
    T        = expert1.shape[1]

    # --- sample N rollouts for each agent ----------
    N = 100
    rollouts1, rollouts2 = [], []
    for _ in range(N):
        traj1, traj2 = rollout_two_agents(G1, G2, init1, init2, goal1, goal2, T, state_mean, state_std, action_mean, action_std, device)
        rollouts1.append(traj1)
        rollouts2.append(traj2)
    for i in range(N):
        np.save(os.path.join(path_static, f"mpc_traj1_{i}.npy"), rollouts1[i])
        np.save(os.path.join(path_static, f"mpc_traj2_{i}.npy"), rollouts2[i])

    # --- plot all rollouts --------------------------------
    # plt.figure(figsize=(6,6))
    # for r in rollouts1:
    #     plt.plot(r[:,0], r[:,1], color='blue', alpha=0.3)
    # for r in rollouts2:
    #     plt.plot(r[:,0], r[:,1], color='orange', alpha=0.3)
    # plt.title("Mode Collapse Check (fixed start→goal)")
    # plt.xlabel("x"); plt.ylabel("y"); plt.axis('equal'); plt.grid()
    # plt.show()
